"""Stage 8: figures (PNG at 300 dpi + PDF). Drawn only from measured tables.

Colour rules: one fixed colour and marker per series (colour never carries meaning alone),
a diverging blue-red scale with a neutral midpoint for correlations, one axis per plot.
Figures produced from synthetic smoke-test data are watermarked so they can never be
mistaken for a result.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .collect import load_table
from .config import Paths
from .stats import bootstrap_ci
from .utils import log

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
DIVERGING = LinearSegmentedColormap.from_list("bwr_neutral", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b3261e"])
INDEX_ORDER = ["flat", "hnsw", "ivf", "ivfpq"]
INDEX_LABEL = {"flat": "Flat (exact)", "hnsw": "HNSW", "ivf": "IVF-Flat", "ivfpq": "IVF-PQ"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.8, "lines.markersize": 5.5, "figure.dpi": 110, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def _save(fig, paths: Paths, name: str, synthetic: bool) -> None:
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC SMOKE TEST - NOT A RESULT", rotation=24, ha="center", va="center",
                 fontsize=22, color="#d03b3b", alpha=0.22, weight="bold", zorder=100)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(paths.figures, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    log.info(f"[figures] {name}")


def fig_recall_qps(bench: pd.DataFrame, paths, synthetic, latency_k: int, max_panels: int = 6,
                   metric: str = "qps_1t", ylabel: str = "queries / s (1 thread)", name: str = "recall_qps_native") -> None:
    """Recall-cost trade-off of each index in the native space of several backbones."""
    df = bench[(bench["k"] == latency_k) & (bench["reduction"] == "native")]
    models = list(dict.fromkeys(df["model"]))[:max_panels]
    if not models:
        return
    keys = [k for k in dict.fromkeys(df.sort_values(["index", "build_key"])["build_key"]) if k != "flat"][:8]
    ncol = min(3, len(models))
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.9 * nrow), squeeze=False, sharey=False)
    for ax, model in zip(axes.ravel(), models):
        g = df[df["model"] == model]
        for i, key in enumerate(keys):
            c = g[g["build_key"] == key].groupby("search_value")[["recall", metric]].mean().sort_index()
            if len(c):
                ax.plot(c["recall"], c[metric], color=SERIES[i], marker=MARKERS[i], label=key)
        flat = g[g["build_key"] == "flat"][metric]
        if len(flat):
            ax.axhline(flat.mean(), color=INK2, lw=1, ls="--")
            ax.annotate("exact search", (ax.get_xlim()[0], flat.mean()), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, color=INK2)
        ax.set_yscale("log")
        ax.set_title(model, loc="left")
        ax.set_xlabel(f"recall@{latency_k}")
        ax.set_ylabel(ylabel)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    h, lab = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(4, len(lab)), bbox_to_anchor=(0.5, -0.02 - 0.03 * nrow))
    fig.tight_layout()
    _save(fig, paths, name, synthetic)


def fig_geometry_heatmap(corr: pd.DataFrame, paths, synthetic, name: str = "geometry_vs_cost",
                         cost_label: str = "log QPS") -> None:
    """Which geometry property goes with search cost, per index type."""
    if not len(corr):
        return
    kinds = [k for k in INDEX_ORDER if k in set(corr["index"])]
    piv = corr.pivot(index="feature", columns="index", values="spearman").reindex(columns=kinds)
    q = corr.pivot(index="feature", columns="index", values="q_fdr").reindex(columns=kinds)
    order = piv.abs().max(1).sort_values(ascending=False).index
    piv, q = piv.loc[order], q.loc[order]
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(kinds), 0.3 * len(piv) + 1.2))
    im = ax.imshow(piv.values, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(kinds)), [INDEX_LABEL[k] for k in kinds])
    ax.set_yticks(range(len(piv)), piv.index)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                star = "*" if np.isfinite(q.values[i, j]) and q.values[i, j] < 0.05 else ""
                ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(f"Spearman correlation with {cost_label}")
    ax.set_title("Geometry feature vs search efficiency at the recall target\n(* = FDR < 0.05)", loc="left")
    _save(fig, paths, name, synthetic)


def fig_reduction_recall(qual: pd.DataFrame, paths, synthetic) -> None:
    """How much of the native neighbourhood survives each reduction method, by dimension."""
    red = qual[qual["reduction"] != "native"]
    if "reduction_recall@10" not in red.columns or not len(red):
        return
    methods = sorted(red["reduction"].unique())[:8]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, m in enumerate(methods):
        g = red[red["reduction"] == m].groupby("dim")["reduction_recall@10"]
        dims = sorted(g.groups)
        mean = np.array([g.get_group(d).mean() for d in dims])
        ci = np.array([bootstrap_ci(g.get_group(d).values) for d in dims])
        ax.plot(dims, mean, color=SERIES[i], marker=MARKERS[i], label=m)
        if np.isfinite(ci).all():
            ax.errorbar(dims, mean, yerr=[mean - ci[:, 0], ci[:, 1] - mean], color=SERIES[i], lw=0,
                        elinewidth=1, capsize=2.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(red["dim"].unique()), [str(d) for d in sorted(red["dim"].unique())])
    ax.set_xlabel("dimension after reduction")
    ax.set_ylabel("recall@10 vs native space")
    ax.set_title("Neighbourhood kept after reduction, exact search\n(mean over backbones, 95% bootstrap interval)",
                 loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    _save(fig, paths, "reduction_recall_vs_dim", synthetic)


def fig_pareto(op: pd.DataFrame, paths, synthetic, basis: str, target: float) -> None:
    """Memory per vector against throughput at the end-to-end recall target."""
    m = op[(op["basis"] == basis) & np.isclose(op["target"], target) & op["qps_1t"].notna()]
    if not len(m):
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for i, kind in enumerate([k for k in INDEX_ORDER if k in set(m["index"])]):
        g = m[m["index"] == kind]
        ax.scatter(g["bytes_per_vec"], g["qps_1t"], s=26, color=SERIES[i], marker=MARKERS[i],
                   edgecolor="white", linewidth=0.6, label=INDEX_LABEL[kind], alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("index size (bytes per vector)")
    ax.set_ylabel("queries / s (1 thread)")
    ax.set_title(f"Memory vs throughput at {basis.replace('_', ' ')}@10 >= {target:g}", loc="left")
    ax.legend(loc="best")
    _save(fig, paths, "pareto_memory_qps", synthetic)


def fig_recommender(rec: pd.DataFrame, paths, synthetic) -> None:
    methods = [("ours", "Geometry surrogate (ours)"), ("prior_only", "Surrogate without geometry"),
               ("nn_dataset", "Nearest backbone"), ("global_best", "Global best config"),
               ("random", "Random config")]
    methods = [m for m in methods if m[0] in rec.columns]
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    rng = np.random.default_rng(0)
    for i, (key, label) in enumerate(methods):
        v = rec[key].values
        ax.barh(i, v.mean(), height=0.55, color=SERIES[0] if key == "ours" else "#b9b8b2")
        ax.scatter(v, i + rng.uniform(-0.15, 0.15, len(v)), s=10, color=INK, alpha=0.55, zorder=3, clip_on=False)
        ax.text(1.09, i, f"{v.mean():.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(methods)), [m[1] for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("throughput of chosen config / oracle (1 = grid search)")
    ax.set_title("Leave-one-family-out recommendation\n(bar = mean, dots = held-out backbones)", loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, paths, "recommender", synthetic)


def fig_feature_scatter(op, geom, corr, paths, synthetic, target: float) -> None:
    """The three features most associated with HNSW throughput, shown as raw points."""
    if not len(corr) or "hnsw" not in set(corr["index"]):
        return
    top = corr[corr["index"] == "hnsw"].assign(a=lambda d: d["spearman"].abs()).nlargest(3, "a")["feature"].tolist()
    m = op[(op["basis"] == "recall") & np.isclose(op["target"], target) & (op["index"] == "hnsw") & op["qps_1t"].notna()]
    m = m.loc[m.groupby("space_id")["qps_1t"].idxmax()].merge(geom[["space_id"] + top], on="space_id")
    if not len(m):
        return
    fig, axes = plt.subplots(1, len(top), figsize=(3.3 * len(top), 2.9), squeeze=False)
    for ax, f in zip(axes[0], top):
        for i, (lab, sel) in enumerate((("native", m["reduction"] == "native"), ("reduced", m["reduction"] != "native"))):
            ax.scatter(m.loc[sel, f], m.loc[sel, "qps_1t"], s=24, color=SERIES[i], marker=MARKERS[i],
                       edgecolor="white", linewidth=0.6, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel(f)
        ax.set_ylabel("HNSW queries / s (1 thread)")
    axes[0][0].legend(loc="best")
    fig.suptitle(f"HNSW throughput at recall@10 >= {target:g} vs the three most associated geometry features",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, paths, "feature_scatter_hnsw", synthetic)


def run_figures(cfg: dict) -> None:
    paths = Paths(cfg)
    syn = bool(cfg["synthetic"])
    rc = cfg["recommender"]

    def maybe(name):
        try:
            return load_table(paths, name)
        except FileNotFoundError:
            return pd.DataFrame()

    bench, op, geom, qual = maybe("bench"), maybe("operating_points"), maybe("geometry"), maybe("quality")
    corr, rec = maybe("stats_geometry_vs_cost"), maybe("recommender")
    if len(bench):
        fig_recall_qps(bench, paths, syn, int(cfg["bench"]["latency_k"]))
        if "ndis" in bench.columns:
            fig_recall_qps(bench, paths, syn, int(cfg["bench"]["latency_k"]), metric="ndis",
                           ylabel="distance evaluations / query", name="recall_ndis_native")
    if len(corr):
        fig_geometry_heatmap(corr, paths, syn)
    corr_n = maybe("stats_geometry_vs_cost_ndis")
    if len(corr_n):
        fig_geometry_heatmap(corr_n, paths, syn, name="geometry_vs_cost_ndis",
                             cost_label="log(1 / distance evaluations)")
    if len(qual):
        fig_reduction_recall(qual, paths, syn)
    if len(op):
        fig_pareto(op, paths, syn, "e2e_recall", float(rc["target"]))
    if len(rec):
        fig_recommender(rec, paths, syn)
    if len(op) and len(geom) and len(corr):
        fig_feature_scatter(op, geom, corr, paths, syn, float(rc["target"]))
