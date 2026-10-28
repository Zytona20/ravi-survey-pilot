"""Stage `survey`: the measurements behind the empirical section of the survey paper.

Every number is computed from this run's tables and embeddings; nothing is estimated or filled in.

- Geometry cards (the template of Table 9 in the survey), one per backbone.
- P1: at equal dimension, do query difficulty measures (local intrinsic dimensionality, inverse contrast)
  rank the backbones by search work (distance evaluations per query at a fixed recall)?
- P3: how does search work (distance evaluations x dimension) at a fixed end-to-end recall change as the
  dimension falls, and where is its minimum?
- P4 / P5: at the same dimension, what do the hubness and KoLeo heads change relative to the plain
  neighbour-ranking head: hubness, intrinsic dimensionality, contrast, result quality and search work?
- Compression profile of each backbone: end-to-end recall of exact search after PCA and scalar quantization.

Outputs: results/survey_*.csv, results/survey_geometry_cards.tex, results/survey_report.md, figures/survey_*.png
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

from .collect import load_table, save_table
from .config import Paths, model_specs, native_id
from .groundtruth import load_gt
from .metrics import recall_at_k
from .utils import log, read_json

CARD_FIELDS = [  # (column, label in the LaTeX card, number format)
    ("native_dim", "Native dimension", "{:.0f}"),
    ("pca_pr", "Participation ratio (global ID)", "{:.1f}"),
    ("twonn_id", "TwoNN intrinsic dimension", "{:.1f}"),
    ("lid_median", "LID, median (database)", "{:.1f}"),
    ("lid_p90", "LID, 90th percentile (database)", "{:.1f}"),
    ("qlid_median", "LID, median (queries)", "{:.1f}"),
    ("rc_q10", "Relative contrast of queries, $k=10$", "{:.3f}"),
    ("hub_skew", "Hubness: skewness of $O_{10}$", "{:.2f}"),
    ("hub_antihub", "Share of antihubs", "{:.3f}"),
    ("aniso_meancos", "Mean cosine of random pairs", "{:.3f}"),
    ("aniso_top1", "Share of largest eigenvalue", "{:.3f}"),
    ("clus_qerr", "Relative $k$-means error", "{:.3f}"),
    ("clus_imbalance", "Cluster-size imbalance", "{:.2f}"),
    ("qshift_nn", "Query--base shift ratio", "{:.3f}"),
    ("rr10_pca256", "End-to-end recall@10, PCA 256", "{:.3f}"),
    ("rr10_pca128", "End-to-end recall@10, PCA 128", "{:.3f}"),
    ("rr10_pca64", "End-to-end recall@10, PCA 64", "{:.3f}"),
    ("rr10_sq8", "End-to-end recall@10, 8-bit SQ", "{:.3f}"),
    ("rr10_sq4", "End-to-end recall@10, 4-bit SQ", "{:.3f}"),
    ("rr10_bin1", "End-to-end recall@10, 1-bit", "{:.3f}"),
    ("hnsw_ndis", "HNSW distance evaluations at recall@10 $=0.9$", "{:,.0f}"),
    ("ivf_ndis", "IVF distance evaluations at recall@10 $=0.9$", "{:,.0f}"),
    ("label_p10", "Class precision@10 (exact search)", "{:.3f}"),
]


def _is_test_run(cfg: dict) -> bool:
    return bool(cfg["synthetic"]) or not bool(cfg["extract"].get("pretrained", True))


def _maybe(paths, name):
    try:
        return load_table(paths, name)
    except FileNotFoundError:
        return pd.DataFrame()


def spearman_exact(x, y) -> tuple[int, float, float]:
    """Spearman's rho with a two-sided permutation p-value (exact for n <= 8, else 20,000 permutations)."""
    from scipy.stats import rankdata

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return n, float("nan"), float("nan")
    rx, ry = rankdata(x), rankdata(y)

    def rho(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    r = rho(rx, ry)
    if n <= 8:
        perms = itertools.permutations(ry)
        null = np.array([rho(rx, np.asarray(p)) for p in perms])
    else:
        rng = np.random.default_rng(0)
        null = np.array([rho(rx, rng.permutation(ry)) for _ in range(20000)])
    p = float((np.abs(null) >= abs(r) - 1e-12).mean())
    return n, r, p


def best_ndis(ops: pd.DataFrame, basis: str, target: float, index: str) -> pd.DataFrame:
    """Per space: fewest distance evaluations per query that reach `target` (best build setting per seed),
    then mean, std and count over seeds."""
    if not len(ops) or "ndis" not in ops:
        return pd.DataFrame(columns=["space_id", "ndis", "ndis_std", "n_seeds"])
    m = ops[(ops["basis"] == basis) & np.isclose(ops["target"], target) & (ops["index"] == index) & ops["ndis"].notna()]
    if not len(m):
        return pd.DataFrame(columns=["space_id", "ndis", "ndis_std", "n_seeds"])
    per_seed = m.groupby(["space_id", "seed"])["ndis"].min().reset_index()
    out = per_seed.groupby("space_id")["ndis"].agg(ndis="mean", ndis_std="std", n_seeds="count").reset_index()
    return out


def compression_profile(paths: Paths, model: str, k: int = 10, train_n: int = 50000, seed: int = 0) -> dict:
    """End-to-end recall@k of exact search on scalar-quantized and 1-bit codes of the native embeddings,
    against exact search on the float vectors (the run's native ground truth)."""
    import faiss

    from .data import load_native

    xb, xq, _, _ = load_native(paths, model)
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    gt = load_gt(paths, native_id(model))[:, :k]
    d = xb.shape[1]
    rng = np.random.default_rng(seed)
    tr = xb[np.sort(rng.choice(len(xb), min(len(xb), train_n), replace=False))]
    out = {}
    for name, qt in (("sq8", faiss.ScalarQuantizer.QT_8bit), ("sq4", faiss.ScalarQuantizer.QT_4bit)):
        index = faiss.IndexScalarQuantizer(d, qt, faiss.METRIC_INNER_PRODUCT)
        index.train(tr)
        index.add(xb)
        _, found = index.search(xq, k)
        out[f"rr10_{name}"] = recall_at_k(found, gt, k)
    mu = xb.mean(0, keepdims=True)  # centre first: post-ReLU features are all non-negative
    bits_b = np.packbits(xb - mu > 0, axis=1)
    bits_q = np.packbits(xq - mu > 0, axis=1)
    bindex = faiss.IndexBinaryFlat(bits_b.shape[1] * 8)
    bindex.add(bits_b)
    _, found = bindex.search(bits_q, k)
    out["rr10_bin1"] = recall_at_k(found, gt, k)
    return out


def _space_tag(reduction: str, dim) -> str:
    return "native" if reduction == "native" else f"{reduction}{int(dim)}"


def geometry_cards(cfg, paths, geom, qual, ops) -> pd.DataFrame:
    rows = []
    hnsw = best_ndis(ops, "recall", 0.90, "hnsw").set_index("space_id")
    ivf = best_ndis(ops, "recall", 0.90, "ivf").set_index("space_id")
    for spec in model_specs(cfg):
        model = spec["name"]
        sid = native_id(model)
        g = geom[geom["space_id"] == sid]
        if not len(g):
            continue
        row = {"model": model, "paradigm": spec.get("paradigm", "na")}
        row.update({c: g.iloc[0].get(c, np.nan) for c, _, _ in CARD_FIELDS if c in g.columns})
        row["native_dim"] = int(g.iloc[0]["dim"])
        q = qual[qual["model"] == model] if len(qual) else qual
        for d in (256, 128, 64):
            r = q[(q["reduction"] == "pca") & (q["dim"] == d)] if len(q) else q
            row[f"rr10_pca{d}"] = float(r.iloc[0]["reduction_recall@10"]) if len(r) else np.nan
        nat = q[q["reduction"] == "native"] if len(q) else q
        row["label_p10"] = float(nat.iloc[0]["label_p@10"]) if len(nat) and "label_p@10" in nat else np.nan
        try:
            row.update(compression_profile(paths, model, seed=int(cfg["seed"])))
        except Exception as e:  # never lose the rest of the card
            log.info(f"[survey] compression profile of {model} failed: {e}")
        row["hnsw_ndis"] = float(hnsw.loc[sid, "ndis"]) if sid in hnsw.index else np.nan
        row["hnsw_ndis_std"] = float(hnsw.loc[sid, "ndis_std"]) if sid in hnsw.index else np.nan
        row["ivf_ndis"] = float(ivf.loc[sid, "ndis"]) if sid in ivf.index else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def cards_latex(cards: pd.DataFrame) -> str:
    """The cards as one LaTeX table: fields as rows, backbones as columns."""
    cols = list(cards["model"])
    short = [m.split(".")[0].replace("_", "\\_") for m in cols]
    lines = ["\\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", "\\toprule",
             "\\textbf{Field} & " + " & ".join(f"\\textbf{{{s}}}" for s in short) + "\\\\", "\\midrule"]
    for col, label, fmt in CARD_FIELDS:
        if col not in cards:
            continue
        vals = []
        for v in cards[col]:
            vals.append("--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v))
        lines.append(f"{label} & " + " & ".join(vals) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def p1_table(cfg, geom, ops) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per backbone and space type: difficulty measures and search work; then rank correlations."""
    dim = int(cfg["survey"]["card_dim"])
    hnsw = best_ndis(ops, "recall", 0.90, "hnsw").rename(columns={"ndis": "hnsw_ndis", "ndis_std": "hnsw_ndis_std"})
    ivf = best_ndis(ops, "recall", 0.90, "ivf").rename(columns={"ndis": "ivf_ndis", "ndis_std": "ivf_ndis_std"})
    rows, corr = [], []
    for space, sel in (("pca%d" % dim, (geom["reduction"] == "pca") & (geom["dim"] == dim)),
                       ("native", geom["reduction"] == "native")):
        g = geom[sel][["space_id", "model", "paradigm", "dim"] + [c for c in ("qlid_median", "lid_median", "rc_q10",
                                                                              "hub_skew") if c in geom]]
        g = g.merge(hnsw[["space_id", "hnsw_ndis", "hnsw_ndis_std"]], on="space_id", how="left")
        g = g.merge(ivf[["space_id", "ivf_ndis", "ivf_ndis_std"]], on="space_id", how="left")
        g["space"] = space
        if "rc_q10" in g:
            g["inv_rc_q10"] = 1.0 / g["rc_q10"]
        rows.append(g)
        for pred in ("qlid_median", "lid_median", "inv_rc_q10"):
            for cost in ("hnsw_ndis", "ivf_ndis"):
                if pred in g and cost in g:
                    n, r, p = spearman_exact(g[pred], g[cost])
                    corr.append({"space": space, "predictor": pred, "cost": cost, "n_backbones": n,
                                 "spearman": r, "p_perm": p})
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(corr)


def p3_table(cfg, qual, ops) -> pd.DataFrame:
    rows = []
    q = qual[qual["reduction"].isin(["native", "pca"])] if len(qual) else qual
    for t in cfg["survey"]["targets"]:
        for kind in ("hnsw", "ivf"):
            b = best_ndis(ops, "e2e_recall", float(t), kind).set_index("space_id")
            for _, r in q.iterrows():
                sid = r["space_id"]
                ndis = float(b.loc[sid, "ndis"]) if sid in b.index else np.nan
                rows.append({"model": r["model"], "index": kind, "target": float(t), "space": _space_tag(r["reduction"], r["dim"]),
                             "dim": int(r["dim"]), "ceiling_recall10": float(r.get("reduction_recall@10", np.nan))
                             if r["reduction"] != "native" else 1.0,
                             "reaches_target": np.isfinite(ndis), "ndis": ndis, "work": ndis * int(r["dim"])})
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    return df


def p3_summary(p3: pd.DataFrame) -> pd.DataFrame:
    """Per backbone, index and target: where search work is lowest, and what happens below that dimension.

    rises_below: a smaller dimension still reaches the target but needs more work (the interior optimum of P3);
    ceiling_below: a smaller dimension cannot reach the target at all (its exact-search ceiling is too low)."""
    rows = []
    if not len(p3):
        return pd.DataFrame()
    for (model, kind, t), g in p3.groupby(["model", "index", "target"]):
        ok = g[g["reaches_target"]].sort_values("dim")
        row = {"model": model, "index": kind, "target": t, "dims_tested": ",".join(str(d) for d in sorted(g["dim"]))}
        if not len(ok):
            rows.append({**row, "argmin_dim": np.nan, "smallest_feasible_dim": np.nan, "rises_below": False,
                         "ceiling_below": True, "work_native_over_min": np.nan})
            continue
        dmin = int(ok.loc[ok["work"].idxmin(), "dim"])
        below = g[g["dim"] < dmin]
        rows.append({**row, "argmin_dim": dmin, "smallest_feasible_dim": int(ok["dim"].min()),
                     "rises_below": bool(below["reaches_target"].any()),
                     "ceiling_below": bool((~below["reaches_target"]).any()),
                     "work_native_over_min": float(ok["work"].iloc[-1] / ok["work"].min())})
    return pd.DataFrame(rows)


def heads_table(cfg, geom, qual, ops) -> pd.DataFrame:
    dim = int(cfg["survey"]["card_dim"])
    reds = ["pca"] + [h for h in cfg["survey"]["heads"] if h in cfg["reductions"]]
    if len(reds) < 2:
        return pd.DataFrame()
    hnsw = best_ndis(ops, "recall", 0.90, "hnsw").rename(columns={"ndis": "hnsw_ndis"})
    hnsw_e2e = best_ndis(ops, "e2e_recall", 0.90, "hnsw").rename(columns={"ndis": "hnsw_e2e_ndis"})
    g = geom[(geom["reduction"].isin(reds)) & (geom["dim"] == dim)]
    cols = [c for c in ("hub_skew", "lid_median", "qlid_median", "rc_q10", "aniso_meancos") if c in g]
    g = g[["space_id", "model", "reduction"] + cols]
    if len(qual):
        qc = [c for c in ("reduction_recall@10", "label_p@10") if c in qual]
        g = g.merge(qual[["space_id"] + qc], on="space_id", how="left")
    g = g.merge(hnsw[["space_id", "hnsw_ndis"]], on="space_id", how="left")
    g = g.merge(hnsw_e2e[["space_id", "hnsw_e2e_ndis"]], on="space_id", how="left")
    return g.sort_values(["model", "reduction"]).reset_index(drop=True)


def heads_summary(h: pd.DataFrame) -> pd.DataFrame:
    """Change of each head relative to head_rank, per backbone, and how many backbones agree in sign."""
    if not len(h) or "head_rank" not in set(h["reduction"]):
        return pd.DataFrame()
    out = []
    metrics = [c for c in ("hub_skew", "qlid_median", "rc_q10", "label_p@10", "reduction_recall@10", "hnsw_ndis")
               if c in h]
    for red in sorted(set(h["reduction"]) - {"head_rank"}):
        for m in metrics:
            deltas = []
            for model, g in h.groupby("model"):
                a = g[g["reduction"] == red][m]
                b = g[g["reduction"] == "head_rank"][m]
                if len(a) and len(b) and np.isfinite(a.iloc[0]) and np.isfinite(b.iloc[0]):
                    deltas.append(float(a.iloc[0] - b.iloc[0]))
            if deltas:
                d = np.asarray(deltas)
                out.append({"reduction": red, "metric": m, "n_backbones": len(d), "mean_change": float(d.mean()),
                            "n_increase": int((d > 0).sum()), "n_decrease": int((d < 0).sum())})
    return pd.DataFrame(out)


def _figures(cfg, paths, p1, p3, test_run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .figures import INK2, MARKERS, SERIES

    def save(fig, name):
        if test_run:
            fig.text(0.5, 0.5, "TEST RUN - NOT A RESULT", rotation=24, ha="center", va="center", fontsize=22,
                     color="#d03b3b", alpha=0.22, weight="bold", zorder=100)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(paths.figures, f"{name}.{ext}"), dpi=300)
        plt.close(fig)

    dim = int(cfg["survey"]["card_dim"])
    g = p1[p1["space"] == f"pca{dim}"] if len(p1) else p1
    if len(g) and g["hnsw_ndis"].notna().any() and "qlid_median" in g:
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for i, (_, r) in enumerate(g.iterrows()):
            ax.scatter(r["qlid_median"], r["hnsw_ndis"], color=SERIES[i % 8], marker=MARKERS[i % 8], s=40)
            ax.annotate(str(r["model"]).split(".")[0], (r["qlid_median"], r["hnsw_ndis"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=INK2)
        ax.set_xlabel("median local intrinsic dimension of queries")
        ax.set_ylabel("HNSW distance evaluations / query\nat recall@10 = 0.9")
        ax.set_title(f"P1: all backbones reduced to {dim} dimensions (PCA)", loc="left", fontsize=9)
        save(fig, "survey_p1")
    t = 0.9 if len(p3) and (p3["target"] == 0.9).any() else (p3["target"].max() if len(p3) else None)
    d = p3[(p3["index"] == "hnsw") & (p3["target"] == t)] if t is not None else p3
    if len(d) and d["reaches_target"].any():
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        for i, (model, gm) in enumerate(d.groupby("model")):
            gm = gm.sort_values("dim")
            ok = gm[gm["reaches_target"]]
            ax.plot(ok["dim"], ok["work"], color=SERIES[i % 8], marker=MARKERS[i % 8], label=str(model).split(".")[0])
            miss = gm[~gm["reaches_target"]]
            if len(miss):
                ax.scatter(miss["dim"], [ok["work"].max() if len(ok) else 1] * len(miss), marker="x",
                           color=SERIES[i % 8], s=30)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("dimension (PCA; rightmost = native)")
        ax.set_ylabel("distance evaluations x dimension\nper query (HNSW)")
        ax.set_title(f"P3: search work at end-to-end recall@10 = {t:g} (x: target not reached)", loc="left", fontsize=8)
        ax.legend(fontsize=6.5, ncol=2)
        save(fig, "survey_p3")


def _md(df: pd.DataFrame, floatfmt="{:.3g}") -> str:
    if df is None or not len(df):
        return "_(no data)_\n"
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def run_survey(cfg: dict) -> None:
    paths = Paths(cfg)
    geom, qual, ops = _maybe(paths, "geometry"), _maybe(paths, "quality"), _maybe(paths, "operating_points_seeds")
    if not len(geom) or not len(ops):
        log.info("[survey] geometry or operating points missing: run the earlier stages first")
        return
    test_run = _is_test_run(cfg)
    cards = geometry_cards(cfg, paths, geom, qual, ops)
    save_table(cards, os.path.join(paths.results, "survey_geometry_cards"))
    with open(os.path.join(paths.results, "survey_geometry_cards.tex"), "w") as f:
        if test_run:
            f.write("% TEST RUN: random weights or synthetic data. These numbers are not results.\n")
        f.write(cards_latex(cards))
    p1, p1c = p1_table(cfg, geom, ops)
    save_table(p1, os.path.join(paths.results, "survey_p1"))
    save_table(p1c, os.path.join(paths.results, "survey_p1_correlations"))
    p3 = p3_table(cfg, qual, ops)
    p3s = p3_summary(p3)
    save_table(p3, os.path.join(paths.results, "survey_p3"))
    save_table(p3s, os.path.join(paths.results, "survey_p3_summary"))
    heads = heads_table(cfg, geom, qual, ops)
    hsum = heads_summary(heads)
    save_table(heads, os.path.join(paths.results, "survey_p4p5_heads"))
    save_table(hsum, os.path.join(paths.results, "survey_p4p5_summary"))
    try:
        _figures(cfg, paths, p1, p3, test_run)
    except Exception as e:
        log.info(f"[survey] figures failed: {e}")

    out = []
    if test_run:
        out.append("> **TEST RUN (random weights or synthetic data). These numbers are not results.**\n")
    out.append(f"# Survey measurements: {cfg['name']}\n")
    env_path = os.path.join(paths.results, "env.json")
    if os.path.exists(env_path):
        env = read_json(env_path)
        out.append("Benchmark machine: " + ", ".join(f"{k}={env.get(k)}" for k in ("cpu_model", "cpu_count", "faiss")
                                                       if k in env) + "\n")
    ds = cfg.get("dataset") or {}
    out.append(f"Dataset: {ds.get('name', 'synthetic')}; backbones: {len(cards)}; seeds: {cfg['bench']['seeds']}; "
               f"HNSW M={cfg['bench']['hnsw']['M']}, efConstruction={cfg['bench']['hnsw']['efConstruction']}; "
               f"IVF nlist={cfg['bench']['ivf']['nlist']}.\n")
    out.append("## Geometry cards (native space of each backbone)\n")
    show = cards[[c for c in ["model", "native_dim", "qlid_median", "rc_q10", "hub_skew", "aniso_meancos",
                              "qshift_nn", "rr10_pca256", "rr10_sq8", "rr10_bin1", "hnsw_ndis"] if c in cards]]
    out.append(_md(show))
    out.append("Full card: results/survey_geometry_cards.csv; LaTeX: results/survey_geometry_cards.tex\n")
    out.append(f"## P1: query difficulty and search work at {cfg['survey']['card_dim']} dimensions\n")
    out.append(_md(p1[[c for c in ["space", "model", "qlid_median", "lid_median", "rc_q10", "hnsw_ndis",
                                   "hnsw_ndis_std", "ivf_ndis"] if c in p1]]))
    out.append("\nSpearman correlation across backbones, two-sided permutation p-value. With 6 backbones only "
               "|rho| >= 0.89 reaches p < 0.05; treat the result as an illustration, not a test of generality.\n")
    out.append(_md(p1c))
    out.append("\n## P3: search work at a fixed end-to-end recall as the dimension falls (HNSW)\n")
    if len(p3):
        t = p3[(p3["index"] == "hnsw")]
        out.append(_md(t[["model", "target", "space", "dim", "ceiling_recall10", "ndis", "work"]]
                       .sort_values(["target", "model", "dim"])))
        out.append("\nWhere work is lowest. rises_below: a smaller dimension still reaches the target but needs more "
                   "work (the interior optimum of P3); ceiling_below: a smaller dimension cannot reach the target "
                   "because exact search there already loses too many neighbours.\n")
        out.append(_md(p3s[p3s["index"] == "hnsw"]))
    out.append("\n## P4 and P5: heads at the same dimension, relative to the plain ranking head\n")
    out.append(_md(heads))
    out.append("\nChanges relative to head_rank (n_increase / n_decrease = number of backbones). P4 expects head_hub "
               "to lower hub_skew and, if hubs help navigation, to raise hnsw_ndis; P5 expects head_koleo to change "
               "qlid_median or rc_q10 and, through them, hnsw_ndis. The pca rows compare PCA with head_rank.\n")
    out.append(_md(hsum))
    p2 = _maybe(paths, "survey_p2")
    if len(p2):
        out.append("\n## P2: text queries against image embeddings (CLIP)\n")
        out.append(_md(p2))
    path = os.path.join(paths.results, "survey_report.md")
    with open(path, "w") as f:
        f.write("\n".join(out))
    log.info(f"[survey] written to {path}")
