"""`report`: a Markdown summary of the measured tables, for checking results and drafting a paper.

Every number in the report is read from results/*.csv of this run. Nothing is estimated or filled in.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .collect import load_table
from .config import Paths
from .utils import log, read_json


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df is None or not len(df):
        return "_(no data)_\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _maybe(paths, name):
    try:
        return load_table(paths, name)
    except FileNotFoundError:
        return pd.DataFrame()


def run_report(cfg: dict) -> str:
    paths = Paths(cfg)
    rc = cfg["recommender"]
    target = float(rc["target"])
    out = []
    synthetic = bool(cfg["synthetic"])
    if synthetic:
        out.append("> **SYNTHETIC SMOKE TEST. These numbers come from random vectors and are not research results.**\n")
    out.append(f"# RAVI results: {cfg['name']}\n")
    env_path = os.path.join(paths.results, "env.json")
    if os.path.exists(env_path):
        env = read_json(env_path)
        keys = ["time", "cpu_model", "cpu_count", "ram_gb", "gpu", "faiss", "faiss_max_threads", "torch", "python", "platform"]
        out.append("## Environment of the benchmark stage\n")
        out.append("\n".join(f"- **{k}**: {env.get(k)}" for k in keys if k in env) + "\n")
        out.append("Timing numbers (QPS, latency) are only comparable within this environment. "
                   "Distance evaluations per query (`ndis`) do not depend on the hardware.\n")
    ds = cfg.get("dataset") or {}
    out.append("## Setup\n")
    out.append(f"- dataset: {ds.get('name', 'synthetic' if synthetic else '?')}\n"
               f"- backbones: {', '.join(m['name'] for m in (cfg['models'] or cfg['synthetic']['spaces']))}\n"
               f"- reduced dimensions: {cfg['dims']}; reductions: {cfg['reductions']}\n"
               f"- benchmark seeds: {cfg['bench']['seeds']}; recall targets: {cfg['bench']['recall_targets']}\n")

    qual, op, geom = _maybe(paths, "quality"), _maybe(paths, "operating_points"), _maybe(paths, "geometry")
    if len(qual):
        out.append("## 1. Neighborhoods kept after reduction (exact search)\n")
        out.append("Mean over backbones of recall@10 of exact search in the reduced space against exact search "
                   "in the native space. 1.0 means the reduction keeps every true neighbor.\n")
        red = qual[qual["reduction"] != "native"]
        if len(red) and "reduction_recall@10" in red:
            t = red.pivot_table(index="reduction", columns="dim", values="reduction_recall@10", aggfunc="mean")
            t.columns = [f"d={c}" for c in t.columns]
            out.append(_md(t.reset_index()))
        if "label_p@10" in qual:
            out.append("\nClass precision@10 of exact search (share of the 10 results with the query's label), per backbone:\n")
            q2 = qual.assign(space=np.where(qual["reduction"] == "native", "native",
                                            qual["reduction"] + qual["dim"].astype(str)))
            t = q2.pivot_table(index="model", columns="space", values="label_p@10")
            t = t[["native"] + [c for c in t.columns if c != "native"]] if "native" in t.columns else t
            out.append(_md(t.reset_index()))
    if len(op):
        has_ndis = "ndis" in op.columns and op["ndis"].notna().any()
        key = "ndis" if has_ndis else "qps_1t"
        m = op[(op["basis"] == "recall") & np.isclose(op["target"], target) & (op["reduction"] == "native")]
        if len(m):
            out.append(f"\n## 2a. Native spaces: cost at index recall@10 >= {target:g}\n")
            out.append("Best build setting of each index type; mean over seeds. `ndis` = distance evaluations per query "
                       "(lower is better, hardware-independent); `qps_1t` = single-thread queries per second.\n")
            best = m[m[key].notna()]
            if len(best):
                idx = best.groupby(["space_id", "index"])[key].idxmin() if has_ndis else \
                    best.groupby(["space_id", "index"])[key].idxmax()
                cols = [c for c in ["model", "dim", "index", "build_key", "ndis", "ndis_std", "qps_1t", "qps_1t_std",
                                    "lat_p99_ms", "bytes_per_vec", "build_time_s"] if c in best.columns]
                out.append(_md(best.loc[idx][cols].sort_values(["index", "model"])))
        m = op[(op["basis"] == "e2e_recall") & np.isclose(op["target"], target)]
        if len(m):
            out.append(f"\n## 2b. All spaces: cost at end-to-end recall@10 >= {target:g} (against the native space)\n")
            out.append(f"Median over backbones of the best `{key}` per index; in brackets, how many backbones reached the "
                       "target at all. A reduced space that cannot reach the target end to end is reported as 0/N.\n")
            best = m.copy()
            best["space"] = np.where(best["reduction"] == "native", "native", best["reduction"] + best["dim"].astype(str))
            agg = (best.groupby(["space", "index", "model"])[key].min() if has_ndis
                   else best.groupby(["space", "index", "model"])[key].max()).reset_index()
            n_models = best["model"].nunique()
            rows = []
            for (space, kind), g in agg.groupby(["space", "index"]):
                ok = g[key].notna()
                med = float(np.median(g.loc[ok, key])) if ok.any() else float("nan")
                rows.append({"space": space, "index": kind,
                             "value": f"- [0/{n_models}]" if not ok.any() else f"{med:,.0f} [{int(ok.sum())}/{n_models}]"})
            t = pd.DataFrame(rows).pivot(index="space", columns="index", values="value").fillna("")
            order = ["native"] + sorted(i for i in t.index if i != "native")
            out.append(_md(t.reindex([o for o in order if o in t.index]).reset_index()))
            out.append("\nThe full per-backbone, per-seed table is results/operating_points.csv.\n")
    for metric, suffix in (("log QPS", ""), ("log(1/ndis)", "_ndis")):
        corr = _maybe(paths, f"stats_geometry_vs_cost{suffix}")
        if len(corr):
            out.append(f"\n## 3. Geometry features most associated with {metric} (Spearman, index recall target)\n")
            top = corr.assign(a=corr["spearman"].abs()).sort_values("a", ascending=False).groupby("index").head(5)
            out.append(_md(top[["index", "feature", "n_spaces", "n_models", "spearman", "ci_lo", "ci_hi",
                                "partial_spearman_given_dim", "q_fdr"]]))
            out.append("CI: bootstrap over backbones. With few backbones these intervals are wide; "
                       "treat them as exploratory.\n")
    for tag, title in (("reductionrecall10", "neighborhood kept (exact search)"), ("ndis", "distance evaluations at the end-to-end target"),
                       ("qps", "single-thread QPS at the end-to-end target")):
        ranks, pairs = _maybe(paths, f"stats_reductions_{tag}_ranks"), _maybe(paths, f"stats_reductions_{tag}_pairs")
        if len(ranks):
            out.append(f"\n## 4. Reduction methods compared on {title}\n")
            out.append(_md(ranks))
            if len(pairs):
                out.append("\nPairwise Wilcoxon signed-rank tests, Holm-corrected:\n")
                out.append(_md(pairs))
    rec = _maybe(paths, "recommender_summary")
    if len(rec):
        out.append("\n## 5. Configuration recommender (leave one family out)\n")
        out.append("Score = efficiency of the chosen configuration / best configuration of that backbone (1 = oracle).\n")
        out.append(_md(rec))
    text = "\n".join(out)
    path = os.path.join(paths.results, "report.md")
    with open(path, "w") as f:
        f.write(text)
    log.info(f"[report] written to {path}")
    return text
