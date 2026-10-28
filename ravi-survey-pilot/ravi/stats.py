"""Stage 7: statistics. Nothing here invents data; every test reads the measured tables.

  geometry_vs_cost     Spearman correlation of each geometry feature with log QPS at a recall
                       target, per index type; partial correlation controlling for dimension;
                       cluster bootstrap over backbones (spaces of one backbone are not independent);
                       Benjamini-Hochberg FDR over features
  compare_methods      Friedman test + pairwise Wilcoxon signed-rank with Holm correction
                       (reduction methods compared on matched backbone x dimension blocks)
  paradigm_effect      Kruskal-Wallis across training paradigms at the native dimension
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats as ss

from .collect import load_table, save_table
from .config import Paths
from .geometry import FEATURES
from .utils import log


def _rank(a):
    return ss.rankdata(a)


def spearman(x, y) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y after removing the (rank-linear) effect of z."""
    if len(x) < 4 or np.ptp(z) == 0:
        return spearman(x, y)
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    A = np.c_[np.ones(len(rz)), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    adj = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate(p[order] * (m - np.arange(m)))
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def best_per_space(op: pd.DataFrame, basis: str, target: float, metric: str = "qps_1t") -> pd.DataFrame:
    """Best build setting of each index type in each space at the recall target."""
    m = op[(op["basis"] == basis) & np.isclose(op["target"], target) & op[metric].notna()]
    idx = m.groupby(["space_id", "index"])[metric].idxmax()
    return m.loc[idx].reset_index(drop=True)


def geometry_vs_cost(op, geom, basis, target, n_boot, seed, metric: str = "qps_1t") -> pd.DataFrame:
    best = best_per_space(op, basis, target, metric)
    feats = [f for f in FEATURES if f in geom.columns]
    j = best.merge(geom[["space_id"] + feats], on="space_id")
    rng = np.random.default_rng(seed)
    rows = []
    for kind, g in j.groupby("index"):
        y = np.log(g[metric].values)
        z = np.log(g["dim"].values.astype(float))
        models = g["model"].values
        uniq = np.unique(models)
        member = {u: np.flatnonzero(models == u) for u in uniq}
        for f in feats:
            x = g[f].values.astype(float)
            ok = np.isfinite(x)
            if ok.sum() < 5:
                continue
            rho = spearman(x[ok], y[ok])
            pval = ss.spearmanr(x[ok], y[ok])[1] if np.isfinite(rho) else np.nan
            boots = []
            for _ in range(n_boot):  # resample whole backbones
                pick = np.concatenate([member[u] for u in rng.choice(uniq, len(uniq))])
                pick = pick[np.isfinite(x[pick])]
                boots.append(spearman(x[pick], y[pick]))
            boots = np.asarray(boots, dtype=float)
            boots = boots[np.isfinite(boots)]
            lo, hi = (np.quantile(boots, [0.025, 0.975]) if len(boots) > 10 else (np.nan, np.nan))
            rows.append({"index": kind, "feature": f, "n_spaces": int(ok.sum()), "n_models": len(uniq),
                         "spearman": rho, "ci_lo": lo, "ci_hi": hi, "p": pval,
                         "partial_spearman_given_dim": partial_spearman(x[ok], y[ok], z[ok])})
    res = pd.DataFrame(rows)
    if len(res):
        res["q_fdr"] = res.groupby("index")["p"].transform(lambda p: bh_fdr(p.values))
    return res


def compare_methods(wide: pd.DataFrame, higher_is_better: bool = True):
    """wide: rows = matched blocks, columns = methods. Returns (rank table, pairwise table, friedman dict)."""
    wide = wide.dropna()
    methods = list(wide.columns)
    if len(wide) < 3 or len(methods) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}
    vals = wide.values if higher_is_better else -wide.values
    ranks = np.apply_along_axis(lambda r: ss.rankdata(-r), 1, vals)
    rank_tab = pd.DataFrame({"method": methods, "mean_rank": ranks.mean(0), "mean_value": wide.values.mean(0),
                             "n_blocks": len(wide)}).sort_values("mean_rank")
    fried = {}
    if len(methods) >= 3:
        try:
            st, p = ss.friedmanchisquare(*[wide[m].values for m in methods])
            fried = {"friedman_chi2": float(st), "friedman_p": float(p), "n_blocks": len(wide), "k": len(methods)}
        except ValueError:
            pass
    pairs = []
    for a, b in itertools.combinations(methods, 2):
        d = wide[a].values - wide[b].values
        if np.allclose(d, 0):
            p = 1.0
        else:
            p = float(ss.wilcoxon(wide[a].values, wide[b].values, zero_method="wilcox").pvalue)
        pairs.append({"a": a, "b": b, "mean_diff_a_minus_b": float(d.mean()), "wins_a": int((d > 0).sum()),
                      "wins_b": int((d < 0).sum()), "p": p})
    pair_tab = pd.DataFrame(pairs)
    if len(pair_tab):
        pair_tab["p_holm"] = holm(pair_tab["p"].values)
    return rank_tab, pair_tab, fried


def run_stats(cfg: dict) -> None:
    paths = Paths(cfg)
    rc = cfg["recommender"]
    n_boot, seed = int(cfg["stats"]["n_boot"]), cfg["seed"]
    op, geom, qual = (load_table(paths, n) for n in ("operating_points", "geometry", "quality"))
    notes = []

    for metric, suffix in (("qps_1t", ""), ("inv_ndis", "_ndis")):
        if metric not in op.columns or not op[metric].notna().any():
            continue
        corr = geometry_vs_cost(op, geom, "recall", float(rc["target"]), n_boot, seed, metric)
        if len(corr):
            corr["cost_metric"] = metric
            save_table(corr, os.path.join(paths.results, f"stats_geometry_vs_cost{suffix}"))
            notes.append(f"geometry_vs_cost[{metric}]: {len(corr)} feature x index rows")

    red = qual[qual["reduction"] != "native"]
    for metric in ("reduction_recall@10", "label_p@10"):
        if metric not in red.columns or not len(red):
            continue
        wide = red.pivot_table(index=["model", "dim"], columns="reduction", values=metric)
        rank_tab, pair_tab, fried = compare_methods(wide)
        tag = metric.replace("@", "").replace("_", "")
        if len(rank_tab):
            for k, v in fried.items():
                rank_tab[k] = v
            save_table(rank_tab, os.path.join(paths.results, f"stats_reductions_{tag}_ranks"))
            save_table(pair_tab, os.path.join(paths.results, f"stats_reductions_{tag}_pairs"))
            notes.append(f"reductions on {metric}: {len(wide.dropna())} blocks, friedman p={fried.get('friedman_p')}")

    for metric, suffix in (("qps_1t", "qps"), ("inv_ndis", "ndis")):
        if metric not in op.columns or not op[metric].notna().any():
            continue
        best = best_per_space(op, "e2e_recall", float(rc["target"]), metric)
        red_b = best[best["reduction"] != "native"]
        if len(red_b):
            wide = red_b.assign(v=np.log(red_b[metric])).pivot_table(index=["model", "dim", "index"],
                                                                     columns="reduction", values="v")
            rank_tab, pair_tab, fried = compare_methods(wide)
            if len(rank_tab):
                for k, v in fried.items():
                    rank_tab[k] = v
                save_table(rank_tab, os.path.join(paths.results, f"stats_reductions_{suffix}_ranks"))
                save_table(pair_tab, os.path.join(paths.results, f"stats_reductions_{suffix}_pairs"))

    nat = best_per_space(op, "recall", float(rc["target"]))
    nat = nat[nat["reduction"] == "native"]
    rows = []
    for kind, g in nat.groupby("index"):
        groups = [np.log(v["qps_1t"].values) for _, v in g.groupby("paradigm") if len(v) >= 2]
        if len(groups) >= 2:
            st, p = ss.kruskal(*groups)
            rows.append({"index": kind, "kruskal_h": float(st), "p": float(p), "n_groups": len(groups), "n": len(g)})
    if rows:
        save_table(pd.DataFrame(rows), os.path.join(paths.results, "stats_paradigm_effect"))
    log.info("[stats] " + ("; ".join(notes) if notes else "nothing to test yet"))
