"""Stage 6: configuration recommender, evaluated leave-one-family-out.

Question: for a backbone the recommender has never seen, can cheap geometry measurements
of its native space pick the (reduction, dimension, index, build setting) that gives the
highest throughput while still reaching the recall target (and fitting a memory budget)?

  ours          surrogate = configuration prior + gradient-boosted correction from geometry
  prior_only    the same surrogate without the geometry correction (ablation)
  global_best   the single configuration that is best on average over the training backbones
  nn_dataset    copy the best configuration of the most similar training backbone
                (dataset-similarity strategy of Oyamada et al., Information Systems 2023)
  random        expected score of a uniformly random configuration
  oracle        exhaustive grid search (score 1 by definition)

Score = QPS of the chosen configuration / QPS of the oracle; a choice that misses the recall
target scores 0.  Regret = 1 - score.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .collect import load_table, save_table
from .config import Paths
from .geometry import FEATURES
from .utils import log

CONF = ["reduction", "cdim", "index", "build_key"]


def build_meta(op: pd.DataFrame, geom: pd.DataFrame, basis: str, target: float, metric: str,
               mem_budget: float | None) -> tuple[pd.DataFrame, list[str]]:
    m = op[(op["basis"] == basis) & (np.isclose(op["target"], target))].copy()
    m["qps"] = m[metric]
    m["ok"] = m["feasible"] & m["qps"].notna()
    if mem_budget:
        m["ok"] &= m["bytes_per_vec"] <= float(mem_budget)
    m["cdim"] = np.where(m["reduction"] == "native", 0, m["dim"])  # transferable config key
    feats = [f for f in FEATURES if f in geom.columns]
    nat = geom[geom["reduction"] == "native"][["model"] + feats].rename(columns={f: "nat_" + f for f in feats})
    m = m.merge(nat, on="model", how="left")
    return m, ["nat_" + f for f in feats]


def _design(m: pd.DataFrame, nat_cols: list[str], cats: dict) -> pd.DataFrame:
    X = pd.DataFrame(index=m.index)
    X["log2_dim"] = np.log2(m["dim"].astype(float))
    X["log2_native_dim"] = np.log2(m["native_dim"].astype(float))
    X["M"] = m["M"].astype(float).fillna(0)
    X["log2_nlist"] = np.log2(m["nlist"].astype(float).fillna(1))
    X["pq_m"] = m["pq_m"].astype(float).fillna(0)
    for col, values in cats.items():
        for v in values:
            X[f"{col}={v}"] = (m[col] == v).astype(float)
    for c in nat_cols:
        X[c] = m[c].astype(float)
    return X.fillna(0.0)


def _add_prior(df: pd.DataFrame, prior: pd.Series, feas: pd.Series) -> pd.DataFrame:
    key = pd.MultiIndex.from_frame(df[CONF])
    floor = (prior.min() - 1.0) if prior.notna().any() else 0.0
    df = df.copy()
    df["cfg_prior"] = prior.reindex(key).fillna(floor).values
    df["cfg_feas_rate"] = feas.reindex(key).fillna(0.0).values
    return df


def evaluate_lofo(meta: pd.DataFrame, nat_cols: list[str], seed: int = 0) -> pd.DataFrame:
    """Leave one backbone family out; recommend a configuration for each held-out backbone.

    Throughput is expressed as log speed-up over exact search in the native space of the same
    backbone, which makes backbones of different size comparable. The surrogate predicts
        log speed-up = prior(config) + deviation(config, geometry)
        P(reaches the recall target | config, geometry)
    and picks the configuration with the highest expected speed-up, P * exp(log speed-up).
    `prior_only` uses the training averages alone, so (ours - prior_only) isolates what the
    geometry features add.
    """
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    cats = {c: sorted(meta[c].unique()) for c in ("reduction", "index")}
    meta = meta.copy()
    meta["logq"] = np.where(meta["ok"], np.log(meta["qps"].where(meta["ok"], 1.0)), np.nan)
    ref = meta[(meta["index"] == "flat") & (meta["reduction"] == "native")].groupby("model")["logq"].mean()
    if ref.reindex(meta["model"].unique()).notna().all():
        meta["yc"] = meta["logq"] - meta["model"].map(ref)  # log speed-up over exact native search
    else:  # no exact baseline in the grid: centre on the backbone mean instead
        meta["yc"] = meta["logq"] - meta.groupby("model")["logq"].transform("mean")
    rows = []
    for fam in sorted(meta["family"].unique()):
        tr, te = meta[meta["family"] != fam], meta[meta["family"] == fam]
        if tr["model"].nunique() < 2 or not tr["ok"].any():
            continue
        prior = tr[tr["ok"]].groupby(CONF)["yc"].mean()
        feas = tr.groupby(CONF)["ok"].mean()
        tr = _add_prior(tr, prior, feas)
        extra = ["cfg_prior", "cfg_feas_rate"]
        Xtr = pd.concat([_design(tr, nat_cols, cats), tr[extra]], axis=1)
        leaf = max(5, int(0.02 * tr["ok"].sum()))
        reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=leaf,
                                            l2_regularization=1.0, random_state=seed)
        reg.fit(Xtr[tr["ok"]], (tr["yc"] - tr["cfg_prior"])[tr["ok"]])
        clf = None
        if tr["ok"].nunique() == 2:
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                                 min_samples_leaf=leaf, random_state=seed)
            clf.fit(Xtr, tr["ok"].astype(int))
        oracle_tr = tr[tr["ok"]].groupby("model")["qps"].max()
        tr = tr.assign(score=np.where(tr["ok"], tr["qps"] / tr["model"].map(oracle_tr), 0.0))
        global_rank = tr.groupby(CONF)["score"].mean().sort_values(ascending=False)
        nat_tr = tr.groupby("model")[nat_cols].first()
        mu, sd = nat_tr.mean(), nat_tr.std().replace(0, 1).fillna(1)
        for model, g in te.groupby("model"):
            if not g["ok"].any():
                continue
            oracle = g.loc[g["ok"], "qps"].max()
            g = _add_prior(g.assign(score=np.where(g["ok"], g["qps"] / oracle, 0.0)), prior, feas)
            key_score = {tuple(r[c] for c in CONF): r["score"] for _, r in g.iterrows()}

            def first_available(ranking):
                for key in ranking:
                    if key in key_score:
                        return key_score[key]
                return 0.0

            Xte = pd.concat([_design(g, nat_cols, cats), g[extra]], axis=1)
            dev = reg.predict(Xte)
            seen = (g["cfg_feas_rate"].values > 0)  # never feasible on any training backbone -> utility 0
            p_ok = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)] if clf is not None else np.ones(len(g))
            pick = int(np.argmax(np.where(seen, p_ok * np.exp(g["cfg_prior"].values + dev), 0.0)))
            pick0 = int(np.argmax(np.where(seen, g["cfg_feas_rate"].values * np.exp(g["cfg_prior"].values), 0.0)))
            z = ((nat_tr - mu) / sd).fillna(0).values
            zq = ((g[nat_cols].iloc[0] - mu) / sd).fillna(0).values.astype(float)
            nearest = nat_tr.index[int(np.argmin(((z - zq) ** 2).sum(1)))]
            nn_rank = tr[tr["model"] == nearest].sort_values("score", ascending=False)
            nn_keys = [tuple(r[c] for c in CONF) for _, r in nn_rank.iterrows()]
            okm = g["ok"].values
            full = g["cfg_prior"].values + dev
            rho = (spearmanr(full[okm], g.loc[okm, "logq"])[0]
                   if okm.sum() > 2 and np.ptp(full[okm]) > 0 else np.nan)
            rows.append({"family": fam, "model": model, "n_configs": len(g), "n_feasible": int(okm.sum()),
                         "oracle_qps": oracle, "ours": float(g["score"].iloc[pick]),
                         "prior_only": float(g["score"].iloc[pick0]),
                         "global_best": first_available(list(global_rank.index)),
                         "nn_dataset": first_available(nn_keys), "random": float(g["score"].mean()),
                         "nearest_train_model": nearest, "pred_spearman": rho,
                         "picked": "|".join(str(v) for v in g[CONF].iloc[pick].tolist())})
    return pd.DataFrame(rows)


def run_recommend(cfg: dict) -> None:
    paths = Paths(cfg)
    rc = cfg["recommender"]
    op, geom = load_table(paths, "operating_points"), load_table(paths, "geometry")
    meta, nat_cols = build_meta(op, geom, rc["basis"], float(rc["target"]), rc["metric"], rc.get("mem_budget_bytes"))
    res = evaluate_lofo(meta, nat_cols, cfg["seed"])
    if not len(res):
        log.info("[recommend] not enough families / feasible configurations to evaluate")
        return
    save_table(res, os.path.join(paths.results, "recommender"))
    methods = ["ours", "prior_only", "global_best", "nn_dataset", "random"]
    summ = pd.DataFrame({"method": methods,
                         "mean_score": [res[m].mean() for m in methods],
                         "mean_regret": [1 - res[m].mean() for m in methods],
                         "worst_score": [res[m].min() for m in methods],
                         "n_models": len(res)})
    save_table(summ, os.path.join(paths.results, "recommender_summary"))
    what = {"qps_1t": "single-thread QPS", "inv_ndis": "1 / distance evaluations per query"}.get(rc["metric"], rc["metric"])
    log.info(f"[recommend] leave-one-family-out, score = {what} of the chosen configuration / "
             f"that of the best configuration\n" + summ.round(3).to_string(index=False))
