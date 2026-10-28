"""Stage 3: geometry of an embedding space, measured on a fixed-size random sample.

All features are computed on the SAME sample size for every space, because nearest-neighbour
distances (and so LID, hubness, contrast) depend on how densely the space is sampled.

Feature groups
  intrinsic dimension   lid_mean/median/std (MLE, Amsaleg et al. 2015), twonn_id (Facco et al. 2017),
                        pca_pr (participation ratio), pca_d90, pca_erank
  hubness               hub_skew, hub_robinhood, hub_antihub, hub_occ, hub_gini   (k-occurrence, k=10)
  anisotropy            aniso_meancos, aniso_meannorm, aniso_top1
  contrast / density    rc_1, rc_10 (relative contrast, He et al. 2012), dens_r10_mean, dens_r10_cv
  clusterability        clus_qerr, clus_size_cv, clus_imbalance, clus_silhouette
  query shift           qshift_nn (query-to-base NN distance / base-to-base NN distance)
  semantic              label_purity10 (only when labels exist)
"""
from __future__ import annotations

import os

import numpy as np

from .config import Paths, list_spaces
from .data import load_space
from .knn import exact_knn, sim_to_dist
from .utils import log, pick_device, stable_hash, write_json


def lid_mle(dist: np.ndarray, k: int) -> np.ndarray:
    """Per-point MLE of local intrinsic dimensionality from ascending kNN distances."""
    r = np.maximum(dist[:, :k], 1e-12)
    m = np.log(r / r[:, k - 1:k]).mean(1)
    with np.errstate(divide="ignore"):
        lid = -1.0 / m
    return np.where(np.isfinite(lid), lid, np.nan)


def twonn_id(dist: np.ndarray, discard: float = 0.1) -> float:
    """TwoNN intrinsic dimension: slope through the origin of -log(1-F(mu)) vs log(mu), mu = r2/r1."""
    r1, r2 = dist[:, 0], dist[:, 1]
    ok = r1 > 1e-12
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu)
    if n < 10:
        return float("nan")
    keep = int(n * (1 - discard))
    x = np.log(mu[:keep])
    y = -np.log(1.0 - np.arange(1, keep + 1) / n)
    den = float((x * x).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def gini(v: np.ndarray) -> float:
    v = np.sort(np.asarray(v, dtype=np.float64))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def hubness_features(idx: np.ndarray, n: int, k: int = 10) -> dict:
    occ = np.bincount(idx[:, :k].ravel(), minlength=n).astype(np.float64)
    sd = occ.std()
    return {
        "hub_skew": float(((occ - k) ** 3).mean() / sd**3) if sd > 0 else 0.0,
        "hub_robinhood": float(np.abs(occ - k).sum() / (2 * k * n)),
        "hub_antihub": float((occ == 0).mean()),
        "hub_occ": float(occ[occ > 2 * k].sum() / (k * n)),
        "hub_gini": gini(occ),
    }


def geometry_features(xb: np.ndarray, xq: np.ndarray | None = None, yb: np.ndarray | None = None,
                      k: int = 20, seed: int = 0, device: str | None = None) -> dict:
    """xb: [n, d] L2-normalised sample of the base set; xq: optional query sample."""
    import faiss

    n, d = xb.shape
    K = min(max(k, 100), n - 1)
    rng = np.random.default_rng(seed)
    idx, sim = exact_knn(xb, xb, K, device=device, exclude_self=True)
    dist = sim_to_dist(sim)
    f: dict = {"n_sample": n, "d": d}

    # intrinsic dimension
    lid = lid_mle(dist, min(k, K))
    f.update(lid_mean=float(np.nanmean(lid)), lid_median=float(np.nanmedian(lid)), lid_std=float(np.nanstd(lid)))
    f["lid_p90"] = float(np.nanpercentile(lid, 90))
    f["lid100_mean"] = float(np.nanmean(lid_mle(dist, K)))
    f["twonn_id"] = twonn_id(dist)
    xc = xb.astype(np.float64) - xb.mean(0)
    ev = np.clip(np.linalg.eigvalsh(xc.T @ xc / (n - 1))[::-1], 0, None)
    p = ev / ev.sum()
    f["pca_pr"] = float(ev.sum() ** 2 / (ev**2).sum())
    f["pca_d90"] = int(np.searchsorted(np.cumsum(p), 0.90) + 1)
    f["pca_erank"] = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))

    # hubness
    f.update(hubness_features(idx, n, 10))

    # anisotropy
    a, b = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
    keep = a != b
    rand_sim = (xb[a[keep]] * xb[b[keep]]).sum(1)
    f["aniso_meancos"] = float(rand_sim.mean())
    f["aniso_meannorm"] = float(np.linalg.norm(xb.mean(0)))
    f["aniso_top1"] = float(p[0])

    # contrast and density
    rand_dist = sim_to_dist(rand_sim).mean()
    f["rc_1"] = float(rand_dist / max(dist[:, 0].mean(), 1e-12))
    f["rc_10"] = float(rand_dist / max(dist[:, 9].mean(), 1e-12))
    f["dens_r10_mean"] = float(dist[:, 9].mean())
    f["dens_r10_cv"] = float(dist[:, 9].std() / max(dist[:, 9].mean(), 1e-12))

    # clusterability (what IVF relies on)
    kc = int(max(8, round(np.sqrt(n))))
    km = faiss.Kmeans(d, kc, niter=20, seed=int(seed) + 1, verbose=False)
    km.train(np.ascontiguousarray(xb))
    d2, assign = km.index.search(np.ascontiguousarray(xb), 1)
    sizes = np.bincount(assign.ravel(), minlength=kc).astype(np.float64)
    f["clus_qerr"] = float(d2.sum() / max((xc**2).sum(), 1e-12))
    f["clus_size_cv"] = float(sizes.std() / sizes.mean())
    f["clus_imbalance"] = float((sizes**2).sum() * kc / n**2)
    try:
        from sklearn.metrics import silhouette_score

        sub = rng.choice(n, min(n, 3000), replace=False)
        lab = assign.ravel()[sub]
        f["clus_silhouette"] = float(silhouette_score(xb[sub], lab)) if len(np.unique(lab)) > 1 else float("nan")
    except Exception:
        f["clus_silhouette"] = float("nan")

    # query side: shift, local intrinsic dimensionality and relative contrast of the queries
    if xq is not None and len(xq):
        kq = min(k, n)
        _, qs = exact_knn(xb, xq, kq, device=device)
        qdist = sim_to_dist(qs)
        f["qshift_nn"] = float(qdist[:, 0].mean() / max(dist[:, 0].mean(), 1e-12))
        qlid = lid_mle(qdist, kq)
        f["qlid_median"] = float(np.nanmedian(qlid))
        f["qlid_p90"] = float(np.nanpercentile(qlid, 90))
        qa, qb = rng.integers(0, len(xq), 20000), rng.integers(0, n, 20000)
        q_rand = sim_to_dist((xq[qa] * xb[qb]).sum(1)).mean()
        f["rc_q10"] = float(q_rand / max(qdist[:, min(9, kq - 1)].mean(), 1e-12))

    # semantic neighbourhood purity
    if yb is not None and (yb >= 0).any():
        f["label_purity10"] = float((yb[idx[:, :10]] == yb[:, None]).mean())
    return f


FEATURES = ["lid_mean", "lid_median", "lid_std", "lid_p90", "lid100_mean", "twonn_id", "pca_pr", "pca_d90", "pca_erank",
            "hub_skew", "hub_robinhood", "hub_antihub", "hub_occ", "hub_gini",
            "aniso_meancos", "aniso_meannorm", "aniso_top1", "rc_1", "rc_10", "dens_r10_mean", "dens_r10_cv",
            "clus_qerr", "clus_size_cv", "clus_imbalance", "clus_silhouette", "qshift_nn", "qlid_median", "rc_q10"]


def run_geometry(cfg: dict) -> None:
    paths = Paths(cfg)
    device = pick_device(cfg["device"])
    part_dir = paths.parts("geometry")
    for s in list_spaces(cfg, paths):
        out = os.path.join(part_dir, f"{s['space_id']}.json")
        if os.path.exists(out):
            continue
        xb, xq, yb, _ = load_space(paths, s)
        rng = np.random.default_rng(cfg["seed"] + stable_hash(s["model"]) % 100000)
        n = min(len(xb), int(cfg["geometry"]["sample"]))
        ids = np.sort(rng.choice(len(xb), n, replace=False))  # same rows for every space of a model
        qids = np.sort(rng.choice(len(xq), min(len(xq), 5000), replace=False))
        feats = geometry_features(xb[ids], xq[qids], None if yb is None else yb[ids],
                                  k=int(cfg["geometry"]["k"]), seed=cfg["seed"], device=device)
        write_json(out, {**s, **feats})
        log.info(f"[geometry] {s['space_id']}: lid={feats['lid_mean']:.1f} hub_skew={feats['hub_skew']:.2f} "
                 f"clus_qerr={feats['clus_qerr']:.3f}")
