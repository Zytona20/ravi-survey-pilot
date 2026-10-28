"""Unit tests on cases with a known answer.  Run:  python -m pytest -q tests   (about 1 minute, CPU)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ravi.collect import aggregate_seeds, operating_points  # noqa: E402
from ravi.config import DEFAULTS  # noqa: E402
from ravi.geometry import geometry_features, hubness_features, lid_mle, twonn_id  # noqa: E402
from ravi.head import _apply, pca_fit, train_head  # noqa: E402
from ravi.knn import exact_knn, sim_to_dist  # noqa: E402
from ravi.metrics import label_precision_at_k, map_at_k, recall_at_k  # noqa: E402
from ravi.reduce import fit_reducer  # noqa: E402
from ravi.stats import bh_fdr, compare_methods, holm, partial_spearman, spearman  # noqa: E402
from ravi.utils import l2n  # noqa: E402

RNG = np.random.default_rng(0)


def unit(n, d, seed=0):
    return l2n(np.random.default_rng(seed).standard_normal((n, d)))


def test_recall_known_values():
    truth = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    found = np.array([[4, 3, 9, 1], [-1, -1, 5, 0]])
    assert recall_at_k(found, truth, 4) == (3 / 4 + 1 / 4) / 2
    assert recall_at_k(truth, truth, 4) == 1.0
    assert recall_at_k(found, truth, 2) == 0.0  # top-2 of found shares nothing with top-2 of truth


def test_label_metrics():
    yb = np.array([0, 0, 1, 1, 2])
    yq = np.array([0, 1])
    found = np.array([[0, 2, 1], [4, 2, 3]])
    assert label_precision_at_k(found, yb, yq, 3) == (2 / 3 + 2 / 3) / 2
    # query 0: rel = 1,0,1 -> (1/1 + 2/3)/3 ; query 1: rel = 0,1,1 -> (1/2 + 2/3)/3
    assert abs(map_at_k(found, yb, yq, 3) - ((1 + 2 / 3) / 3 + (1 / 2 + 2 / 3) / 3) / 2) < 1e-12


def test_exact_knn_matches_bruteforce_and_excludes_self():
    x = unit(500, 16)
    idx, sim = exact_knn(x, x, 5, device="cpu", exclude_self=True)
    full = x @ x.T
    np.fill_diagonal(full, -np.inf)
    ref = np.argsort(-full, axis=1)[:, :5]
    assert (idx == ref).mean() > 0.999
    assert not (idx == np.arange(500)[:, None]).any()
    assert np.allclose(sim_to_dist(sim), np.sqrt(np.maximum(2 - 2 * np.take_along_axis(full, ref, 1), 0)), atol=1e-3)


def test_intrinsic_dimension_estimators_recover_known_dimension():
    for d_true in (4, 10):
        z = np.random.default_rng(d_true).standard_normal((8000, d_true)).astype(np.float32)
        idx = np.argsort(((z[:1500, None] - z[None]) ** 2).sum(-1), axis=1)[:, 1:21]
        dist = np.sqrt(((z[:1500, None] - z[idx]) ** 2).sum(-1))
        assert abs(np.nanmean(lid_mle(dist, 20)) - d_true) < 0.2 * d_true
        assert abs(twonn_id(dist) - d_true) < 0.2 * d_true


def test_hubness_measures_on_known_graphs():
    n, k = 200, 10
    ring = (np.arange(n)[:, None] + np.arange(1, k + 1)[None, :]) % n  # every point is chosen exactly k times
    f = hubness_features(ring, n, k)
    assert f["hub_skew"] == 0.0 and f["hub_robinhood"] == 0.0 and f["hub_antihub"] == 0.0 and f["hub_gini"] < 1e-9
    star = np.tile(np.arange(k), (n, 1))  # everybody points at the same k points
    f = hubness_features(star, n, k)
    assert f["hub_skew"] > 3 and f["hub_antihub"] == (n - k) / n and f["hub_occ"] == 1.0


def test_hubness_grows_with_dimension_for_gaussian_data():
    """Radovanovic et al. (2010): k-occurrence skewness rises with dimension for i.i.d. data under L2."""
    skew = []
    for d in (3, 60):
        z = np.random.default_rng(d).standard_normal((2000, d))
        d2 = (z**2).sum(1)[:, None] + (z**2).sum(1)[None] - 2 * z @ z.T
        np.fill_diagonal(d2, np.inf)
        skew.append(hubness_features(np.argsort(d2, axis=1)[:, :10], 2000, 10)["hub_skew"])
    assert skew[1] > skew[0] + 1.0


def test_geometry_features_are_finite():
    f = geometry_features(unit(1500, 24), unit(100, 24, seed=1), RNG.integers(0, 5, 1500), device="cpu")
    for k, v in f.items():
        assert np.isfinite(v), k
    assert 0.9 < f["qshift_nn"] < 1.2  # queries drawn from the same distribution as the base


def test_reducers_shape_norm_and_pca_optimality():
    x = l2n(RNG.standard_normal((2000, 5)) @ RNG.standard_normal((5, 40)) + 0.01 * RNG.standard_normal((2000, 40)))
    cfg = {"head": DEFAULTS["head"]}
    for method in ("pca", "pcaw", "rp", "trunc"):
        W, mu, _ = fit_reducer(method, x, 8, cfg, 0, "cpu")
        z = _apply(x, W, mu)
        assert z.shape == (2000, 8) and np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-4)
    W, mu, ev = pca_fit(x, 5)
    assert ev[:5].sum() / ev.sum() > 0.99  # data is 5-dimensional up to noise
    gt, _ = exact_knn(x, x[:200], 10, device="cpu")
    zp = _apply(x, W, mu)
    found, _ = exact_knn(zp, zp[:200], 10, device="cpu")
    assert recall_at_k(found, gt, 10) > 0.9


def test_head_beats_pca_when_pca_keeps_the_wrong_directions():
    """Cluster centres have large variance in 30 directions, neighbours are decided by 10 other
    low-variance directions. PCA to 16-d keeps centre directions and loses the neighbours; a head
    trained to preserve neighbours must do clearly better."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((64, 64)))
    y = rng.integers(0, 300, 9000)
    x = (rng.standard_normal((300, 30))[y] @ q[:, :30].T + (0.35 * rng.standard_normal((9000, 10))) @ q[:, 30:40].T
         + 0.01 * rng.standard_normal((9000, 64)))
    x = l2n(x)
    xb, xq = x[:8000], x[8000:]
    gt, _ = exact_knn(xb, xq, 10, device="cpu")

    def rec(W, mu):
        f, _ = exact_knn(_apply(xb, W, mu), _apply(xq, W, mu), 10, device="cpu")
        return recall_at_k(f, gt, 10)

    Wp, mup, _ = pca_fit(xb, 16)
    hp = {**DEFAULTS["head"]["defaults"], "steps": 300, "batch": 512, "eval_every": 300, "n_val": 400}
    Wh, muh, hist = train_head(xb, 16, hp, "cpu", 0)
    assert rec(Wh, muh) > rec(Wp, mup) + 0.2
    assert hist[-1]["val_recall10"] > hist[0]["val_recall10"]


def test_index_benchmark_is_sane():
    from ravi.bench import bench_one

    x = l2n(np.random.default_rng(1).standard_normal((4000, 8)) @ np.random.default_rng(2).standard_normal((8, 32)))
    xb, xq = x[:3600], x[3600:]
    gt, _ = exact_knn(xb, xq, 100, device="cpu")
    bcfg = {**DEFAULTS["bench"], "n_latency_queries": 50, "repeats": 1, "threads": 1,
            "hnsw": {"M": [16], "efConstruction": 80, "efSearch": [10, 200]},
            "ivf": {"nlist": [32], "nprobe": [1, 32]}}
    os.makedirs("/tmp/ravi_test", exist_ok=True)
    flat = bench_one(xb, xq, gt, gt, None, None, "flat", {}, 0, bcfg, "/tmp/ravi_test")
    assert all(r["recall"] > 0.999 for r in flat)
    hn = [r for r in bench_one(xb, xq, gt, gt, None, None, "hnsw", {"M": 16, "efConstruction": 80}, 0, bcfg,
                               "/tmp/ravi_test") if r["k"] == 10]
    assert hn[-1]["recall"] >= hn[0]["recall"] and hn[-1]["recall"] > 0.95
    iv = [r for r in bench_one(xb, xq, gt, gt, None, None, "ivf", {"nlist": 32}, 0, bcfg, "/tmp/ravi_test")
          if r["k"] == 10]
    assert iv[-1]["recall"] > 0.999  # probing every list is exact search
    assert iv[0]["recall"] < iv[-1]["recall"]
    assert all(r["bytes_per_vec"] >= 4 * 32 for r in flat + hn + iv)  # nothing here compresses vectors


def test_operating_point_picks_fastest_setting_that_reaches_target():
    import pandas as pd

    base = dict(space_id="s", model="m", family="f", paradigm="p", reduction="native", dim=8, native_dim=8,
                index="hnsw", build_key="b", seed=0, k=10, build_time_s=1.0, bytes_per_vec=10.0, M=16,
                nlist=np.nan, pq_m=np.nan, qps_batch=1.0, lat_p50_ms=1.0, lat_p99_ms=1.0, label_p=np.nan)
    rows = [dict(base, search_value=v, recall=r, e2e_recall=r - 0.05, qps_1t=q)
            for v, r, q in ((10, 0.80, 900.0), (20, 0.93, 500.0), (40, 0.97, 300.0), (80, 0.99, 100.0))]
    op = operating_points(pd.DataFrame(rows), [0.9, 0.95, 0.999], 10)
    pick = op.set_index(["basis", "target"])
    assert pick.loc[("recall", 0.9), "qps_1t"] == 500.0
    assert pick.loc[("recall", 0.95), "qps_1t"] == 300.0
    assert not pick.loc[("recall", 0.999), "feasible"]
    assert pick.loc[("e2e_recall", 0.9), "qps_1t"] == 300.0  # end-to-end recall is 0.05 lower, so ef=20 no longer qualifies
    agg = aggregate_seeds(op)
    assert np.isnan(agg[(agg.basis == "recall") & (agg.target == 0.999)]["qps_1t"]).all()


def test_statistics_helpers():
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    assert np.allclose(bh_fdr(np.array([0.01, 0.04, 0.03])), [0.03, 0.04, 0.04])
    x = np.arange(50.0)
    assert abs(spearman(x, x**3) - 1.0) < 1e-12
    z = RNG.standard_normal(400)
    a, b = z + 0.1 * RNG.standard_normal(400), z + 0.1 * RNG.standard_normal(400)
    assert spearman(a, b) > 0.9 and abs(partial_spearman(a, b, z)) < 0.2  # relation is entirely due to z
    import pandas as pd

    wide = pd.DataFrame({"good": RNG.random(30) + 1.0, "bad": RNG.random(30), "mid": RNG.random(30) + 0.5})
    ranks, pairs, fried = compare_methods(wide)
    assert ranks.iloc[0]["method"] == "good" and fried["friedman_p"] < 1e-3
    assert pairs[(pairs.a == "good") & (pairs.b == "bad")]["p_holm"].iloc[0] < 1e-3
