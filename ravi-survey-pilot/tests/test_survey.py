"""Helpers of the survey analyses (stage `survey`) and of the cross-modal experiment (stage `crossmodal`)."""
import numpy as np
import pandas as pd

from ravi.config import DEFAULTS, Paths, list_spaces
from ravi.crossmodal import at_target, hnsw_sweep, query_geometry
from ravi.knn import exact_knn
from ravi.survey import best_ndis, heads_summary, spearman_exact
from ravi.utils import l2n, write_json


def test_spearman_exact_permutation_p():
    n, r, p = spearman_exact([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60])
    assert n == 6 and np.isclose(r, 1.0)
    assert np.isclose(p, 2 / 720)          # only the identity and the reversal are as extreme
    n, r, p = spearman_exact([1, 2, np.nan], [3, 2, 1])
    assert n == 2 and np.isnan(r)


def test_best_ndis_takes_best_build_per_seed_then_mean():
    ops = pd.DataFrame([
        {"space_id": "a", "seed": 0, "basis": "recall", "target": 0.9, "index": "hnsw", "build_key": "M16", "ndis": 100.0},
        {"space_id": "a", "seed": 0, "basis": "recall", "target": 0.9, "index": "hnsw", "build_key": "M32", "ndis": 80.0},
        {"space_id": "a", "seed": 1, "basis": "recall", "target": 0.9, "index": "hnsw", "build_key": "M16", "ndis": 120.0},
        {"space_id": "a", "seed": 1, "basis": "recall", "target": 0.9, "index": "hnsw", "build_key": "M32", "ndis": np.nan},
        {"space_id": "a", "seed": 0, "basis": "e2e_recall", "target": 0.9, "index": "hnsw", "build_key": "M16", "ndis": 5.0},
    ])
    b = best_ndis(ops, "recall", 0.9, "hnsw").set_index("space_id")
    assert np.isclose(b.loc["a", "ndis"], 100.0) and b.loc["a", "n_seeds"] == 2


def test_heads_summary_counts_backbones():
    h = pd.DataFrame([
        {"model": "m1", "reduction": "head_rank", "hub_skew": 2.0, "hnsw_ndis": 100.0},
        {"model": "m1", "reduction": "head_hub", "hub_skew": 1.0, "hnsw_ndis": 120.0},
        {"model": "m2", "reduction": "head_rank", "hub_skew": 3.0, "hnsw_ndis": 90.0},
        {"model": "m2", "reduction": "head_hub", "hub_skew": 2.5, "hnsw_ndis": 80.0},
    ])
    s = heads_summary(h).set_index(["reduction", "metric"])
    assert s.loc[("head_hub", "hub_skew"), "n_decrease"] == 2
    assert s.loc[("head_hub", "hnsw_ndis"), "n_increase"] == 1


def test_reduction_dims_restrict_heads(tmp_path):
    cfg = {**DEFAULTS, "out_dir": str(tmp_path), "models": [{"name": "m"}], "dims": [64, 128, 256],
           "reductions": ["pca", "head_hub"], "reduction_dims": {"head_hub": [256]}}
    paths = Paths(cfg)
    write_json(paths.emb_meta("m"), {"dim": 768})
    ids = {s["space_id"] for s in list_spaces(cfg, paths)}
    assert ids == {"m__native", "m__pca64", "m__pca128", "m__pca256", "m__head_hub256"}


def test_hnsw_sweep_and_query_geometry():
    import faiss

    rng = np.random.default_rng(0)
    a, b = np.zeros(16, np.float32), np.zeros(16, np.float32)
    a[0], b[1] = 3.0, 3.0
    xb = l2n((rng.standard_normal((3000, 16)) + a).astype(np.float32))      # database around direction a
    xq_in = l2n((rng.standard_normal((200, 16)) + a).astype(np.float32))    # queries from the same distribution
    xq_out = l2n((rng.standard_normal((200, 16)) + b).astype(np.float32))   # queries around another direction
    index = faiss.IndexHNSWFlat(16, 8)
    index.add(xb)
    gt, _ = exact_knn(xb, xq_in, 10, device="cpu")
    rows = hnsw_sweep(index, xq_in, gt, 10, [10, 40, 400])
    assert rows[-1]["recall@10"] > 0.95 and rows[0]["ndis"] < rows[-1]["ndis"]
    t = at_target(rows, 0.9)
    assert t["reaches_target"] and t["ndis_at_target"] <= rows[-1]["ndis"]
    assert not at_target(rows, 1.01)["reaches_target"]
    _, s = exact_knn(xb, xb, 1, device="cpu", exclude_self=True)
    nn1 = float(np.sqrt(np.maximum(2 - 2 * s[:, 0], 0)).mean())
    g_in = query_geometry(xb, nn1, xq_in)
    g_out = query_geometry(xb, nn1, xq_out)
    assert 0.8 < g_in["qshift_nn"] < 1.3 and g_out["qshift_nn"] > g_in["qshift_nn"]
