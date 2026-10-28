"""Hugging Face dataset path, disjoint query sampling, and the distance-count cost metric."""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fake_parquet(path, n_classes=5, per_class=12, seed=0):
    import datasets
    from PIL import Image

    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    for c in range(n_classes):
        for _ in range(per_class):
            a = np.full((40, 48, 3), 40 * c, dtype=np.uint8) + rng.integers(0, 30, (40, 48, 3), dtype=np.uint8)
            imgs.append(Image.fromarray(a))
            labels.append(c)
    feats = datasets.Features({"image": datasets.Image(), "label": datasets.ClassLabel(names=[f"c{i}" for i in range(n_classes)])})
    datasets.Dataset.from_dict({"image": imgs, "label": labels}, features=feats).to_parquet(path)


def test_hf_dataset_extract_and_disjoint_queries(tmp_path):
    from ravi.config import DEFAULTS, Paths, _deep_update
    from ravi.extract import run_extract
    from ravi.utils import read_json

    pq = str(tmp_path / "val.parquet")
    _fake_parquet(pq)
    spec = {"type": "hf", "path": "parquet", "data_files": pq, "split": "train", "cache_dir": str(tmp_path / "hf")}
    cfg = _deep_update(copy.deepcopy(DEFAULTS), {
        "out_dir": str(tmp_path / "run"), "device": "cpu",
        "dataset": {"name": "fake", "base": {**spec, "per_class": 8}, "query": {**spec, "per_class": 2}},
        "models": [{"name": "resnet18.a1_in1k", "family": "resnet"}],
        "extract": {"batch_size": 8, "workers": 0, "amp": False, "pretrained": False}})
    run_extract(cfg)
    paths = Paths(cfg)
    base = read_json(os.path.join(paths.lists, "base.json"))
    query = read_json(os.path.join(paths.lists, "query.json"))
    assert len(base["paths"]) == 40 and len(query["paths"]) == 10
    assert not set(base["paths"]) & set(query["paths"])  # same split -> queries must not repeat database images
    assert base["class_to_idx"] == {f"c{i}": i for i in range(5)}
    xb = np.load(paths.emb("resnet18.a1_in1k", "base"))
    yq = np.load(paths.labels("resnet18.a1_in1k", "query"))
    assert xb.shape == (40, 512) and np.isfinite(xb).all() and (np.bincount(yq) == 2).all()


def test_distance_counts():
    from ravi.bench import bench_one
    from ravi.config import DEFAULTS
    from ravi.knn import exact_knn
    from ravi.utils import l2n

    x = l2n(np.random.default_rng(3).standard_normal((3000, 16)))
    xb, xq = x[:2800], x[2800:]
    gt, _ = exact_knn(xb, xq, 100, device="cpu")
    bcfg = {**DEFAULTS["bench"], "n_latency_queries": 20, "repeats": 2, "threads": 1,
            "hnsw": {"M": [8], "efConstruction": 40, "efSearch": [10, 100]}, "ivf": {"nlist": [16], "nprobe": [1, 16]}}
    os.makedirs("/tmp/ravi_test", exist_ok=True)
    flat = bench_one(xb, xq, gt, gt, None, None, "flat", {}, 0, bcfg, "/tmp/ravi_test")
    assert all(r["ndis"] == 2800 for r in flat)
    hn = [r for r in bench_one(xb, xq, gt, gt, None, None, "hnsw", {"M": 8, "efConstruction": 40}, 0, bcfg,
                               "/tmp/ravi_test") if r["k"] == 10]
    assert 0 < hn[0]["ndis"] < hn[-1]["ndis"] < 2800  # a wider beam evaluates more distances
    iv = [r for r in bench_one(xb, xq, gt, gt, None, None, "ivf", {"nlist": 16}, 0, bcfg, "/tmp/ravi_test")
          if r["k"] == 10]
    assert abs(iv[-1]["ndis"] - (16 + 2800)) < 1e-6  # probing every list scans every vector, plus 16 centroids
