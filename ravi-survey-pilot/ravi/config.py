"""Configuration loading and the on-disk layout of a run."""
from __future__ import annotations

import copy
import os

import yaml

from .utils import ensure_dir, read_json, safe_name

DEFAULTS = {
    "name": "run",
    "out_dir": "runs/run",
    "emb_dir": None,  # optional: reuse native embeddings extracted by another run
    "seed": 0,
    "device": "auto",
    "synthetic": None,  # set -> synthetic spaces (smoke test only; never a result)
    "dataset": None,  # {name, base: {...}, query: {...}}
    "models": [],
    "extract": {"batch_size": 256, "workers": 8, "amp": True, "pretrained": True},  # pretrained: false is for tests only
    "dims": [64, 128, 256, 384, 512],  # the native dimension is always included
    "reduction_dims": {},  # optional per-reduction dimensions, e.g. {head_hub: [256]}; others use `dims`
    "reductions": ["pca", "rp", "trunc"],
    "fit_sample": 200000,  # vectors used to fit PCA / train heads
    "gt": {"k": 100},
    "geometry": {"sample": 20000, "k": 20},
    "bench": {
        "seeds": [0, 1, 2],
        "ks": [10, 100],
        "latency_k": 10,
        "n_latency_queries": 2000,
        "repeats": 3,
        "threads": 0,  # 0 = all cores for the batch-throughput measurement
        "early_stop_recall": 0.999,
        "flat": True,
        "hnsw": {"M": [16, 32], "efConstruction": 200, "efSearch": [10, 16, 32, 64, 128, 256, 512]},
        "ivf": {"nlist": [1024, 4096], "nprobe": [1, 2, 4, 8, 16, 32, 64, 128, 256]},
        "ivfpq": {"nlist": [1024], "m": [16, 32], "nbits": 8, "nprobe": [1, 2, 4, 8, 16, 32, 64, 128, 256]},
        "recall_targets": [0.90, 0.95, 0.99],
        "spaces": None,  # optional list of fnmatch patterns to restrict the benchmark
    },
    "head": {
        "variants": {},  # name -> hyper-parameters, see head.py
        "defaults": {
            "steps": 6000,
            "batch": 2048,
            "group": 4,
            "k_pos": 16,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "tau_t": 0.05,
            "tau_s": 0.05,
            "tau_h": 0.05,
            "w_kl": 1.0,
            "w_nce": 1.0,
            "lam_hub": 0.0,
            "lam_koleo": 0.0,
            "lam_lid": 0.0,
            "lam_iso": 0.0,
            "k_lid": 20,
            "init": "pca",
            "eval_every": 1000,
            "n_val": 2000,
        },
    },
    "recommender": {"basis": "e2e_recall", "target": 0.95, "metric": "qps_1t", "mem_budget_bytes": None},
    "stats": {"n_boot": 2000},
    # survey analysis (stage `survey`): tests of predictions P1, P3, P4 and P5 and geometry cards
    "survey": {"card_dim": 256, "targets": [0.80, 0.90], "heads": ["head_rank", "head_hub", "head_koleo"]},
    # cross-modal experiment (stage `crossmodal`, prediction P2): text queries against image embeddings
    "crossmodal": {
        "model": "ViT-B-16",
        "pretrained": "openai",
        "batch_size": 256,
        "templates": ["a photo of {}, a type of food.", "a photo of {}.", "a close-up photo of {}.",
                      "a photo of a plate of {}.", "{}", "a cropped photo of {}.", "a good photo of {}.",
                      "a photo of a delicious {}."],
        "hnsw": {"M": 16, "efConstruction": 200, "efSearch": [10, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]},
        "recall_target": 0.90,
        "k": 10,
        "geometry_sample": 20000,
    },
}


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _deep_update(copy.deepcopy(DEFAULTS), user)
    cfg["_path"] = os.path.abspath(path)
    if not cfg["models"] and not cfg["synthetic"]:
        raise ValueError("config needs either `models` (+ `dataset`) or `synthetic`")
    return cfg


class Paths:
    """Every file a run reads or writes."""

    def __init__(self, cfg: dict):
        self.out = ensure_dir(cfg["out_dir"])
        self.results = ensure_dir(os.path.join(self.out, "results"))
        self.figures = ensure_dir(os.path.join(self.out, "figures"))
        self.lists = ensure_dir(os.path.join(self.out, "lists"))
        self.emb_root = cfg.get("emb_dir") or os.path.join(self.out, "emb")

    def emb_dir(self, model: str) -> str:
        return ensure_dir(os.path.join(self.emb_root, safe_name(model)))

    def emb(self, model: str, split: str) -> str:
        return os.path.join(self.emb_dir(model), f"{split}.npy")

    def labels(self, model: str, split: str) -> str:
        return os.path.join(self.emb_dir(model), f"{split}_labels.npy")

    def emb_meta(self, model: str) -> str:
        return os.path.join(self.emb_dir(model), "meta.json")

    def space_dir(self, space_id: str) -> str:
        return ensure_dir(os.path.join(self.out, "spaces", space_id))

    def gt(self, space_id: str) -> str:
        return os.path.join(ensure_dir(os.path.join(self.out, "gt")), f"{space_id}.npz")

    def parts(self, kind: str) -> str:
        return ensure_dir(os.path.join(self.out, "parts", kind))


def native_id(model: str) -> str:
    return f"{safe_name(model)}__native"


def make_space_id(model: str, reduction: str, dim: int) -> str:
    if reduction == "native":
        return native_id(model)
    return f"{safe_name(model)}__{reduction}{dim}"


def model_specs(cfg: dict) -> list[dict]:
    """Backbones of the run. Synthetic spaces are listed the same way."""
    if cfg["synthetic"]:
        return [dict(s) for s in cfg["synthetic"]["spaces"]]
    return [dict(m) for m in cfg["models"]]


def list_spaces(cfg: dict, paths: Paths, require_native: bool = True) -> list[dict]:
    """All (model, reduction, dim) spaces of the run.

    The native dimension is read from the extraction metadata, so this can only be
    called for models whose embeddings exist. Reduced dimensions >= native are skipped.
    """
    out = []
    for m in model_specs(cfg):
        name = m["name"]
        meta_path = paths.emb_meta(name)
        if not os.path.exists(meta_path):
            if require_native:
                continue
            native_dim = int(m.get("dim", m.get("D", 0)))
        else:
            native_dim = int(read_json(meta_path)["dim"])
        base = {"model": name, "family": m.get("family", "na"), "paradigm": m.get("paradigm", "na"),
                "native_dim": native_dim}
        out.append({**base, "space_id": native_id(name), "reduction": "native", "dim": native_dim})
        for red in cfg["reductions"]:
            for d in (cfg.get("reduction_dims") or {}).get(red, cfg["dims"]):
                if isinstance(d, str) or int(d) >= native_dim:
                    continue
                out.append({**base, "space_id": make_space_id(name, red, int(d)), "reduction": red, "dim": int(d)})
    return out
