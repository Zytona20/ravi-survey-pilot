"""Stage 2: native space -> compact spaces. Every method is one linear map followed by
L2 normalisation,  z = normalise((x - mu) W^T), so all of them cost the same at query time.

  pca    principal components
  pcaw   PCA + whitening
  rp     Gaussian random projection with orthonormal rows
  trunc  keep the first d coordinates (the right baseline for Matryoshka-trained models)
  <head variant name>   learned head from head.py (see `head.variants` in the config)
"""
from __future__ import annotations

import os

import numpy as np

from .config import Paths, list_spaces
from .data import load_native
from .head import _apply, pca_fit, train_head
from .utils import chunks, log, pick_device, save_npy, stable_hash, timer, write_json


def fit_reducer(method: str, x_fit: np.ndarray, d: int, cfg: dict, seed: int, device: str):
    """Returns (W [d, D], mu [D], info dict)."""
    D = x_fit.shape[1]
    if method == "pca":
        W, mu, _ = pca_fit(x_fit, d)
        return W, mu, {}
    if method == "pcaw":
        W, mu, ev = pca_fit(x_fit, d)
        return W / np.sqrt(np.maximum(ev[:d], 1e-12))[:, None].astype(np.float32), mu, {}
    if method == "rp":
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.standard_normal((D, d)))
        return q.T.astype(np.float32), np.zeros(D, np.float32), {}
    if method == "trunc":
        W = np.zeros((d, D), np.float32)
        W[np.arange(d), np.arange(d)] = 1.0
        return W, np.zeros(D, np.float32), {}
    variants = cfg["head"]["variants"] or {}
    if method in variants:
        hp = {**cfg["head"]["defaults"], **(variants[method] or {})}
        W, mu, hist = train_head(x_fit, d, hp, device, seed)
        return W, mu, {"history": hist, "hp": hp}
    raise ValueError(f"unknown reduction '{method}' (not a built-in and not in head.variants)")


def run_reduce(cfg: dict) -> None:
    paths = Paths(cfg)
    device = pick_device(cfg["device"])
    spaces = [s for s in list_spaces(cfg, paths) if s["reduction"] != "native"]
    by_model: dict[str, list[dict]] = {}
    for s in spaces:
        by_model.setdefault(s["model"], []).append(s)
    for model, todo in by_model.items():
        todo = [s for s in todo if not os.path.exists(os.path.join(paths.space_dir(s["space_id"]), "meta.json"))]
        if not todo:
            continue
        xb, xq, _, _ = load_native(paths, model)
        rng = np.random.default_rng(cfg["seed"] + stable_hash(model) % 100000)
        n_fit = min(len(xb), int(cfg["fit_sample"]))
        fit_ids = np.sort(rng.choice(len(xb), n_fit, replace=False))
        x_fit = xb[fit_ids]
        for s in todo:
            sdir = paths.space_dir(s["space_id"])
            log.info(f"[reduce] {s['space_id']}")
            with timer() as t:
                W, mu, info = fit_reducer(s["reduction"], x_fit, s["dim"], cfg, cfg["seed"], device)
            zb = np.empty((len(xb), s["dim"]), np.float32)
            for a, b in chunks(len(xb), 100000):
                zb[a:b] = _apply(xb[a:b], W, mu)
            zq = _apply(xq, W, mu)
            np.savez(os.path.join(sdir, "transform.npz"), W=W, mu=mu)
            save_npy(os.path.join(sdir, "base.npy"), zb)
            save_npy(os.path.join(sdir, "query.npy"), zq)
            write_json(os.path.join(sdir, "meta.json"), {**s, "fit_seconds": t(), "n_fit": n_fit, **info})
