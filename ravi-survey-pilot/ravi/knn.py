"""Exact nearest-neighbour search (the ground truth every recall number is measured against).

All spaces in this project are L2-normalised, so ranking by inner product is identical
to ranking by Euclidean distance, with  d^2 = 2 - 2 * <x, y>.
Uses the GPU through PyTorch when one is available (so only ``faiss-cpu`` is required),
otherwise FAISS IndexFlatIP on the CPU.
"""
from __future__ import annotations

import numpy as np

from .utils import chunks, pick_device


def _knn_torch(xb, xq, k, device, qchunk, bchunk):
    import torch

    nq = xq.shape[0]
    out_i = np.empty((nq, k), dtype=np.int64)
    out_s = np.empty((nq, k), dtype=np.float32)

    def to_dev(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

    # keep the whole base on the GPU when it fits, otherwise stream it chunk by chunk
    cached = None
    if device.startswith("cuda"):
        free = torch.cuda.mem_get_info(torch.device(device))[0]
        if xb.shape[0] * xb.shape[1] * 4 < 0.5 * free:
            cached = [(bs, to_dev(xb[bs:be])) for bs, be in chunks(xb.shape[0], bchunk)]
    with torch.no_grad():
        for qs, qe in chunks(nq, qchunk):
            q = to_dev(xq[qs:qe])
            best_s = best_i = None
            it = cached if cached is not None else ((bs, to_dev(xb[bs:be])) for bs, be in chunks(xb.shape[0], bchunk))
            for bs, b in it:
                s = q @ b.T
                ts, ti = torch.topk(s, min(k, b.shape[0]), dim=1)
                ti = ti + bs
                if best_s is None:
                    best_s, best_i = ts, ti
                else:
                    cs, ci = torch.cat([best_s, ts], 1), torch.cat([best_i, ti], 1)
                    ts2, pos = torch.topk(cs, min(k, cs.shape[1]), dim=1)
                    best_s, best_i = ts2, torch.gather(ci, 1, pos)
                del s
            out_i[qs:qe] = best_i.cpu().numpy()
            out_s[qs:qe] = best_s.cpu().numpy()
    del cached
    return out_i, out_s


def _knn_faiss(xb, xq, k, bchunk):
    import faiss

    index = faiss.IndexFlatIP(xb.shape[1])
    for bs, be in chunks(xb.shape[0], bchunk):
        index.add(np.ascontiguousarray(xb[bs:be], dtype=np.float32))
    s, i = index.search(np.ascontiguousarray(xq, dtype=np.float32), k)
    return i.astype(np.int64), s.astype(np.float32)


def exact_knn(xb, xq, k: int, device: str | None = None, exclude_self: bool = False,
              qchunk: int = 2048, bchunk: int = 131072):
    """Top-k by inner product. Returns (indices [nq,k] int64, similarities [nq,k] float32).

    exclude_self=True assumes row i of xq is row i of xb and removes that match.
    """
    device = pick_device(device)
    kk = k + 1 if exclude_self else k
    kk = min(kk, xb.shape[0])
    if device.startswith("cuda"):
        idx, sim = _knn_torch(xb, xq, kk, device, qchunk, bchunk)
    else:
        idx, sim = _knn_faiss(xb, xq, kk, bchunk)
    if exclude_self:
        n = idx.shape[0]
        rows = np.arange(n)[:, None]
        is_self = idx == rows
        # drop the self match if present, otherwise (exact duplicates) drop the last column
        drop = np.where(is_self.any(1), is_self.argmax(1), kk - 1)
        keep = np.ones_like(idx, dtype=bool)
        keep[np.arange(n), drop] = False
        idx = idx[keep].reshape(n, kk - 1)
        sim = sim[keep].reshape(n, kk - 1)
    return idx, sim


def sim_to_dist(sim: np.ndarray) -> np.ndarray:
    """Euclidean distance between unit vectors from their inner product."""
    return np.sqrt(np.maximum(2.0 - 2.0 * sim.astype(np.float64), 0.0))
