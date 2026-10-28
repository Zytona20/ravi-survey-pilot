"""Retrieval metrics. Three different questions, three different numbers:

  recall      index result vs exact search IN THE SAME space   -> how good is the index
  e2e_recall  index result vs exact search in the NATIVE space  -> index + dimension reduction
  label_*     do retrieved images share the query's class       -> semantic retrieval quality
"""
from __future__ import annotations

import numpy as np



def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int, per_query: bool = False):
    """|found[:, :k] ∩ truth[:, :k]| / k averaged over queries (id-based k-recall@k).

    Missing results (-1, returned by IVF when the probed lists hold fewer than k vectors)
    count as misses. Runs in O(nq * k * log) through one global searchsorted.
    """
    f = np.asarray(found[:, :k], dtype=np.int64)
    t = np.sort(np.asarray(truth[:, :k], dtype=np.int64), axis=1)
    n = f.shape[0]
    big = int(max(f.max(initial=0), t.max(initial=0))) + 2
    off = (np.arange(n, dtype=np.int64) * big)[:, None]
    tf = (t + off).ravel()  # sorted within rows and row offsets ascend -> globally sorted
    ff = (np.where(f >= 0, f, big - 1) + off).ravel()  # big-1 never occurs in truth
    pos = np.minimum(np.searchsorted(tf, ff), len(tf) - 1)
    hits = (tf[pos] == ff).reshape(n, -1).sum(1) / float(k)
    return hits if per_query else float(hits.mean())


def label_precision_at_k(found: np.ndarray, yb: np.ndarray, yq: np.ndarray, k: int, per_query: bool = False):
    """Fraction of the top-k whose label equals the query label. Missing ids (-1) count as wrong."""
    f = found[:, :k]
    ok = (yb[np.clip(f, 0, len(yb) - 1)] == yq[:, None]) & (f >= 0)
    p = ok.mean(1)
    return p if per_query else float(p.mean())


def map_at_k(found: np.ndarray, yb: np.ndarray, yq: np.ndarray, k: int) -> float:
    """mAP@k: mean over queries of (1/k) * sum_i P@i * rel_i  (assumes >= k relevant items exist)."""
    f = found[:, :k]
    rel = ((yb[np.clip(f, 0, len(yb) - 1)] == yq[:, None]) & (f >= 0)).astype(np.float64)
    prec = np.cumsum(rel, 1) / np.arange(1, k + 1)[None, :]
    return float(((prec * rel).sum(1) / k).mean())


def top1_accuracy(found: np.ndarray, yb: np.ndarray, yq: np.ndarray) -> float:
    return label_precision_at_k(found, yb, yq, 1)
