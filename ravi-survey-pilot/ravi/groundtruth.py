"""Exact ground truth for every space, plus the exact-search quality of each space."""
from __future__ import annotations

import os

import numpy as np

from .config import Paths, list_spaces, native_id
from .data import load_space
from .knn import exact_knn
from .metrics import label_precision_at_k, map_at_k, recall_at_k, top1_accuracy
from .utils import log, pick_device, timer, write_json


def load_gt(paths: Paths, space_id: str) -> np.ndarray:
    return np.load(paths.gt(space_id))["I"]


def run_gt(cfg: dict) -> None:
    paths = Paths(cfg)
    device = pick_device(cfg["device"])
    k = int(cfg["gt"]["k"])
    spaces = list_spaces(cfg, paths)
    spaces.sort(key=lambda s: s["reduction"] != "native")  # native first: reduced spaces compare against it
    part_dir = paths.parts("quality")
    for s in spaces:
        out_gt, out_q = paths.gt(s["space_id"]), os.path.join(part_dir, f"{s['space_id']}.json")
        if os.path.exists(out_gt) and os.path.exists(out_q):
            continue
        xb, xq, yb, yq = load_space(paths, s)
        kk = min(k, len(xb))
        with timer() as t:
            idx, sim = exact_knn(xb, xq, kk, device=device)
        tmp = out_gt + f".tmp{os.getpid()}.npz"
        np.savez(tmp, I=idx.astype(np.int32), S=sim)
        os.replace(tmp, out_gt)
        row = {**s, "n_base": len(xb), "n_query": len(xq), "gt_seconds": t()}
        nat = load_gt(paths, native_id(s["model"]))
        for kq in (1, 10, 100):
            if kq <= kk:
                # ceiling for any index built in this space: exact search here vs exact search in native space
                row[f"reduction_recall@{kq}"] = recall_at_k(idx, nat, kq)
        if yb is not None and (yq >= 0).any():
            row["label_top1"] = top1_accuracy(idx, yb, yq)
            row["label_p@10"] = label_precision_at_k(idx, yb, yq, min(10, kk))
            row["label_map@100"] = map_at_k(idx, yb, yq, min(100, kk))
        write_json(out_q, row)
        log.info(f"[gt] {s['space_id']}: reduction_recall@10={row.get('reduction_recall@10', float('nan')):.4f} "
                 f"label_p@10={row.get('label_p@10', float('nan')):.4f}")
