"""Stage 4: FAISS index benchmark.

For every space x index type x build setting x seed:
  build once (time, size), then sweep the search parameter and record for each setting
    ndis        distance (or PQ code) evaluations per query: hardware-independent cost
    recall      vs exact search in the same space
    e2e_recall  vs exact search in the native space of the same backbone
    label_p     class precision of the returned images
    qps_batch   all queries in one call, `threads` OpenMP threads (best of `repeats`)
    latency     one query at a time, ONE thread: mean / P50 / P95 / P99, and qps_1t = 1 / mean

Seeds change (a) the insertion order (HNSW graphs depend on it) and (b) the k-means seed of
the IVF coarse quantiser and of the product quantiser.

All vectors are L2-normalised and indexes use the L2 metric, which ranks identically to cosine.
Timing only means something on a quiet machine: run this stage with nothing else running.
"""
from __future__ import annotations

import fnmatch
import os
import time

import numpy as np

from .config import Paths, list_spaces, native_id
from .data import load_space
from .groundtruth import load_gt
from .metrics import label_precision_at_k, recall_at_k
from .utils import chunks, env_info, log, timer, write_json


def build_grid(bcfg: dict, d: int, n: int) -> list[tuple[str, dict]]:
    """All (index kind, build parameters) to run for a space of dimension d with n vectors."""
    grid: list[tuple[str, dict]] = []
    if bcfg.get("flat", True):
        grid.append(("flat", {}))
    h = bcfg.get("hnsw")
    if h:
        for M in h["M"]:
            grid.append(("hnsw", {"M": int(M), "efConstruction": int(h["efConstruction"])}))
    iv = bcfg.get("ivf")
    if iv:
        for nlist in iv["nlist"]:
            if n >= 39 * int(nlist):  # FAISS wants >= 39 training points per centroid
                grid.append(("ivf", {"nlist": int(nlist)}))
    pq = bcfg.get("ivfpq")
    if pq:
        for nlist in pq["nlist"]:
            for m in pq["m"]:
                if d % int(m) == 0 and d // int(m) >= 2 and n >= 39 * int(nlist):
                    grid.append(("ivfpq", {"nlist": int(nlist), "pq_m": int(m), "pq_nbits": int(pq["nbits"])}))
    return grid


def build_key(kind: str, bp: dict) -> str:
    if kind == "flat":
        return "flat"
    if kind == "hnsw":
        return f"hnsw_M{bp['M']}_efc{bp['efConstruction']}"
    if kind == "ivf":
        return f"ivf_nlist{bp['nlist']}"
    return f"ivfpq_nlist{bp['nlist']}_m{bp['pq_m']}x{bp['pq_nbits']}"


def search_sweep(kind: str, bcfg: dict, bp: dict) -> tuple[str, list[int]]:
    if kind == "flat":
        return "none", [0]
    if kind == "hnsw":
        return "efSearch", [int(v) for v in bcfg["hnsw"]["efSearch"]]
    key = "ivf" if kind == "ivf" else "ivfpq"
    return "nprobe", [int(v) for v in bcfg[key]["nprobe"] if int(v) <= bp["nlist"]]


def _make_index(kind: str, d: int, bp: dict, seed: int):
    import faiss

    if kind == "flat":
        return faiss.IndexFlatL2(d)
    if kind == "hnsw":
        idx = faiss.IndexHNSWFlat(d, bp["M"])
        idx.hnsw.efConstruction = bp["efConstruction"]
        return idx
    quant = faiss.IndexFlatL2(d)
    if kind == "ivf":
        idx = faiss.IndexIVFFlat(quant, d, bp["nlist"])
    else:
        idx = faiss.IndexIVFPQ(quant, d, bp["nlist"], bp["pq_m"], bp["pq_nbits"])
        idx.pq.cp.seed = int(seed) + 7
    idx.cp.seed = int(seed) + 3
    idx.cp.niter = 20
    return idx


def _set_param(index, kind: str, value: int) -> None:
    if kind == "hnsw":
        index.hnsw.efSearch = int(value)
    elif kind in ("ivf", "ivfpq"):
        index.nprobe = int(value)


def _index_bytes(index, tmp_dir: str) -> int:
    import faiss

    path = os.path.join(tmp_dir, f"index_{os.getpid()}.faiss")
    faiss.write_index(index, path)
    size = os.path.getsize(path)
    os.remove(path)
    return int(size)


def _reset_stats() -> None:
    import faiss

    faiss.cvar.hnsw_stats.reset()
    faiss.cvar.indexIVF_stats.reset()


def _distances_per_query(kind: str, index, n: int, n_queries: int) -> float:
    """Hardware-independent cost: distance (or code) evaluations per query, from FAISS counters.

    flat: n.  hnsw: all distances computed during graph search.  ivf/ivfpq: nlist distances to the
    coarse centroids plus the vectors (ivf) or codes (ivfpq) scanned in the probed lists.
    """
    import faiss

    if kind == "flat":
        return float(n)
    if kind == "hnsw":
        return faiss.cvar.hnsw_stats.ndis / max(n_queries, 1)
    return index.nlist + faiss.cvar.indexIVF_stats.ndis / max(n_queries, 1)


def bench_one(xb, xq, gt, gt_native, yb, yq, kind, bp, seed, bcfg, tmp_dir) -> list[dict]:
    import faiss
    import psutil

    n, d = xb.shape
    threads = int(bcfg["threads"]) or faiss.omp_get_max_threads()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)  # seed-dependent insertion order
    xb_p = np.ascontiguousarray(xb[perm])
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    faiss.omp_set_num_threads(threads)
    index = _make_index(kind, d, bp, seed)
    t_train = 0.0
    if not index.is_trained:
        n_train = min(n, max(256 * bp["nlist"], 100000))
        tr = xb_p[:n_train]  # xb_p is already a random permutation
        with timer() as t:
            index.train(tr)
        t_train = t()
    with timer() as t:
        for a, b in chunks(n, 100000):
            index.add(xb_p[a:b])
    t_add = t()
    rss1 = proc.memory_info().rss
    size = _index_bytes(index, tmp_dir)
    base = {"n_base": n, "n_query": len(xq), "index": kind, "build_key": build_key(kind, bp), "seed": seed,
            "M": bp.get("M"), "efConstruction": bp.get("efConstruction"), "nlist": bp.get("nlist"),
            "pq_m": bp.get("pq_m"), "pq_nbits": bp.get("pq_nbits"),
            "train_time_s": t_train, "add_time_s": t_add, "build_time_s": t_train + t_add,
            "index_bytes": size, "bytes_per_vec": size / n, "raw_bytes_per_vec": 4.0 * d,
            "rss_delta_mb": (rss1 - rss0) / 2**20, "threads": threads}
    pname, values = search_sweep(kind, bcfg, bp)
    nq_lat = min(len(xq), int(bcfg["n_latency_queries"]))
    lat_ids = rng.choice(len(xq), nq_lat, replace=False)
    xq_c = np.ascontiguousarray(xq)
    lat_k = int(bcfg["latency_k"])
    rows = []
    for v in values:
        _set_param(index, kind, v)
        row_common = {**base, "search_param": pname, "search_value": v}
        stop = False
        for k in bcfg["ks"]:
            k = int(k)
            if k > n:
                continue
            faiss.omp_set_num_threads(threads)
            index.search(xq_c[: min(64, len(xq_c))], k)  # warm-up
            _reset_stats()
            best, found = float("inf"), None
            reps = int(bcfg["repeats"])
            for _ in range(reps):
                t0 = time.perf_counter()
                _, found = index.search(xq_c, k)
                best = min(best, time.perf_counter() - t0)
            ndis = _distances_per_query(kind, index, n, reps * len(xq_c))
            found = np.where(found >= 0, perm[np.clip(found, 0, n - 1)], -1)  # back to original ids
            row = {**row_common, "k": k, "recall": recall_at_k(found, gt, k),
                   "e2e_recall": recall_at_k(found, gt_native, k), "qps_batch": len(xq_c) / best,
                   "ndis": ndis}
            if yb is not None and (yq >= 0).any():
                row["label_p"] = label_precision_at_k(found, yb, yq, k)
            if k == lat_k:
                faiss.omp_set_num_threads(1)
                lat = np.empty(nq_lat)
                for j, qi in enumerate(lat_ids):
                    q = xq_c[qi:qi + 1]
                    t0 = time.perf_counter()
                    index.search(q, k)
                    lat[j] = time.perf_counter() - t0
                lat *= 1000.0
                row.update(lat_mean_ms=float(lat.mean()), lat_p50_ms=float(np.percentile(lat, 50)),
                           lat_p95_ms=float(np.percentile(lat, 95)), lat_p99_ms=float(np.percentile(lat, 99)),
                           qps_1t=float(1000.0 / lat.mean()))
                stop = row["recall"] >= float(bcfg["early_stop_recall"])
            rows.append(row)
        if stop:
            break  # larger search effort cannot improve an already (near) exact result
    faiss.omp_set_num_threads(threads)
    return rows


def run_bench(cfg: dict) -> None:
    paths = Paths(cfg)
    bcfg = cfg["bench"]
    part_dir = paths.parts("bench")
    tmp_dir = paths.parts("tmp")
    write_json(os.path.join(paths.results, "env.json"), env_info())
    spaces = list_spaces(cfg, paths)
    if bcfg.get("spaces"):
        spaces = [s for s in spaces if any(fnmatch.fnmatch(s["space_id"], p) for p in bcfg["spaces"])]
    for si, s in enumerate(spaces):
        todo = []
        for kind, bp in build_grid(bcfg, s["dim"], int(1e18)):
            seeds = bcfg["seeds"][:1] if kind == "flat" else bcfg["seeds"]
            for seed in seeds:
                out = os.path.join(part_dir, f"{s['space_id']}__{build_key(kind, bp)}__s{seed}.json")
                if not os.path.exists(out):
                    todo.append((kind, bp, int(seed), out))
        if not todo:
            continue
        xb, xq, yb, yq = load_space(paths, s)
        gt = load_gt(paths, s["space_id"])
        gt_native = load_gt(paths, native_id(s["model"]))
        valid = {build_key(k, b) for k, b in build_grid(bcfg, s["dim"], len(xb))}
        for kind, bp, seed, out in todo:
            if build_key(kind, bp) not in valid:
                write_json(out, {"skipped": "too few vectors for this nlist", "rows": []})
                continue
            with timer() as t:
                rows = bench_one(xb, xq, gt, gt_native, yb, yq, kind, bp, seed, bcfg, tmp_dir)
            for r in rows:
                r.update({k: s[k] for k in ("space_id", "model", "family", "paradigm", "reduction", "dim", "native_dim")})
            write_json(out, {"rows": rows, "wall_seconds": t(), "synthetic": bool(cfg["synthetic"])})
            top = max((r["recall"] for r in rows if r["k"] == int(bcfg["latency_k"])), default=float("nan"))
            log.info(f"[bench {si + 1}/{len(spaces)}] {s['space_id']} {build_key(kind, bp)} seed={seed}: "
                     f"build {rows[0]['build_time_s']:.1f}s, max recall@{bcfg['latency_k']} {top:.3f}, {t():.0f}s")
