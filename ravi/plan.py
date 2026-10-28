"""`plan`: size of a run before starting it, and time left once some of it has run."""
from __future__ import annotations

import glob
import os
from collections import defaultdict

from .bench import build_grid, search_sweep
from .config import Paths, list_spaces, model_specs
from .utils import log, read_json


def run_plan(cfg: dict, pilot_dir: str | None = None) -> dict:
    paths = Paths(cfg)
    bcfg = cfg["bench"]
    spaces = list_spaces(cfg, paths, require_native=False)
    missing = [s["model"] for s in spaces if s["reduction"] == "native" and not s["native_dim"]]
    if cfg["synthetic"]:
        n_base, n_query = int(cfg["synthetic"]["n_base"]), int(cfg["synthetic"]["n_query"])
    else:
        ds = cfg["dataset"] or {}
        n_base, n_query = int(ds.get("n_base", 0)), int(ds.get("n_query", 0))
    builds, searches, per_kind = 0, 0, defaultdict(int)
    disk_native = disk_reduced = 0
    for s in spaces:
        if not s["dim"]:
            continue
        if s["reduction"] == "native":
            disk_native += (n_base + n_query) * s["dim"] * 4
        else:
            disk_reduced += (n_base + n_query) * s["dim"] * 4
        for kind, bp in build_grid(bcfg, s["dim"], n_base or int(1e18)):
            ns = 1 if kind == "flat" else len(bcfg["seeds"])
            builds += ns
            per_kind[kind] += ns
            searches += ns * len(search_sweep(kind, bcfg, bp)[1]) * len(bcfg["ks"])
    out = {"models": len(model_specs(cfg)), "spaces": len(spaces), "index_builds": builds,
           "builds_per_index": dict(per_kind), "search_measurements": searches,
           "disk_native_gb": disk_native / 1e9, "disk_reduced_gb": disk_reduced / 1e9}
    log.info(f"[plan] {out['models']} backbones, {out['spaces']} spaces, {builds} index builds "
             f"{dict(per_kind)}, {searches} search measurements")
    if n_base:
        log.info(f"[plan] disk: native embeddings {out['disk_native_gb']:.0f} GB + compact spaces "
                 f"{out['disk_reduced_gb']:.0f} GB (float32, {n_base} base + {n_query} query vectors)")
    else:
        log.info("[plan] set dataset.n_base / dataset.n_query in the config to get a disk estimate")
    if missing:
        log.info(f"[plan] native dimension unknown for: {missing} (add `dim:` to the model entry)")

    # measured time so far in THIS run, and what is left at the same per-build cost
    done = defaultdict(list)
    for f in glob.glob(os.path.join(paths.parts("bench"), "*.json")):
        o = read_json(f)
        if o.get("rows"):
            done[o["rows"][0]["index"]].append(o["wall_seconds"])
    if done:
        spent = sum(sum(v) for v in done.values())
        left = sum((per_kind[k] - len(v)) * (sum(v) / len(v)) for k, v in done.items() if per_kind[k] > len(v))
        log.info(f"[plan] benchmark so far: {sum(len(v) for v in done.values())} builds, {spent / 3600:.2f} h measured; "
                 f"about {left / 3600:.2f} h left at the same cost per build")
        out.update(bench_hours_measured=spent / 3600, bench_hours_left=left / 3600)
    if pilot_dir:
        pil = defaultdict(list)
        n_pilot = None
        for f in glob.glob(os.path.join(pilot_dir, "parts", "bench", "*.json")):
            o = read_json(f)
            if o.get("rows"):
                pil[o["rows"][0]["index"]].append(o["wall_seconds"])
                n_pilot = o["rows"][0]["n_base"]
        if pil and n_pilot and n_base:
            scale = n_base / n_pilot
            est = sum(per_kind[k] * (sum(v) / len(v)) * scale for k, v in pil.items())
            log.info(f"[plan] rough extrapolation from pilot ({n_pilot} -> {n_base} vectors, linear in N, "
                     f"same machine assumed): about {est / 3600:.1f} h of benchmark time. "
                     "HNSW build grows slightly faster than linear, so treat this as a lower bound.")
            out["bench_hours_extrapolated"] = est / 3600
    return out
