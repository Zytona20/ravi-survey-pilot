"""Stage 1: images -> native embeddings (resumable, one .npy memmap per backbone and split)."""
from __future__ import annotations

import os
import time

import numpy as np

from .config import Paths, model_specs
from .data import build_image_list, make_dataset, make_synthetic_space
from .models import forward_features, load_backbone
from .utils import log, pick_device, read_json, save_npy, write_json


def _extract_split(model, transform, spec, rel_paths, out_path, device, ecfg):
    import torch

    n = len(rel_paths)
    prog_path = out_path + ".progress.json"
    done = read_json(prog_path)["done"] if os.path.exists(prog_path) and os.path.exists(out_path) else 0
    ds = make_dataset(spec, rel_paths, transform)
    if done:
        ds = torch.utils.data.Subset(ds, range(done, n))
        log.info(f"  resuming at image {done}/{n}")
    dl = torch.utils.data.DataLoader(ds, batch_size=int(ecfg["batch_size"]), shuffle=False,
                                     num_workers=int(ecfg["workers"]), pin_memory=device.startswith("cuda"))
    mm, bad_total, pos, t0, last = None, 0, done, time.time(), time.time()
    if done:
        mm = np.lib.format.open_memmap(out_path, mode="r+")
    for images, bad in dl:
        feats = forward_features(model, images, device, bool(ecfg["amp"]))
        if mm is None:
            mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, feats.shape[1]))
        mm[pos:pos + len(feats)] = feats
        pos += len(feats)
        bad_total += int(bad.sum())
        if time.time() - last > 60:
            mm.flush()
            write_json(prog_path, {"done": pos})
            rate = (pos - done) / (time.time() - t0)
            log.info(f"  {pos}/{n} images  {rate:.0f} img/s  eta {(n - pos) / max(rate, 1e-9) / 60:.1f} min")
            last = time.time()
    mm.flush()
    dim = int(mm.shape[1])
    del mm
    if os.path.exists(prog_path):
        os.remove(prog_path)
    return dim, bad_total, (n - done) / max(time.time() - t0, 1e-9)


def run_extract(cfg: dict) -> None:
    paths = Paths(cfg)
    if cfg["synthetic"]:
        return _run_synthetic(cfg, paths)
    device = pick_device(cfg["device"])
    ds = cfg["dataset"]
    if all(os.path.exists(paths.emb_meta(m["name"])) for m in model_specs(cfg)):
        log.info("[extract] all embeddings exist, nothing to do")
        return
    base_paths, yb, c2i = build_image_list(ds["base"], os.path.join(paths.lists, "base.json"), cfg["seed"])
    same_split = {k: v for k, v in ds["base"].items() if k not in ("per_class", "max_images")} == \
        {k: v for k, v in ds["query"].items() if k not in ("per_class", "max_images")}
    query_paths, yq, _ = build_image_list(ds["query"], os.path.join(paths.lists, "query.json"), cfg["seed"] + 1, c2i,
                                          exclude=base_paths if same_split else None)
    for spec in model_specs(cfg):
        name = spec["name"]
        if os.path.exists(paths.emb_meta(name)):
            log.info(f"[extract] {name}: done, skipping")
            continue
        log.info(f"[extract] {name} on {device}")
        ecfg = {**cfg["extract"], **{k: spec[k] for k in ("batch_size",) if k in spec}}
        model, transform, data_cfg = load_backbone(spec, device, pretrained=bool(cfg["extract"].get("pretrained", True)))
        t0 = time.time()
        dim, bad_b, ips = _extract_split(model, transform, ds["base"], base_paths, paths.emb(name, "base"), device, ecfg)
        _, bad_q, _ = _extract_split(model, transform, ds["query"], query_paths, paths.emb(name, "query"), device, ecfg)
        save_npy(paths.labels(name, "base"), yb)
        save_npy(paths.labels(name, "query"), yq)
        write_json(paths.emb_meta(name), {
            "model": name, "dim": dim, "n_base": len(base_paths), "n_query": len(query_paths),
            "input_size": list(data_cfg["input_size"]), "unreadable_images": bad_b + bad_q,
            "images_per_second": ips, "seconds": time.time() - t0, "dataset": ds.get("name"),
            "synthetic": False, "pretrained": bool(cfg["extract"].get("pretrained", True)),
            **{k: spec.get(k) for k in ("family", "paradigm", "keep_head")}})
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        log.info(f"[extract] {name}: dim={dim}, {ips:.0f} img/s, {bad_b + bad_q} unreadable images")


def _run_synthetic(cfg: dict, paths: Paths) -> None:
    syn = cfg["synthetic"]
    for spec in syn["spaces"]:
        name = spec["name"]
        if os.path.exists(paths.emb_meta(name)):
            continue
        xb, xq, yb, yq = make_synthetic_space(spec, int(syn["n_base"]), int(syn["n_query"]), cfg["seed"])
        save_npy(paths.emb(name, "base"), xb)
        save_npy(paths.emb(name, "query"), xq)
        save_npy(paths.labels(name, "base"), yb)
        save_npy(paths.labels(name, "query"), yq)
        write_json(paths.emb_meta(name), {"model": name, "dim": int(xb.shape[1]), "n_base": len(xb),
                                          "n_query": len(xq), "synthetic": True, "spec": spec})
        log.info(f"[synthetic] {name}: {xb.shape}")
