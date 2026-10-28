"""Stage `crossmodal` (prediction P2 of the survey): text queries against image embeddings.

One CLIP model from open_clip embeds the database images, held-out query images and text prompts built from the
class names, so that all three live in the same space. The same HNSW index is then searched with image queries and
with text queries, and we record, per query type:
  - distance evaluations and graph hops per query needed to reach recall@10 = target (search work);
  - query--base shift (mean query nearest-neighbour distance over the database's own);
  - median local intrinsic dimensionality and relative contrast of the queries (query difficulty).
If text queries need more work while their local intrinsic dimensionality is not higher, the extra cost comes from
the shift (navigation) rather than from discrimination, as P2 predicts.

Requires `pip install open_clip_torch`. Outputs: results/survey_p2.csv, results/survey_p2_curves.csv,
figures/survey_p2.png. Resumable: embeddings are cached in <out_dir>/crossmodal/.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .collect import save_table
from .config import Paths
from .data import build_image_list, make_dataset
from .geometry import lid_mle
from .knn import exact_knn, sim_to_dist
from .metrics import label_precision_at_k, recall_at_k
from .utils import ensure_dir, l2n, log, pick_device, read_json, save_npy, write_json


def _lists(cfg: dict, paths: Paths):
    ds = cfg["dataset"]
    base, yb, c2i = build_image_list(ds["base"], os.path.join(paths.lists, "base.json"), cfg["seed"])
    same = {k: v for k, v in ds["base"].items() if k not in ("per_class", "max_images")} == \
        {k: v for k, v in ds["query"].items() if k not in ("per_class", "max_images")}
    query, yq, _ = build_image_list(ds["query"], os.path.join(paths.lists, "query.json"), cfg["seed"] + 1, c2i,
                                    exclude=base if same else None)
    return base, yb, query, yq, c2i


def _encode_images(model, preprocess, spec, rows, device, batch_size, workers):
    import torch

    dl = torch.utils.data.DataLoader(make_dataset(spec, rows, preprocess), batch_size=batch_size, shuffle=False,
                                     num_workers=workers, pin_memory=device.startswith("cuda"))
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
        for i, (x, _) in enumerate(dl):
            out.append(model.encode_image(x.to(device)).float().cpu().numpy())
            if i % 50 == 0:
                log.info(f"  images {min((i + 1) * batch_size, len(rows))}/{len(rows)}")
    return l2n(np.concatenate(out).astype(np.float32))


def _encode_texts(model, tokenizer, texts, device, batch_size=256):
    import torch

    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
        for s in range(0, len(texts), batch_size):
            tok = tokenizer(texts[s:s + batch_size]).to(device)
            out.append(model.encode_text(tok).float().cpu().numpy())
    return l2n(np.concatenate(out).astype(np.float32))


def query_geometry(xb_sample: np.ndarray, db_nn1: float, xq: np.ndarray, k: int = 20, seed: int = 0) -> dict:
    """Shift, local intrinsic dimensionality and relative contrast of a query set against a database sample."""
    kq = min(k, len(xb_sample))
    _, qs = exact_knn(xb_sample, xq, kq)
    qd = sim_to_dist(qs)
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(xq), 20000), rng.integers(0, len(xb_sample), 20000)
    rand = sim_to_dist((xq[a] * xb_sample[b]).sum(1)).mean()
    lid = lid_mle(qd, kq)
    return {"qshift_nn": float(qd[:, 0].mean() / max(db_nn1, 1e-12)), "qlid_median": float(np.nanmedian(lid)),
            "rc_q10": float(rand / max(qd[:, min(9, kq - 1)].mean(), 1e-12))}


def hnsw_sweep(index, xq: np.ndarray, gt: np.ndarray, k: int, ef_values) -> list[dict]:
    """Recall@k, distance evaluations and hops per query of an HNSW index for each efSearch."""
    import faiss

    rows = []
    for ef in ef_values:
        index.hnsw.efSearch = int(ef)
        faiss.cvar.hnsw_stats.reset()
        _, found = index.search(np.ascontiguousarray(xq, dtype=np.float32), k)
        st = faiss.cvar.hnsw_stats
        rows.append({"efSearch": int(ef), "recall@10": recall_at_k(found, gt, k),
                     "ndis": st.ndis / len(xq), "nhops": getattr(st, "nhops", float("nan")) / len(xq)})
    return rows


def at_target(rows: list[dict], target: float) -> dict:
    ok = [r for r in rows if r["recall@10"] >= target]
    if not ok:
        return {"reaches_target": False, "ndis_at_target": float("nan"), "nhops_at_target": float("nan"),
                "ef_at_target": float("nan"), "max_recall@10": max(r["recall@10"] for r in rows)}
    best = min(ok, key=lambda r: r["ndis"])
    return {"reaches_target": True, "ndis_at_target": best["ndis"], "nhops_at_target": best["nhops"],
            "ef_at_target": best["efSearch"], "max_recall@10": max(r["recall@10"] for r in rows)}


def run_crossmodal(cfg: dict) -> None:
    import faiss

    paths = Paths(cfg)
    cc = cfg["crossmodal"]
    if not cfg.get("dataset"):
        log.info("[crossmodal] needs an image dataset; skipping")
        return
    device = pick_device(cfg["device"])
    out = ensure_dir(os.path.join(paths.out, "crossmodal"))
    base, yb, query, yq, c2i = _lists(cfg, paths)
    f_b, f_qi, f_qt = (os.path.join(out, f) for f in ("base.npy", "query_img.npy", "query_txt.npy"))
    f_meta = os.path.join(out, "meta.json")

    if not all(os.path.exists(f) for f in (f_b, f_qi, f_qt, f_meta)):
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(cc["model"], pretrained=cc["pretrained"])
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(cc["model"])
        bs, workers = int(cc["batch_size"]), int(cfg["extract"]["workers"])
        if not os.path.exists(f_b):
            log.info(f"[crossmodal] embedding {len(base)} database images with {cc['model']} ({cc['pretrained']})")
            save_npy(f_b, _encode_images(model, preprocess, cfg["dataset"]["base"], base, device, bs, workers))
        if not os.path.exists(f_qi):
            save_npy(f_qi, _encode_images(model, preprocess, cfg["dataset"]["query"], query, device, bs, workers))
        names = [n for n, _ in sorted(c2i.items(), key=lambda kv: kv[1])]
        idx_of = {n: i for n, i in c2i.items()}
        texts, ylab = [], []
        for n in names:
            clean = str(n).replace("_", " ")
            for t in cc["templates"]:
                texts.append(t.format(clean))
                ylab.append(idx_of[n])
        save_npy(f_qt, _encode_texts(model, tokenizer, texts, device))
        save_npy(os.path.join(out, "query_txt_labels.npy"), np.asarray(ylab, dtype=np.int64))
        write_json(f_meta, {"model": cc["model"], "pretrained": cc["pretrained"], "n_base": len(base),
                            "n_query_img": len(query), "n_query_txt": len(texts), "templates": cc["templates"]})
        del model
    xb, xqi, xqt = np.load(f_b), np.load(f_qi), np.load(f_qt)
    yqt = np.load(os.path.join(out, "query_txt_labels.npy"))
    k = int(cc["k"])

    # geometry of each query set against a fixed database sample
    rng = np.random.default_rng(cfg["seed"])
    sid = np.sort(rng.choice(len(xb), min(len(xb), int(cc["geometry_sample"])), replace=False))
    sample = np.ascontiguousarray(xb[sid])
    _, ss = exact_knn(sample, sample, 1, exclude_self=True)
    db_nn1 = float(sim_to_dist(ss)[:, 0].mean())

    # one HNSW index, searched with both query types
    h = cc["hnsw"]
    index = faiss.IndexHNSWFlat(xb.shape[1], int(h["M"]))
    index.hnsw.efConstruction = int(h["efConstruction"])
    index.add(np.ascontiguousarray(xb, dtype=np.float32))
    rows, curves = [], []
    for qtype, xq, yqq in (("image", xqi, yq), ("text", xqt, yqt)):
        gt, _ = exact_knn(xb, xq, k, device=device)
        sweep = hnsw_sweep(index, xq, gt, k, h["efSearch"])
        for r in sweep:
            curves.append({"query_type": qtype, **r})
        row = {"query_type": qtype, "n_queries": len(xq), **at_target(sweep, float(cc["recall_target"])),
               **query_geometry(sample, db_nn1, xq, seed=cfg["seed"])}
        if yb is not None and (np.asarray(yqq) >= 0).any():
            row["label_p@10_exact"] = label_precision_at_k(gt, np.asarray(yb), np.asarray(yqq), k)
        rows.append(row)
        log.info(f"[crossmodal] {qtype}: " + ", ".join(f"{a}={b:.3g}" for a, b in row.items()
                                                       if isinstance(b, (int, float)) and not isinstance(b, bool)))
    res = pd.DataFrame(rows)
    if len(res) == 2 and res["ndis_at_target"].notna().all():
        res["ndis_ratio_vs_image"] = res["ndis_at_target"] / res.loc[res["query_type"] == "image", "ndis_at_target"].iloc[0]
    save_table(res, os.path.join(paths.results, "survey_p2"))
    save_table(pd.DataFrame(curves), os.path.join(paths.results, "survey_p2_curves"))
    _figure(cfg, paths, pd.DataFrame(curves))


def _figure(cfg, paths, curves: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .figures import MARKERS, SERIES

    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    for i, (qt, g) in enumerate(curves.groupby("query_type")):
        g = g.sort_values("ndis")
        ax.plot(g["ndis"], g["recall@10"], color=SERIES[i], marker=MARKERS[i], label=f"{qt} queries")
    ax.axhline(float(cfg["crossmodal"]["recall_target"]), color="#52514e", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("distance evaluations / query (HNSW)")
    ax.set_ylabel("recall@10")
    ax.set_title("P2: the same index, image vs text queries", loc="left", fontsize=9)
    ax.legend()
    if (not cfg["extract"].get("pretrained", True)) or cfg["crossmodal"].get("pretrained") in (None, "", "none"):
        fig.text(0.5, 0.5, "TEST RUN - NOT A RESULT", rotation=24, ha="center", va="center", fontsize=20,
                 color="#d03b3b", alpha=0.22, weight="bold", zorder=100)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(paths.figures, f"survey_p2.{ext}"), dpi=300)
    plt.close(fig)
