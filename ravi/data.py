"""Data sources: image folders / file lists for real runs, synthetic vectors for the smoke test."""
from __future__ import annotations

import os

import numpy as np

from .utils import l2n, log, read_json, stable_hash, write_json

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG")


# --------------------------------------------------------------------------- synthetic
def make_synthetic_space(spec: dict, n_base: int, n_query: int, seed: int):
    """A controllable fake 'embedding space' used ONLY to test that the code runs.

    Gaussian clusters living on an `intrinsic`-dimensional subspace of R^D, plus ambient
    noise, an optional common offset (`cone`, makes the space anisotropic) and optional
    Zipf-distributed cluster sizes. Returns raw (un-normalised) vectors and cluster labels.
    """
    rng = np.random.default_rng(seed * 1000003 + stable_hash(spec["name"]))
    D, m = int(spec["D"]), int(spec.get("intrinsic", 16))
    C = int(spec.get("clusters", 50))
    noise, cone = float(spec.get("noise", 0.05)), float(spec.get("cone", 0.0))
    sep, zipf = float(spec.get("sep", 1.0)), float(spec.get("zipf", 0.0))
    spread = float(spec.get("spread", 0.0))
    centres = rng.standard_normal((C, m)) * sep
    sig = 1.0 + spread * rng.random(C)
    basis, _ = np.linalg.qr(rng.standard_normal((D, m)))
    offset = rng.standard_normal(D)
    offset /= np.linalg.norm(offset)
    pri = 1.0 / np.arange(1, C + 1) ** zipf
    pri /= pri.sum()

    def draw(n):
        y = rng.choice(C, size=n, p=pri)
        z = centres[y] + rng.standard_normal((n, m)) * sig[y, None]
        x = z @ basis.T / np.sqrt(m) + noise * rng.standard_normal((n, D)) + cone * offset
        return x.astype(np.float32), y.astype(np.int64)

    xb, yb = draw(n_base)
    xq, yq = draw(n_query)
    return xb, xq, yb, yq


# --------------------------------------------------------------------------- images
def _scan_imagefolder(root: str):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not classes:
        raise FileNotFoundError(
            f"{root} has no class sub-folders. Expected <root>/<class>/<image>. "
            "For the ImageNet validation set, sort the images into class folders first (see README)."
        )
    paths, names = [], []
    for c in classes:
        cdir = os.path.join(root, c)
        for fn in sorted(os.listdir(cdir)):
            if fn.endswith(IMG_EXT):
                paths.append(os.path.join(c, fn))
                names.append(c)
    return paths, names


def _hf_token(spec: dict):
    env = spec.get("token_env", "HF_TOKEN")
    return os.environ.get(env) or None


def load_hf_split(spec: dict):
    """A Hugging Face image dataset split. `path` may be a hub id (e.g. "ethz/food101",
    "ILSVRC/imagenet-1k"), a builder name with `data_files` (e.g. "parquet"), or a folder
    written by `save_to_disk`."""
    import datasets

    path = spec["path"]
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_info.json")):
        ds = datasets.load_from_disk(path)
        return ds[spec["split"]] if isinstance(ds, datasets.DatasetDict) else ds
    return datasets.load_dataset(path, name=spec.get("name"), split=spec["split"], data_files=spec.get("data_files"),
                                 cache_dir=spec.get("cache_dir"), token=_hf_token(spec))


def _hf_labels(spec: dict):
    ds = load_hf_split(spec)
    key = spec.get("label_key", "label")
    labels = np.asarray(ds[key], dtype=np.int64)
    feat = ds.features.get(key)
    names = list(getattr(feat, "names", []) or [])
    return labels, names


def build_image_list(spec: dict, cache_path: str, seed: int, class_to_idx: dict | None = None,
                     exclude: list[str] | None = None):
    """Deterministic list of (path or row id, label) for one split; cached as JSON.

    spec: {type: imagefolder|filelist|hf, root, list, path, split, per_class, max_images}
    A seeded per-class subsample makes pilot subsets reproducible. Labels are -1 when unknown.
    `exclude` removes items already used elsewhere (e.g. queries drawn from the same split as the base).
    """
    if os.path.exists(cache_path):
        c = read_json(cache_path)
        if c.get("spec") == spec:
            return c["paths"], np.asarray(c["labels"], dtype=np.int64), c["class_to_idx"]
    kind = spec.get("type", "imagefolder")
    if kind == "hf":
        lab, names_list = _hf_labels(spec)
        paths = [str(i) for i in range(len(lab))]  # row ids of the split
        if class_to_idx is None:
            uniq = sorted(set(lab.tolist()) - {-1})
            class_to_idx = {(names_list[c] if c < len(names_list) else str(c)): int(c) for c in uniq}
        names = [None] * len(lab)
        labels = lab.copy()
    else:
        if kind == "imagefolder":
            paths, names = _scan_imagefolder(spec["root"])
        elif kind == "filelist":
            paths, names = [], []
            with open(spec["list"]) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    p, _, lab_s = line.partition(",")
                    paths.append(p)
                    names.append(lab_s if lab_s else None)
        else:
            raise ValueError(f"unknown dataset type {kind}")
        if class_to_idx is None:
            uniq = sorted({n for n in names if n is not None})
            class_to_idx = {n: i for i, n in enumerate(uniq)}
        labels = np.asarray([class_to_idx.get(n, -1) if n is not None else -1 for n in names], dtype=np.int64)
    rng = np.random.default_rng(seed)
    keep = np.arange(len(paths))
    if exclude:
        ex = set(exclude)
        keep = np.asarray([i for i in keep if paths[i] not in ex], dtype=np.int64)
    per_class = spec.get("per_class")
    if per_class:
        sel = []
        sub = labels[keep]
        for c in np.unique(sub):
            idx = keep[sub == c]
            sel.append(rng.permutation(idx)[: int(per_class)])
        keep = np.sort(np.concatenate(sel)) if sel else keep
    if spec.get("max_images") and len(keep) > int(spec["max_images"]):
        keep = np.sort(rng.permutation(keep)[: int(spec["max_images"])])
    paths = [paths[i] for i in keep]
    labels = labels[keep]
    write_json(cache_path, {"spec": spec, "paths": paths, "labels": labels.tolist(), "class_to_idx": class_to_idx})
    log.info(f"image list: {len(paths)} images, {len(class_to_idx)} classes -> {cache_path}")
    return paths, labels, class_to_idx


class ImageList:
    """torch Dataset over (root, relative paths). Unreadable files become a black image and are counted."""

    def __init__(self, root: str, paths: list[str], transform):
        self.root, self.paths, self.transform = root, paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        bad = 0
        try:
            img = Image.open(os.path.join(self.root, self.paths[i])).convert("RGB")
        except Exception:  # corrupt / truncated file
            img = Image.new("RGB", (256, 256))
            bad = 1
        return self.transform(img), bad


class HFImageList:
    """torch Dataset over rows of a Hugging Face image dataset (images decoded on access, no copies on disk)."""

    def __init__(self, spec: dict, rows: list[str], transform):
        self.spec, self.rows, self.transform = spec, [int(r) for r in rows], transform
        self.key = spec.get("image_key", "image")
        self._ds = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image

        if self._ds is None:  # opened lazily so that every DataLoader worker has its own handle
            self._ds = load_hf_split(self.spec)
        bad = 0
        try:
            img = self._ds[self.rows[i]][self.key]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            img = img.convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
            bad = 1
        return self.transform(img), bad


def make_dataset(spec: dict, items: list[str], transform):
    if spec.get("type") == "hf":
        return HFImageList(spec, items, transform)
    return ImageList(spec.get("root", ""), items, transform)


def load_native(paths, model: str, normalise: bool = True):
    """Native embeddings of a backbone: (base, query, base_labels, query_labels)."""
    xb = np.load(paths.emb(model, "base"), mmap_mode="r")
    xq = np.load(paths.emb(model, "query"), mmap_mode="r")
    yb = np.load(paths.labels(model, "base")) if os.path.exists(paths.labels(model, "base")) else None
    yq = np.load(paths.labels(model, "query")) if os.path.exists(paths.labels(model, "query")) else None
    if normalise:
        xb, xq = l2n(xb), l2n(xq)
    return xb, xq, yb, yq


def load_space(paths, space: dict):
    """L2-normalised vectors of any space (native or reduced), fully in RAM."""
    if space["reduction"] == "native":
        return load_native(paths, space["model"])
    d = paths.space_dir(space["space_id"])
    xb = np.load(os.path.join(d, "base.npy"))
    xq = np.load(os.path.join(d, "query.npy"))
    yb = np.load(paths.labels(space["model"], "base")) if os.path.exists(paths.labels(space["model"], "base")) else None
    yq = np.load(paths.labels(space["model"], "query")) if os.path.exists(paths.labels(space["model"], "query")) else None
    return xb, xq, yb, yq
