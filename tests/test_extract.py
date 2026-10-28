"""Image -> embedding path on a tiny fake image folder (random-weight ResNet-18, CPU, no download)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fake_folder(root, n_classes, per_class, seed):
    from PIL import Image

    rng = np.random.default_rng(seed)
    for c in range(n_classes):
        d = os.path.join(root, f"class{c:02d}")
        os.makedirs(d, exist_ok=True)
        for i in range(per_class):
            Image.fromarray(rng.integers(0, 255, (72, 96, 3), dtype=np.uint8)).save(os.path.join(d, f"im{i:03d}.jpg"))
    with open(os.path.join(root, "class00", "broken.jpg"), "w") as f:  # an unreadable file must not crash a run
        f.write("not an image")


def test_extract_resume_and_labels(tmp_path):
    from ravi.config import DEFAULTS, Paths, _deep_update
    from ravi.data import build_image_list
    from ravi.extract import _extract_split, run_extract
    from ravi.models import load_backbone
    from ravi.utils import read_json, write_json
    import copy

    tr, va = str(tmp_path / "train"), str(tmp_path / "val")
    _fake_folder(tr, 4, 6, 0)
    _fake_folder(va, 4, 2, 1)
    cfg = _deep_update(copy.deepcopy(DEFAULTS), {
        "out_dir": str(tmp_path / "run"), "device": "cpu",
        "dataset": {"name": "fake", "base": {"type": "imagefolder", "root": tr},
                    "query": {"type": "imagefolder", "root": va}},
        "models": [{"name": "resnet18.a1_in1k", "family": "resnet", "paradigm": "supervised"}],
        "extract": {"batch_size": 5, "workers": 0, "amp": False, "pretrained": False}})
    run_extract(cfg)
    paths = Paths(cfg)
    meta = read_json(paths.emb_meta("resnet18.a1_in1k"))
    xb = np.load(paths.emb("resnet18.a1_in1k", "base"))
    yb = np.load(paths.labels("resnet18.a1_in1k", "base"))
    assert xb.shape == (25, 512) and meta["dim"] == 512 and meta["unreadable_images"] == 2
    assert np.isfinite(xb).all() and (np.bincount(yb) == [7, 6, 6, 6]).all()

    # resume: pretend the job died after 10 images and check the result is identical
    rel, _, _ = build_image_list(cfg["dataset"]["base"], os.path.join(paths.lists, "base.json"), 0)
    model, transform, _ = load_backbone(cfg["models"][0], "cpu", pretrained=False)
    ref_path, res_path = str(tmp_path / "ref.npy"), str(tmp_path / "res.npy")
    _extract_split(model, transform, cfg["dataset"]["base"], rel, ref_path, "cpu", cfg["extract"])
    ref = np.load(ref_path)
    part = np.lib.format.open_memmap(res_path, mode="w+", dtype=np.float32, shape=ref.shape)
    part[:10] = ref[:10]
    part.flush()
    del part
    write_json(res_path + ".progress.json", {"done": 10})
    _extract_split(model, transform, cfg["dataset"]["base"], rel, res_path, "cpu", cfg["extract"])
    assert np.allclose(np.load(res_path), ref, atol=1e-5)
    assert not os.path.exists(res_path + ".progress.json")
