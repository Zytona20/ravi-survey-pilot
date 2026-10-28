"""Small shared helpers: logging, seeding, atomic writes, timing, environment capture."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import re
import sys
import time
import zlib
from contextlib import contextmanager

import numpy as np


def get_logger(name: str = "ravi") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


log = get_logger()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_hash(text: str) -> int:
    """Deterministic 32-bit hash (Python's hash() is salted per process)."""
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "-", str(text))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def _clean(o):
    """Replace NaN/inf floats by None so the JSON is strict."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def write_json(path: str, obj) -> None:
    """Atomic JSON write: a crash can never leave a half-written result file."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(_clean(json.loads(json.dumps(obj, cls=_NpEncoder))), f, indent=1)
    os.replace(tmp, path)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def save_npy(path: str, arr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = f"{path}.tmp{os.getpid()}.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


@contextmanager
def timer():
    """``with timer() as t: ...`` then ``t()`` returns elapsed seconds."""
    t0 = time.perf_counter()
    box = {"t": None}
    yield lambda: box["t"] if box["t"] is not None else time.perf_counter() - t0
    box["t"] = time.perf_counter() - t0


def l2n(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return an L2-normalised float32 copy (rows)."""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def chunks(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(n, s + size)


def pick_device(pref: str | None = None) -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def env_info() -> dict:
    """Hardware / software fingerprint stored next to every result table."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        import faiss

        info["faiss"] = faiss.__version__
        info["faiss_compile_options"] = faiss.get_compile_options()
        info["faiss_max_threads"] = faiss.omp_get_max_threads()
    except ImportError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    return info
