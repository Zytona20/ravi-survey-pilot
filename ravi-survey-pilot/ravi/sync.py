"""Copying a run folder between fast local disk and Google Drive (or any slower, persistent folder).

Stages write to local disk. `push(local, drive)` copies new and changed files to Drive while a stage runs and after
it ends; `restore(drive, local)` brings a run back after Colab disconnects and the local disk is wiped.

Design choices, so that the copy on Drive is always usable for resuming, even if Colab disconnects mid-copy:
- each file is written to `<name>.partial` on Drive and then renamed, so a half-copied file never has its real name;
- the pipeline writes results, metadata and progress records (`.json`) only after the data they describe is on disk.
  A push copies data files first and, while a stage is running, only the `.json` files that already existed when the
  push started. A `.json` on Drive therefore never describes data that is not on Drive;
- whether a local file changed is decided from a manifest kept on the local disk (size and modification time of each
  local file when it was last copied), not from the timestamps on Drive, which Drive may not preserve. The manifest
  lives and dies with the local copy, so after a wipe everything restored is recorded afresh.
Nothing is ever deleted on either side.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time

MANIFEST = ".ravi_sync.json"
SKIP_PARTS = (".tmp", ".partial")
_lock = threading.Lock()


def _files(root: str):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in files:
            if f == MANIFEST or any(p in f for p in SKIP_PARTS):
                continue
            yield os.path.join(d, f)


def _sig(path: str):
    s = os.stat(path)
    return [s.st_size, s.st_mtime_ns]


def _load(local: str) -> dict:
    try:
        with open(os.path.join(local, MANIFEST)) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def _save(local: str, man: dict) -> None:
    os.makedirs(local, exist_ok=True)
    tmp = os.path.join(local, MANIFEST + ".tmp")
    with open(tmp, "w") as f:
        json.dump(man, f)
    os.replace(tmp, os.path.join(local, MANIFEST))


def _copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".partial"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def push(local: str, remote: str, markers_before: float | None = None) -> dict:
    """Copy local files that are new or changed since they were last copied, to `remote`.

    markers_before: copy `.json` files only if last modified before this time (seconds since the epoch). Use the
    current time when a stage may be writing to `local` during the push; use None when nothing is writing.
    """
    if not os.path.isdir(local):
        return {"files": 0, "bytes": 0}
    with _lock:
        man = _load(local)
        files = list(_files(local))
        data = [p for p in files if not p.endswith(".json")]
        markers = [p for p in files if p.endswith(".json")]
        n = nbytes = 0
        for group, is_marker in ((data, False), (markers, True)):
            for p in group:
                rel = os.path.relpath(p, local)
                q = os.path.join(remote, rel)
                try:
                    sig = _sig(p)  # taken before copying: a write during the copy changes it, so the next push recopies
                    if is_marker and markers_before is not None and sig[1] / 1e9 >= markers_before:
                        continue
                    if man.get(rel) == sig and os.path.exists(q):
                        continue
                    _copy(p, q)
                except FileNotFoundError:  # removed by the running stage (e.g. a finished progress record)
                    continue
                man[rel] = sig
                n += 1
                nbytes += sig[0]
        _save(local, man)
    return {"files": n, "bytes": nbytes}


def restore(remote: str, local: str) -> dict:
    """Copy files that exist on `remote` but not in `local`. Local files are never overwritten: if both exist, the
    local one is the same or newer (written after the last push) and will be pushed later."""
    if not os.path.isdir(remote):
        return {"files": 0, "bytes": 0}
    with _lock:
        man = _load(local)
        n = nbytes = 0
        for p in _files(remote):
            rel = os.path.relpath(p, remote)
            q = os.path.join(local, rel)
            if os.path.exists(q):
                continue
            _copy(p, q)
            man[rel] = _sig(q)
            n += 1
            nbytes += man[rel][0]
        _save(local, man)
    return {"files": n, "bytes": nbytes}


class PeriodicPush:
    """`with PeriodicPush(local, remote, every_s):` pushes every `every_s` seconds in a background thread while the
    block runs, and once more, completely, when it ends."""

    def __init__(self, local: str, remote: str, every_s: float = 600, log=print):
        self.local, self.remote, self.every_s, self.log = local, remote, every_s, log
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _once(self, final: bool) -> None:
        t0 = time.time()
        try:
            r = push(self.local, self.remote, markers_before=None if final else t0 - 1.0)
            if r["files"]:
                self.log(f"[sync] {r['files']} files ({r['bytes'] / 1e6:.0f} MB) copied to {self.remote} "
                         f"in {time.time() - t0:.0f}s")
        except Exception as e:  # a failed copy must not stop the experiment; the next push retries
            self.log(f"[sync] copy to {self.remote} failed ({type(e).__name__}: {e}); will retry")

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            self._once(final=False)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._once(final=True)
        return False
