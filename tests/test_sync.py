"""Local disk <-> Drive copying used by the Colab notebook."""
import os
import time

from ravi.sync import MANIFEST, PeriodicPush, push, restore


def _write(path, text, mtime=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _read(path):
    with open(path) as f:
        return f.read()


def test_push_orders_markers_after_data_and_detects_changes(tmp_path):
    local, remote = str(tmp_path / "local"), str(tmp_path / "drive")
    old = time.time() - 100
    _write(f"{local}/emb/m/base.npy", "AAAA", old)
    _write(f"{local}/emb/m/base.npy.progress.json", '{"done": 2}', old)
    _write(f"{local}/emb/m/meta.json", "{}")                       # written "during" the push below
    _write(f"{local}/spaces/s/base.npy.tmp123.npy", "partial")     # atomic-write temp file: never copied
    r = push(local, remote, markers_before=time.time() - 10)
    assert r["files"] == 2
    assert os.path.exists(f"{remote}/emb/m/base.npy.progress.json")
    assert not os.path.exists(f"{remote}/emb/m/meta.json")         # newer than the push start: left for later
    assert not os.path.exists(f"{remote}/spaces/s/base.npy.tmp123.npy")
    assert not os.path.exists(f"{remote}/{MANIFEST}")

    assert push(local, remote)["files"] == 1                       # final push: the marker now goes too
    assert push(local, remote)["files"] == 0                       # nothing changed

    _write(f"{local}/emb/m/base.npy", "BBBB", old + 5)             # same size, new content (like a memmap)
    assert push(local, remote)["files"] == 1
    assert _read(f"{remote}/emb/m/base.npy") == "BBBB"

    os.remove(f"{remote}/emb/m/meta.json")                         # file lost on the remote side: copied again
    assert push(local, remote)["files"] == 1


def test_restore_after_wipe_then_push_copies_nothing(tmp_path):
    local, remote, local2 = str(tmp_path / "local"), str(tmp_path / "drive"), str(tmp_path / "local2")
    _write(f"{local}/a/x.npy", "x")
    _write(f"{local}/a/meta.json", "{}")
    push(local, remote)
    r = restore(remote, local2)                                    # a new Colab machine with an empty disk
    assert r["files"] == 2 and _read(f"{local2}/a/x.npy") == "x"
    assert push(local2, remote)["files"] == 0
    _write(f"{local2}/a/y.npy", "y")
    assert push(local2, remote)["files"] == 1


def test_restore_never_overwrites_local(tmp_path):
    local, remote = str(tmp_path / "local"), str(tmp_path / "drive")
    _write(f"{remote}/r/results.csv", "old")
    _write(f"{local}/r/results.csv", "newer")
    assert restore(remote, local)["files"] == 0
    assert _read(f"{local}/r/results.csv") == "newer"
    assert push(local, remote)["files"] == 1
    assert _read(f"{remote}/r/results.csv") == "newer"


def test_periodic_push_runs_in_background_and_at_exit(tmp_path):
    local, remote = str(tmp_path / "local"), str(tmp_path / "drive")
    msgs = []
    _write(f"{local}/a.npy", "a", time.time() - 100)
    with PeriodicPush(local, remote, every_s=0.2, log=msgs.append):
        time.sleep(0.6)
        assert os.path.exists(f"{remote}/a.npy")
        _write(f"{local}/b.json", "{}")
    assert os.path.exists(f"{remote}/b.json")
    assert all("failed" not in m for m in msgs)
