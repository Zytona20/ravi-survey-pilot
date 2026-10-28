"""Stage 5: merge part files into tables and derive operating points.

An operating point answers: "for this index, what is the fastest setting that still reaches
recall >= target?"  It is computed twice, with two recall definitions:
  basis = recall      (index quality inside its own space)
  basis = e2e_recall  (against the native space: what a user of the backbone actually gets)
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from .config import Paths
from .utils import log, read_json

KEYS = ["space_id", "model", "family", "paradigm", "reduction", "dim", "native_dim", "index", "build_key"]


def save_table(df: pd.DataFrame, path_no_ext: str) -> None:
    df.to_csv(path_no_ext + ".csv", index=False)
    try:
        df.to_parquet(path_no_ext + ".parquet", index=False)
    except Exception as e:  # pyarrow missing: CSV is still written
        log.info(f"(parquet not written: {e})")


def load_table(paths: Paths, name: str) -> pd.DataFrame:
    p = os.path.join(paths.results, name)
    if os.path.exists(p + ".parquet"):
        try:
            return pd.read_parquet(p + ".parquet")
        except Exception:
            pass
    return pd.read_csv(p + ".csv")


def _load_parts(part_dir: str, rows_key: str | None) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(part_dir, "*.json"))):
        obj = read_json(f)
        rows.extend(obj[rows_key] if rows_key else [obj])
    return pd.DataFrame(rows)


def operating_points(bench: pd.DataFrame, targets, latency_k: int) -> pd.DataFrame:
    df = bench[bench["k"] == latency_k]
    out = []
    for keys, g in df.groupby(KEYS + ["seed"], dropna=False):
        first = g.iloc[0]
        for basis in ("recall", "e2e_recall"):
            for t in targets:
                ok = g[g[basis] >= t]
                row = dict(zip(KEYS + ["seed"], keys))
                row.update(basis=basis, target=float(t), feasible=len(ok) > 0,
                           build_time_s=first["build_time_s"], bytes_per_vec=first["bytes_per_vec"],
                           M=first["M"], nlist=first["nlist"], pq_m=first["pq_m"])
                if len(ok):
                    b = ok.loc[ok["qps_1t"].idxmax()]
                    row.update(qps_1t=b["qps_1t"], qps_batch=b["qps_batch"], lat_p50_ms=b["lat_p50_ms"],
                               lat_p99_ms=b["lat_p99_ms"], search_value=b["search_value"],
                               recall=b["recall"], e2e_recall=b["e2e_recall"], label_p=b.get("label_p", np.nan))
                    if "ndis" in ok.columns and ok["ndis"].notna().any():
                        # hardware-independent operating point: fewest distance evaluations reaching the target
                        c = ok.loc[ok["ndis"].idxmin()]
                        row.update(ndis=c["ndis"], inv_ndis=1e6 / max(c["ndis"], 1e-9),
                                   search_value_ndis=c["search_value"], ndis_frac=c["ndis"] / c["n_base"])
                out.append(row)
    return pd.DataFrame(out)


def aggregate_seeds(op: pd.DataFrame) -> pd.DataFrame:
    """Mean / std over seeds. A setting counts as feasible only if every seed reached the target."""
    num = ["qps_1t", "qps_batch", "lat_p50_ms", "lat_p99_ms", "recall", "e2e_recall", "label_p",
           "build_time_s", "bytes_per_vec", "search_value", "ndis", "inv_ndis", "ndis_frac"]
    num = [c for c in num if c in op.columns]
    gcols = KEYS + ["basis", "target"]
    agg = op.groupby(gcols, dropna=False).agg(
        n_seeds=("seed", "nunique"), feasible=("feasible", "all"),
        M=("M", "first"), nlist=("nlist", "first"), pq_m=("pq_m", "first"),
        **{c: (c, "mean") for c in num}, **{c + "_std": (c, "std") for c in ("qps_1t", "lat_p99_ms", "ndis") if c in num},
    ).reset_index()
    agg.loc[~agg["feasible"], [c for c in ("qps_1t", "qps_batch", "ndis", "inv_ndis", "ndis_frac") if c in agg]] = np.nan
    return agg


def run_collect(cfg: dict) -> None:
    paths = Paths(cfg)
    tables = {}
    geom = _load_parts(paths.parts("geometry"), None)
    qual = _load_parts(paths.parts("quality"), None)
    bench = _load_parts(paths.parts("bench"), "rows")
    for name, df in (("geometry", geom), ("quality", qual), ("bench", bench)):
        if len(df):
            save_table(df, os.path.join(paths.results, name))
            tables[name] = len(df)
    if len(bench):
        op = operating_points(bench, cfg["bench"]["recall_targets"], int(cfg["bench"]["latency_k"]))
        save_table(op, os.path.join(paths.results, "operating_points_seeds"))
        save_table(aggregate_seeds(op), os.path.join(paths.results, "operating_points"))
        tables["operating_points"] = len(op)
    log.info(f"[collect] rows written: {tables}")
