"""Command line:  python -m ravi <stage> -c configs/<file>.yaml"""
from __future__ import annotations

import argparse
import time

from .config import load_config
from .utils import log, set_seed

ORDER = ["extract", "reduce", "gt", "geometry", "bench", "collect", "recommend", "stats", "figures", "report",
         "survey"]
EXTRA = ["crossmodal"]  # run on request only: needs open_clip_torch and an image dataset


def _stage(name: str):
    if name == "extract":
        from .extract import run_extract as f
    elif name == "reduce":
        from .reduce import run_reduce as f
    elif name == "gt":
        from .groundtruth import run_gt as f
    elif name == "geometry":
        from .geometry import run_geometry as f
    elif name == "bench":
        from .bench import run_bench as f
    elif name == "collect":
        from .collect import run_collect as f
    elif name == "recommend":
        from .recommender import run_recommend as f
    elif name == "stats":
        from .stats import run_stats as f
    elif name == "figures":
        from .figures import run_figures as f
    elif name == "report":
        from .report import run_report as f
    elif name == "survey":
        from .survey import run_survey as f
    elif name == "crossmodal":
        from .crossmodal import run_crossmodal as f
    else:
        raise SystemExit(f"unknown stage {name}")
    return f


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ravi", description=__doc__)
    ap.add_argument("stage", choices=ORDER + EXTRA + ["all", "plan", "analyse"],
                    help="`all` runs every stage in order; `analyse` = collect + recommend + stats + figures + report + survey; "
                         "`crossmodal` (text vs image queries) runs only on request")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--pilot", default=None, help="plan only: run directory of a finished pilot to extrapolate from")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    if args.stage == "plan":
        from .plan import run_plan

        run_plan(cfg, args.pilot)
        return
    stages = ORDER if args.stage == "all" else ORDER[5:] if args.stage == "analyse" else [args.stage]
    for s in stages:
        t0 = time.time()
        log.info(f"===== {s} =====")
        _stage(s)(cfg)
        log.info(f"===== {s} finished in {time.time() - t0:.1f}s =====")


if __name__ == "__main__":
    main()
