"""Development phase (section 9): 64 dynamic + 64 static worlds.

Runs every strategy plus the tuning grids, selects
  * fixed* (max mean G, ties -> smaller p),
  * tuned-output theta (ties -> smaller theta),
  * output-trend (L, theta) (ties -> smaller L, then smaller theta),
separately for dynamic and static, and writes frozen/settings.json.

Usage:  python scripts/run_development.py
"""
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bm import params as P                                    # noqa: E402
from bm.experiment import ROOT, environment_info, run_many, save_json, write_csv  # noqa: E402
from bm.tape import dev_seeds                                 # noqa: E402


def mean_G(rows, strategy, config=None):
    vals = [r["G"] for r in rows if r["strategy"] == strategy
            and (config is None or r["config"] == config)]
    assert len(vals) == P.DEV_N, (strategy, config, len(vals))
    return float(np.mean(vals))


def select(rows):
    """Selections for one mode; candidate order encodes the tie-break."""
    fixed = [(p, mean_G(rows, f"fixed_{p}")) for p in P.FIXED_PERIODS]
    best_fixed = None
    for p, g in fixed:                      # ascending p; strict '>' keeps the smaller p on ties
        if best_fixed is None or g > best_fixed[1]:
            best_fixed = (p, g)

    tuned = [(th, mean_G(rows, "tuned_output_grid", f"theta={th:g}")) for th in P.THETA_GRID]
    best_tuned = None
    for th, g in tuned:
        if best_tuned is None or g > best_tuned[1]:
            best_tuned = (th, g)

    trend = [((L, th), mean_G(rows, "output_trend_grid", f"L={L:g};theta={th:g}"))
             for L in P.L_GRID for th in P.THETA_GRID]
    best_trend = None
    for cfg, g in trend:
        if best_trend is None or g > best_trend[1]:
            best_trend = (cfg, g)

    return {
        "fixed_star": {"p": best_fixed[0], "dev_mean_G": best_fixed[1]},
        "tuned_output": {"theta": best_tuned[0], "dev_mean_G": best_tuned[1]},
        "output_trend": {"L": best_trend[0][0], "theta": best_trend[0][1],
                         "dev_mean_G": best_trend[1]},
        "candidates": {
            "fixed": [{"p": p, "dev_mean_G": g} for p, g in fixed],
            "tuned_output": [{"theta": th, "dev_mean_G": g} for th, g in tuned],
            "output_trend": [{"L": c[0], "theta": c[1], "dev_mean_G": g} for c, g in trend],
        },
    }


def main():
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    all_rows = []
    settings = {"selection_rule": "max mean G on development; ties: smaller p / "
                                  "smaller theta / smaller L then smaller theta",
                "note": "C is not tuned. Comparison strategies get separate settings per "
                        "environment type (an advantage for them).",
                "optional_own_agent": "not used"}
    for mode in ("dynamic", "static"):
        seeds = dev_seeds(mode)
        rows = run_many("development", mode, seeds, settings=None, with_grid=True)
        all_rows.extend(rows)
        settings[mode] = select(rows)
        print(mode, settings[mode]["fixed_star"], settings[mode]["tuned_output"],
              settings[mode]["output_trend"])
    all_rows.sort(key=lambda r: (r["mode"] != "dynamic", r["world_index"]))
    write_csv(all_rows, os.path.join(ROOT, "results", "development", "development.csv"))
    save_json(settings, os.path.join(ROOT, "frozen", "settings.json"))
    save_json({"started_utc": started, "wall_seconds": time.time() - t0,
               "n_rows": len(all_rows), "environment": environment_info()},
              os.path.join(ROOT, "results", "development", "run_info.json"))
    print(f"development done: {len(all_rows)} rows in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
