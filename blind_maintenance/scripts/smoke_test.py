"""Technical smoke test: a few DEVELOPMENT worlds, all strategies, a printed table.
This is NOT an experiment and its numbers must not be reported as results.

Usage:  python scripts/smoke_test.py [--worlds 2]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bm.experiment import ROOT                                     # noqa: E402
from bm.policies import base_strategies                            # noqa: E402
from bm.runner import run_world                                    # noqa: E402
from bm.tape import dev_seeds                                      # noqa: E402

DEFAULT = {"tuned_output": {"theta": 0.965}, "output_trend": {"L": 1, "theta": 0.965}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=2)
    args = ap.parse_args()
    sp = os.path.join(ROOT, "frozen", "settings.json")
    settings = json.load(open(sp)) if os.path.exists(sp) else None
    print("SMOKE TEST - development worlds only, not an experiment")
    t0 = time.time()
    for mode in ("dynamic", "static"):
        st = settings[mode] if settings else DEFAULT
        for seed in dev_seeds(mode)[:args.worlds]:
            for r in run_world(seed, mode, base_strategies(st)):
                print(f"{mode:8s} {seed} {r['strategy']:13s} G={r['G']:8.2f} "
                      f"alive={r['survived']} S={r['final_S']:.3f} SERVICE={r['n_service']}")
    print(f"ok in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
