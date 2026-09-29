"""Re-execution check: reruns the confirmatory worlds with the frozen settings and
seeds (in memory, nothing is overwritten) and compares every value with
results/confirmatory/confirmatory.csv.

Usage:  python scripts/reproduce_check.py            (all 512 + 256 worlds)
        python scripts/reproduce_check.py --quick    (first 20 worlds of each mode)
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bm.experiment import ROOT, ROW_COLUMNS, _fmt, read_csv, run_many   # noqa: E402

REL_TOL = 1e-9     # last-digit differences of libm/BLAS between platforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    with open(os.path.join(ROOT, "frozen", "settings.json")) as f:
        settings = json.load(f)
    with open(os.path.join(ROOT, "frozen", "confirmatory_seeds.json")) as f:
        seeds = json.load(f)
    ref = {(r["mode"], r["world_seed"], r["strategy"]): r
           for r in read_csv(os.path.join(ROOT, "results", "confirmatory", "confirmatory.csv"))}

    t0 = time.time()
    n_rows = exact = close = 0
    bad = []
    for mode in ("dynamic", "static"):
        lst = seeds[mode][:20] if args.quick else seeds[mode]
        print(f"{mode}: {len(lst)} worlds ...", flush=True)
        for r in run_many("confirmatory", mode, lst, settings=settings[mode]):
            n_rows += 1
            old_row = ref[(mode, str(r["world_seed"]), r["strategy"])]
            for k in ROW_COLUMNS:
                new, old = str(_fmt(r[k])), old_row[k]
                if new == old:
                    exact += 1
                    continue
                try:
                    a, b = float(new), float(old)
                    if math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-12):
                        close += 1
                        continue
                except ValueError:
                    pass
                bad.append((mode, r["world_seed"], r["strategy"], k, old, new))

    total = exact + close + len(bad)
    print(f"\nrows recomputed: {n_rows} ({time.time() - t0:.0f} s)")
    print(f"values identical:              {exact} of {total}")
    print(f"values equal within {REL_TOL:g}:   {close}")
    print(f"values different:              {len(bad)}")
    for b in bad[:20]:
        print("  ", b)
    if bad:
        print("RESULT: MISMATCH - the published numbers were NOT reproduced exactly.")
        sys.exit(1)
    print("RESULT: OK - the published confirmatory results are reproduced.")


if __name__ == "__main__":
    main()
