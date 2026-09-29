"""Confirmatory run (sections 9-11).  Runs exactly once on the frozen code/settings.

Checks the SHA-256 of every enforced file against frozen/manifest.json,
runs all strategies on 512 dynamic + 256 static worlds, writes
results/confirmatory/confirmatory.csv and the analyzer step logs for the first
three listed worlds of each mode.

Usage:  python scripts/run_confirmatory.py
        python scripts/run_confirmatory.py --rerun-reason "..."   (keeps the old results)
"""
import argparse
import csv
import gzip
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bm.experiment import (ROOT, environment_info, run_many, save_json,  # noqa: E402
                           sha256_file, write_csv)
from bm.policies import base_strategies                            # noqa: E402
from bm.runner import LOG_COLUMNS, run_world                       # noqa: E402

OUT = os.path.join(ROOT, "results", "confirmatory")


def verify_freeze():
    with open(os.path.join(ROOT, "frozen", "manifest.json")) as f:
        man = json.load(f)
    bad = []
    for rel, digest in man["enforced_files_sha256"].items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p) or sha256_file(p) != digest:
            bad.append(rel)
    if bad:
        sys.exit("frozen files changed since the manifest: " + ", ".join(bad))
    with open(os.path.join(ROOT, "frozen", "confirmatory_seeds.json")) as f:
        seeds = json.load(f)
    if seeds["manifest_sha256_at_draw"] != sha256_file(os.path.join(ROOT, "frozen", "manifest.json")):
        sys.exit("seeds were not drawn against the current manifest")
    with open(os.path.join(ROOT, "frozen", "settings.json")) as f:
        settings = json.load(f)
    return man, seeds, settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-reason", default=None)
    args = ap.parse_args()
    csv_path = os.path.join(OUT, "confirmatory.csv")
    man, seeds, settings = verify_freeze()
    if os.path.exists(csv_path):
        if not args.rerun_reason:
            sys.exit("confirmatory results exist; a rerun needs --rerun-reason and is logged")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        keep = os.path.join(ROOT, "results", f"confirmatory_superseded_{stamp}")
        shutil.move(OUT, keep)
        with open(os.path.join(ROOT, "CHANGELOG.md"), "a") as f:
            f.write(f"\n- {stamp}: confirmatory rerun. Reason: {args.rerun_reason}. "
                    f"Previous results kept in {os.path.relpath(keep, ROOT)}.\n")

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    rows = []
    for mode in ("dynamic", "static"):
        rows += run_many("confirmatory", mode, seeds[mode], settings=settings[mode])
    wall_main = time.time() - t0
    rows.sort(key=lambda r: (r["mode"] != "dynamic", r["world_index"]))
    write_csv(rows, csv_path)

    # analyzer step logs for the first three pre-listed worlds of each mode
    log_dir = os.path.join(OUT, "step_logs")
    os.makedirs(log_dir, exist_ok=True)
    by_key = {(r["mode"], r["world_seed"], r["strategy"]): r for r in rows}
    for mode in ("dynamic", "static"):
        for i, seed in enumerate(seeds["step_log_worlds"][mode]):
            logs = {}
            res = run_world(seed, mode, base_strategies(settings[mode]), log_rows_by_policy=logs)
            for r in res:     # logged rerun must reproduce the batch result exactly
                ref = by_key[(mode, seed, r["strategy"])]
                for k, v in r.items():
                    rv = ref[k]
                    assert (isinstance(v, float) and math.isnan(v) and math.isnan(rv)) or v == rv, \
                        (mode, seed, r["strategy"], k)
            path = os.path.join(log_dir, f"{mode}_world{i}_seed{seed}.csv.gz")
            with gzip.open(path, "wt", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=["strategy"] + LOG_COLUMNS)
                wr.writeheader()
                for name, lrows in logs.items():
                    for lr in lrows:
                        wr.writerow({"strategy": name, **{k: (repr(v) if isinstance(v, float) else v)
                                                          for k, v in lr.items()}})
    save_json({"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "wall_seconds_main_run": wall_main, "wall_seconds_total": time.time() - t0,
               "n_rows": len(rows), "master_seed": seeds["master_seed"],
               "manifest_sha256": sha256_file(os.path.join(ROOT, "frozen", "manifest.json")),
               "environment": environment_info()},
              os.path.join(OUT, "run_info.json"))
    print(f"confirmatory done: {len(rows)} rows, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
