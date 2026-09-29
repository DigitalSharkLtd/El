"""POST-HOC DESCRIPTIVE look at the confirmatory data (written after the confirmatory
run; NOT part of the frozen analysis plan; no new hypothesis tests, no intervals).

Usage:  python scripts/posthoc_descriptive.py  -> results/confirmatory/posthoc_descriptive.md
"""
import csv
import glob
import gzip
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bm.experiment import ROOT, read_csv                            # noqa: E402
from bm.tape import make_tape                                       # noqa: E402

S_SYMPTOM = 0.45 + 0.005 / 1.55     # c_S < 0.995  <=>  S > 0.4532...
STRATS = ["fixed_5", "output_trend", "C", "C_level", "C_scrambled", "oracle_D"]


def main():
    rows = read_csv(os.path.join(ROOT, "results", "confirmatory", "confirmatory.csv"))
    by = {}
    for r in rows:
        by.setdefault((r["mode"], r["strategy"]), []).append(r)
    L = ["# Post-hoc descriptive notes (NOT in the frozen plan)\n",
         "Computed after the confirmatory run. Descriptive only; no intervals, no tests.\n"]

    L.append("## Deaths in dynamic worlds: count and hidden regime at the death step\n")
    L.append("| strategy | deaths | regime at death slow/mid/fast | worlds with final S > 0.4532 (y=1 for ever) |")
    L.append("|---|---|---|---|")
    died_sets = {}
    for s in STRATS:
        rr = by[("dynamic", s)]
        cnt = Counter()
        died = set()
        for r in rr:
            if r["died"] == "1":
                reg = int(make_tape(int(r["world_seed"]), "dynamic").regime[int(r["death_step"]) - 1])
                cnt[reg] += 1
                died.add(int(r["world_seed"]))
        died_sets[s] = died
        n_sym = sum(float(r["final_S"]) > S_SYMPTOM for r in rr)
        L.append(f"| {s} | {len(died)} | {cnt[0]}/{cnt[1]}/{cnt[2]} | {n_sym} of {len(rr)} |")
    both = died_sets["C"] & died_sets["fixed_5"]
    L.append(f"\nWorlds where both C and fixed_5 died: {len(both)}; only C: "
             f"{len(died_sets['C'] - died_sets['fixed_5'])}; only fixed_5: "
             f"{len(died_sets['fixed_5'] - died_sets['C'])}.\n")

    L.append("## What triggered C's SERVICE requests (step logs of the 6 logged worlds)\n")
    L.append("| mode | strategy | living steps | requests | via reactive E<0.965 | via risk>0.24 only | executed SERVICE |")
    L.append("|---|---|---|---|---|---|---|")
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "confirmatory", "step_logs", "*.csv.gz"))):
        mode = os.path.basename(path).split("_")[0]
        agg = {}
        with gzip.open(path, "rt") as f:
            for r in csv.DictReader(f):
                if r["strategy"] not in ("C", "C_level", "C_scrambled"):
                    continue
                a = agg.setdefault(r["strategy"], Counter())
                a["steps"] += 1
                dbg = dict(kv.split("=") for kv in r["policy_debug"].split(";"))
                if r["requested"] == "1":
                    a["req"] += 1
                    if dbg["reactive"] == "1":
                        a["reactive"] += 1
                    else:
                        a["risk"] += 1
                if r["action"] == "1":
                    a["svc"] += 1
        for s, a in agg.items():
            key = (mode, s)
            by.setdefault(("_log",) + key, Counter()).update(a)
    for mode in ("dynamic", "static"):
        for s in ("C", "C_level", "C_scrambled"):
            a = by.get(("_log", mode, s))
            if a:
                L.append(f"| {mode} | {s} | {a['steps']} | {a['req']} | {a['reactive']} | "
                         f"{a['risk']} | {a['svc']} |")
    L.append("")
    out = os.path.join(ROOT, "results", "confirmatory", "posthoc_descriptive.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
