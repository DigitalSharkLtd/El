"""Analysis (section 10).

Usage:
    python scripts/analyze.py --phase confirmatory
    python scripts/analyze.py --phase development     (descriptive only)

Writes results/<phase>/summary.json and results/<phase>/summary_tables.md.
All intervals: paired percentile bootstrap, 10 000 resamples of worlds with
replacement, fixed analysis seed (bm.params.ANALYSIS_SEED). Only the main
estimate (dynamic, C - fixed*, G) is primary; everything else is
exploratory / unadjusted.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bm import params as P                                        # noqa: E402
from bm.experiment import ROOT, read_csv, save_json                # noqa: E402
from bm.stats import boot_index, boot_means, percentile_ci, summarize, wilson  # noqa: E402

BASE_ORDER = (["work_only"] + [f"fixed_{p}" for p in P.FIXED_PERIODS]
              + ["output_only", "tuned_output", "output_trend", "C", "C_scrambled",
                 "C_level", "oracle_D"])
NUM = ["G", "survived", "died", "steps_alive", "final_S", "final_R", "final_D", "n_work",
       "n_service", "B", "n_denied_service_requests", "alive_steps_r0", "alive_steps_r1",
       "alive_steps_r2", "services_r0", "services_r1", "services_r2", "n_train_C"]
PAIR_METRICS = [("G", 1.0), ("survived", 100.0), ("final_S", 1.0), ("n_service", 1.0),
                ("B", 1.0), ("steps_alive", 1.0)]


def load(phase, settings):
    rows = read_csv(os.path.join(ROOT, "results", phase, f"{phase}.csv"))
    data = {}
    for r in rows:
        name = r["strategy"]
        mode = r["mode"]
        if phase == "development":
            if name == "tuned_output_grid":
                if r["config"] != f"theta={settings[mode]['tuned_output']['theta']:g}":
                    continue
                name = "tuned_output"
            elif name == "output_trend_grid":
                s = settings[mode]["output_trend"]
                if r["config"] != f"L={s['L']:g};theta={s['theta']:g}":
                    continue
                name = "output_trend"
        d = data.setdefault(mode, {}).setdefault(name, {})
        idx = int(r["world_index"])
        d[idx] = {k: (math.nan if r[k] == "NA" else float(r[k])) for k in NUM}
        d[idx]["world_seed"] = int(r["world_seed"])
    out = {}
    for mode, strat in data.items():
        out[mode] = {}
        n = None
        seeds_ref = None
        for name, per in strat.items():
            ids = sorted(per)
            assert ids == list(range(len(ids))), (mode, name)
            n = n or len(ids)
            assert len(ids) == n, (mode, name, len(ids), n)
            seeds = [per[i]["world_seed"] for i in ids]
            if seeds_ref is None:
                seeds_ref = seeds
            assert seeds == seeds_ref, "worlds must be aligned for pairing"
            out[mode][name] = {k: np.array([per[i][k] for i in ids]) for k in NUM}
    return out


def strategy_table(mode_data, idx):
    tab = {}
    for name in BASE_ORDER:
        if name not in mode_data:
            continue
        d = mode_data[name]
        n = len(d["G"])
        k_surv = int(d["survived"].sum())
        surv = summarize(d["survived"] * 100.0, idx)
        wl, wh = wilson(k_surv, n)
        tab[name] = {
            "n_worlds": n,
            "G": summarize(d["G"], idx),
            "survival_pct": surv,
            "survival_wilson_pct": [100 * wl, 100 * wh],
            "survival_bootstrap_degenerate": surv["degenerate"],
            "n_survived": k_surv,
            "mean_steps_alive": float(d["steps_alive"].mean()),
            "mean_final_S": float(d["final_S"].mean()),
            "mean_final_R": float(d["final_R"].mean()),
            "mean_n_work": float(d["n_work"].mean()),
            "mean_n_service": float(d["n_service"].mean()),
            "mean_B": float(d["B"].mean()),
            "B": summarize(d["B"], idx),
            "mean_denied_service_requests": float(d["n_denied_service_requests"].mean()),
        }
        if not np.all(np.isnan(d["n_train_C"])):
            tab[name]["mean_n_train_C"] = float(np.nanmean(d["n_train_C"]))
    return tab


def paired(mode_data, a, b, idx):
    out = {}
    for metric, scale in PAIR_METRICS:
        diff = (mode_data[a][metric] - mode_data[b][metric]) * scale
        s = summarize(diff, idx)
        s["n_worlds"] = len(diff)
        s["n_a_better"] = int((diff > 0).sum())
        s["n_b_better"] = int((diff < 0).sum())
        s["n_equal"] = int((diff == 0).sum())
        out[metric if metric != "survived" else "survival_pp"] = s
    return out


def pace(mode_data, mode, idx_all):
    """100 * SERVICE_j / alive_steps_j per hidden regime (analyzer-only diagnostic)."""
    res = {}
    for name in BASE_ORDER:
        if name not in mode_data:
            continue
        d = mode_data[name]
        rates = {}
        for j in range(3):
            al = d[f"alive_steps_r{j}"]
            sv = d[f"services_r{j}"]
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(al > 0, 100.0 * sv / al, np.nan)
            rates[j] = r
        entry = {}
        for j, lab in enumerate(("slow_m0.48", "mid_m0.92", "fast_m1.65")):
            r = rates[j]
            ok = ~np.isnan(r)
            entry[lab] = {"n_worlds_observed": int(ok.sum()),
                          "mean_rate": float(np.nanmean(r)) if ok.any() else None}
        both = ~np.isnan(rates[0]) & ~np.isnan(rates[2])
        nb = int(both.sum())
        if nb >= 2:
            diff = (rates[2] - rates[0])[both]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                [P.ANALYSIS_SEED, 99, nb])))
            bidx = rng.integers(0, nb, size=(P.N_BOOT, nb))
            lo, hi = percentile_ci(boot_means(diff, bidx))
            entry["fast_minus_slow"] = {"n_worlds_both_regimes": nb, "mean": float(diff.mean()),
                                        "ci_low": lo, "ci_high": hi}
        else:
            entry["fast_minus_slow"] = {"n_worlds_both_regimes": nb, "mean": None}
        res[name] = entry
    return res


def analyze(phase):
    with open(os.path.join(ROOT, "frozen", "settings.json")) as f:
        settings = json.load(f)
    data = load(phase, settings)
    summary = {"phase": phase,
               "bootstrap": {"type": "paired percentile", "n_boot": P.N_BOOT,
                             "analysis_seed": P.ANALYSIS_SEED,
                             "unit": "world (resampled with replacement)"},
               "frozen_settings": {m: {k: settings[m][k] for k in
                                       ("fixed_star", "tuned_output", "output_trend")}
                                   for m in ("dynamic", "static")},
               "per_strategy": {}, "comparisons": {}, "mode_contrast": {}, "pace": {}}
    idxs = {}
    for mode in ("dynamic", "static"):
        md = data[mode]
        n = len(md["C"]["G"])
        idx = boot_index(n, mode)
        idxs[mode] = idx
        fixed_star = f"fixed_{settings[mode]['fixed_star']['p']}"
        summary["per_strategy"][mode] = strategy_table(md, idx)
        comps = {}
        others = [fixed_star, "C_scrambled", "C_level", "output_only", "tuned_output",
                  "output_trend", "oracle_D", "work_only"] + \
                 [f"fixed_{p}" for p in P.FIXED_PERIODS if f"fixed_{p}" != fixed_star]
        for o in others:
            key = f"C - {o}" + (" (fixed*)" if o == fixed_star else "")
            comps[key] = paired(md, "C", o, idx)
        # a few extra descriptive pairings among comparison strategies
        for a, b in (("tuned_output", fixed_star), ("output_trend", fixed_star),
                     ("oracle_D", fixed_star), ("C_level", "C_scrambled")):
            comps[f"{a} - {b}"] = paired(md, a, b, idx)
        summary["comparisons"][mode] = comps
        summary["pace"][mode] = pace(md, mode, idx)

    # main estimate
    fs_dyn = f"fixed_{settings['dynamic']['fixed_star']['p']}"
    main = summary["comparisons"]["dynamic"][f"C - {fs_dyn} (fixed*)"]
    summary["main"] = {
        "estimand": f"mean(G_C - G_fixed*) in dynamic, fixed* = {fs_dyn} chosen on development",
        "G": main["G"],
        "companion": {"survival_pp": main["survival_pp"], "final_S": main["final_S"],
                      "n_service": main["n_service"]},
    }

    # dynamic/static contrast of paired effects (independent world samples)
    fs_st = f"fixed_{settings['static']['fixed_star']['p']}"
    for label, od, os_ in (("C - fixed*", fs_dyn, fs_st),
                           ("C - C_scrambled", "C_scrambled", "C_scrambled"),
                           ("C - C_level", "C_level", "C_level"),
                           ("C - output_only", "output_only", "output_only"),
                           ("C - tuned_output", "tuned_output", "tuned_output"),
                           ("C - output_trend", "output_trend", "output_trend"),
                           ("C - oracle_D", "oracle_D", "oracle_D")):
        entry = {}
        for metric, scale in (("G", 1.0), ("survived", 100.0)):
            dd = (data["dynamic"]["C"][metric] - data["dynamic"][od][metric]) * scale
            ds = (data["static"]["C"][metric] - data["static"][os_][metric]) * scale
            b = boot_means(dd, idxs["dynamic"]) - boot_means(ds, idxs["static"])
            lo, hi = percentile_ci(b)
            entry[metric if metric != "survived" else "survival_pp"] = {
                "dynamic_effect": float(dd.mean()), "static_effect": float(ds.mean()),
                "difference": float(dd.mean() - ds.mean()), "ci_low": lo, "ci_high": hi}
        summary["mode_contrast"][label] = entry
    return summary


# ------------------------------------------------------------------ markdown
def f2(x):
    return "NA" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def ci(s, nd=2):
    return f"{s['mean']:.{nd}f} [{s['ci_low']:.{nd}f}; {s['ci_high']:.{nd}f}]"


def to_markdown(summary):
    L = []
    ph = summary["phase"]
    L.append(f"# Summary tables — {ph}\n")
    L.append("Bootstrap: paired percentile, 10 000 resamples of worlds, analysis seed "
             f"{P.ANALYSIS_SEED}. Only the main estimate is primary; all other intervals are "
             "exploratory / unadjusted.\n")
    fs = summary["frozen_settings"]
    L.append("Frozen settings: " + "; ".join(
        f"{m}: fixed*=p{fs[m]['fixed_star']['p']}, tuned θ={fs[m]['tuned_output']['theta']}, "
        f"trend L={fs[m]['output_trend']['L']} θ={fs[m]['output_trend']['theta']}"
        for m in ("dynamic", "static")) + "\n")
    m = summary["main"]
    L.append("## Main estimate\n")
    L.append(f"{m['estimand']}\n")
    L.append(f"* ΔG = {ci(m['G'])} (C better in {m['G']['n_a_better']}, worse in "
             f"{m['G']['n_b_better']}, equal in {m['G']['n_equal']} of {m['G']['n_worlds']} worlds)")
    c = m["companion"]
    L.append(f"* Δsurvival, p.p. = {ci(c['survival_pp'], 1)}"
             + (" (bootstrap degenerate)" if c['survival_pp']['degenerate'] else ""))
    L.append(f"* Δfinal S = {ci(c['final_S'], 4)}")
    L.append(f"* ΔSERVICE count = {ci(c['n_service'], 1)}\n")
    for mode in ("dynamic", "static"):
        L.append(f"## Per-strategy results — {mode}\n")
        L.append("| strategy | N | mean G [95% CI] | survival % [boot CI] | Wilson CI % | "
                 "mean min(death,1000) | final S | final R | WORK | SERVICE | B | denied req. |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, t in summary["per_strategy"][mode].items():
            deg = " ⚠" if t["survival_bootstrap_degenerate"] else ""
            L.append(f"| {name} | {t['n_worlds']} | {ci(t['G'])} | {ci(t['survival_pct'], 1)}{deg} | "
                     f"[{t['survival_wilson_pct'][0]:.1f}; {t['survival_wilson_pct'][1]:.1f}] | "
                     f"{t['mean_steps_alive']:.1f} | {t['mean_final_S']:.3f} | {t['mean_final_R']:.2f} | "
                     f"{t['mean_n_work']:.1f} | {t['mean_n_service']:.1f} | {t['mean_B']:.1f} | "
                     f"{t['mean_denied_service_requests']:.1f} |")
        L.append("\n⚠ = zero sample variation in survival; bootstrap interval is degenerate, "
                 "use the Wilson interval (zero sample variation ≠ zero uncertainty).\n")
        L.append(f"## Paired comparisons — {mode} (exploratory/unadjusted except the main one)\n")
        L.append("| comparison | ΔG [95% CI] | Δsurvival p.p. [CI] | Δfinal S [CI] | ΔSERVICE [CI] | ΔB [CI] | a>b / a<b / = (G) |")
        L.append("|---|---|---|---|---|---|---|")
        for name, cc in summary["comparisons"][mode].items():
            L.append(f"| {name} | {ci(cc['G'])} | {ci(cc['survival_pp'], 1)} | "
                     f"{ci(cc['final_S'], 4)} | {ci(cc['n_service'], 1)} | {ci(cc['B'], 1)} | "
                     f"{cc['G']['n_a_better']} / {cc['G']['n_b_better']} / {cc['G']['n_equal']} |")
        L.append("")
        L.append(f"## Service pace per hidden regime — {mode} (analyzer-only diagnostic)\n")
        L.append("Rate = 100·SERVICE_j / alive_steps_j; NA if regime not met while alive. "
                 "fast−slow only for worlds that met both regimes while alive.\n")
        L.append("| strategy | slow (n) | mid (n) | fast (n) | fast−slow [95% CI] (n both) |")
        L.append("|---|---|---|---|---|")
        for name, e in summary["pace"][mode].items():
            cells = []
            for lab in ("slow_m0.48", "mid_m0.92", "fast_m1.65"):
                cells.append(f"{f2(e[lab]['mean_rate'])} ({e[lab]['n_worlds_observed']})")
            fsd = e["fast_minus_slow"]
            if fsd["mean"] is None:
                cells.append(f"NA ({fsd['n_worlds_both_regimes']})")
            else:
                cells.append(f"{fsd['mean']:.2f} [{fsd['ci_low']:.2f}; {fsd['ci_high']:.2f}] "
                             f"({fsd['n_worlds_both_regimes']})")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    L.append("## Dynamic − static contrast of paired effects (exploratory; independent samples; "
             "static uses m=0.92 only, so this is NOT a clean isolation of non-stationarity)\n")
    L.append("| effect | dynamic | static | dyn − static [95% CI] | Δsurvival p.p.: dyn / static / diff [CI] |")
    L.append("|---|---|---|---|---|")
    for name, e in summary["mode_contrast"].items():
        g = e["G"]
        s = e["survival_pp"]
        L.append(f"| {name} | {g['dynamic_effect']:.2f} | {g['static_effect']:.2f} | "
                 f"{g['difference']:.2f} [{g['ci_low']:.2f}; {g['ci_high']:.2f}] | "
                 f"{s['dynamic_effect']:.1f} / {s['static_effect']:.1f} / {s['difference']:.1f} "
                 f"[{s['ci_low']:.1f}; {s['ci_high']:.1f}] |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("development", "confirmatory"), required=True)
    args = ap.parse_args()
    summary = analyze(args.phase)
    out_dir = os.path.join(ROOT, "results", args.phase)
    save_json(summary, os.path.join(out_dir, "summary.json"))
    md = to_markdown(summary)
    with open(os.path.join(out_dir, "summary_tables.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
