"""Freeze manifest (section 9): code, parameters, selected settings, versions, RNG,
metrics and SHA-256 of files.  Must be run before confirmatory seeds are drawn.

Usage:  python scripts/freeze.py
"""
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bm import params as P                                        # noqa: E402
from bm.experiment import ROOT, environment_info, save_json, sha256_file  # noqa: E402
from bm.runner import METRIC_COLUMNS                               # noqa: E402
from bm.seeds import DERIVATION                                    # noqa: E402
from bm.tape import STREAM_NAMES                                   # noqa: E402

# files whose hashes are enforced by run_confirmatory.py
ENFORCED_GLOBS = ["bm/*.py", "scripts/*.py", "tests/*.py", "requirements.txt",
                  "protocol/*.md", "frozen/settings.json"]
# files recorded for provenance only
RECORDED_GLOBS = ["README.md", "results/development/*", "results/tests/*"]


def collect(globs):
    out = {}
    for g in globs:
        for p in sorted(glob.glob(os.path.join(ROOT, g))):
            if os.path.isfile(p) and "__pycache__" not in p:
                out[os.path.relpath(p, ROOT)] = sha256_file(p)
    return out


def main():
    path = os.path.join(ROOT, "frozen", "manifest.json")
    if os.path.exists(path):
        sys.exit("frozen/manifest.json already exists - refusing to overwrite a freeze")
    with open(os.path.join(ROOT, "frozen", "settings.json")) as f:
        settings = json.load(f)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "protocol/Blind_Maintenance_Benchmark_RU_v1.md (package A, v1.0)",
        "environment": environment_info(),
        "rng": {
            "algorithm": "numpy.random.Generator(PCG64)",
            "world_streams": "SeedSequence(world_seed).spawn(6) in order " + ", ".join(STREAM_NAMES),
            "bootstrap": f"SeedSequence([{P.ANALYSIS_SEED}, mode_code]) with mode_code dynamic=1, "
                         "static=2; one index matrix per mode shared by all comparisons",
            "confirmatory_seed_derivation": DERIVATION,
        },
        "development_seeds": {m: {"base": P.DEV_BASE[m], "stride": P.DEV_STRIDE, "n": P.DEV_N}
                              for m in ("dynamic", "static")},
        "confirmatory_sizes": P.CONF_N,
        "selected_settings": {m: {k: settings[m][k] for k in
                                  ("fixed_star", "tuned_output", "output_trend")}
                              for m in ("dynamic", "static")},
        "optional_own_agent": "not used (no exploratory agent)",
        "metrics": METRIC_COLUMNS,
        "analysis_plan": {
            "primary": "mean(G_C - G_fixed*) in dynamic confirmatory worlds, paired percentile "
                       "bootstrap 95% CI, 10000 resamples of worlds",
            "companion": ["survival difference (p.p.)", "final S difference",
                          "SERVICE count difference"],
            "secondary_exploratory_unadjusted": [
                "C vs C_scrambled, C_level, output_only, tuned_output, output_trend, oracle_D, "
                "all fixed (both modes; static uses its own fixed*)",
                "dynamic - static contrast of paired effects",
                "service pace per hidden regime (fast - slow), subsample size reported"],
            "no_pass_fail": True,
        },
        "enforced_files_sha256": collect(ENFORCED_GLOBS),
        "recorded_files_sha256": collect(RECORDED_GLOBS),
    }
    save_json(manifest, path)
    digest = sha256_file(path)
    with open(os.path.join(ROOT, "frozen", "MANIFEST_SHA256.txt"), "w") as f:
        f.write(f"{digest}  frozen/manifest.json\n")
    print("manifest sha256:", digest)


if __name__ == "__main__":
    main()
