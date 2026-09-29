"""Paired percentile bootstrap over worlds and Wilson intervals (section 10)."""
import math

import numpy as np

from . import params as P

MODE_CODE = {"dynamic": 1, "static": 2}


def boot_index(n, mode, n_boot=P.N_BOOT, seed=P.ANALYSIS_SEED):
    """One fixed resampling matrix per mode (worlds with replacement).
    The same matrix is used for every strategy and comparison of that mode,
    which keeps all comparisons paired within world."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, MODE_CODE[mode]])))
    return rng.integers(0, n, size=(n_boot, n))


def boot_means(values, idx):
    v = np.asarray(values, dtype=np.float64)
    return v[idx].mean(axis=1)


def percentile_ci(boot, level=0.95):
    lo, hi = np.percentile(boot, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def summarize(values, idx):
    v = np.asarray(values, dtype=np.float64)
    b = boot_means(v, idx)
    lo, hi = percentile_ci(b)
    return {"mean": float(v.mean()), "ci_low": lo, "ci_high": hi,
            "degenerate": bool(np.all(v == v[0]))}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))
