"""Exogenous environment tape (spec section 3).

The tape is generated once per world before any strategy runs.  Actions never
consume environment randomness.

RNG: numpy.random.Generator(PCG64).  Streams are derived from the world seed as
    SeedSequence(world_seed).spawn(6)
in the fixed order
    0 availability, 1 wear, 2 sensor noise, 3 other noise,
    4 channel mapping, 5 permutations (q_t).
Array index j = 0..T-1 corresponds to step t = j + 1.
"""
from dataclasses import dataclass

import numpy as np

from . import params as P

STREAM_NAMES = ("availability", "wear", "sensor_noise", "other_noise",
                "channel_map", "permutations")


@dataclass(frozen=True)
class Tape:
    seed: int
    mode: str               # "dynamic" | "static"
    b: np.ndarray           # (T,) availability state after the step-start flip
    a: np.ndarray           # (T,) availability
    regime: np.ndarray      # (T,) wear index 0/1/2
    m: np.ndarray           # (T,) wear multiplier
    ar: np.ndarray          # (T,) AR(1) process, already updated for the step
    walk: np.ndarray        # (T,) clipped random walk, already updated for the step
    h: np.ndarray           # (T,) wear shocks
    eps_prod: np.ndarray    # (T,)
    eps_sensor: np.ndarray  # (T, 8)
    q: np.ndarray           # (T, 8) per-step permutation for C-scrambled
    pi: np.ndarray          # (8,) channel placement permutation
    signs: np.ndarray       # (8,) channel signs, indexed by target position

    def fingerprint(self) -> str:
        import hashlib
        hsh = hashlib.sha256()
        for name in ("b", "a", "regime", "m", "ar", "walk", "h", "eps_prod",
                     "eps_sensor", "q", "pi", "signs"):
            hsh.update(np.ascontiguousarray(getattr(self, name)).tobytes())
        return hsh.hexdigest()


def _streams(seed: int):
    children = np.random.SeedSequence(int(seed)).spawn(len(STREAM_NAMES))
    return [np.random.Generator(np.random.PCG64(c)) for c in children]


def make_tape(seed: int, mode: str) -> Tape:
    if mode not in ("dynamic", "static"):
        raise ValueError(mode)
    T = P.T
    rng_av, rng_wear, rng_sens, rng_oth, rng_map, rng_perm = _streams(seed)

    # 3.1 availability: initial b uniform; at the start of every step (incl. the
    # first) flip with prob 0.04; then a = clip(N(mu_b, 0.16^2), 0, 1.5).
    b0 = int(rng_av.integers(0, 2))
    flips = rng_av.random(T) < P.P_FLIP_B
    b = (b0 + np.cumsum(flips)) % 2
    mu = np.where(b == 1, P.MU_B[1], P.MU_B[0])
    a = np.clip(mu + P.SD_A * rng_av.standard_normal(T), 0.0, P.A_MAX)

    # 3.2 wear regime
    regime = np.empty(T, dtype=np.int64)
    if mode == "static":
        regime[:] = P.STATIC_INDEX
    else:
        idx = int(rng_wear.integers(0, 3))
        dwell = 0
        for j in range(T):
            if dwell >= P.MIN_DWELL:
                if rng_wear.random() < P.P_SWITCH:
                    others = [i for i in range(3) if i != idx]
                    idx = others[int(rng_wear.integers(0, 2))]
                    dwell = 0
            dwell += 1
            regime[j] = idx
    m = np.asarray(P.M_LEVELS, dtype=np.float64)[regime]

    # 3.3 auxiliary processes (ar, walk updated at the start of each step,
    # then used by that step's sensors)
    ar_noise = P.AR_SD * rng_oth.standard_normal(T)
    walk_noise = P.WALK_SD * rng_oth.standard_normal(T)
    h_nonzero = rng_oth.random(T) >= P.P_H_ZERO
    h_draw = np.maximum(0.0, P.H_SD * rng_oth.standard_normal(T))
    h = np.where(h_nonzero, h_draw, 0.0)
    eps_prod = P.EPS_PROD_SD * rng_oth.standard_normal(T)
    ar = np.empty(T)
    walk = np.empty(T)
    ar_v = 0.0
    walk_v = 0.0
    for j in range(T):
        ar_v = P.AR_COEF * ar_v + ar_noise[j]
        walk_v = min(1.0, max(-1.0, walk_v + walk_noise[j]))
        ar[j] = ar_v
        walk[j] = walk_v

    eps_sensor = P.EPS_SENSOR_SD * rng_sens.standard_normal((T, P.N_CH))

    pi = rng_map.permutation(P.N_CH)
    signs = np.where(rng_map.integers(0, 2, size=P.N_CH) == 1, 1.0, -1.0)

    q = np.stack([rng_perm.permutation(P.N_CH) for _ in range(T)])

    arrays = dict(b=b, a=a, regime=regime, m=m, ar=ar, walk=walk, h=h,
                  eps_prod=eps_prod, eps_sensor=eps_sensor, q=q, pi=pi,
                  signs=signs)
    for arr in arrays.values():
        arr.setflags(write=False)
    return Tape(seed=int(seed), mode=mode, **arrays)


def dev_seeds(mode: str):
    base = P.DEV_BASE[mode]
    return [base + P.DEV_STRIDE * i for i in range(P.DEV_N)]
