"""Deterministic derivation of the confirmatory world lists from a master seed."""
import numpy as np

from . import params as P
from .tape import dev_seeds

DERIVATION = (
    "rng = numpy.random.Generator(PCG64(SeedSequence(master_seed))); "
    "dynamic list: draw rng.integers(0, 2**63 - 1) repeatedly, skipping values already "
    "used (development seeds of both modes, or earlier draws), until 512 values; "
    "static list: continue with the same generator until 256 further values."
)


def derive_confirmatory_seeds(master_seed: int):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(master_seed))))
    used = set(dev_seeds("dynamic")) | set(dev_seeds("static"))
    out = {}
    for mode in ("dynamic", "static"):
        lst = []
        while len(lst) < P.CONF_N[mode]:
            s = int(rng.integers(0, 2**63 - 1))
            if s in used:
                continue
            used.add(s)
            lst.append(s)
        out[mode] = lst
    return out
