"""Batch execution helpers shared by the scripts."""
import csv
import hashlib
import json
import math
import os
import platform
import sys
from multiprocessing import Pool

import numpy as np

from . import params as P
from .policies import base_strategies, tuning_grid
from .runner import METRIC_COLUMNS, run_world

ROW_COLUMNS = ["phase", "mode", "world_index", "world_seed", "strategy", "config",
               "tape_sha256"] + METRIC_COLUMNS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _job(args):
    phase, mode, idx, seed, settings, with_grid = args
    pols = base_strategies(settings)
    if with_grid:
        pols += tuning_grid()
    rows = run_world(seed, mode, pols)
    for r in rows:
        r["phase"] = phase
        r["world_index"] = idx
    return rows


def run_many(phase, mode, seeds, settings=None, with_grid=False, processes=None):
    jobs = [(phase, mode, i, s, settings, with_grid) for i, s in enumerate(seeds)]
    processes = processes or min(4, os.cpu_count() or 1)
    out = []
    with Pool(processes) as pool:
        for rows in pool.imap(_job, jobs, chunksize=2):
            out.extend(rows)
    return out


def write_csv(rows, path, columns=ROW_COLUMNS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: _fmt(r.get(k)) for k in columns})


def _fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return "NA"
        return repr(v)
    return v


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_info():
    info = {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            info["mem_total"] = f.readline().split(":", 1)[1].strip()
    except OSError:
        pass
    return info


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
        f.write("\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
