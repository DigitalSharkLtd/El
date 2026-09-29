"""Draw the confirmatory master seed AFTER the manifest has been saved (section 9).

Without an operator the master seed is generated here from the OS entropy source
(secrets.randbits(63)).  An operator may instead supply it:
    python scripts/draw_confirmatory_seeds.py --master-seed 123456789
"""
import argparse
import os
import secrets
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bm.experiment import ROOT, save_json, sha256_file             # noqa: E402
from bm.seeds import DERIVATION, derive_confirmatory_seeds         # noqa: E402
from bm.tape import dev_seeds                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-seed", type=int, default=None)
    args = ap.parse_args()
    out = os.path.join(ROOT, "frozen", "confirmatory_seeds.json")
    man = os.path.join(ROOT, "frozen", "manifest.json")
    if not os.path.exists(man):
        sys.exit("freeze first: frozen/manifest.json is missing")
    if os.path.exists(out):
        sys.exit("confirmatory seeds already drawn - refusing to redraw")
    source = "operator" if args.master_seed is not None else \
        "self-generated (secrets.randbits(63)) after the manifest was saved; no external operator"
    master = args.master_seed if args.master_seed is not None else secrets.randbits(63)
    lists = derive_confirmatory_seeds(master)
    dev = set(dev_seeds("dynamic")) | set(dev_seeds("static"))
    assert not (set(lists["dynamic"]) & dev) and not (set(lists["static"]) & dev)
    assert not (set(lists["dynamic"]) & set(lists["static"]))
    save_json({
        "drawn_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": master,
        "master_seed_source": source,
        "derivation": DERIVATION,
        "manifest_sha256_at_draw": sha256_file(man),
        "disjoint_from_development": True,
        "dynamic": lists["dynamic"],
        "static": lists["static"],
        "step_log_worlds": {"dynamic": lists["dynamic"][:3], "static": lists["static"][:3]},
    }, out)
    print("master seed:", master, "| source:", source)


if __name__ == "__main__":
    main()
