#!/usr/bin/env python3
"""Run an independent subset of the full SYN -> SVHN CIFAR-matched grid."""

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from run.run_syn_svhn_pilot import ensure_clean_model, run_configuration
from run.syn_svhn_full_config import ATTACKS, FULL_ROOT, MODELS, configuration_count, spec_for_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["4", "5", "6", "7"], required=True)
    parser.add_argument("--attacks", choices=sorted(ATTACKS), nargs="+", required=True)
    parser.add_argument("--models", choices=list(MODELS), nargs="+", default=list(MODELS))
    parser.add_argument("--prepare-clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    split_files = [
        Path("clean_set/syn/clean_split/clean_labels"),
        Path("clean_set/syn/test_split/labels"),
    ]
    if not args.dry_run and not all(path.is_file() for path in split_files):
        raise FileNotFoundError("SYN clean/test defense splits are missing; run create_clean_set.py first")

    worker_root = Path(FULL_ROOT) / "workers" / f"gpu{args.device}"
    print(
        f"[worker] device={args.device} attacks={args.attacks} models={args.models} "
        f"configurations={configuration_count(args.attacks, args.models)}"
    )
    ok = True
    for model in args.models:
        model_root = worker_root / model
        if args.prepare_clean:
            clean_root = model_root / "clean"
            try:
                ensure_clean_model(
                    clean_root, clean_root / "failures.csv", args.dry_run,
                    no_normalize=False, model=model, devices=args.device,
                )
            except (OSError, RuntimeError) as error:
                print(f"[failed] normalized clean model={model}: {error}", file=sys.stderr)
                ok = False

        raw_clean_checkpoint = None
        if "upgd" in args.attacks:
            raw_root = model_root / "upgd_raw_clean"
            try:
                raw_clean_checkpoint = ensure_clean_model(
                    raw_root, raw_root / "failures.csv", args.dry_run,
                    no_normalize=True, model=model, devices=args.device,
                )
            except (OSError, RuntimeError) as error:
                print(f"[failed] UPGD raw clean model={model}: {error}", file=sys.stderr)
                ok = False

        for attack in args.attacks:
            if attack == "upgd" and raw_clean_checkpoint is None:
                print(f"[skip] UPGD model={model}: raw clean checkpoint unavailable", file=sys.stderr)
                ok = False
                continue
            base_spec = ATTACKS[attack]
            for poison_rate in base_spec["poison_rates"]:
                spec = spec_for_rate(attack, poison_rate)
                scope_root = model_root / attack / f"poison_rate={poison_rate:g}"
                failure_csv = scope_root / "failures.csv"
                for strength in spec["strengths"]:
                    completed = run_configuration(
                        attack, spec, strength, scope_root, failure_csv,
                        raw_clean_checkpoint, args.dry_run, model=model,
                        devices=args.device, poison_rate=poison_rate,
                    )
                    ok = completed and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
