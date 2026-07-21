#!/usr/bin/env python3
"""Run one three-point SYN -> SVHN attack pilot with resumable stages."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import config
from run.syn_svhn_pilot_config import ATTACKS, PILOT_ROOT, POISON_RATE, POISON_SEED, TRAINING_SEED
from utils import supervisor


DEFENSES = {
    "SentiNet": "sentinet_defense_results.json",
    "ScaleUp": "scaleup_defense_results.json",
    "STRIP": "strip_defense_results.json",
    "IBD_PSC": "ibd_psc_defense_results.json",
}


def experiment_args(spec, strength, poison_type=None, poison_rate=POISON_RATE):
    values = {
        "dataset": "syn", "poison_type": poison_type or spec["poison_type"],
        "poison_rate": poison_rate, "cover_rate": spec.get("cover_rate", 0.0),
        "alpha": spec.get("alpha", 0.2), "s": spec.get("s", 0.5), "k": spec.get("k", 4),
        "delta": spec.get("delta", 30.0), "f": spec.get("f", 6),
        "eps": spec.get("eps", 8.0), "constraint": spec.get("constraint", "Linf"),
        "upgd_steps": spec.get("upgd_steps", 100),
        "upgd_steps_multiplier": spec.get("upgd_steps_multiplier", 5),
        "mask_rate": spec.get("mask_rate", 0.2), "label_mode": spec.get("label_mode", "clean"),
        "model": "resnet18", "poison_seed": POISON_SEED,
        "sample_cap": None, "test_alpha": None, "no_normalize": spec.get("no_normalize", False),
    }
    values[spec["strength_name"]] = strength
    values["trigger"] = config.trigger_default["syn"][values["poison_type"]]
    return SimpleNamespace(**values)


def cli_args(args):
    pairs = [
        ("dataset", args.dataset), ("poison_type", args.poison_type),
        ("poison_rate", args.poison_rate), ("cover_rate", args.cover_rate),
        ("alpha", args.alpha), ("label_mode", args.label_mode), ("model", args.model),
        ("s", args.s), ("k", args.k),
        ("delta", args.delta), ("f", args.f), ("eps", args.eps),
        ("constraint", args.constraint), ("upgd_steps", args.upgd_steps),
        ("upgd_steps_multiplier", args.upgd_steps_multiplier), ("mask_rate", args.mask_rate),
    ]
    result = []
    for name, value in pairs:
        result.extend([f"-{name}", str(value)])
    return result


def record_failure(path, attack, strength, stage, command, returncode, log_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "attack", "strength", "stage", "returncode", "command", "log_path",
        ])
        if new_file:
            writer.writeheader()
        writer.writerow({
            "attack": attack, "strength": strength, "stage": stage,
            "returncode": returncode, "command": " ".join(command), "log_path": str(log_path),
        })


def existing_stage_is_valid(stage, path):
    if stage.endswith("create"):
        return path.is_file() and path.stat().st_size > 0
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    if stage.endswith("train"):
        model_path = payload.get("model_path")
        return payload.get("final_epoch") == 200 and model_path and Path(model_path).is_file()
    if stage.endswith("source_test"):
        total = payload.get("source_clean_total", payload.get("clean_total"))
        return total == 9_553 and payload.get("model_path") and Path(payload["model_path"]).is_file()
    if stage.endswith("target_test"):
        return (payload.get("target_clean_total") == 10_000
                and payload.get("bn_recalibration") is False
                and payload.get("model_path") and Path(payload["model_path"]).is_file())
    if stage.startswith("defense_"):
        try:
            tpr, fpr = float(payload["tpr"]), float(payload["fpr"])
        except (KeyError, TypeError, ValueError):
            return False
        return math.isfinite(tpr) and math.isfinite(fpr) and 0 <= tpr <= 100 and 0 <= fpr <= 100
    return True


def run_stage(stage, command, expected, log_dir, failure_csv, attack, strength,
              dry_run=False):
    log_path = log_dir / f"{stage}.log"
    if expected is not None and expected.exists():
        if existing_stage_is_valid(stage, expected):
            print(f"[skip] {stage}: {expected}")
            return True
        print(f"[invalid existing artifact] {stage}: {expected}", file=sys.stderr)
        if not dry_run:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"Invalid existing artifact: {expected}\n", encoding="utf-8")
            record_failure(failure_csv, attack, strength, stage, command, -2, log_path)
        return False
    print("[run]", " ".join(command))
    if dry_run:
        return True
    log_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    artifact_ok = (expected is None or
                   (expected.exists() and existing_stage_is_valid(stage, expected)))
    if result.returncode != 0 or not artifact_ok:
        record_failure(failure_csv, attack, strength, stage, command, result.returncode, log_path)
        print(f"[failed] {stage}; see {log_path}", file=sys.stderr)
        return False
    return True


def result_paths(args):
    poison_dir = Path(supervisor.get_poison_set_dir(args))
    if args.poison_type == "belt":
        checkpoint = poison_dir / f"{supervisor.get_arch(args).__name__}_belt_aug_model_seed={TRAINING_SEED}.pt"
    else:
        checkpoint = Path(supervisor.get_model_dir(args))
    if args.poison_type == "SIG":
        source = poison_dir / f"test_results_seed={TRAINING_SEED}_delta={args.delta}.json"
    elif args.poison_type == "WaNet":
        source = poison_dir / f"test_results_seed={TRAINING_SEED}_s={args.s}.json"
    else:
        source = poison_dir / f"test_results_seed={TRAINING_SEED}.json"
    return poison_dir, checkpoint, source, poison_dir / f"test_svhn_results_seed={TRAINING_SEED}.json"


def run_configuration(attack, spec, strength, root, failure_csv, clean_checkpoint, dry_run):
    args = experiment_args(spec, strength)
    poison_dir, checkpoint, source_json, target_json = result_paths(args)
    log_dir = root / "logs" / attack / f"{spec['strength_name']}={strength:g}"
    common = cli_args(args)
    create = [sys.executable, "create_poisoned_set.py", *common]
    if args.poison_type == "upgd":
        create.extend(["-upgd_model_path", str(clean_checkpoint)])
    stages = [
        ("create", create, poison_dir / "data"),
    ]
    train = [sys.executable, "train_on_poisoned_set.py", *common, "-seed", str(TRAINING_SEED)]
    stages.extend([
        ("train", train, poison_dir / f"train_results_seed={TRAINING_SEED}.json"),
        ("source_test", [sys.executable, "test_model.py", *common, "-seed", str(TRAINING_SEED)], source_json),
        ("target_test", [sys.executable, "test_svhn.py", *common, "-seed", str(TRAINING_SEED)], target_json),
    ])
    for stage, command, expected in stages:
        if stage != "create" and args.no_normalize:
            command.append("-no_normalize")
        if not run_stage(stage, command, expected, log_dir, failure_csv, attack, strength,
                         dry_run):
            return False
    for defense, filename in DEFENSES.items():
        command = [sys.executable, "other_defense.py", *common, "-seed", str(TRAINING_SEED), "-defense", defense]
        if args.no_normalize:
            command.append("-no_normalize")
        if not run_stage(f"defense_{defense}", command, poison_dir / filename,
                         log_dir, failure_csv, attack, strength, dry_run):
            return False
    return checkpoint.exists() or dry_run


def ensure_clean_model(root, failure_csv, dry_run, no_normalize=False):
    spec = {
        "poison_type": "none", "strength_name": "alpha", "alpha": 0.0,
        "no_normalize": no_normalize,
    }
    args = experiment_args(spec, 0.0, poison_type="none", poison_rate=0.0)
    poison_dir, checkpoint, source_json, target_json = result_paths(args)
    log_dir = root / "logs" / "clean"
    common = cli_args(args)
    create = ("create", [sys.executable, "create_poisoned_set.py", *common], poison_dir / "data")
    if no_normalize:
        arch_name = supervisor.get_arch(args).__name__
        raw_dir = (Path(supervisor.get_poisoned_train_set_root()) / "syn" /
                   f"upgd_raw_base_0.000_poison_seed={POISON_SEED}_arch={arch_name}")
        checkpoint = raw_dir / f"upgd_raw_base_{arch_name}.pt"
        train = [sys.executable, "train_on_poisoned_set.py", *common,
                 "-seed", str(TRAINING_SEED), "-no_normalize", "-model_path", str(checkpoint)]
        stages = [
            create,
            ("train", train, raw_dir / f"train_results_seed={TRAINING_SEED}.json"),
        ]
    else:
        stages = [
            create,
            ("train", [sys.executable, "train_on_poisoned_set.py", *common,
                       "-seed", str(TRAINING_SEED)],
             poison_dir / f"train_results_seed={TRAINING_SEED}.json"),
            ("source_test", [sys.executable, "test_model.py", *common,
                             "-seed", str(TRAINING_SEED)], source_json),
            ("target_test", [sys.executable, "test_svhn.py", *common,
                             "-seed", str(TRAINING_SEED)], target_json),
        ]
    for stage, command, expected in stages:
        if not run_stage(f"clean_{stage}", command, expected, log_dir, failure_csv,
                         "clean", 0.0, dry_run):
            raise RuntimeError("Clean SYN model preparation failed")
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", choices=sorted(ATTACKS), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(PILOT_ROOT)
    failure_csv = root / "pilot_failures.csv"
    if not args.dry_run:
        if not (Path("clean_set/syn/clean_split/clean_labels").exists()
                and Path("clean_set/syn/test_split/labels").exists()):
            subprocess.run([sys.executable, "create_clean_set.py", "-dataset", "syn"], check=True)
    clean_checkpoint = ensure_clean_model(
        root, failure_csv, args.dry_run, no_normalize=args.attack == "upgd"
    )
    spec = ATTACKS[args.attack]
    ok = True
    for strength in spec["strengths"]:
        ok = run_configuration(args.attack, spec, strength, root, failure_csv,
                               clean_checkpoint, args.dry_run) and ok
        if not args.dry_run:
            subprocess.run([
                sys.executable,
                "analysis-transfer-asr2/paper_analysis/syn_svhn_pilot/collect_syn_svhn_pilot.py",
            ], check=False)
    if not args.dry_run:
        subprocess.run([
            sys.executable,
            "analysis-transfer-asr2/paper_analysis/syn_svhn_pilot/collect_syn_svhn_pilot.py",
        ], check=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
