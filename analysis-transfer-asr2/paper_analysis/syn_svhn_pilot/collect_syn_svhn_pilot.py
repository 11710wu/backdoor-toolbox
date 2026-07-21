#!/usr/bin/env python3
"""Collect SYN -> SVHN pilot artifacts without touching paper master results."""

import csv
import json
import math
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from run.run_syn_svhn_pilot import experiment_args
from run.syn_svhn_pilot_config import ATTACKS, PILOT_ROOT, POISON_SEED, TRAINING_SEED
from utils import supervisor


DEFENSE_FILES = {
    "sentinet": "sentinet_defense_results.json",
    "scaleup": "scaleup_defense_results.json",
    "strip": "strip_defense_results.json",
    "ibd_psc": "ibd_psc_defense_results.json",
}


def read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def find_source_result(directory, poison_type):
    if poison_type == "SIG":
        candidates = sorted(directory.glob(f"test_results_seed={TRAINING_SEED}_delta=*.json"))
    elif poison_type == "WaNet":
        candidates = sorted(directory.glob(f"test_results_seed={TRAINING_SEED}_s=*.json"))
    else:
        candidates = [directory / f"test_results_seed={TRAINING_SEED}.json"]
    return read_json(candidates[-1]) if candidates else None


def main():
    output = Path(PILOT_ROOT)
    run_rows = []
    detector_rows = []
    summary_rows = []
    for attack_key, spec in ATTACKS.items():
        for level_index, strength in enumerate(spec["strengths"]):
            args = experiment_args(spec, strength)
            directory = Path(supervisor.get_poison_set_dir(args))
            if not (directory / "data").is_file():
                continue
            poison_type = args.poison_type
            strength_name = spec["strength_name"]
            strength = float(strength)
            source = find_source_result(directory, poison_type)
            target = read_json(directory / f"test_svhn_results_seed={TRAINING_SEED}.json")
            source_acc = finite(source.get("source_clean_acc", source.get("clean_acc"))) if source else None
            source_asr = finite(source.get("source_asr", source.get("asr"))) if source else None
            target_acc = finite(target.get("target_clean_acc")) if target else None
            transfer_asr = finite(target.get("target_transfer_asr")) if target else None
            run = {
                "attack": attack_key, "poison_type": poison_type,
                "strength_level": ["low", "medium", "high"][level_index],
                "strength_name": strength_name, "strength_value": strength,
                "poison_rate": args.poison_rate, "cover_rate": args.cover_rate,
                "training_seed": TRAINING_SEED, "poison_seed": POISON_SEED,
                "source_clean_acc": source_acc, "source_asr": source_asr,
                "source_difficulty": 1.0 - source_acc if source_acc is not None else None,
                "target_clean_acc": target_acc, "target_transfer_asr": transfer_asr,
                "target_difficulty": 1.0 - target_acc if target_acc is not None else None,
                "source_asr_ge_5pct": source_asr is not None and source_asr >= 0.05,
                "source_asr_ge_10pct": source_asr is not None and source_asr >= 0.10,
                "source_clean_correct": source.get("source_clean_correct", source.get("clean_correct")) if source else None,
                "source_clean_total": source.get("source_clean_total", source.get("clean_total")) if source else None,
                "source_asr_success": source.get("source_asr_success", source.get("asr_success")) if source else None,
                "source_asr_eligible": source.get("source_asr_eligible", source.get("asr_eligible")) if source else None,
                "target_clean_correct": target.get("target_clean_correct") if target else None,
                "target_clean_total": target.get("target_clean_total") if target else None,
                "transfer_success": target.get("transfer_success") if target else None,
                "target_transfer_eligible": target.get("transfer_eligible") if target else None,
                "result_dir": str(directory),
            }
            run_rows.append(run)
            tprs = {}
            for detector, filename in DEFENSE_FILES.items():
                payload = read_json(directory / filename)
                tpr_percent = finite(payload.get("tpr")) if payload else None
                fpr_percent = finite(payload.get("fpr")) if payload else None
                if tpr_percent is not None and not 0.0 <= tpr_percent <= 100.0:
                    raise ValueError(f"Invalid percentage TPR in {directory / filename}: {tpr_percent}")
                if fpr_percent is not None and not 0.0 <= fpr_percent <= 100.0:
                    raise ValueError(f"Invalid percentage FPR in {directory / filename}: {fpr_percent}")
                tpr = tpr_percent / 100.0 if tpr_percent is not None else None
                fpr = fpr_percent / 100.0 if fpr_percent is not None else None
                tprs[detector] = tpr
                detector_rows.append({
                    "attack": attack_key, "strength_name": strength_name,
                    "strength_value": strength, "detector": detector,
                    "tpr": tpr, "fpr": fpr, "tpr_storage_unit": "percent_0_100",
                    "status": ("complete" if tpr is not None and fpr is not None
                               else "missing" if payload is None else "missing_metric"),
                    "result_file": str(directory / filename),
                })
            complete = all(value is not None for value in tprs.values())
            summary_rows.append({
                **run,
                **{f"{name}_tpr": value for name, value in tprs.items()},
                "stealthiness": 1.0 - sum(tprs.values()) / 4.0 if complete else None,
                "stealthiness_status": "complete" if complete else "NA_missing_detector",
            })

    level_order = {"low": 0, "medium": 1, "high": 2}
    run_rows.sort(key=lambda row: (row["attack"], level_order[row["strength_level"]]))
    detector_rows.sort(key=lambda row: (row["attack"], row["strength_value"], row["detector"]))
    summary_rows.sort(key=lambda row: (row["attack"], level_order[row["strength_level"]]))
    run_fields = [
        "attack", "poison_type", "strength_level", "strength_name", "strength_value",
        "poison_rate", "cover_rate", "training_seed", "poison_seed", "source_clean_acc",
        "source_asr", "source_difficulty", "target_clean_acc", "target_transfer_asr",
        "target_difficulty", "source_asr_ge_5pct", "source_asr_ge_10pct",
        "source_clean_correct", "source_clean_total", "source_asr_success",
        "source_asr_eligible", "target_clean_correct", "target_clean_total",
        "transfer_success", "target_transfer_eligible", "result_dir",
    ]
    write_csv(output / "pilot_runs.csv", run_rows, run_fields)
    write_csv(output / "pilot_detectors.csv", detector_rows, [
        "attack", "strength_name", "strength_value", "detector", "tpr", "fpr",
        "tpr_storage_unit", "status", "result_file",
    ])
    write_csv(output / "pilot_summary.csv", summary_rows, run_fields + [
        "sentinet_tpr", "scaleup_tpr", "strip_tpr", "ibd_psc_tpr",
        "stealthiness", "stealthiness_status",
    ])
    failure_path = output / "pilot_failures.csv"
    if not failure_path.exists():
        write_csv(failure_path, [], ["attack", "strength", "stage", "returncode", "command", "log_path"])
    print(f"[pilot collector] {len(run_rows)} runs, {len(detector_rows)} detector rows -> {output}")


if __name__ == "__main__":
    main()
