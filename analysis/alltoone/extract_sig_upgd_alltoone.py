#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提取 SIG/UPGD all-to-one 对比数据（new vs baseline）。"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.extract_all_results import ARCH_TO_OUTPUT, extract_folder_results, parse_folder_name

DATASETS = ["cifar10", "tiny_imagenet", "mnistm"]
ARCH_NORMALIZE = {
    "ResNet18": "resnet18",
    "resnet18": "resnet18",
    "mobilenetv2": "mobilenet",
    "mobilenet": "mobilenet",
    "vgg19_bn": "vgg",
    "vgg": "vgg",
}
ATTACK_ALLOWLIST = {"sig", "upgd"}


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_arch(params: Dict[str, Any]) -> Optional[str]:
    arch_raw = params.get("arch_raw")
    if arch_raw in ARCH_TO_OUTPUT:
        return ARCH_TO_OUTPUT[arch_raw]
    return ARCH_NORMALIZE.get(str(arch_raw).lower()) if arch_raw is not None else None


def _extract_target_label(folder_name: str) -> Optional[int]:
    for pattern in [r"(?:target_label|target_class|target)=(-?\d+)", r"(?:y_target|tgt)=(-?\d+)"]:
        m = re.search(pattern, folder_name, flags=re.I)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def _extract_seed(folder_name: str) -> Optional[int]:
    m = re.search(r"(?:poison_seed|seed)=(-?\d+)", folder_name, flags=re.I)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_extra_attack_params(folder_name: str) -> Dict[str, Optional[float]]:
    def _grab(pattern: str) -> Optional[float]:
        m = re.search(pattern, folder_name, flags=re.I)
        return _safe_float(m.group(1)) if m else None

    return {
        "f": _grab(r"(?:^|_)f=([0-9.]+)"),
        "upgd_steps": _grab(r"upgd_steps=([0-9.]+)"),
        "upgd_steps_multiplier": _grab(r"upgd_steps_multiplier=([0-9.]+)"),
    }


def _build_rows_for_folder(folder: Path, dataset: str, source_set: str) -> List[Dict[str, Any]]:
    params = parse_folder_name(folder.name)
    attack_type = str(params.get("attack_type", "")).lower()
    if attack_type not in ATTACK_ALLOWLIST:
        return []
    params["target_label"] = _extract_target_label(folder.name)
    params["poison_seed"] = _extract_seed(folder.name)
    params.update(_extract_extra_attack_params(folder.name))
    params["model_norm"] = _normalize_arch(params)

    rows = extract_folder_results(folder, params, include_nc=True, dataset=dataset)
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        row_attack = str(row.get("attack_type", "")).lower()
        if row_attack not in ATTACK_ALLOWLIST:
            continue
        row["source_set"] = source_set
        row["attack_family"] = row_attack
        row["mode"] = "all-to-one"
        row["mode_inferred"] = row.get("target_label") is None
        row["dataset"] = dataset
        row["model_norm"] = row.get("model_norm")
        row["folder_name"] = folder.name
        out_rows.append(row)
    return out_rows


def collect_rows(root: Path, source_set: str) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_dir = root / dataset
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            continue
        for folder in dataset_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith("none_"):
                all_rows.extend(_build_rows_for_folder(folder, dataset, source_set))
    return all_rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "source_set", "dataset", "folder_name", "attack_type", "attack_family", "mode", "mode_inferred",
        "model_norm", "poison_rate", "train_param_value", "test_param_type", "test_param_value", "target_label",
        "poison_seed", "delta", "eps", "f", "upgd_steps", "upgd_steps_multiplier", "alpha", "cover_rate",
        "stealth_tpr_avg", "stealth_auc_avg", "transfer_rate", "asr", "nc_max_anomaly_index", "nc_is_poisoned",
        "S_stealth", "S_stealth_tpr",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 SIG/UPGD all-to-one 对比原始数据")
    parser.add_argument("--new-root", type=Path, default=Path("poisoned_train_set"))
    parser.add_argument("--baseline-root", type=Path, default=Path("poisoned_train_set1"))
    parser.add_argument("--output-csv", type=Path, default=Path("analysis/data_sig_upgd_alltoone_raw.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.new_root, "new") + collect_rows(args.baseline_root, "baseline")
    rows.sort(
        key=lambda item: (
            item.get("dataset", ""),
            item.get("attack_family", ""),
            item.get("model_norm") or "",
            _safe_float(item.get("poison_rate")) or -1,
            _safe_float(item.get("test_param_value")) or -1,
            item.get("source_set", ""),
            item.get("folder_name", ""),
        )
    )
    write_csv(rows, args.output_csv)
    print(f"✓ 提取完成: {args.output_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
