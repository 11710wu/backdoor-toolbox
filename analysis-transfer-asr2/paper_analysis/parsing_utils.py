#!/usr/bin/env python3
"""Filesystem parsers for result folders."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from config import ATTACK_ORDER, DATASETS, DEFENSES, MASTER_COLUMNS, SOURCE_ASR_MAIN_THRESHOLD
from metrics import (
    chance_rate_for_dataset,
    compute_chance_adjusted_rate,
    compute_difficulty,
    compute_joint_transfer,
    compute_stealth_from_tprs,
    compute_transfer_gap,
    compute_transfer_rate,
    compute_transfer_retention_rate,
    compute_transferability,
    normalize_percent_rate,
    normalize_rate,
    stealth_component,
)


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def parse_folder_name(name: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
    attack_type = "unknown"
    for attack in sorted(ATTACK_ORDER, key=len, reverse=True):
        if name == attack or name.startswith(f"{attack}_"):
            attack_type = attack
            break
    if attack_type == "basic":
        attack_family = "badnet"
    else:
        attack_family = attack_type

    poison_rate = float("nan")
    m_rate = re.match(r"^[A-Za-z_]+_([0-9]+(?:\.[0-9]+)?)", name)
    if m_rate:
        poison_rate = safe_float(m_rate.group(1))
    if attack_type == "none":
        poison_rate = 0.0

    params: Dict[str, Any] = {
        "attack_type": attack_type,
        "attack_family": attack_family,
        "poison_rate": poison_rate,
        "strength_name": "",
        "strength_value": float("nan"),
        "cover_rate": float("nan"),
        "mask_rate": float("nan"),
        "label_mode": "clean",
        "input_noise_type": "",
        "input_noise_level": float("nan"),
        "arch": "",
        "arch_base": "",
        "dataset": dataset_hint or "",
    }

    if m := re.search(r"(?:^|_)mask=([0-9.]+)", name):
        params["mask_rate"] = safe_float(m.group(1).rstrip("."))

    # BELT folders encode the intended trigger-strength axis as alpha while
    # mask_rate is a fixed trigger-shape parameter used for strict matching.
    strength_patterns = (
        [("alpha", r"(?:^|_)alpha=([0-9.]+)"), ("mask_rate", r"(?:^|_)mask=([0-9.]+)")]
        if attack_type == "belt"
        else [
            ("eps", r"(?:^|_)eps=([0-9.]+)"),
            ("delta", r"(?:^|_)delta=([0-9.]+)"),
            ("s", r"(?:^|_)s=([0-9.]+)"),
            ("alpha", r"(?:^|_)alpha=([0-9.]+)"),
            ("mask_rate", r"(?:^|_)mask=([0-9.]+)"),
        ]
    )
    for key, pattern in strength_patterns:
        m = re.search(pattern, name)
        if m:
            params["strength_name"] = key
            params["strength_value"] = safe_float(m.group(1).rstrip("."))
            break

    if m := re.search(r"(?:^|_)cover=([0-9.]+)", name):
        params["cover_rate"] = safe_float(m.group(1).rstrip("."))
    if m := re.search(r"(?:^|_)mode=(clean|all2one)(?:_|$)", name):
        params["label_mode"] = m.group(1)
    elif m := re.search(r"(?:^|_)mode=([^_]+)", name):
        params["label_mode"] = m.group(1)
    if m := re.search(r"(?:^|_)noise=([A-Za-z0-9_]+)_level=([0-9.]+)", name):
        params["input_noise_type"] = m.group(1)
        params["input_noise_level"] = safe_float(m.group(2).rstrip("."))

    if m := re.search(r"arch=(.+?)_(cifar10|mnistm|tiny_imagenet)$", name):
        params["arch_base"] = m.group(1)
        params["dataset"] = m.group(2)
        params["arch"] = f"{params['arch_base']}_{params['dataset']}"
    elif dataset_hint:
        params["dataset"] = dataset_hint

    return params


def choose_source_file(folder: Path) -> Optional[Path]:
    base = folder / "test_results_seed=2333.json"
    if base.exists():
        return base
    files = sorted(folder.glob("test_results_seed=2333*.json"))
    return files[0] if files else None


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def transfer_variant_from_name(filename: str, prefix: str) -> str:
    stem = filename[:-4] if filename.endswith(".txt") else filename
    suffix = stem[len(prefix) :] if stem.startswith(prefix) else ""
    return suffix.lstrip("_") or "default"


def list_transfer_files(folder: Path, dataset: str) -> List[Tuple[Path, str, str]]:
    candidates: List[Tuple[Path, str, str]] = []
    if dataset == "cifar10":
        prefix = "test_stl10_results"
        for path in _unique_paths(sorted(folder.glob(f"{prefix}*.txt"))):
            candidates.append((path, "stl10", transfer_variant_from_name(path.name, prefix)))
    elif dataset == "mnistm":
        for prefix, transfer_dataset in [
            ("test_mnist_cross_results", "mnist_cross"),
            ("test_mnistm_results", "mnist_cross"),
        ]:
            for path in _unique_paths(sorted(folder.glob(f"{prefix}*.txt"))):
                candidates.append((path, transfer_dataset, transfer_variant_from_name(path.name, prefix)))
    elif dataset == "tiny_imagenet":
        for prefix, transfer_dataset in [
            ("test_tiny_target_domain_results", "imagenetv2_tiny"),
            ("test_tiny_target_domain_qwen_results", "qwen"),
        ]:
            for path in _unique_paths(sorted(folder.glob(f"{prefix}*.txt"))):
                candidates.append((path, transfer_dataset, transfer_variant_from_name(path.name, prefix)))
    else:
        prefix = "test_"
        for path in _unique_paths(sorted(folder.glob("test_*results*.txt"))):
            if "corruption=" in path.name:
                continue
            candidates.append((path, "unknown", transfer_variant_from_name(path.name, prefix)))
    return sorted(candidates, key=lambda item: (item[1], item[2], item[0].name))


def default_transfer_dataset(dataset: str) -> str:
    if dataset == "cifar10":
        return "stl10"
    if dataset == "mnistm":
        return "mnist_cross"
    if dataset == "tiny_imagenet":
        return "imagenetv2_tiny"
    return "unknown"


def is_main_transfer_dataset(dataset: str, transfer_dataset: str) -> bool:
    return transfer_dataset == default_transfer_dataset(dataset)


def parse_transfer_text(path: Path) -> Tuple[float, float]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return float("nan"), float("nan")

    acc = float("nan")
    asr = float("nan")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if ("准确率" in line or lower.startswith("accuracy") or "clean acc" in lower) and re.search(r"[:：]", line):
            if m := re.search(r"[:：]\s*([0-9.]+)", line):
                acc = normalize_rate(m.group(1))
        if ("攻击成功率" in line or "asr" in lower) and re.search(r"[:：]", line):
            if m := re.search(r"[:：]\s*([0-9.]+)", line):
                asr = normalize_rate(m.group(1))
    return acc, asr


def parse_defense_file(path: Optional[Path]) -> Dict[str, float]:
    if path is None:
        return {"tpr": float("nan"), "auc": float("nan"), "fpr": float("nan"), "threshold": float("nan")}
    data = load_json(path)
    if not data:
        return {"tpr": float("nan"), "auc": float("nan"), "fpr": float("nan"), "threshold": float("nan")}
    return {
        # Defense implementations persist TPR/FPR on a 0--100 percentage scale,
        # including legitimate values below 1%.
        "tpr": normalize_percent_rate(data.get("tpr")),
        "auc": normalize_rate(data.get("auc")),
        "fpr": normalize_percent_rate(data.get("fpr")),
        "threshold": safe_float(data.get("threshold", data.get("threshold_low", float("nan")))),
    }


def choose_defense_file(folder: Path, prefix: str) -> Optional[Path]:
    base = folder / f"{prefix}_defense_results.json"
    if base.exists():
        return base
    files = sorted(folder.glob(f"{prefix}_defense_results*.json"))
    return files[0] if files else None


def parse_result_folder(folder: Path, result_group: str, dataset_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    params = parse_folder_name(folder.name, dataset_hint=dataset_hint)
    dataset = params.get("dataset") or dataset_hint or folder.parent.name

    base_row: Dict[str, Any] = {
        "result_group": result_group,
        "dataset": dataset,
        "transfer_dataset": "",
        "arch": params.get("arch", ""),
        "arch_base": params.get("arch_base", ""),
        "attack_type": params.get("attack_type", "unknown"),
        "poison_rate": params.get("poison_rate", float("nan")),
        "strength_name": params.get("strength_name", ""),
        "strength_value": params.get("strength_value", float("nan")),
        "cover_rate": params.get("cover_rate", float("nan")),
        "mask_rate": params.get("mask_rate", float("nan")),
        "label_mode": params.get("label_mode", "clean"),
        "input_noise_type": params.get("input_noise_type", ""),
        "input_noise_level": params.get("input_noise_level", float("nan")),
        "clean_acc": float("nan"),
        "difficulty": float("nan"),
        "source_asr": float("nan"),
        "transfer_acc": float("nan"),
        "transfer_asr": float("nan"),
        "transferability": float("nan"),
        "transfer_asr_chance_adjusted": float("nan"),
        "transfer_rate": float("nan"),
        "legacy_transfer_rate": float("nan"),
        "transfer_retention_rate": float("nan"),
        "transfer_gap": float("nan"),
        "joint_transfer": float("nan"),
        "transfer_variant": "default",
        "transfer_file": "",
        "is_main_transfer_dataset": False,
        "complete_analysis_row": False,
        "result_dir": str(folder),
        "folder_name": folder.name,
    }

    base_missing: List[str] = []

    source_file = choose_source_file(folder)
    if source_file is None:
        base_missing.append("source_test")
    else:
        source_data = load_json(source_file)
        if not source_data:
            base_missing.append("source_test_json_parse")
        else:
            base_row["clean_acc"] = normalize_rate(source_data.get("clean_acc", source_data.get("acc", source_data.get("test_acc"))))
            base_row["source_asr"] = normalize_rate(source_data.get("asr", source_data.get("attack_success_rate", source_data.get("test_asr"))))
            base_row["difficulty"] = compute_difficulty(base_row["clean_acc"])

    for defense_name, prefix in DEFENSES.items():
        defense_path = choose_defense_file(folder, prefix)
        if defense_path is None:
            base_missing.append(f"{defense_name}_defense")
        parsed = parse_defense_file(defense_path)
        base_row[f"{defense_name}_tpr"] = parsed["tpr"]
        base_row[f"stealth_{defense_name}"] = stealth_component(parsed["tpr"])

    base_row["stealthiness"] = compute_stealth_from_tprs(
        [base_row["sentinet_tpr"], base_row["scaleup_tpr"], base_row["strip_tpr"], base_row["ibd_psc_tpr"]]
    )

    transfer_files = list_transfer_files(folder, dataset)
    if not transfer_files:
        transfer_files = [(None, default_transfer_dataset(dataset), "default")]

    rows: List[Dict[str, Any]] = []
    for transfer_file, transfer_dataset, transfer_variant in transfer_files:
        row = dict(base_row)
        missing = list(base_missing)
        row["transfer_dataset"] = transfer_dataset
        row["transfer_variant"] = transfer_variant
        row["is_main_transfer_dataset"] = is_main_transfer_dataset(dataset, transfer_dataset)
        if transfer_file is None:
            missing.append("transfer_test")
        else:
            row["transfer_file"] = str(transfer_file)
            row["transfer_acc"], row["transfer_asr"] = parse_transfer_text(transfer_file)
            if pd.isna(row["transfer_asr"]):
                missing.append("transfer_asr_parse")

        chance_rate = chance_rate_for_dataset(dataset, transfer_dataset)
        row["transferability"] = compute_transferability(row["transfer_asr"])
        row["transfer_asr_chance_adjusted"] = compute_chance_adjusted_rate(row["transfer_asr"], chance_rate)
        row["legacy_transfer_rate"] = compute_transfer_rate(row["transfer_asr"], row["source_asr"])
        row["transfer_retention_rate"] = compute_transfer_retention_rate(row["transfer_asr"], row["source_asr"])
        row["transfer_gap"] = compute_transfer_gap(row["transfer_asr"], row["source_asr"])
        row["joint_transfer"] = compute_joint_transfer(row["source_asr"], row["transfer_asr"], chance_rate)
        # Compatibility with the original paper_analysis scripts: transfer_rate
        # now means the main transferability metric, i.e. target-domain ASR.
        row["transfer_rate"] = row["transferability"]

        row["complete_source"] = bool(not pd.isna(row["clean_acc"]) and not pd.isna(row["source_asr"]))
        row["complete_transfer"] = bool(not pd.isna(row["transfer_asr"]))
        row["complete_defense_results"] = bool(
            not pd.isna(row["sentinet_tpr"])
            and not pd.isna(row["scaleup_tpr"])
            and not pd.isna(row["strip_tpr"])
            and not pd.isna(row["ibd_psc_tpr"])
        )
        row["complete_analysis_row"] = bool(
            row["complete_source"]
            and row["complete_transfer"]
            and row["complete_defense_results"]
            and not pd.isna(row["transferability"])
            and not pd.isna(row["stealthiness"])
            and row["source_asr"] >= SOURCE_ASR_MAIN_THRESHOLD
        )
        row["include_main_analysis"] = bool(row["complete_analysis_row"] and row["is_main_transfer_dataset"])

        if row["complete_analysis_row"]:
            row["analysis_status"] = "complete"
        elif not row["complete_source"] or not row["complete_transfer"] or not row["complete_defense_results"]:
            row["analysis_status"] = "pending"
        else:
            row["analysis_status"] = "partial"
            if row["source_asr"] < SOURCE_ASR_MAIN_THRESHOLD:
                missing.append("source_asr_below_main_threshold")

        row["missing_items"] = ";".join(sorted(set(missing)))
        for col in MASTER_COLUMNS:
            row.setdefault(col, "")
        rows.append(row)
    return rows


def iter_result_folders(root: Path, dataset_hint: Optional[str] = None) -> Iterable[Tuple[Path, Optional[str]]]:
    if not root.exists():
        return []
    if dataset_hint:
        return [(p, dataset_hint) for p in sorted(root.iterdir()) if p.is_dir()]
    out: List[Tuple[Path, Optional[str]]] = []
    for dataset in DATASETS:
        dataset_root = root / dataset
        if dataset_root.exists():
            out.extend((p, dataset) for p in sorted(dataset_root.iterdir()) if p.is_dir())
    if not out:
        out.extend((p, None) for p in sorted(root.iterdir()) if p.is_dir())
    return out
