#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 all-to-one 与 clean-label 两种模式下 SIG/UPGD 的迁移性、隐蔽性和防御效果。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFENSE_PREFIX = {
    "STRIP": "strip",
    "SCaLe-Up": "scaleup",
    "SentiNet": "sentinet",
    "IBD_PSC": "ibd_psc",
}
DATASET_ALIASES = {
    "cifar10": "cifar10",
    "mnistm": "mnistm",
    "tiny-imagenet": "tiny_imagenet",
    "tiny_imagenet": "tiny_imagenet",
}
MODE_ROOTS = {
    "all-to-one": "poisoned_train_set2",
    "clean-label": "poisoned_train_set1",
}


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _parse_attack(folder_name: str) -> Optional[str]:
    name = folder_name.lower()
    if name.startswith("sig_"):
        return "sig"
    if name.startswith("upgd_"):
        return "upgd"
    return None


def _parse_transfer_rate(text: str) -> Optional[float]:
    for line in text.splitlines():
        line = line.strip()
        m = re.search(r"攻击成功率(?:\s*\([^)]*\))?[:：]\s*([\d.]+)", line)
        if m:
            return float(m.group(1))
        if "ASR" in line and ":" in line:
            m = re.search(r"([\d.]+)", line)
            if m:
                return float(m.group(1))
    m = re.search(r"攻击成功率[:：]\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


def _transfer_patterns(dataset: str) -> List[str]:
    if dataset == "cifar10":
        return ["test_stl10_results*.txt"]
    if dataset == "mnistm":
        return ["test_mnistm_results*.txt", "test_mnist_cross_results*.txt"]
    if dataset == "tiny_imagenet":
        return ["test_tiny_target_domain_results*.txt"]
    return []


def _parse_defense_json(path: Path) -> Optional[Tuple[float, float]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        tpr = float(data.get("tpr", 0) or 0)
        if tpr > 1:
            tpr /= 100.0
        auc = float(data.get("auc", 0) or 0)
        return tpr, auc
    except Exception:
        return None


def collect_rows(project_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    mode_rows: List[Dict[str, Any]] = []
    defense_rows: List[Dict[str, Any]] = []

    for mode, root_name in MODE_ROOTS.items():
        base = project_root / root_name
        if not base.exists():
            continue
        for ds_dir in base.iterdir():
            if not ds_dir.is_dir():
                continue
            dataset = DATASET_ALIASES.get(ds_dir.name)
            if not dataset:
                continue
            for folder in ds_dir.iterdir():
                if not folder.is_dir():
                    continue
                attack = _parse_attack(folder.name)
                if attack not in {"sig", "upgd"}:
                    continue

                all_tprs: List[float] = []
                all_aucs: List[float] = []
                per_def_tpr: Dict[str, List[float]] = defaultdict(list)
                per_def_auc: Dict[str, List[float]] = defaultdict(list)
                for defense, prefix in DEFENSE_PREFIX.items():
                    for result_file in folder.glob(f"{prefix}_defense_results*.json"):
                        parsed = _parse_defense_json(result_file)
                        if parsed is None:
                            continue
                        tpr, auc = parsed
                        all_tprs.append(tpr)
                        all_aucs.append(auc)
                        per_def_tpr[defense].append(tpr)
                        per_def_auc[defense].append(auc)

                transfer_rates: List[float] = []
                seen_files = set()
                for pattern in _transfer_patterns(dataset):
                    for result_file in folder.glob(pattern):
                        if result_file in seen_files:
                            continue
                        seen_files.add(result_file)
                        try:
                            rate = _parse_transfer_rate(result_file.read_text(encoding="utf-8"))
                        except Exception:
                            rate = None
                        if rate is not None:
                            transfer_rates.append(rate)

                if all_tprs and all_aucs and transfer_rates:
                    mode_rows.append({
                        "mode": mode,
                        "dataset": dataset,
                        "attack_family": attack,
                        "folder_name": folder.name,
                        "transfer_rate": _mean(transfer_rates),
                        "stealth_tpr": 1.0 - (_mean(all_tprs) or 0.0),
                        "stealth_auc": 1.0 - (_mean(all_aucs) or 0.0),
                    })

                for defense in DEFENSE_PREFIX:
                    if per_def_tpr[defense] and per_def_auc[defense]:
                        defense_rows.append({
                            "mode": mode,
                            "dataset": dataset,
                            "attack_family": attack,
                            "defense": defense,
                            "folder_name": folder.name,
                            "tpr": _mean(per_def_tpr[defense]),
                            "auc": _mean(per_def_auc[defense]),
                        })
    return mode_rows, defense_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_mode_summary(mode_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in mode_rows:
        grouped[(row["mode"], row["dataset"], row["attack_family"])].append(row)

    out: List[Dict[str, Any]] = []
    for (mode, dataset, attack), rows in sorted(grouped.items()):
        out.append({
            "mode": mode,
            "dataset": dataset,
            "attack_family": attack,
            "n": len(rows),
            "transfer_rate_mean": _mean(float(r["transfer_rate"]) for r in rows),
            "stealth_auc_mean": _mean(float(r["stealth_auc"]) for r in rows),
            "stealth_tpr_mean": _mean(float(r["stealth_tpr"]) for r in rows),
        })
    return out


def build_mode_delta(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index = {(r["mode"], r["dataset"], r["attack_family"]): r for r in summary_rows}
    out: List[Dict[str, Any]] = []
    datasets = sorted({r["dataset"] for r in summary_rows})
    for dataset in datasets:
        for attack in ["sig", "upgd"]:
            all2one = index.get(("all-to-one", dataset, attack))
            clean = index.get(("clean-label", dataset, attack))
            if not all2one or not clean:
                continue
            out.append({
                "dataset": dataset,
                "attack_family": attack,
                "transfer_delta": all2one["transfer_rate_mean"] - clean["transfer_rate_mean"],
                "stealth_auc_delta": all2one["stealth_auc_mean"] - clean["stealth_auc_mean"],
                "stealth_tpr_delta": all2one["stealth_tpr_mean"] - clean["stealth_tpr_mean"],
            })
    return out


def build_defense_summary(defense_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in defense_rows:
        grouped[(row["mode"], row["dataset"], row["attack_family"], row["defense"])].append(row)

    out: List[Dict[str, Any]] = []
    for (mode, dataset, attack, defense), rows in sorted(grouped.items()):
        out.append({
            "mode": mode,
            "dataset": dataset,
            "attack_family": attack,
            "defense": defense,
            "n": len(rows),
            "tpr_mean": _mean(float(r["tpr"]) for r in rows),
            "auc_mean": _mean(float(r["auc"]) for r in rows),
        })
    return out


def build_defense_delta(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index = {(r["mode"], r["dataset"], r["attack_family"], r["defense"]): r for r in summary_rows}
    out: List[Dict[str, Any]] = []
    keys = sorted({(r["dataset"], r["attack_family"], r["defense"]) for r in summary_rows})
    for dataset, attack, defense in keys:
        all2one = index.get(("all-to-one", dataset, attack, defense))
        clean = index.get(("clean-label", dataset, attack, defense))
        if not all2one or not clean:
            continue
        out.append({
            "dataset": dataset,
            "attack_family": attack,
            "defense": defense,
            "tpr_delta": all2one["tpr_mean"] - clean["tpr_mean"],
            "auc_delta": all2one["auc_mean"] - clean["auc_mean"],
        })
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 all-to-one vs clean-label 的迁移/隐蔽/防御差异")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/alltoone"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    mode_rows, defense_rows = collect_rows(project_root)
    mode_summary = build_mode_summary(mode_rows)
    mode_delta = build_mode_delta(mode_summary)
    defense_summary = build_defense_summary(defense_rows)
    defense_delta = build_defense_delta(defense_summary)

    _write_csv(output_dir / "mode_transfer_stealth_summary.csv", mode_summary)
    _write_csv(output_dir / "mode_transfer_stealth_delta.csv", mode_delta)
    _write_csv(output_dir / "mode_defense_dataset_summary.csv", defense_summary)
    _write_csv(output_dir / "mode_defense_dataset_delta.csv", defense_delta)
    print(f"✓ mode summary: {output_dir / 'mode_transfer_stealth_summary.csv'}")
    print(f"✓ mode delta: {output_dir / 'mode_transfer_stealth_delta.csv'}")
    print(f"✓ defense summary: {output_dir / 'mode_defense_dataset_summary.csv'}")
    print(f"✓ defense delta: {output_dir / 'mode_defense_dataset_delta.csv'}")


if __name__ == "__main__":
    main()
