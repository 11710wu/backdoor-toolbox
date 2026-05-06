#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 SIG/UPGD all-to-one 的 pairwise/group/unmatched 对比结果。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


NUMERIC_METRICS = [
    "stealth_tpr_avg",
    "stealth_auc_avg",
    "transfer_rate",
    "asr",
    "nc_max_anomaly_index",
    "S_stealth",
    "S_stealth_tpr",
]


def _to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_model(value: Any) -> str:
    if value is None:
        return ""
    v = str(value).strip().lower()
    if "resnet" in v:
        return "resnet18"
    if "mobile" in v:
        return "mobilenet"
    if "vgg" in v:
        return "vgg"
    return v


def _pair_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    attack = str(row.get("attack_family", "")).lower()
    return (
        str(row.get("dataset", "")).lower(),
        _norm_model(row.get("model_norm")),
        attack,
        str(row.get("poison_rate", "")),
        str(row.get("target_label", "")),
        str(row.get("mode", "")).lower(),
        str(row.get("f", "")) if attack == "sig" else "",
        str(row.get("delta", "")) if attack == "sig" else "",
        str(row.get("eps", "")) if attack == "upgd" else "",
        str(row.get("upgd_steps", "")) if attack == "upgd" else "",
        str(row.get("upgd_steps_multiplier", "")) if attack == "upgd" else "",
    )


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pairwise(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, ...], Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: {"new": [], "baseline": []})
    for row in rows:
        source = row.get("source_set", "")
        if source not in {"new", "baseline"}:
            continue
        grouped[_pair_key(row)][source].append(row)

    pairwise_rows: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []

    for key, bucket in grouped.items():
        new_rows = bucket["new"]
        baseline_rows = bucket["baseline"]
        if not new_rows or not baseline_rows:
            side = "new_only" if new_rows else "baseline_only"
            for row in (new_rows or baseline_rows):
                unmatched_rows.append(
                    {
                        "unmatched_side": side,
                        "dataset": row.get("dataset"),
                        "model_norm": _norm_model(row.get("model_norm")),
                        "attack_family": row.get("attack_family"),
                        "poison_rate": row.get("poison_rate"),
                        "target_label": row.get("target_label"),
                        "mode": row.get("mode"),
                        "folder_name": row.get("folder_name"),
                    }
                )
            continue

        pair_count = min(len(new_rows), len(baseline_rows))
        for idx in range(pair_count):
            new_row = new_rows[idx]
            baseline_row = baseline_rows[idx]
            row: Dict[str, Any] = {
                "pair_id": f"{'|'.join(key)}#{idx + 1}",
                "dataset": new_row.get("dataset"),
                "model_norm": _norm_model(new_row.get("model_norm")),
                "attack_family": new_row.get("attack_family"),
                "poison_rate": new_row.get("poison_rate"),
                "target_label": new_row.get("target_label"),
                "mode": new_row.get("mode"),
                "f": new_row.get("f"),
                "delta": new_row.get("delta"),
                "eps": new_row.get("eps"),
                "upgd_steps": new_row.get("upgd_steps"),
                "upgd_steps_multiplier": new_row.get("upgd_steps_multiplier"),
                "folder_new": new_row.get("folder_name"),
                "folder_baseline": baseline_row.get("folder_name"),
            }
            for metric in NUMERIC_METRICS:
                n = _to_float(new_row.get(metric))
                b = _to_float(baseline_row.get(metric))
                row[f"{metric}_new"] = n
                row[f"{metric}_baseline"] = b
                row[f"{metric}_delta"] = (n - b) if (n is not None and b is not None) else ""
            pairwise_rows.append(row)

        if len(new_rows) > pair_count:
            for row in new_rows[pair_count:]:
                unmatched_rows.append(
                    {
                        "unmatched_side": "new_only",
                        "dataset": row.get("dataset"),
                        "model_norm": _norm_model(row.get("model_norm")),
                        "attack_family": row.get("attack_family"),
                        "poison_rate": row.get("poison_rate"),
                        "target_label": row.get("target_label"),
                        "mode": row.get("mode"),
                        "folder_name": row.get("folder_name"),
                    }
                )
        if len(baseline_rows) > pair_count:
            for row in baseline_rows[pair_count:]:
                unmatched_rows.append(
                    {
                        "unmatched_side": "baseline_only",
                        "dataset": row.get("dataset"),
                        "model_norm": _norm_model(row.get("model_norm")),
                        "attack_family": row.get("attack_family"),
                        "poison_rate": row.get("poison_rate"),
                        "target_label": row.get("target_label"),
                        "mode": row.get("mode"),
                        "folder_name": row.get("folder_name"),
                    }
                )

    return pairwise_rows, unmatched_rows


def build_group_summary(pairwise_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in pairwise_rows:
        grouped[(row["dataset"], row["model_norm"], row["attack_family"])].append(row)

    result: List[Dict[str, Any]] = []
    for (dataset, model_norm, attack_family), rows in sorted(grouped.items()):
        out: Dict[str, Any] = {
            "dataset": dataset,
            "model_norm": model_norm,
            "attack_family": attack_family,
            "pair_count": len(rows),
        }
        for metric in NUMERIC_METRICS:
            deltas = [_to_float(r.get(f"{metric}_delta")) for r in rows]
            deltas = [d for d in deltas if d is not None]
            out[f"{metric}_delta_mean"] = mean(deltas) if deltas else ""
        result.append(out)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIG/UPGD all-to-one 对比统计")
    parser.add_argument("--input-csv", type=Path, default=Path("analysis/data_sig_upgd_alltoone_raw.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/report_tables"),
        help="输出目录（pairwise/group/unmatched）",
    )
    parser.add_argument("--prefix", default="sig_upgd_alltoone")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_csv(args.input_csv)
    pairwise_rows, unmatched_rows = build_pairwise(rows)
    group_rows = build_group_summary(pairwise_rows)

    pairwise_path = args.output_dir / f"{args.prefix}_pairwise_comparison.csv"
    group_path = args.output_dir / f"{args.prefix}_group_summary.csv"
    unmatched_path = args.output_dir / f"{args.prefix}_unmatched_cases.csv"

    pairwise_fields = [
        "pair_id",
        "dataset",
        "model_norm",
        "attack_family",
        "poison_rate",
        "target_label",
        "mode",
        "f",
        "delta",
        "eps",
        "upgd_steps",
        "upgd_steps_multiplier",
        "folder_new",
        "folder_baseline",
    ]
    for metric in NUMERIC_METRICS:
        pairwise_fields.extend([f"{metric}_new", f"{metric}_baseline", f"{metric}_delta"])

    group_fields = ["dataset", "model_norm", "attack_family", "pair_count"] + [
        f"{metric}_delta_mean" for metric in NUMERIC_METRICS
    ]
    unmatched_fields = [
        "unmatched_side",
        "dataset",
        "model_norm",
        "attack_family",
        "poison_rate",
        "target_label",
        "mode",
        "folder_name",
    ]

    _write_csv(pairwise_path, pairwise_rows, pairwise_fields)
    _write_csv(group_path, group_rows, group_fields)
    _write_csv(unmatched_path, unmatched_rows, unmatched_fields)

    print(f"✓ pairwise: {pairwise_path}")
    print(f"✓ group:    {group_path}")
    print(f"✓ unmatched:{unmatched_path}")


if __name__ == "__main__":
    main()
