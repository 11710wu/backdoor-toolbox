#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 SIG/UPGD all-to-one 提取结果完整性与可配对覆盖率。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _to_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.8f}"
    except (TypeError, ValueError):
        return str(value)


def _normalize_model(value: Any) -> str:
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
    attack_specific = (
        _to_float(row.get("f")) if attack == "sig" else "",
        _to_float(row.get("delta")) if attack == "sig" else "",
        _to_float(row.get("eps")) if attack == "upgd" else "",
        _to_float(row.get("upgd_steps")) if attack == "upgd" else "",
        _to_float(row.get("upgd_steps_multiplier")) if attack == "upgd" else "",
    )
    return (
        str(row.get("dataset", "")).lower(),
        _normalize_model(row.get("model_norm")),
        attack,
        _to_float(row.get("poison_rate")),
        *attack_specific,
    )


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _missing_fields(row: Dict[str, Any]) -> List[str]:
    attack = str(row.get("attack_family", "")).lower()
    source_set = str(row.get("source_set", "")).strip().lower()
    missing: List[str] = []
    base_fields = ["dataset", "model_norm", "poison_rate", "attack_family"]
    for field in base_fields:
        if row.get(field, "") in ("", None):
            missing.append(field)
    # 仅对 new(all-to-one) 强制要求 mode/target_label，baseline(clean-label)不强制
    if source_set == "new":
        if str(row.get("mode", "")).strip().lower() != "all-to-one":
            missing.append("mode_not_all_to_one")
        if row.get("target_label", "") in ("", None):
            missing.append("target_label")
    if attack == "sig":
        for field in ["delta", "f"]:
            if row.get(field, "") in ("", None):
                missing.append(field)
    elif attack == "upgd":
        for field in ["eps", "upgd_steps", "upgd_steps_multiplier"]:
            if row.get(field, "") in ("", None):
                missing.append(field)
    else:
        missing.append("attack_family_invalid")
    return missing


def run_validation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "total_rows": len(rows),
        "by_source_set": Counter(row.get("source_set", "") for row in rows),
        "by_attack_family": Counter(str(row.get("attack_family", "")).lower() for row in rows),
        "by_dataset": Counter(str(row.get("dataset", "")).lower() for row in rows),
    }

    missing_cases: List[Dict[str, Any]] = []
    missing_counter: Counter[str] = Counter()
    for row in rows:
        missing = _missing_fields(row)
        if missing:
            missing_counter.update(missing)
            missing_cases.append(
                {
                    "source_set": row.get("source_set"),
                    "folder_name": row.get("folder_name"),
                    "attack_family": row.get("attack_family"),
                    "missing_fields": missing,
                }
            )

    by_key = defaultdict(lambda: {"new": 0, "baseline": 0})
    for row in rows:
        key = _pair_key(row)
        source = row.get("source_set", "")
        if source in {"new", "baseline"}:
            by_key[key][source] += 1

    matched_keys = [key for key, counts in by_key.items() if counts["new"] > 0 and counts["baseline"] > 0]
    new_only = [key for key, counts in by_key.items() if counts["new"] > 0 and counts["baseline"] == 0]
    baseline_only = [key for key, counts in by_key.items() if counts["baseline"] > 0 and counts["new"] == 0]

    return {
        "summary": {
            "total_rows": summary["total_rows"],
            "rows_new": summary["by_source_set"].get("new", 0),
            "rows_baseline": summary["by_source_set"].get("baseline", 0),
            "rows_by_attack_family": dict(summary["by_attack_family"]),
            "rows_by_dataset": dict(summary["by_dataset"]),
            "rows_with_missing_fields": len(missing_cases),
            "missing_field_counts": dict(missing_counter),
            "pair_keys_total": len(by_key),
            "pair_keys_matched": len(matched_keys),
            "pair_keys_new_only": len(new_only),
            "pair_keys_baseline_only": len(baseline_only),
        },
        "unmatched_cases": {
            "new_only_keys": [list(key) for key in new_only],
            "baseline_only_keys": [list(key) for key in baseline_only],
        },
        "missing_cases": missing_cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验 SIG/UPGD all-to-one 提取结果")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("analysis/alltoone/data_sig_upgd_alltoone_raw.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("analysis/alltoone/validation_sig_upgd_alltoone.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.input_csv)
    result = run_validation(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 校验完成: {args.output_json}")
    print(
        "  matched/new_only/baseline_only = "
        f"{result['summary']['pair_keys_matched']}/"
        f"{result['summary']['pair_keys_new_only']}/"
        f"{result['summary']['pair_keys_baseline_only']}"
    )


if __name__ == "__main__":
    main()
