#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 SIG/UPGD all-to-one 专项 Markdown 报告。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _top_changes(pairwise_rows: List[Dict[str, Any]], metric_delta: str, top_k: int = 10) -> List[Dict[str, Any]]:
    vals: List[Dict[str, Any]] = []
    for row in pairwise_rows:
        v = row.get(metric_delta, "")
        if v in ("", None):
            continue
        try:
            fv = float(v)
        except ValueError:
            continue
        vals.append({**row, "_sort": abs(fv)})
    vals.sort(key=lambda x: x["_sort"], reverse=True)
    return vals[:top_k]


def build_report(
    validation: Dict[str, Any],
    group_rows: List[Dict[str, Any]],
    unmatched_rows: List[Dict[str, Any]],
    pairwise_rows: List[Dict[str, Any]],
) -> str:
    summary = validation.get("summary", {})
    lines: List[str] = []
    lines.append("# SIG/UPGD all-to-one 对比分析报告")
    lines.append("")
    lines.append("## 1. 数据覆盖与配对成功率")
    lines.append(f"- 提取总行数: {summary.get('total_rows', 0)}")
    lines.append(f"- new 行数: {summary.get('rows_new', 0)}")
    lines.append(f"- baseline 行数: {summary.get('rows_baseline', 0)}")
    lines.append(f"- 配对键总数: {summary.get('pair_keys_total', 0)}")
    lines.append(f"- 成功配对键数: {summary.get('pair_keys_matched', 0)}")
    lines.append(f"- 仅 new 存在键数: {summary.get('pair_keys_new_only', 0)}")
    lines.append(f"- 仅 baseline 存在键数: {summary.get('pair_keys_baseline_only', 0)}")
    lines.append(f"- 字段缺失行数: {summary.get('rows_with_missing_fields', 0)}")
    lines.append("")

    lines.append("## 2. 分组变化汇总（按 dataset/model/attack）")
    if not group_rows:
        lines.append("- 无可配对样本，无法统计分组变化。")
    else:
        lines.append("|dataset|model|attack|pair_count|transfer_delta_mean|asr_delta_mean|S_stealth_delta_mean|")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in group_rows:
            lines.append(
                f"|{row.get('dataset','')}|{row.get('model_norm','')}|{row.get('attack_family','')}|"
                f"{row.get('pair_count','')}|{row.get('transfer_rate_delta_mean','')}|"
                f"{row.get('asr_delta_mean','')}|{row.get('S_stealth_delta_mean','')}|"
            )
    lines.append("")

    lines.append("## 3. 关键变化 Top10（按 |transfer_rate_delta|）")
    top_transfer = _top_changes(pairwise_rows, "transfer_rate_delta", top_k=10)
    if not top_transfer:
        lines.append("- 无可用 transfer_rate 对比数据。")
    else:
        lines.append("|pair_id|dataset|model|attack|poison_rate|target_label|transfer_delta|asr_delta|")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for row in top_transfer:
            lines.append(
                f"|{row.get('pair_id','')}|{row.get('dataset','')}|{row.get('model_norm','')}|"
                f"{row.get('attack_family','')}|{row.get('poison_rate','')}|{row.get('target_label','')}|"
                f"{row.get('transfer_rate_delta','')}|{row.get('asr_delta','')}|"
            )
    lines.append("")

    lines.append("## 4. 不可比与异常样本")
    lines.append(f"- unmatched 行数: {len(unmatched_rows)}")
    sample_unmatched = unmatched_rows[:20]
    if sample_unmatched:
        lines.append("")
        lines.append("|side|dataset|model|attack|poison_rate|target_label|folder_name|")
        lines.append("|---|---|---|---|---:|---:|---|")
        for row in sample_unmatched:
            lines.append(
                f"|{row.get('unmatched_side','')}|{row.get('dataset','')}|{row.get('model_norm','')}|"
                f"{row.get('attack_family','')}|{row.get('poison_rate','')}|{row.get('target_label','')}|"
                f"{row.get('folder_name','')}|"
            )
    missing_counts = summary.get("missing_field_counts", {})
    if missing_counts:
        lines.append("")
        lines.append("- 字段缺失统计：")
        for key, value in sorted(missing_counts.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"  - {key}: {value}")

    lines.append("")
    lines.append("## 5. 输出文件索引")
    lines.append("- `analysis/data_sig_upgd_alltoone_raw.csv`")
    lines.append("- `analysis/validation_sig_upgd_alltoone.json`")
    lines.append("- `analysis/report_tables/sig_upgd_alltoone_pairwise_comparison.csv`")
    lines.append("- `analysis/report_tables/sig_upgd_alltoone_group_summary.csv`")
    lines.append("- `analysis/report_tables/sig_upgd_alltoone_unmatched_cases.csv`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 SIG/UPGD all-to-one 对比报告")
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=Path("analysis/validation_sig_upgd_alltoone.json"),
    )
    parser.add_argument(
        "--pairwise-csv",
        type=Path,
        default=Path("analysis/report_tables/sig_upgd_alltoone_pairwise_comparison.csv"),
    )
    parser.add_argument(
        "--group-csv",
        type=Path,
        default=Path("analysis/report_tables/sig_upgd_alltoone_group_summary.csv"),
    )
    parser.add_argument(
        "--unmatched-csv",
        type=Path,
        default=Path("analysis/report_tables/sig_upgd_alltoone_unmatched_cases.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("analysis/sig_upgd_alltoone_comparison_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = _read_json(args.validation_json)
    pairwise_rows = _read_csv(args.pairwise_csv)
    group_rows = _read_csv(args.group_csv)
    unmatched_rows = _read_csv(args.unmatched_csv)

    report = build_report(validation, group_rows, unmatched_rows, pairwise_rows)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"✓ 报告完成: {args.output_md}")


if __name__ == "__main__":
    main()
