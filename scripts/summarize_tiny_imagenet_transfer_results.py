#!/usr/bin/env python3
"""Summarize Tiny-ImageNet poison experiment results into Markdown.

This script scans experiment folders under poisoned_train_set/tiny_imagenet,
matches the original test JSON and target-domain transfer TXT for each run,
and writes a Markdown report with:

- overall summary by attack type
- per-attack configuration means across architectures
- per-attack per-model detailed rows
- directories missing result files
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


KNOWN_ATTACKS = [
    "adaptive_blend",
    "adaptive_patch",
    "WaNet",
    "SIG",
    "basic",
    "blend",
    "belt",
    "upgd",
    "none",
]

ATTACK_PARAM_ORDER: Dict[str, List[str]] = {
    "adaptive_blend": ["poison_rate", "alpha", "cover", "trigger"],
    "adaptive_patch": ["poison_rate", "alpha", "cover"],
    "basic": ["poison_rate", "alpha", "trigger"],
    "belt": ["poison_rate", "alpha", "cover", "mask"],
    "blend": ["poison_rate", "alpha", "trigger"],
    "SIG": ["poison_rate", "delta", "f"],
    "WaNet": ["poison_rate", "cover", "s", "k"],
    "upgd": ["poison_rate", "eps", "constraint", "steps", "mult"],
}

ATTACK_CONFIG_COLUMNS: Dict[str, List[Tuple[str, str]]] = {
    "adaptive_blend": [("poison_rate", "中毒率"), ("alpha", "Alpha"), ("cover", "Cover"), ("trigger", "Trigger")],
    "adaptive_patch": [("poison_rate", "中毒率"), ("alpha", "Alpha"), ("cover", "Cover")],
    "basic": [("poison_rate", "中毒率"), ("alpha", "Alpha"), ("trigger", "Trigger")],
    "belt": [("poison_rate", "中毒率"), ("alpha", "Alpha"), ("cover", "Cover"), ("mask", "Mask")],
    "blend": [("poison_rate", "中毒率"), ("alpha", "Alpha"), ("trigger", "Trigger")],
    "SIG": [("poison_rate", "中毒率"), ("delta", "Delta"), ("f", "F")],
    "WaNet": [("poison_rate", "中毒率"), ("cover", "Cover"), ("s", "S"), ("k", "K")],
    "upgd": [("poison_rate", "中毒率"), ("eps", "Eps"), ("constraint", "Constraint"), ("steps", "Steps"), ("mult", "Mult")],
}


@dataclass
class Record:
    attack_type: str
    experiment_dir: str
    model: str
    poison_seed: Optional[str]
    params: Dict[str, str]
    clean_acc: float
    clean_asr: float
    transfer_acc: float
    transfer_asr: float
    json_path: str
    txt_path: str
    target_domain_path: Optional[str]


@dataclass
class MissingRecord:
    experiment_dir: str
    attack_type: str
    json_count: int
    txt_count: int


def _detect_attack_type(name: str) -> str:
    for attack in sorted(KNOWN_ATTACKS, key=len, reverse=True):
        if name == attack or name.startswith(attack + "_"):
            return attack
    raise ValueError(f"Unknown attack type for directory: {name}")


def _extract_string(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _parse_experiment_dir(name: str) -> Tuple[str, Dict[str, str]]:
    attack_type = _detect_attack_type(name)
    after_prefix = name[len(attack_type) + 1 :] if name != attack_type else ""
    poison_rate, _, _ = after_prefix.partition("_")
    params: Dict[str, str] = {}
    if poison_rate:
        params["poison_rate"] = poison_rate

    for key in ("alpha", "cover", "mask", "delta", "f", "s", "k", "eps", "constraint", "steps", "mult"):
        value = _extract_string(rf"_{key}=([^_]+)", "_" + name)
        if value is not None:
            params[key] = value

    trigger = _extract_string(r"_trigger=(.+?)_poison_seed=", name)
    if trigger is not None:
        params["trigger"] = trigger

    poison_seed = _extract_string(r"_poison_seed=([^_]+)", name)
    if poison_seed is not None:
        params["poison_seed"] = poison_seed

    arch = _extract_string(r"_arch=(.+)$", name)
    if arch is not None:
        params["arch"] = arch

    return attack_type, params


def _normalize_model_name(arch: Optional[str]) -> str:
    if not arch:
        return "unknown"
    if arch.endswith("_tiny_imagenet"):
        arch = arch[: -len("_tiny_imagenet")]
    return arch


def _parse_original_json(path: Path) -> Tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clean_acc = float(data["clean_acc"])
    clean_asr = float(data["asr"])
    return clean_acc, clean_asr


def _parse_transfer_txt(path: Path) -> Tuple[float, float, Optional[str]]:
    text = path.read_text(encoding="utf-8")
    acc_match = re.search(r"准确率:\s*([0-9.]+)", text)
    asr_match = re.search(r"攻击成功率 \(ASR\):\s*([0-9.]+)", text)
    target_path_match = re.search(r"目标域路径:\s*(.+)", text)
    if not acc_match or not asr_match:
        raise ValueError(f"Cannot parse ACC/ASR from {path}")
    target_domain_path = target_path_match.group(1).strip() if target_path_match else None
    return float(acc_match.group(1)), float(asr_match.group(1)), target_domain_path


def load_records(root: Path) -> Tuple[List[Record], List[MissingRecord]]:
    records: List[Record] = []
    missing: List[MissingRecord] = []

    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        attack_type, params = _parse_experiment_dir(exp_dir.name)
        json_files = sorted(exp_dir.glob("test_results_seed=*.json"))
        txt_files = sorted(exp_dir.glob("test_tiny_target_domain_results*.txt"))

        if len(json_files) != 1 or len(txt_files) != 1:
            missing.append(
                MissingRecord(
                    experiment_dir=exp_dir.name,
                    attack_type=attack_type,
                    json_count=len(json_files),
                    txt_count=len(txt_files),
                )
            )
            continue

        clean_acc, clean_asr = _parse_original_json(json_files[0])
        transfer_acc, transfer_asr, target_domain_path = _parse_transfer_txt(txt_files[0])

        poison_seed = params.get("poison_seed")
        model = _normalize_model_name(params.get("arch"))
        records.append(
            Record(
                attack_type=attack_type,
                experiment_dir=exp_dir.name,
                model=model,
                poison_seed=poison_seed,
                params=params,
                clean_acc=clean_acc,
                clean_asr=clean_asr,
                transfer_acc=transfer_acc,
                transfer_asr=transfer_asr,
                json_path=str(json_files[0]),
                txt_path=str(txt_files[0]),
                target_domain_path=target_domain_path,
            )
        )

    return records, missing


def _to_float(value: Optional[str]) -> float:
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except ValueError:
        return float("-inf")


def _sort_key(record: Record) -> Tuple:
    order = ATTACK_PARAM_ORDER.get(record.attack_type, ["poison_rate"])
    key: List[object] = [record.attack_type]
    for field in order:
        value = record.params.get(field)
        try:
            key.append(float(value) if value is not None else float("-inf"))
        except ValueError:
            key.append(value or "")
    key.append(record.model.lower())
    return tuple(key)


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_value(value: Optional[str]) -> str:
    if value is None:
        return "-"
    if value == "":
        return "-"
    return value


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _config_key(record: Record) -> Tuple:
    order = ATTACK_PARAM_ORDER.get(record.attack_type, ["poison_rate"])
    return tuple(record.params.get(field, "") for field in order)


def _collect_attack_summary(records: Sequence[Record]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Record]] = defaultdict(list)
    for rec in records:
        grouped[rec.attack_type].append(rec)

    out: Dict[str, Dict[str, float]] = {}
    for attack_type, rows in grouped.items():
        out[attack_type] = {
            "count": float(len(rows)),
            "clean_acc_mean": mean(rec.clean_acc for rec in rows),
            "clean_asr_mean": mean(rec.clean_asr for rec in rows),
            "transfer_acc_mean": mean(rec.transfer_acc for rec in rows),
            "transfer_asr_mean": mean(rec.transfer_asr for rec in rows),
        }
    return out


def build_markdown(records: Sequence[Record], missing: Sequence[MissingRecord], root: Path) -> str:
    lines: List[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    target_paths = sorted({rec.target_domain_path for rec in records if rec.target_domain_path})

    lines.append("# Tiny-ImageNet 原始测试与迁移测试汇总")
    lines.append("")
    lines.append(f"- 扫描目录: `{root}`")
    lines.append(f"- 生成时间: `{now}`")
    lines.append(f"- 完整实验数: `{len(records)}`")
    lines.append(f"- 缺失结果文件的目录数: `{len(missing)}`")
    if target_paths:
        lines.append(f"- 检测到的目标域路径: `{target_paths[0]}`" if len(target_paths) == 1 else "- 检测到多个目标域路径:")
        if len(target_paths) > 1:
            for path in target_paths:
                lines.append(f"  - `{path}`")
    lines.append("")

    attack_summary = _collect_attack_summary(records)
    summary_headers = ["攻击类型", "实验数", "原始ACC均值", "原始ASR均值", "迁移ACC均值", "迁移ASR均值"]
    summary_rows = []
    for attack_type in sorted(attack_summary.keys()):
        item = attack_summary[attack_type]
        summary_rows.append(
            [
                attack_type,
                str(int(item["count"])),
                _format_pct(item["clean_acc_mean"]),
                _format_pct(item["clean_asr_mean"]),
                _format_pct(item["transfer_acc_mean"]),
                _format_pct(item["transfer_asr_mean"]),
            ]
        )
    lines.append("## 总览")
    lines.append("")
    lines.extend(_md_table(summary_headers, summary_rows))
    lines.append("")

    for attack_type in sorted({rec.attack_type for rec in records}):
        attack_records = sorted([rec for rec in records if rec.attack_type == attack_type], key=_sort_key)
        config_columns = ATTACK_CONFIG_COLUMNS[attack_type]

        lines.append(f"## {attack_type}")
        lines.append("")
        lines.append(f"- 完整实验数: `{len(attack_records)}`")
        lines.append("")

        config_grouped: Dict[Tuple, List[Record]] = defaultdict(list)
        for rec in attack_records:
            config_grouped[_config_key(rec)].append(rec)

        config_headers = [label for _, label in config_columns] + [
            "模型数",
            "原始ACC均值",
            "原始ASR均值",
            "迁移ACC均值",
            "迁移ASR均值",
        ]
        config_rows: List[List[str]] = []
        for key in sorted(config_grouped.keys(), key=lambda items: tuple(_to_float(x) if re.fullmatch(r"-?[0-9.]+", x or "") else (x or "") for x in items)):
            grouped_rows = config_grouped[key]
            head = grouped_rows[0]
            config_rows.append(
                [
                    _format_value(head.params.get(field))
                    for field, _ in config_columns
                ]
                + [
                    str(len(grouped_rows)),
                    _format_pct(mean(row.clean_acc for row in grouped_rows)),
                    _format_pct(mean(row.clean_asr for row in grouped_rows)),
                    _format_pct(mean(row.transfer_acc for row in grouped_rows)),
                    _format_pct(mean(row.transfer_asr for row in grouped_rows)),
                ]
            )

        lines.append("### 按配置统计")
        lines.append("")
        lines.extend(_md_table(config_headers, config_rows))
        lines.append("")

        detail_headers = ["模型"] + [label for _, label in config_columns] + ["原始ACC", "原始ASR", "迁移ACC", "迁移ASR"]
        detail_rows: List[List[str]] = []
        for rec in attack_records:
            detail_rows.append(
                [rec.model]
                + [_format_value(rec.params.get(field)) for field, _ in config_columns]
                + [
                    _format_pct(rec.clean_acc),
                    _format_pct(rec.clean_asr),
                    _format_pct(rec.transfer_acc),
                    _format_pct(rec.transfer_asr),
                ]
            )

        lines.append("### 按模型明细")
        lines.append("")
        lines.extend(_md_table(detail_headers, detail_rows))
        lines.append("")

    lines.append("## 缺失结果文件")
    lines.append("")
    if not missing:
        lines.append("无。")
    else:
        missing_headers = ["实验目录", "攻击类型", "原始JSON数", "迁移TXT数"]
        missing_rows = [
            [row.experiment_dir, row.attack_type, str(row.json_count), str(row.txt_count)]
            for row in sorted(missing, key=lambda x: (x.attack_type, x.experiment_dir))
        ]
        lines.extend(_md_table(missing_headers, missing_rows))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Tiny-ImageNet original and transfer test results.")
    parser.add_argument(
        "--root",
        type=str,
        default="/workspace/backdoor-toolbox-new1/poisoned_train_set/tiny_imagenet",
        help="Root directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/backdoor-toolbox-new1/poisoned_train_set/tiny_imagenet/tiny_imagenet_original_vs_transfer_summary.md",
        help="Output Markdown path.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment root does not exist: {root}")

    records, missing = load_records(root)
    markdown = build_markdown(records, missing, root)

    output_path = Path(args.output).resolve()
    output_path.write_text(markdown, encoding="utf-8")
    print(f"[OK] records={len(records)} missing={len(missing)}")
    print(f"[OK] wrote markdown to {output_path}")


if __name__ == "__main__":
    main()
