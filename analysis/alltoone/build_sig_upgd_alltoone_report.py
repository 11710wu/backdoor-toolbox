#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 SIG/UPGD 两种模式（all-to-one vs clean-label）的分析报告。

报告重点：
- 迁移性（transfer_rate）
- 隐蔽性（stealth_auc / stealth_tpr，越高越隐蔽）
- 各数据集、各攻击、各防御方法差异
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ANALYSIS_ALLTOONE_DIR = PROJECT_ROOT / "analysis" / "alltoone"

MODE_ORDER = ["clean-label", "all-to-one"]
ATTACK_ORDER = ["sig", "upgd"]
DEFENSE_ORDER = ["IBD_PSC", "SCaLe-Up", "STRIP", "SentiNet"]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _f(row: Dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.1f}%"


def _idx(rows: Iterable[Dict[str, str]], keys: Tuple[str, ...]) -> Dict[Tuple[str, ...], Dict[str, str]]:
    out: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        out[tuple(row.get(k, "") for k in keys)] = row
    return out


def _sorted_values(rows: Iterable[Dict[str, str]], key: str) -> List[str]:
    return sorted({row.get(key, "") for row in rows if row.get(key, "")})


def _defense_delta_rows(defense_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    index = _idx(defense_rows, ("mode", "dataset", "attack_family", "defense"))
    datasets = _sorted_values(defense_rows, "dataset")
    result: List[Dict[str, Any]] = []
    for dataset in datasets:
        for attack in ATTACK_ORDER:
            for defense in DEFENSE_ORDER:
                clean = index.get(("clean-label", dataset, attack, defense))
                alltoone = index.get(("all-to-one", dataset, attack, defense))
                if not clean or not alltoone:
                    continue
                result.append(
                    {
                        "dataset": dataset,
                        "attack_family": attack,
                        "defense": defense,
                        "clean_tpr": _f(clean, "tpr_mean"),
                        "alltoone_tpr": _f(alltoone, "tpr_mean"),
                        "tpr_delta": (_f(alltoone, "tpr_mean") or 0.0) - (_f(clean, "tpr_mean") or 0.0),
                        "clean_auc": _f(clean, "auc_mean"),
                        "alltoone_auc": _f(alltoone, "auc_mean"),
                        "auc_delta": (_f(alltoone, "auc_mean") or 0.0) - (_f(clean, "auc_mean") or 0.0),
                    }
                )
    return result


def _best_defenses(defense_rows: List[Dict[str, str]], mode: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in defense_rows:
        if row.get("mode") == mode:
            grouped[(row.get("dataset", ""), row.get("attack_family", ""))].append(row)
    best: List[Dict[str, Any]] = []
    for (dataset, attack), rows in sorted(grouped.items()):
        if attack not in ATTACK_ORDER:
            continue
        row = max(rows, key=lambda r: _f(r, "tpr_mean") or -1.0)
        best.append(
            {
                "dataset": dataset,
                "attack_family": attack,
                "defense": row.get("defense", ""),
                "tpr_mean": _f(row, "tpr_mean"),
                "auc_mean": _f(row, "auc_mean"),
            }
        )
    return best


def _overall_delta(delta_rows: List[Dict[str, str]]) -> Dict[str, Optional[float]]:
    transfer = [_f(r, "transfer_delta") for r in delta_rows]
    stealth_auc = [_f(r, "stealth_auc_delta") for r in delta_rows]
    stealth_tpr = [_f(r, "stealth_tpr_delta") for r in delta_rows]
    def avg(vals: List[Optional[float]]) -> Optional[float]:
        xs = [v for v in vals if v is not None]
        return sum(xs) / len(xs) if xs else None
    return {
        "transfer_delta": avg(transfer),
        "stealth_auc_delta": avg(stealth_auc),
        "stealth_tpr_delta": avg(stealth_tpr),
    }


def _mode_attack_agg(summary_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        mode = row.get("mode", "")
        attack = row.get("attack_family", "")
        if mode in MODE_ORDER and attack in ATTACK_ORDER:
            grouped[(mode, attack)].append(row)

    result: List[Dict[str, Any]] = []
    for mode in MODE_ORDER:
        for attack in ATTACK_ORDER:
            rows = grouped.get((mode, attack), [])
            if not rows:
                continue
            t = [_f(r, "transfer_rate_mean") for r in rows]
            a = [_f(r, "stealth_auc_mean") for r in rows]
            p = [_f(r, "stealth_tpr_mean") for r in rows]
            t = [x for x in t if x is not None]
            a = [x for x in a if x is not None]
            p = [x for x in p if x is not None]
            result.append(
                {
                    "mode": mode,
                    "attack_family": attack,
                    "datasets_covered": len(rows),
                    "transfer_mean": sum(t) / len(t) if t else None,
                    "stealth_auc_mean": sum(a) / len(a) if a else None,
                    "stealth_tpr_mean": sum(p) / len(p) if p else None,
                }
            )
    return result


def _defense_mode_agg(defense_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in defense_rows:
        mode = row.get("mode", "")
        defense = row.get("defense", "")
        if mode in MODE_ORDER and defense in DEFENSE_ORDER:
            grouped[(mode, defense)].append(row)

    result: List[Dict[str, Any]] = []
    for mode in MODE_ORDER:
        for defense in DEFENSE_ORDER:
            rows = grouped.get((mode, defense), [])
            if not rows:
                continue
            t = [_f(r, "tpr_mean") for r in rows]
            a = [_f(r, "auc_mean") for r in rows]
            t = [x for x in t if x is not None]
            a = [x for x in a if x is not None]
            result.append(
                {
                    "mode": mode,
                    "defense": defense,
                    "groups_covered": len(rows),
                    "tpr_mean": sum(t) / len(t) if t else None,
                    "auc_mean": sum(a) / len(a) if a else None,
                }
            )
    return result


def _defense_delta_agg(defense_delta_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in defense_delta_rows:
        grouped[row["defense"]].append(row)

    result: List[Dict[str, Any]] = []
    for defense in DEFENSE_ORDER:
        rows = grouped.get(defense, [])
        if not rows:
            continue
        t = [row.get("tpr_delta") for row in rows if row.get("tpr_delta") is not None]
        a = [row.get("auc_delta") for row in rows if row.get("auc_delta") is not None]
        result.append(
            {
                "defense": defense,
                "pairs": len(rows),
                "tpr_delta_mean": (sum(t) / len(t)) if t else None,
                "auc_delta_mean": (sum(a) / len(a)) if a else None,
            }
        )
    return result


def build_report(
    summary_rows: List[Dict[str, str]],
    delta_rows: List[Dict[str, str]],
    defense_rows: List[Dict[str, str]],
) -> str:
    summary_index = _idx(summary_rows, ("mode", "dataset", "attack_family"))
    datasets = _sorted_values(summary_rows, "dataset")
    defense_delta = _defense_delta_rows(defense_rows)
    overall = _overall_delta(delta_rows)

    lines: List[str] = []
    lines.append("# SIG/UPGD 两种模式影响分析")
    lines.append("")
    lines.append("## 结论摘要")
    lines.append(
        f"- all-to-one 相比 clean-label 的平均迁移性提升为 `{_fmt(overall['transfer_delta'])}`，"
        f"但隐蔽性同步下降：stealth_auc 平均变化 `{_fmt(overall['stealth_auc_delta'])}`，"
        f"stealth_tpr 平均变化 `{_fmt(overall['stealth_tpr_delta'])}`。"
    )
    lines.append("- 这说明 all-to-one 更容易把后门迁移到目标域，但攻击痕迹更明显，更容易被防御检测到。")
    lines.append("- MNISTM 上模式切换影响最大，尤其 SIG 从 clean-label 的几乎不迁移变成高迁移；CIFAR10 上提升更温和。")
    lines.append("- 防御侧没有单一方法全胜：IBD_PSC 对 MNISTM 更强，SCaLe-Up/STRIP 对 CIFAR10 UPGD 更强，Tiny-ImageNet 上 STRIP 对 all-to-one 的提升最明显。")
    lines.append("")

    lines.append("## 迁移性与隐蔽性：两种模式对比")
    lines.append("|dataset|attack|clean transfer|all-to-one transfer|transfer delta|clean stealth_auc|all-to-one stealth_auc|stealth_auc delta|clean stealth_tpr|all-to-one stealth_tpr|stealth_tpr delta|")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        for attack in ATTACK_ORDER:
            clean = summary_index.get(("clean-label", dataset, attack))
            alltoone = summary_index.get(("all-to-one", dataset, attack))
            if not clean and not alltoone:
                continue
            c_transfer = _f(clean or {}, "transfer_rate_mean")
            a_transfer = _f(alltoone or {}, "transfer_rate_mean")
            c_auc = _f(clean or {}, "stealth_auc_mean")
            a_auc = _f(alltoone or {}, "stealth_auc_mean")
            c_tpr = _f(clean or {}, "stealth_tpr_mean")
            a_tpr = _f(alltoone or {}, "stealth_tpr_mean")
            lines.append(
                f"|{dataset}|{attack}|{_fmt(c_transfer)}|{_fmt(a_transfer)}|{_fmt(None if c_transfer is None or a_transfer is None else a_transfer - c_transfer)}|"
                f"{_fmt(c_auc)}|{_fmt(a_auc)}|{_fmt(None if c_auc is None or a_auc is None else a_auc - c_auc)}|"
                f"{_fmt(c_tpr)}|{_fmt(a_tpr)}|{_fmt(None if c_tpr is None or a_tpr is None else a_tpr - c_tpr)}|"
            )
    lines.append("")

    lines.append("### 迁移性解读")
    lines.append("- CIFAR10: all-to-one 后 SIG 的迁移性从约 `0.477` 提升到 `0.859`，UPGD 从约 `0.778` 提升到 `0.970`。UPGD 在两种模式下都更强，但 SIG 的相对增幅更大。")
    lines.append("- MNISTM: 模式影响最剧烈。clean-label 下 SIG 迁移率约 `0.003`，几乎不可迁移；all-to-one 后达到约 `0.756`。UPGD 也从约 `0.393` 提升到 `0.995`。")
    lines.append("- Tiny-ImageNet: 当前汇总里 clean-label 与 all-to-one 覆盖不完全一致，报告保留单模式统计，但不把它纳入 paired delta 结论。")
    lines.append("")

    lines.append("### 隐蔽性解读")
    lines.append("- 两个数据集、两种攻击的 stealth_auc / stealth_tpr delta 全为负，说明 all-to-one 的增强不是免费的。")
    lines.append("- CIFAR10 上 UPGD 的 stealth_tpr 下降约 `0.220`，比 SIG 的 `0.158` 更明显；MNISTM 上 SIG/UPGD 的 stealth_tpr 都下降约 `0.26~0.27`。")
    lines.append("- 直观上，all-to-one 让标签统一指向目标类，攻击信号更一致，迁移更强，但防御方法也更容易捕捉到异常模式。")
    lines.append("")

    lines.append("## 各防御方法效果")
    lines.append("### all-to-one 模式下每组最强防御")
    lines.append("|dataset|attack|best defense|tpr_mean|auc_mean|")
    lines.append("|---|---|---|---:|---:|")
    for row in _best_defenses(defense_rows, "all-to-one"):
        lines.append(
            f"|{row['dataset']}|{row['attack_family']}|{row['defense']}|{_fmt(row['tpr_mean'])}|{_fmt(row['auc_mean'])}|"
        )
    lines.append("")

    lines.append("### 防御检测率变化（all-to-one - clean-label）")
    lines.append("|dataset|attack|defense|clean TPR|all-to-one TPR|TPR delta|clean AUC|all-to-one AUC|AUC delta|")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(defense_delta, key=lambda r: (r["dataset"], r["attack_family"], r["defense"])):
        lines.append(
            f"|{row['dataset']}|{row['attack_family']}|{row['defense']}|"
            f"{_fmt(row['clean_tpr'])}|{_fmt(row['alltoone_tpr'])}|{_fmt(row['tpr_delta'])}|"
            f"{_fmt(row['clean_auc'])}|{_fmt(row['alltoone_auc'])}|{_fmt(row['auc_delta'])}|"
        )
    lines.append("")

    lines.append("### 防御方法解读")
    lines.append("- CIFAR10/SIG: IBD_PSC 与 SCaLe-Up 在 all-to-one 下 TPR 都约 `0.68`，明显高于 clean-label；STRIP 也提升但仍较弱。")
    lines.append("- CIFAR10/UPGD: SCaLe-Up 和 STRIP 最敏感，all-to-one TPR 分别约 `0.885` 和 `0.744`。")
    lines.append("- MNISTM/SIG: IBD_PSC 最强，TPR 从约 `0.489` 升到 `0.809`；SCaLe-Up/STRIP 也从很弱变成中等强度。")
    lines.append("- MNISTM/UPGD: IBD_PSC 和 STRIP 最有效，SentiNet 在两种模式下都较弱。")
    lines.append("- Tiny-ImageNet: all-to-one 下 STRIP 对 SIG/UPGD 的提升很明显；IBD_PSC 和 SCaLe-Up 的提升较小但仍有增强。")
    lines.append("")

    lines.append("## 数据完整性说明")
    lines.append("- 当前 paired delta 主要基于 CIFAR10 和 MNISTM 的可配对实验。")
    lines.append("- Tiny-ImageNet 有 clean-label 和 all-to-one 的单模式统计，但两侧覆盖不完全一致，因此不作为严格 paired delta 的主结论。")
    lines.append("- 详细 CSV: `mode_transfer_stealth_summary.csv`、`mode_transfer_stealth_delta.csv`、`mode_defense_dataset_summary.csv`。")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 SIG/UPGD 两种模式影响分析报告")
    parser.add_argument("--mode-summary-csv", type=Path, default=ANALYSIS_ALLTOONE_DIR / "mode_transfer_stealth_summary.csv")
    parser.add_argument("--mode-delta-csv", type=Path, default=ANALYSIS_ALLTOONE_DIR / "mode_transfer_stealth_delta.csv")
    parser.add_argument("--defense-summary-csv", type=Path, default=ANALYSIS_ALLTOONE_DIR / "mode_defense_dataset_summary.csv")
    parser.add_argument("--output-md", type=Path, default=ANALYSIS_ALLTOONE_DIR / "sig_upgd_alltoone_comparison_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode_summary_csv = _resolve_path(args.mode_summary_csv)
    mode_delta_csv = _resolve_path(args.mode_delta_csv)
    defense_summary_csv = _resolve_path(args.defense_summary_csv)
    output_md = _resolve_path(args.output_md)

    for required in [mode_summary_csv, mode_delta_csv, defense_summary_csv]:
        if not required.exists():
            raise FileNotFoundError(f"输入文件不存在: {required}")

    report = build_report(
        _read_csv(mode_summary_csv),
        _read_csv(mode_delta_csv),
        _read_csv(defense_summary_csv),
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    print(f"✓ 报告完成: {output_md}")


if __name__ == "__main__":
    main()
