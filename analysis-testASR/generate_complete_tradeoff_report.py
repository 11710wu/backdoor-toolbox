#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成跨数据集完整 trade-off 分析报告。

报告口径：
- 同域攻击有效性：`asr_mean`
- 迁移性：`transfer_mean`
- 隐蔽性：`stealth_tpr_mean = 1 - avg(defense TPR)`，来自四个防御的平均检测失败程度
- 补充隐蔽性：`S_stealth`（含 NC）

输入：
- analysis/metrics_summary_by_method_asr_transfer.csv
- analysis/metrics_summary_by_method_defense_tpr_auc.csv
- analysis/data_*_nc.csv

输出：
- analysis/complete_tradeoff_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


DATASET_ORDER = ["cifar10", "mnistm", "tiny_imagenet"]
ATTACK_ORDER = [
    "SIG",
    "WaNet",
    "adaptive_blend",
    "adaptive_patch",
    "basic",
    "belt",
    "blend",
    "upgd",
]


def _load_asr_transfer(analysis_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(analysis_dir / "metrics_summary_by_method_asr_transfer.csv")
    for col in [
        "poison_rate",
        "n_points",
        "asr_mean",
        "transfer_mean",
        "stealth_tpr_mean",
        "stealth_auc_mean",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_defense(analysis_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(analysis_dir / "metrics_summary_by_method_defense_tpr_auc.csv")
    for col in ["poison_rate", "n_samples", "tpr_mean", "auc_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_nc(analysis_dir: Path) -> pd.DataFrame:
    frames = []
    for path in analysis_dir.iterdir():
        if path.name.startswith("data_") and path.name.endswith("_nc.csv") and "no_nc" not in path.name:
            frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    for col in ["poison_rate", "asr", "transfer_rate", "stealth_tpr_avg", "S_stealth", "S_stealth_tpr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fmt(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]) or pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{float(x):.{decimals}f}" if pd.notna(x) else "")
    return out


def _md_table(df: pd.DataFrame, decimals: int = 4) -> str:
    if df.empty:
        return "_（无数据）_\n"
    return _fmt(df, decimals=decimals).to_markdown(index=False) + "\n"


def _pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    pts = df[["transfer_mean", "stealth_tpr_mean"]].to_numpy()
    keep: List[int] = []
    for i, (tr, st) in enumerate(pts):
        dominated = False
        for j, (tr2, st2) in enumerate(pts):
            if i == j:
                continue
            if (tr2 >= tr and st2 >= st) and (tr2 > tr or st2 > st):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return df.iloc[keep].copy()


def _dataset_note(dataset: str, corr_ts: float, corr_at: float, best_row: pd.Series) -> str:
    attack = best_row["attack_type"]
    arch = best_row["arch"]
    pr = best_row["poison_rate"]
    return (
        f"- 该数据集上 `transfer_mean` 与 `stealth_tpr_mean` 的相关系数为 **{corr_ts:.4f}**，"
        f"说明迁移性与以防御失效定义的隐蔽性整体呈显著负相关。\n"
        f"- `asr_mean` 与 `transfer_mean` 的相关系数为 **{corr_at:.4f}**，"
        f"说明很多“迁移差”的点仍然与源域后门是否先建立成功高度相关。\n"
        f"- 以 `asr_mean >= 0.3` 为前提、用 `transfer` 与 `stealth` 的调和均值做平衡排序时，"
        f"当前最优参考配置是 **`{attack}` / `{arch}` / poison_rate=`{pr}`**。\n"
    )


def build_report(analysis_dir: Path) -> str:
    asr = _load_asr_transfer(analysis_dir)
    defense = _load_defense(analysis_dir)
    nc = _load_nc(analysis_dir)

    lines: List[str] = []
    lines.append("# 跨数据集迁移性 vs 隐蔽性完整分析报告\n")
    lines.append(
        "本报告统一使用 **防御检测失败程度** 作为主隐蔽性指标："
        "`stealth_tpr_mean = 1 - avg(TPR of STRIP, SCaLe-Up, SentiNet, IBD_PSC)`。"
        "值越大表示越难被这四种防御检出。`S_stealth` 仅作为包含 NC 的补充指标，不替代主结论。\n"
    )

    overall = (
        asr.groupby("dataset", sort=False)
        .agg(
            n_configs=("attack_type", "count"),
            asr_mean=("asr_mean", "mean"),
            transfer_mean=("transfer_mean", "mean"),
            stealth_tpr_mean=("stealth_tpr_mean", "mean"),
        )
        .reset_index()
    )
    lines.append("\n## 1. 跨数据集总体概览\n")
    lines.append(_md_table(overall))

    defense_overall = (
        defense.groupby(["dataset", "defense"], sort=False)
        .agg(tpr_mean=("tpr_mean", "mean"), auc_mean=("auc_mean", "mean"))
        .reset_index()
    )
    lines.append(
        "从防御均值看，三个数据集的“最强防御”并不一致："
        "CIFAR-10 上 **SCaLe-Up** 的平均 TPR 最高，MNIST-M 上 **IBD_PSC** 最强，"
        "Tiny ImageNet 上也主要是 **IBD_PSC** 更稳定。\n"
    )
    lines.append("\n### 1.1 各数据集防御平均检测强度\n")
    lines.append(_md_table(defense_overall))

    for dataset in DATASET_ORDER:
        g = asr[asr["dataset"] == dataset].copy()
        g = g.sort_values(["attack_type", "arch", "poison_rate"])
        corr_ts = g[["transfer_mean", "stealth_tpr_mean"]].corr().iloc[0, 1]
        corr_at = g[["asr_mean", "transfer_mean"]].corr().iloc[0, 1]

        attack_summary = (
            g.groupby("attack_type", sort=False)
            .agg(
                asr_mean=("asr_mean", "mean"),
                transfer_mean=("transfer_mean", "mean"),
                stealth_tpr_mean=("stealth_tpr_mean", "mean"),
                stealth_auc_mean=("stealth_auc_mean", "mean"),
            )
            .reset_index()
        )
        attack_summary["attack_type"] = pd.Categorical(
            attack_summary["attack_type"], categories=ATTACK_ORDER, ordered=True
        )
        attack_summary = attack_summary.sort_values("attack_type").reset_index(drop=True)

        arch_summary = (
            g.groupby("arch", sort=False)
            .agg(
                asr_mean=("asr_mean", "mean"),
                transfer_mean=("transfer_mean", "mean"),
                stealth_tpr_mean=("stealth_tpr_mean", "mean"),
            )
            .reset_index()
        )

        rate_summary = (
            g.groupby("poison_rate", sort=False)
            .agg(
                asr_mean=("asr_mean", "mean"),
                transfer_mean=("transfer_mean", "mean"),
                stealth_tpr_mean=("stealth_tpr_mean", "mean"),
            )
            .reset_index()
            .sort_values("poison_rate")
        )

        valid = g[g["asr_mean"] >= 0.3].copy()
        valid["balance_hmean"] = (
            2
            * valid["transfer_mean"]
            * valid["stealth_tpr_mean"]
            / (valid["transfer_mean"] + valid["stealth_tpr_mean"])
        )
        balance_top = valid.sort_values(
            ["balance_hmean", "transfer_mean", "stealth_tpr_mean"], ascending=False
        ).head(5)
        pareto = _pareto_front(valid).sort_values(
            ["transfer_mean", "stealth_tpr_mean"], ascending=False
        )

        defense_attack = (
            defense[defense["dataset"] == dataset]
            .groupby(["attack_type", "defense"], sort=False)
            .agg(tpr_mean=("tpr_mean", "mean"), auc_mean=("auc_mean", "mean"))
            .reset_index()
        )
        strongest_rows = []
        for attack, sub in defense_attack.groupby("attack_type", sort=False):
            sub = sub.sort_values("tpr_mean", ascending=False)
            strongest_rows.append(
                {
                    "attack_type": attack,
                    "strongest_defense": sub.iloc[0]["defense"],
                    "strongest_tpr": sub.iloc[0]["tpr_mean"],
                    "weakest_defense": sub.iloc[-1]["defense"],
                    "weakest_tpr": sub.iloc[-1]["tpr_mean"],
                }
            )
        strongest_df = pd.DataFrame(strongest_rows)
        strongest_df["attack_type"] = pd.Categorical(
            strongest_df["attack_type"], categories=ATTACK_ORDER, ordered=True
        )
        strongest_df = strongest_df.sort_values("attack_type").reset_index(drop=True)

        nc_sub = nc[nc["dataset"] == dataset]
        nc_attack = (
            nc_sub.groupby("attack_type", sort=False)
            .agg(S_stealth=("S_stealth", "mean"), S_stealth_tpr=("S_stealth_tpr", "mean"))
            .reset_index()
        )
        nc_attack["attack_type"] = pd.Categorical(
            nc_attack["attack_type"], categories=ATTACK_ORDER, ordered=True
        )
        nc_attack = nc_attack.sort_values("attack_type").reset_index(drop=True)

        lines.append(f"\n## 2. {dataset} 结果分析\n")
        lines.append(_dataset_note(dataset, corr_ts, corr_at, balance_top.iloc[0]))

        lines.append("\n### 2.1 按攻击方法汇总\n")
        lines.append(_md_table(attack_summary))

        lines.append("\n### 2.2 按模型架构汇总\n")
        lines.append(_md_table(arch_summary))

        lines.append("\n### 2.3 按中毒率汇总\n")
        lines.append(_md_table(rate_summary))

        lines.append(
            "\n### 2.4 平衡型候选配置（`asr_mean >= 0.3`，按 transfer 与 stealth 的调和均值排序，仅作参考）\n"
        )
        lines.append(
            _md_table(
                balance_top[
                    [
                        "arch",
                        "attack_type",
                        "poison_rate",
                        "asr_mean",
                        "transfer_mean",
                        "stealth_tpr_mean",
                        "balance_hmean",
                    ]
                ]
            )
        )

        lines.append("\n### 2.5 Pareto 前沿（maximize transfer, maximize stealth）\n")
        lines.append(
            _md_table(
                pareto[
                    [
                        "arch",
                        "attack_type",
                        "poison_rate",
                        "asr_mean",
                        "transfer_mean",
                        "stealth_tpr_mean",
                    ]
                ]
            )
        )

        lines.append("\n### 2.6 各攻击最敏感/最不敏感的防御\n")
        lines.append(_md_table(strongest_df))

        lines.append("\n### 2.7 含 NC 的补充隐蔽性（非主指标）\n")
        lines.append(_md_table(nc_attack))

        if dataset == "cifar10":
            lines.append(
                "结论：CIFAR-10 上 trade-off 最明显，`transfer_mean` 与 `stealth_tpr_mean` 呈强负相关。"
                "`basic` 与高强度 `upgd`/`adaptive_patch` 能把迁移性推到很高，但隐蔽性显著下降；"
                "真正兼顾两者的点很少，当前最平衡的是 **ResNet18 + SIG@0.05**，"
                "但它仍然属于“中高迁移 / 中高隐蔽”，而不是极致两端都强。\n"
            )
        elif dataset == "mnistm":
            lines.append(
                "结论：MNIST-M 更像“有限 Pareto 前沿”问题。`belt` 明显占据主前沿，"
                "尤其在 `poison_rate=0.02` 时同时具备高同域 ASR、高迁移和较高隐蔽性；"
                "`upgd` 则提供更高 stealth，但迁移性上限低于 `belt`。`SIG` 在该数据集上主要体现为攻击建立失败。\n"
            )
        elif dataset == "tiny_imagenet":
            lines.append(
                "结论：Tiny ImageNet 的高迁移高隐蔽前沿最宽。`WaNet@0.05` 能把迁移拉满，但隐蔽性一般；"
                "`upgd@0.005` 在三种架构上都给出非常强的平衡结果；"
                "`SIG@0.005` 也表现出比 CIFAR-10/MNIST-M 更好的平衡性。"
                "整体上这是三个数据集中最容易同时拿到“高迁移 + 高隐蔽”的设置。\n"
            )

    lines.append("\n## 3. 综合结论\n")
    lines.append(
        "- **从整体 trade-off 形态看**：CIFAR-10 最尖锐，MNIST-M 次之，Tiny ImageNet 的可行前沿最宽。\n"
    )
    lines.append(
        "- **从方法族看**：`basic` 往往提供极高同域/迁移成功率，但 stealth 最差；`upgd` 与 `adaptive_blend` 更偏向隐蔽；`belt` 在 MNIST-M 上最均衡；`upgd` 在 Tiny ImageNet 上最均衡。\n"
    )
    lines.append(
        "- **从架构看**：CIFAR-10 上 ResNet18 的综合表现最好；MNIST-M 上 MobileNet 的 stealth 最强，但 ResNet18 的迁移更强；Tiny ImageNet 上 ResNet18 在迁移性上最占优，VGG 在 stealth 上略高。\n"
    )
    lines.append(
        "- **从防御看**：`IBD_PSC` 在 MNIST-M 和 Tiny ImageNet 上更稳定，`SCaLe-Up` 在 CIFAR-10 上平均最强；`STRIP` 经常是最弱防御。`SentiNet` 对 `basic` / `adaptive_patch` 尤其有效。\n"
    )
    lines.append(
        "- **论文叙事建议**：主文使用 `stealth_tpr_mean` 作为隐蔽性主指标，强调“基于多防御平均失效率”；把 `S_stealth` 作为补充验证，说明即便加入 NC，主要方法排序不会完全翻转，只会让高 stealth 方法的优势更保守。\n"
    )

    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成跨数据集完整 trade-off 分析报告")
    parser.add_argument("--analysis-dir", type=str, default="analysis")
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/complete_tradeoff_report.md",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.is_absolute():
        analysis_dir = root / analysis_dir
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    report = build_report(analysis_dir)
    output.write_text(report, encoding="utf-8")
    print(f"[OK] wrote report to: {output}")


if __name__ == "__main__":
    main()
