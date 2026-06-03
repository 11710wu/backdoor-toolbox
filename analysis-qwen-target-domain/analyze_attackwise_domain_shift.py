"""Attack-wise ImageNetV2 vs Qwen ACC/transfer/stealth analysis.

This script reads the paired Tiny-ImageNet target-domain table and generates:
- attackwise_domain_shift_summary.csv
- attackwise_domain_shift_report.md
- figures_attackwise/<attack>_domain_metrics.png
- figures_attackwise/<attack>_paired_metric_lines.png
- figures_attackwise/<attack>_delta_scatter.png
- figures_attackwise/attack_delta_overview.png

The defense/stealth values are configuration-level properties in the current
pipeline: the same defense AUC is associated with the ImageNetV2 and Qwen
target-domain rows for a paired configuration.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis-qwen-target-domain"
PAIRED_CSV = ANALYSIS_DIR / "acc_joint_transfer_paired_rows.csv"
OUT_DIR = ANALYSIS_DIR / "figures_attackwise"
SUMMARY_CSV = ANALYSIS_DIR / "attackwise_domain_shift_summary.csv"
REPORT_MD = ANALYSIS_DIR / "attackwise_domain_shift_report.md"

ATTACK_ORDER = [
    "WaNet",
    "basic",
    "SIG",
    "upgd",
    "belt",
    "adaptive_patch",
    "blend",
    "adaptive_blend",
]

COLORS = {
    "ImageNetV2": "#4C78A8",
    "Qwen": "#F58518",
    "delta": "#E45756",
    "stealth": "#54A24B",
}


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def pp(x: float) -> str:
    return f"{x * 100:+.2f}pp"


def num(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:+.{digits}f}"


def pearson(x: pd.Series, y: pd.Series) -> float:
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    a = df.iloc[:, 0].to_numpy(dtype=float)
    b = df.iloc[:, 1].to_numpy(dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def slope(x: pd.Series, y: pd.Series) -> float:
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    a = df.iloc[:, 0].to_numpy(dtype=float)
    b = df.iloc[:, 1].to_numpy(dtype=float)
    if np.std(a) == 0:
        return float("nan")
    return float(np.polyfit(a, b, 1)[0])


def summarize_attack(df: pd.DataFrame, attack: str) -> dict[str, float | str | int]:
    sub = df[df["attack"] == attack].copy()
    return {
        "attack": attack,
        "n": len(sub),
        "iv2_acc": sub["iv2_acc"].mean(),
        "qwen_acc": sub["qwen_acc"].mean(),
        "delta_acc": sub["delta_acc"].mean(),
        "iv2_transfer_asr": sub["iv2_transfer_asr"].mean(),
        "qwen_transfer_asr": sub["qwen_transfer_asr"].mean(),
        "delta_transfer_asr": sub["delta_transfer_asr"].mean(),
        "iv2_joint_transfer": sub["iv2_joint_transfer"].mean(),
        "qwen_joint_transfer": sub["qwen_joint_transfer"].mean(),
        "delta_joint_transfer": sub["delta_joint_transfer"].mean(),
        "detection_auc": sub["detection_auc"].mean(),
        "stealth_auc": sub["stealth_auc"].mean(),
        "delta_acc_vs_delta_transfer_r": pearson(sub["delta_acc"], sub["delta_transfer_asr"]),
        "delta_acc_vs_delta_joint_r": pearson(sub["delta_acc"], sub["delta_joint_transfer"]),
        "delta_acc_to_delta_transfer_slope": slope(sub["delta_acc"], sub["delta_transfer_asr"]),
        "delta_acc_to_delta_joint_slope": slope(sub["delta_acc"], sub["delta_joint_transfer"]),
    }


def build_long_for_attack(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in sub.iterrows():
        common = {
            "config": r["config"],
            "arch": r["arch"],
            "attack": r["attack"],
            "source_asr": r["source_asr"],
            "stealth_auc": r["stealth_auc"],
            "detection_auc": r["detection_auc"],
        }
        rows.append({
            **common,
            "domain": "ImageNetV2",
            "target_acc": r["iv2_acc"],
            "transfer_asr": r["iv2_transfer_asr"],
            "joint_transfer": r["iv2_joint_transfer"],
        })
        rows.append({
            **common,
            "domain": "Qwen",
            "target_acc": r["qwen_acc"],
            "transfer_asr": r["qwen_transfer_asr"],
            "joint_transfer": r["qwen_joint_transfer"],
        })
    return pd.DataFrame(rows)


def plot_domain_metrics(df: pd.DataFrame, attack: str) -> None:
    sub = df[df["attack"] == attack].copy()
    means = {
        "ACC": [sub["iv2_acc"].mean(), sub["qwen_acc"].mean()],
        "transfer_ASR": [sub["iv2_transfer_asr"].mean(), sub["qwen_transfer_asr"].mean()],
        "joint_transfer": [sub["iv2_joint_transfer"].mean(), sub["qwen_joint_transfer"].mean()],
        # Stealth is shared per config in this pipeline, duplicated intentionally.
        "stealth_AUC": [sub["stealth_auc"].mean(), sub["stealth_auc"].mean()],
    }
    deltas = {
        "ACC": sub["delta_acc"].mean(),
        "transfer_ASR": sub["delta_transfer_asr"].mean(),
        "joint_transfer": sub["delta_joint_transfer"].mean(),
        "stealth_AUC": 0.0,
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    labels = list(means.keys())
    x = np.arange(len(labels))
    width = 0.34
    iv2_vals = [means[k][0] * 100 for k in labels]
    qwen_vals = [means[k][1] * 100 for k in labels]
    ax.bar(x - width / 2, iv2_vals, width, label="ImageNetV2", color=COLORS["ImageNetV2"])
    ax.bar(x + width / 2, qwen_vals, width, label="Qwen", color=COLORS["Qwen"])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(max(iv2_vals), max(qwen_vals), 100) * 1.12)
    ax.set_ylabel("Mean value (%)")
    ax.set_title(f"{attack}: target-domain metrics by target dataset")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    for i, k in enumerate(labels):
        d = deltas[k] * 100
        y = max(iv2_vals[i], qwen_vals[i]) + 2.2
        ax.text(i, y, f"{d:+.1f}pp", ha="center", va="bottom", fontsize=9)
    ax.text(
        0.01,
        -0.18,
        "Note: stealth_AUC is a shared per-configuration defense metric in this analysis.",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{attack}_domain_metrics.png", dpi=180)
    plt.close(fig)


def plot_paired_metric_lines(df: pd.DataFrame, attack: str) -> None:
    sub = df[df["attack"] == attack].copy()
    metrics = [
        ("ACC", "iv2_acc", "qwen_acc"),
        ("transfer_ASR", "iv2_transfer_asr", "qwen_transfer_asr"),
        ("joint_transfer", "iv2_joint_transfer", "qwen_joint_transfer"),
        ("stealth_AUC", "stealth_auc", "stealth_auc"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.8), sharey=True)
    rng = np.random.default_rng(20240529)
    arch_colors = {
        "resnet18": "#4C78A8",
        "mobilenetv2": "#F58518",
        "vgg19_bn": "#54A24B",
    }
    arch_seen = set()

    for ax, (label, iv2_col, qwen_col) in zip(axes, metrics):
        for _, r in sub.iterrows():
            jitter = rng.uniform(-0.035, 0.035)
            x = np.array([0 + jitter, 1 + jitter])
            y = np.array([r[iv2_col], r[qwen_col]], dtype=float) * 100
            arch = r.get("arch", "unknown")
            color = arch_colors.get(arch, "#999999")
            line_label = arch if arch not in arch_seen else None
            ax.plot(x, y, color=color, alpha=0.22, linewidth=0.9, label=line_label)
            ax.scatter(x, y, color=color, alpha=0.42, s=12)
            arch_seen.add(arch)

        mean_vals = np.array([sub[iv2_col].mean(), sub[qwen_col].mean()]) * 100
        ax.plot([0, 1], mean_vals, color="black", linewidth=2.4, marker="o", markersize=5)
        ax.text(
            0.5,
            max(mean_vals) + 4.0,
            f"{mean_vals[1] - mean_vals[0]:+.1f}pp",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )
        ax.set_xticks([0, 1], ["ImageNetV2", "Qwen"])
        ax.set_title(label)
        ax.set_xlim(-0.28, 1.28)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)
        if label == "ACC":
            ax.set_ylabel("Per-configuration value (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.96), fontsize=8)
    fig.suptitle(f"{attack}: paired ImageNetV2 -> Qwen metric shifts", y=1.04)
    fig.text(
        0.01,
        0.02,
        "Each thin line is one paired configuration. Black line is the attack-wise mean. "
        "stealth_AUC is shared per configuration in this pipeline.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.savefig(OUT_DIR / f"{attack}_paired_metric_lines.png", dpi=180)
    plt.close(fig)


def plot_delta_scatter(df: pd.DataFrame, attack: str) -> None:
    sub = df[df["attack"] == attack].copy()
    arch_markers = {"resnet18": "o", "mobilenetv2": "s", "vgg19_bn": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True)
    settings = [
        ("delta_transfer_asr", "Delta transfer_ASR (Qwen - ImageNetV2)", axes[0]),
        ("delta_joint_transfer", "Delta joint_transfer (Qwen - ImageNetV2)", axes[1]),
    ]
    for y_key, ylabel, ax in settings:
        for arch, marker in arch_markers.items():
            part = sub[sub["arch"] == arch]
            if part.empty:
                continue
            ax.scatter(
                part["delta_acc"] * 100,
                part[y_key] * 100,
                label=arch,
                marker=marker,
                s=42,
                alpha=0.75,
            )
        if len(sub) >= 3 and sub["delta_acc"].std() > 0:
            x = sub["delta_acc"].to_numpy(dtype=float) * 100
            y = sub[y_key].to_numpy(dtype=float) * 100
            beta, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xs, beta * xs + intercept, color="black", linewidth=1.4)
            r = pearson(sub["delta_acc"], sub[y_key])
            ax.text(0.03, 0.95, f"r={r:+.3f}", transform=ax.transAxes, ha="left", va="top")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.55)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.55)
        ax.set_xlabel("Delta ACC (Qwen - ImageNetV2, pp)")
        ax.set_ylabel(ylabel + " (pp)")
        ax.grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(f"{attack}: does the ACC increase explain transfer change?", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{attack}_delta_scatter.png", dpi=180)
    plt.close(fig)


def plot_overview(summary: pd.DataFrame) -> None:
    s = summary.copy()
    s["attack"] = pd.Categorical(s["attack"], categories=ATTACK_ORDER, ordered=True)
    s = s.sort_values("attack")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharex=True)
    for ax, col, title in [
        (axes[0], "delta_acc", "Delta ACC"),
        (axes[1], "delta_transfer_asr", "Delta transfer_ASR"),
        (axes[2], "delta_joint_transfer", "Delta joint_transfer"),
    ]:
        vals = s[col] * 100
        colors = [COLORS["Qwen"] if v >= 0 else COLORS["delta"] for v in vals]
        ax.bar(s["attack"].astype(str), vals, color=colors)
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_title(title)
        ax.set_ylabel("Qwen - ImageNetV2 (pp)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Attack-wise effect of switching target domain from ImageNetV2 to Qwen", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attack_delta_overview.png", dpi=180)
    plt.close(fig)


def write_summary_csv(rows: list[dict[str, object]]) -> pd.DataFrame:
    fieldnames = [
        "attack", "n",
        "iv2_acc", "qwen_acc", "delta_acc",
        "iv2_transfer_asr", "qwen_transfer_asr", "delta_transfer_asr",
        "iv2_joint_transfer", "qwen_joint_transfer", "delta_joint_transfer",
        "detection_auc", "stealth_auc",
        "delta_acc_vs_delta_transfer_r", "delta_acc_vs_delta_joint_r",
        "delta_acc_to_delta_transfer_slope", "delta_acc_to_delta_joint_slope",
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, df: pd.DataFrame) -> None:
    s = summary.copy()
    s["attack"] = pd.Categorical(s["attack"], categories=ATTACK_ORDER, ordered=True)
    s = s.sort_values("attack")

    table_lines = []
    for _, r in s.iterrows():
        table_lines.append(
            f"| {r['attack']} | {int(r['n'])} | {pct(r['iv2_acc'])} | {pct(r['qwen_acc'])} | "
            f"{pp(r['delta_acc'])} | {pct(r['iv2_transfer_asr'])} | {pct(r['qwen_transfer_asr'])} | "
            f"{pp(r['delta_transfer_asr'])} | {pct(r['iv2_joint_transfer'])} | "
            f"{pct(r['qwen_joint_transfer'])} | {pp(r['delta_joint_transfer'])} | "
            f"{pct(r['stealth_auc'])} | {num(r['delta_acc_vs_delta_transfer_r'])} |"
        )

    def attack_block(attack: str) -> str:
        r = s[s["attack"] == attack].iloc[0]
        if attack == "WaNet":
            reading = (
                "WaNet 是最特殊的个案：Qwen 目标域让 ACC 上升约 "
                f"{pp(r['delta_acc'])}，但 transfer_ASR 没有下降，反而轻微上升 "
                f"{pp(r['delta_transfer_asr'])}。这说明 WaNet 的空间形变触发器在更高 "
                "ACC 的 Qwen 域中仍保持触发能力；它不是“ACC 上升导致 ASR 下降”的普通例子，"
                "而是 ACC、形变强度、迁移和检测共同耦合的机制交互。"
            )
        elif attack in {"SIG", "upgd"}:
            reading = (
                f"{attack} 对目标域变化最敏感：ACC 上升 {pp(r['delta_acc'])} 后，"
                f"transfer_ASR 下降 {pp(r['delta_transfer_asr'])}，joint_transfer 下降 "
                f"{pp(r['delta_joint_transfer'])}。它们说明 ImageNetV2 低 ACC 目标域上"
                "的一部分 ASR 可能来自目标域统计不稳定或分类不可靠性。"
            )
        elif attack in {"basic", "adaptive_patch", "belt"}:
            reading = (
                f"{attack} 属于迁移强、检测强的主 trade-off 证据。Qwen 上 ACC 上升 "
                f"{pp(r['delta_acc'])} 后，transfer_ASR 下降 {pp(r['delta_transfer_asr'])}，"
                f"但 stealth_AUC 仍然很低或偏低（均值 {pct(r['stealth_auc'])}），说明"
                "高迁移局部/强触发器仍然容易被检测。"
            )
        else:
            reading = (
                f"{attack} 呈现中等幅度的 ACC-迁移调整：ACC 上升 {pp(r['delta_acc'])}，"
                f"transfer_ASR 下降 {pp(r['delta_transfer_asr'])}，joint_transfer 下降 "
                f"{pp(r['delta_joint_transfer'])}。它支持“更高 ACC 目标域会压低表观迁移性”，"
                "但幅度小于 SIG/UPGD。"
            )
        return (
            f"### {attack}\n\n"
            f"- 图：`figures_attackwise/{attack}_domain_metrics.png`，"
            f"`figures_attackwise/{attack}_paired_metric_lines.png`，"
            f"`figures_attackwise/{attack}_delta_scatter.png`\n"
            f"- ImageNetV2 ACC / Qwen ACC：{pct(r['iv2_acc'])} -> {pct(r['qwen_acc'])} "
            f"({pp(r['delta_acc'])})\n"
            f"- transfer_ASR：{pct(r['iv2_transfer_asr'])} -> {pct(r['qwen_transfer_asr'])} "
            f"({pp(r['delta_transfer_asr'])})\n"
            f"- joint_transfer：{pct(r['iv2_joint_transfer'])} -> {pct(r['qwen_joint_transfer'])} "
            f"({pp(r['delta_joint_transfer'])})\n"
            f"- stealth_AUC：{pct(r['stealth_auc'])}。注意它是配置级防御属性，"
            "不是两个目标域分别重测的 stealth。\n\n"
            f"{reading}\n"
        )

    md = f"""# Attack-wise ACC, Transfer, and Stealth Shift Analysis

本报告只分析 Tiny-ImageNet 源模型在两个目标域上的配对结果：

- ImageNetV2 aligned target domain
- Qwen generated target domain

每个攻击方法先单独看 ImageNetV2 与 Qwen 的 ACC、迁移性和隐蔽性，再比较不同方法。这里的 `stealth_AUC` 是当前管线中每个源模型/触发器配置的共享防御属性，同一个配置在 ImageNetV2 和 Qwen 两条目标域记录中使用同一个 stealth 值；因此本报告主要看“新目标域导致 ACC 上升后，transfer_ASR / joint_transfer 是否变化”。

## Overall Attack-wise Summary

| attack | n | IV2 ACC | Qwen ACC | delta ACC | IV2 transfer | Qwen transfer | delta transfer | IV2 joint | Qwen joint | delta joint | stealth AUC | r(delta ACC, delta transfer) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_lines)}

## Overview Figure

![overview](figures_attackwise/attack_delta_overview.png)

## Method-by-method Reading

{chr(10).join(attack_block(a) for a in ATTACK_ORDER if a in set(s['attack'].astype(str)))}

## Main Takeaways

1. 逐方法看以后，不能简单说“ACC 上升一定让 ASR 下降”。多数方法确实下降，但 WaNet 是明显例外。
2. SIG 和 UPGD 对新目标域最敏感，Qwen ACC 上升后 transfer_ASR 大幅下降，说明 ImageNetV2 低 ACC 可能显著放大了它们的表观迁移性。
3. basic、adaptive_patch、belt 在 Qwen 上 ASR 有下降，但仍然保持较高迁移性，同时 stealth_AUC 低，继续支持“高迁移、低隐蔽”的主 trade-off。
4. WaNet 需要单独解释：Qwen ACC 上升后 transfer_ASR 基本不降，说明空间形变触发器与目标域 ACC 的关系不是普通单调关系，而是攻击机制交互。
5. 当前 stealth_AUC 不是目标域重测指标，而是源模型/触发器配置的共享检测属性，所以两个目标域比较主要解释 ACC 对迁移性，而不是 ACC 对 stealth 的直接因果影响。
"""
    REPORT_MD.write_text(md, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PAIRED_CSV)
    rows = []
    for attack in ATTACK_ORDER:
        if attack not in set(df["attack"]):
            continue
        rows.append(summarize_attack(df, attack))
        plot_domain_metrics(df, attack)
        plot_paired_metric_lines(df, attack)
        plot_delta_scatter(df, attack)
    summary = write_summary_csv(rows)
    plot_overview(summary)
    write_report(summary, df)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
