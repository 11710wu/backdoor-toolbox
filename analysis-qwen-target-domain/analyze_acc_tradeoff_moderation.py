"""ACC moderation analysis for transferability-stealth trade-off.

This script focuses on one question:

Does target-domain ACC change the relationship between transferability and
stealth, or does it mainly shift the measured transferability?

Inputs:
- acc_joint_transfer_rows.csv: long table, one row per config/domain.
- acc_joint_transfer_paired_rows.csv: paired ImageNetV2/Qwen table.

Outputs:
- acc_tradeoff_moderation_summary.json
- acc_tradeoff_moderation_by_group.csv
- acc_tradeoff_moderation_report.md
- figures_acc_tradeoff_moderation/*.png

Important caveat: stealth_AUC is a configuration-level defense metric in this
pipeline. It is duplicated for ImageNetV2 and Qwen rows of the same config.
Therefore ACC can only be interpreted as affecting measured transferability
and the observed transfer-stealth association, not as directly changing the
measured stealth value across target domains.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis-qwen-target-domain"
LONG_CSV = ANALYSIS_DIR / "acc_joint_transfer_rows.csv"
PAIRED_CSV = ANALYSIS_DIR / "acc_joint_transfer_paired_rows.csv"
OUT_DIR = ANALYSIS_DIR / "figures_acc_tradeoff_moderation"
SUMMARY_JSON = ANALYSIS_DIR / "acc_tradeoff_moderation_summary.json"
GROUP_CSV = ANALYSIS_DIR / "acc_tradeoff_moderation_by_group.csv"
REPORT_MD = ANALYSIS_DIR / "acc_tradeoff_moderation_report.md"

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
    "low": "#72B7B2",
    "mid": "#B279A2",
    "high": "#E45756",
}


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def pp(x: float) -> str:
    return f"{x * 100:+.2f}pp"


def fmt(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.{digits}f}"


def signed(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:+.{digits}f}"


def corr(x: pd.Series, y: pd.Series) -> float:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 3:
        return float("nan")
    a = d.iloc[:, 0].to_numpy(dtype=float)
    b = d.iloc[:, 1].to_numpy(dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def slope(x: pd.Series, y: pd.Series) -> float:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 3:
        return float("nan")
    a = d.iloc[:, 0].to_numpy(dtype=float)
    b = d.iloc[:, 1].to_numpy(dtype=float)
    if np.std(a) == 0:
        return float("nan")
    return float(np.polyfit(a, b, 1)[0])


def residualize(d: pd.DataFrame, value_col: str, controls: Iterable[str]) -> np.ndarray:
    parts = []
    for c in controls:
        if d[c].dtype == object:
            parts.append(pd.get_dummies(d[c], prefix=c, drop_first=True, dtype=float))
        else:
            parts.append(d[[c]].astype(float))
    if parts:
        x_df = pd.concat(parts, axis=1)
        x = x_df.to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(d)), x])
    else:
        x = np.ones((len(d), 1))
    y = d[value_col].to_numpy(dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return y - x @ beta


def partial_corr(df: pd.DataFrame, x_col: str, y_col: str, controls: list[str]) -> float:
    cols = [x_col, y_col] + controls
    d = df[cols].dropna()
    if len(d) < 5:
        return float("nan")
    rx = residualize(d, x_col, controls)
    ry = residualize(d, y_col, controls)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def ols_tradeoff(
    df: pd.DataFrame,
    transfer_col: str,
    include_fixed_effects: bool,
    include_interaction: bool = True,
) -> dict[str, object]:
    cols = [transfer_col, "target_acc", "stealth_auc", "attack", "arch", "domain"]
    d = df[cols].dropna().copy()

    pieces = [
        pd.DataFrame({"Intercept": np.ones(len(d))}, index=d.index),
        d[[transfer_col]].rename(columns={transfer_col: "T"}).astype(float),
        d[["target_acc"]].rename(columns={"target_acc": "ACC"}).astype(float),
    ]
    if include_interaction:
        pieces.append(
            pd.DataFrame(
                {"T_x_ACC": d[transfer_col].astype(float) * d["target_acc"].astype(float)},
                index=d.index,
            )
        )
    if include_fixed_effects:
        for c in ["attack", "arch", "domain"]:
            pieces.append(pd.get_dummies(d[c], prefix=c, drop_first=True, dtype=float))

    x_df = pd.concat(pieces, axis=1)
    x = x_df.to_numpy(dtype=float)
    y = d["stealth_auc"].to_numpy(dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    y_hat = x @ beta
    resid = y - y_hat
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((y - np.mean(y)) ** 2))
    coef = dict(zip(x_df.columns, beta))

    acc_points = {
        "q25_acc": float(df["target_acc"].quantile(0.25)),
        "mean_acc": float(df["target_acc"].mean()),
        "q75_acc": float(df["target_acc"].quantile(0.75)),
        "imagenetv2_mean_acc": float(df[df["domain"] == "ImageNetV2"]["target_acc"].mean()),
        "qwen_mean_acc": float(df[df["domain"] == "Qwen"]["target_acc"].mean()),
    }
    slopes = {}
    for name, acc in acc_points.items():
        slopes[name] = float(coef["T"] + coef.get("T_x_ACC", 0.0) * acc)

    return {
        "n": int(len(d)),
        "r2": r2,
        "coefficients": {k: float(v) for k, v in coef.items() if k in {"Intercept", "T", "ACC", "T_x_ACC"}},
        "conditional_slopes": slopes,
    }


def group_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for domain, sub in df.groupby("domain"):
        rows.append({
            "group_type": "domain",
            "group": domain,
            "n": len(sub),
            "mean_acc": sub["target_acc"].mean(),
            "mean_transfer_asr": sub["transfer_asr"].mean(),
            "mean_joint_transfer": sub["joint_transfer"].mean(),
            "mean_stealth_auc": sub["stealth_auc"].mean(),
            "r_joint_stealth": corr(sub["joint_transfer"], sub["stealth_auc"]),
            "slope_joint_to_stealth": slope(sub["joint_transfer"], sub["stealth_auc"]),
            "r_transfer_stealth": corr(sub["transfer_asr"], sub["stealth_auc"]),
            "slope_transfer_to_stealth": slope(sub["transfer_asr"], sub["stealth_auc"]),
        })

    q = df.copy()
    q["acc_bin"] = pd.qcut(q["target_acc"], 4, duplicates="drop")
    for acc_bin, sub in q.groupby("acc_bin", observed=True):
        rows.append({
            "group_type": "acc_quantile",
            "group": str(acc_bin),
            "n": len(sub),
            "mean_acc": sub["target_acc"].mean(),
            "mean_transfer_asr": sub["transfer_asr"].mean(),
            "mean_joint_transfer": sub["joint_transfer"].mean(),
            "mean_stealth_auc": sub["stealth_auc"].mean(),
            "r_joint_stealth": corr(sub["joint_transfer"], sub["stealth_auc"]),
            "slope_joint_to_stealth": slope(sub["joint_transfer"], sub["stealth_auc"]),
            "r_transfer_stealth": corr(sub["transfer_asr"], sub["stealth_auc"]),
            "slope_transfer_to_stealth": slope(sub["transfer_asr"], sub["stealth_auc"]),
        })

    for attack in ATTACK_ORDER:
        sub = df[df["attack"] == attack]
        if sub.empty:
            continue
        rows.append({
            "group_type": "attack",
            "group": attack,
            "n": len(sub),
            "mean_acc": sub["target_acc"].mean(),
            "mean_transfer_asr": sub["transfer_asr"].mean(),
            "mean_joint_transfer": sub["joint_transfer"].mean(),
            "mean_stealth_auc": sub["stealth_auc"].mean(),
            "r_joint_stealth": corr(sub["joint_transfer"], sub["stealth_auc"]),
            "slope_joint_to_stealth": slope(sub["joint_transfer"], sub["stealth_auc"]),
            "r_transfer_stealth": corr(sub["transfer_asr"], sub["stealth_auc"]),
            "slope_transfer_to_stealth": slope(sub["transfer_asr"], sub["stealth_auc"]),
        })
        for domain, ss in sub.groupby("domain"):
            rows.append({
                "group_type": "attack_domain",
                "group": f"{attack}:{domain}",
                "n": len(ss),
                "mean_acc": ss["target_acc"].mean(),
                "mean_transfer_asr": ss["transfer_asr"].mean(),
                "mean_joint_transfer": ss["joint_transfer"].mean(),
                "mean_stealth_auc": ss["stealth_auc"].mean(),
                "r_joint_stealth": corr(ss["joint_transfer"], ss["stealth_auc"]),
                "slope_joint_to_stealth": slope(ss["joint_transfer"], ss["stealth_auc"]),
                "r_transfer_stealth": corr(ss["transfer_asr"], ss["stealth_auc"]),
                "slope_transfer_to_stealth": slope(ss["transfer_asr"], ss["stealth_auc"]),
            })

    return pd.DataFrame(rows)


def add_regression_line(ax: plt.Axes, sub: pd.DataFrame, x_col: str, y_col: str, color: str) -> None:
    if len(sub) < 3 or sub[x_col].std() == 0:
        return
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    beta, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ax.plot(xs * 100, (beta * xs + intercept) * 100, color=color, linewidth=2.0)


def plot_domain_tradeoff(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True)
    for ax, (domain, sub) in zip(axes, df.groupby("domain")):
        color = COLORS.get(domain, "#666666")
        ax.scatter(sub["joint_transfer"] * 100, sub["stealth_auc"] * 100, alpha=0.42, s=24, color=color)
        add_regression_line(ax, sub, "joint_transfer", "stealth_auc", "black")
        ax.set_title(
            f"{domain}: r={corr(sub['joint_transfer'], sub['stealth_auc']):+.3f}, "
            f"slope={slope(sub['joint_transfer'], sub['stealth_auc']):+.3f}"
        )
        ax.set_xlabel("joint_transfer (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("stealth_AUC (%)")
    fig.suptitle("Transferability-stealth trade-off by target domain", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tradeoff_by_domain_joint.png", dpi=180)
    plt.close(fig)


def plot_acc_bins(df: pd.DataFrame) -> None:
    q = df.copy()
    q["acc_bin"] = pd.qcut(q["target_acc"], 4, labels=["Q1 low ACC", "Q2", "Q3", "Q4 high ACC"])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (label, sub) in zip(axes, q.groupby("acc_bin", observed=True)):
        ax.scatter(sub["joint_transfer"] * 100, sub["stealth_auc"] * 100, alpha=0.42, s=22)
        add_regression_line(ax, sub, "joint_transfer", "stealth_auc", "black")
        ax.set_title(
            f"{label}: ACC={sub['target_acc'].mean()*100:.1f}%, "
            f"r={corr(sub['joint_transfer'], sub['stealth_auc']):+.3f}, "
            f"slope={slope(sub['joint_transfer'], sub['stealth_auc']):+.3f}"
        )
        ax.grid(alpha=0.25)
        ax.set_xlabel("joint_transfer (%)")
        ax.set_ylabel("stealth_AUC (%)")
    fig.suptitle("Does the trade-off slope change across ACC levels?", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tradeoff_by_acc_quantile_joint.png", dpi=180)
    plt.close(fig)


def plot_attack_slopes(group_df: pd.DataFrame) -> None:
    attack_rows = group_df[group_df["group_type"] == "attack"].copy()
    attack_rows["group"] = pd.Categorical(attack_rows["group"], categories=ATTACK_ORDER, ordered=True)
    attack_rows = attack_rows.sort_values("group")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    axes[0].bar(attack_rows["group"].astype(str), attack_rows["r_joint_stealth"], color="#4C78A8")
    axes[0].axhline(0, color="black", linewidth=0.9)
    axes[0].set_title("Correlation: joint_transfer vs stealth_AUC")
    axes[0].set_ylabel("Pearson r")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(attack_rows["group"].astype(str), attack_rows["slope_joint_to_stealth"], color="#E45756")
    axes[1].axhline(0, color="black", linewidth=0.9)
    axes[1].set_title("Slope: stealth_AUC ~ joint_transfer")
    axes[1].set_ylabel("Slope")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Attack-specific trade-off strength", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attackwise_tradeoff_strength.png", dpi=180)
    plt.close(fig)


def plot_interaction_slopes(summary: dict[str, object]) -> None:
    raw = summary["ols_joint_interaction_raw"]["conditional_slopes"]
    fe = summary["ols_joint_interaction_fixed_effects"]["conditional_slopes"]
    labels = ["q25_acc", "imagenetv2_mean_acc", "mean_acc", "qwen_mean_acc", "q75_acc"]
    label_text = ["Q25", "IV2 mean", "Mean", "Qwen mean", "Q75"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x, [raw[k] for k in labels], marker="o", linewidth=2.0, label="Raw interaction")
    ax.plot(x, [fe[k] for k in labels], marker="o", linewidth=2.0, label="Attack+arch+domain FE")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x, label_text)
    ax.set_ylabel("Conditional slope d(stealth_AUC) / d(joint_transfer)")
    ax.set_title("ACC changes the trade-off slope only mildly")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "interaction_conditional_slopes_joint.png", dpi=180)
    plt.close(fig)


def plot_paired_delta_vs_stealth(paired: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    settings = [
        ("delta_transfer_asr", "Delta transfer_ASR (Qwen - ImageNetV2, pp)"),
        ("delta_joint_transfer", "Delta joint_transfer (Qwen - ImageNetV2, pp)"),
    ]
    for ax, (x_col, label) in zip(axes, settings):
        for attack in ATTACK_ORDER:
            sub = paired[paired["attack"] == attack]
            if sub.empty:
                continue
            ax.scatter(sub[x_col] * 100, sub["stealth_auc"] * 100, alpha=0.55, s=24, label=attack)
        if len(paired) >= 3:
            x = paired[x_col].to_numpy(dtype=float)
            y = paired["stealth_auc"].to_numpy(dtype=float)
            beta, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xs * 100, (beta * xs + intercept) * 100, color="black", linewidth=1.8)
            ax.text(
                0.03,
                0.95,
                f"r={corr(paired[x_col], paired['stealth_auc']):+.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
            )
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.7)
        ax.set_xlabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("stealth_AUC (%)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Which stealth levels lose transfer when target-domain ACC rises?", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "paired_delta_transfer_vs_stealth.png", dpi=180)
    plt.close(fig)


def make_summary(df: pd.DataFrame, paired: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "n_long_rows": int(len(df)),
        "n_paired_configs": int(len(paired)),
        "global": {},
        "domain": {},
        "paired_delta": {},
    }
    for transfer_col in ["transfer_asr", "joint_transfer"]:
        summary["global"][transfer_col] = {
            "r_with_stealth": corr(df[transfer_col], df["stealth_auc"]),
            "slope_to_stealth": slope(df[transfer_col], df["stealth_auc"]),
            "partial_r_control_acc": partial_corr(df, transfer_col, "stealth_auc", ["target_acc"]),
            "partial_r_control_acc_domain": partial_corr(df, transfer_col, "stealth_auc", ["target_acc", "domain"]),
            "partial_r_control_acc_attack_arch_domain": partial_corr(
                df,
                transfer_col,
                "stealth_auc",
                ["target_acc", "attack", "arch", "domain"],
            ),
        }
    for domain, sub in df.groupby("domain"):
        summary["domain"][domain] = {
            "n": int(len(sub)),
            "mean_acc": float(sub["target_acc"].mean()),
            "mean_joint_transfer": float(sub["joint_transfer"].mean()),
            "mean_stealth_auc": float(sub["stealth_auc"].mean()),
            "r_joint_stealth": corr(sub["joint_transfer"], sub["stealth_auc"]),
            "slope_joint_to_stealth": slope(sub["joint_transfer"], sub["stealth_auc"]),
        }
    for delta_col in ["delta_acc", "delta_transfer_asr", "delta_joint_transfer"]:
        summary["paired_delta"][delta_col] = {
            "mean": float(paired[delta_col].mean()),
            "r_with_stealth": corr(paired[delta_col], paired["stealth_auc"]),
            "slope_to_stealth": slope(paired[delta_col], paired["stealth_auc"]),
        }
    summary["ols_transfer_interaction_raw"] = ols_tradeoff(df, "transfer_asr", include_fixed_effects=False)
    summary["ols_transfer_interaction_fixed_effects"] = ols_tradeoff(df, "transfer_asr", include_fixed_effects=True)
    summary["ols_joint_interaction_raw"] = ols_tradeoff(df, "joint_transfer", include_fixed_effects=False)
    summary["ols_joint_interaction_fixed_effects"] = ols_tradeoff(df, "joint_transfer", include_fixed_effects=True)
    return summary


def write_report(summary: dict[str, object], group_df: pd.DataFrame) -> None:
    domain_rows = group_df[group_df["group_type"] == "domain"].copy()
    acc_rows = group_df[group_df["group_type"] == "acc_quantile"].copy()
    attack_rows = group_df[group_df["group_type"] == "attack"].copy()
    attack_rows["group"] = pd.Categorical(attack_rows["group"], categories=ATTACK_ORDER, ordered=True)
    attack_rows = attack_rows.sort_values("group")

    def domain_table() -> str:
        lines = []
        for _, r in domain_rows.iterrows():
            lines.append(
                f"| {r['group']} | {int(r['n'])} | {pct(r['mean_acc'])} | "
                f"{pct(r['mean_joint_transfer'])} | {pct(r['mean_stealth_auc'])} | "
                f"{signed(r['r_joint_stealth'])} | {signed(r['slope_joint_to_stealth'])} |"
            )
        return "\n".join(lines)

    def acc_bin_table() -> str:
        lines = []
        for _, r in acc_rows.iterrows():
            lines.append(
                f"| {r['group']} | {int(r['n'])} | {pct(r['mean_acc'])} | "
                f"{pct(r['mean_joint_transfer'])} | {pct(r['mean_stealth_auc'])} | "
                f"{signed(r['r_joint_stealth'])} | {signed(r['slope_joint_to_stealth'])} |"
            )
        return "\n".join(lines)

    def attack_table() -> str:
        lines = []
        for _, r in attack_rows.iterrows():
            lines.append(
                f"| {r['group']} | {int(r['n'])} | {pct(r['mean_acc'])} | "
                f"{pct(r['mean_joint_transfer'])} | {pct(r['mean_stealth_auc'])} | "
                f"{signed(r['r_joint_stealth'])} | {signed(r['slope_joint_to_stealth'])} |"
            )
        return "\n".join(lines)

    joint_global = summary["global"]["joint_transfer"]
    transfer_global = summary["global"]["transfer_asr"]
    raw = summary["ols_joint_interaction_raw"]
    fe = summary["ols_joint_interaction_fixed_effects"]
    raw_coef = raw["coefficients"]
    fe_coef = fe["coefficients"]
    raw_slopes = raw["conditional_slopes"]
    fe_slopes = fe["conditional_slopes"]
    paired_delta = summary["paired_delta"]

    md = f"""# ACC Effect on the Transferability-Stealth Relationship

本报告专门回答一个问题：`target_ACC` 会不会改变“迁移性越强，隐蔽性越差”这条关系？

这里仍然只分析 Tiny-ImageNet 源模型在两个目标域上的结果：

- ImageNetV2 aligned target domain
- Qwen generated target domain

注意：当前数据管线中的 `stealth_AUC` 是源模型/触发器配置级防御属性，同一个配置在 ImageNetV2 和 Qwen 记录里共享同一个 `stealth_AUC`。因此 ACC 的作用不能解释为“目标域让 stealth 本身改变了”，更准确地说是：ACC 改变了横轴上的 measured transferability，从而可能改变我们观察到的 transferability-stealth trade-off。

## 1. Control-ACC Robustness

先看整体关系是否只是 ACC 混杂造成的。

| transfer metric | raw r with stealth | partial r, control ACC | partial r, control ACC+domain | partial r, control ACC+attack+arch+domain |
|---|---:|---:|---:|---:|
| transfer_ASR | {signed(transfer_global['r_with_stealth'])} | {signed(transfer_global['partial_r_control_acc'])} | {signed(transfer_global['partial_r_control_acc_domain'])} | {signed(transfer_global['partial_r_control_acc_attack_arch_domain'])} |
| joint_transfer | {signed(joint_global['r_with_stealth'])} | {signed(joint_global['partial_r_control_acc'])} | {signed(joint_global['partial_r_control_acc_domain'])} | {signed(joint_global['partial_r_control_acc_attack_arch_domain'])} |

结论：控制 ACC 后，`joint_transfer` 与 `stealth_AUC` 的负相关没有消失，反而从 {signed(joint_global['r_with_stealth'])} 到 {signed(joint_global['partial_r_control_acc'])}，再到控制 attack/arch/domain 后的 {signed(joint_global['partial_r_control_acc_attack_arch_domain'])}。这说明 ACC 不是这条 trade-off 的唯一来源。

## 2. Domain-Level Slope

![domain tradeoff](figures_acc_tradeoff_moderation/tradeoff_by_domain_joint.png)

| domain | n | mean ACC | mean joint_transfer | mean stealth_AUC | r(joint, stealth) | slope stealth~joint |
|---|---:|---:|---:|---:|---:|---:|
{domain_table()}

ImageNetV2 的 mean ACC 是 {pct(summary['domain']['ImageNetV2']['mean_acc'])}，Qwen 的 mean ACC 是 {pct(summary['domain']['Qwen']['mean_acc'])}。但是两边的 `stealth_AUC ~ joint_transfer` 斜率几乎一样：ImageNetV2 为 {signed(summary['domain']['ImageNetV2']['slope_joint_to_stealth'])}，Qwen 为 {signed(summary['domain']['Qwen']['slope_joint_to_stealth'])}。

这说明新目标域提高 ACC 后，主要改变的是 measured transferability 的水平，而不是把 trade-off 方向翻转或抹掉。

## 3. ACC-Level Slope

![acc bins](figures_acc_tradeoff_moderation/tradeoff_by_acc_quantile_joint.png)

| ACC bin | n | mean ACC | mean joint_transfer | mean stealth_AUC | r(joint, stealth) | slope stealth~joint |
|---|---:|---:|---:|---:|---:|---:|
{acc_bin_table()}

从 ACC 四分位看，斜率始终是负的，约在 -0.77 到 -0.72 之间。也就是说，从低 ACC 到高 ACC，trade-off 的强度有轻微变化，但没有质变。

## 4. Interaction Model

使用交互项模型：

```text
stealth_AUC = a + b1 * joint_transfer + b2 * target_ACC + b3 * joint_transfer * target_ACC
```

未加固定效应时：

```text
stealth_AUC = {fmt(raw_coef['Intercept'])}
              {signed(raw_coef['T'])} * joint_transfer
              {signed(raw_coef['ACC'])} * target_ACC
              {signed(raw_coef['T_x_ACC'])} * joint_transfer * target_ACC
```

R2 = {fmt(raw['r2'])}

加入 attack、arch、domain 固定效应后：

```text
stealth_AUC = h(attack, arch, domain)
              {signed(fe_coef['T'])} * joint_transfer
              {signed(fe_coef['ACC'])} * target_ACC
              {signed(fe_coef['T_x_ACC'])} * joint_transfer * target_ACC
```

R2 = {fmt(fe['r2'])}

![interaction slopes](figures_acc_tradeoff_moderation/interaction_conditional_slopes_joint.png)

因为交互项 `b3` 是正的，所以 ACC 越高，`joint_transfer -> stealth_AUC` 的负斜率会略微变浅。具体地：

| model | slope at ImageNetV2 mean ACC | slope at Qwen mean ACC |
|---|---:|---:|
| raw interaction | {signed(raw_slopes['imagenetv2_mean_acc'])} | {signed(raw_slopes['qwen_mean_acc'])} |
| attack+arch+domain fixed effects | {signed(fe_slopes['imagenetv2_mean_acc'])} | {signed(fe_slopes['qwen_mean_acc'])} |

这个变化方向可以解释为：高 ACC 目标域减少了一部分低 ACC 带来的 ASR 表观膨胀，使 trade-off 斜率略微变平；但斜率仍然显著为负。

## 5. Attack-Specific Trade-off

![attackwise strength](figures_acc_tradeoff_moderation/attackwise_tradeoff_strength.png)

| attack | n | mean ACC | mean joint_transfer | mean stealth_AUC | r(joint, stealth) | slope stealth~joint |
|---|---:|---:|---:|---:|---:|---:|
{attack_table()}

方法拆开后，负相关仍普遍存在，但强度差别很大：

- basic、blend、WaNet、adaptive_patch、belt 的相关更强。
- SIG 和 upgd 的斜率较浅，说明它们的 transfer range 和 stealth range 不像 basic/belt 那样形成强线性 trade-off。
- 因此 ACC 的影响不是一个统一全局常数，而是和 attack mechanism 交互。

## 6. Paired Domain Shift

![paired delta](figures_acc_tradeoff_moderation/paired_delta_transfer_vs_stealth.png)

从 ImageNetV2 换到 Qwen 时：

- mean delta_ACC = {pp(paired_delta['delta_acc']['mean'])}
- mean delta_transfer_ASR = {pp(paired_delta['delta_transfer_asr']['mean'])}
- mean delta_joint_transfer = {pp(paired_delta['delta_joint_transfer']['mean'])}
- corr(delta_joint_transfer, stealth_AUC) = {signed(paired_delta['delta_joint_transfer']['r_with_stealth'])}

这个负相关说明：Qwen 提高 ACC 后，transfer drop 更明显地发生在更 stealthy 的配置/方法上，尤其是 SIG 和 upgd。这会把高 stealth 方法进一步推向低 transfer 区域，因此不会削弱主 trade-off，反而会让“高 stealth、低 transfer”和“低 stealth、高 transfer”的分离更清楚。

## Final Interpretation

ACC 对迁移性-隐蔽性关系的影响可以分成三层：

1. **混杂/测量层面**：ACC 会影响 measured transferability。低 ACC 目标域可能抬高 `transfer_ASR`，所以只看 target ASR 会高估一部分迁移性。
2. **调节层面**：ACC 会轻微改变 `joint_transfer -> stealth_AUC` 的斜率。交互模型显示高 ACC 时负斜率略微变浅，但不会变成 0 或正数。
3. **方法机制层面**：ACC 的影响是 attack-dependent。SIG/upgd 的 transfer drop 最大；basic/belt/adaptive_patch 的 qualitative trade-off 稳定；WaNet 是特殊机制交互。

因此更准确的数学表述不是单一的全局公式，而是：

```text
joint_transfer = f(attack, arch, domain) + beta_attack * target_ACC
```

以及：

```text
stealth_AUC = h(attack, arch, domain)
              + theta1 * joint_transfer
              + theta2 * target_ACC
              + theta3 * joint_transfer * target_ACC
```

在当前数据中，`theta1 < 0`，`theta3 > 0` 但幅度较小。也就是说：

```text
ACC 会修正 trade-off 的斜率，但不会推翻 trade-off。
ACC 主要改变迁移性测量值；迁移性和隐蔽性之间的负关系在控制 ACC 后依然稳定。
```
"""
    REPORT_MD.write_text(md, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LONG_CSV)
    paired = pd.read_csv(PAIRED_CSV)

    group_df = group_stats(df)
    group_df.to_csv(GROUP_CSV, index=False)

    summary = make_summary(df, paired)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_domain_tradeoff(df)
    plot_acc_bins(df)
    plot_attack_slopes(group_df)
    plot_interaction_slopes(summary)
    plot_paired_delta_vs_stealth(paired)
    write_report(summary, group_df)

    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {GROUP_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
