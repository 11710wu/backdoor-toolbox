#!/usr/bin/env python3
"""cover-rate 独立消融实验综合分析脚本（数据来源：poisoned_train_set）。

实验设计
--------
poisoned_train_set 是 cover_rate 的真正独立消融：
  - adaptive_patch：alpha=0.000 固定，cover_rate 独立变化（0, 0.001, 0.005, 0.01, 0.025, 0.05）
  - adaptive_blend：alpha=0.150 固定，cover_rate 独立变化（0, 0.001, 0.005, 0.01, 0.025, 0.05）
  - WaNet：s=0.5 固定，cover_rate 独立变化（0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1）

功能
----
  1. 统计汇总表（按 dataset / arch / attack_type / cover_rate）
  2. Pearson/Spearman 相关系数 & 线性趋势斜率
  3. 趋势折线图（均值±SE）/ 箱线图 / 散点图 / 热力图（cover_rate × arch）
  4. 综合对比图（3×3：攻击 × 指标）
  5. cover_rate 分析报告（Markdown）
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────── 路径配置 ───────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
TABLES_DIR = BASE_DIR / "report_tables"
FIGURES_DIR = BASE_DIR / "report_figures"

COVER_ABLATION_ATTACKS = ["WaNet", "adaptive_blend", "adaptive_patch"]

ATTACK_LABELS  = {"WaNet": "WaNet", "adaptive_blend": "Adaptive Blend", "adaptive_patch": "Adaptive Patch"}
DATASET_LABELS = {"cifar10": "CIFAR-10", "tiny_imagenet": "Tiny-ImageNet", "mnistm": "MNIST-M"}
ARCH_LABELS    = {"resnet18": "ResNet-18", "mobilenet": "MobileNetV2", "vgg": "VGG-19"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# ─────────────────────── 数据加载 ───────────────────────

def load_data(min_asr: float = 0.0) -> pd.DataFrame:
    """加载 data/ 中所有 no_nc CSV，计算 weighted_tpr/auc，可选 ASR 过滤。"""
    dfs: List[pd.DataFrame] = []
    for f in sorted(DATA_DIR.glob("data_*_no_nc.csv")):
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  [WARN] 无法读取 {f.name}: {e}")
    if not dfs:
        raise FileNotFoundError(f"在 {DATA_DIR} 中未找到数据文件，请先运行 extract_results.py")

    df = pd.concat(dfs, ignore_index=True)

    for col in ["asr", "transfer_rate", "stealth_tpr_avg", "stealth_auc_avg", "cover_rate", "train_param_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 计算检测率（与 analysis-cover-rate 保持一致）
    df["weighted_tpr"] = 1.0 - df["stealth_tpr_avg"]
    df["weighted_auc"] = 1.0 - df["stealth_auc_avg"]

    # 仅保留 cover-rate 消融攻击
    df = df[df["attack_type"].isin(COVER_ABLATION_ATTACKS)].copy()

    if min_asr > 0:
        df = df[df["asr"].fillna(0) >= min_asr].copy()

    return df


# ─────────────────────── 统计汇总 ───────────────────────

def build_cover_rate_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (ds, arch, attack, cr), grp in df.groupby(
        ["dataset", "arch", "attack_type", "cover_rate"], dropna=False
    ):
        n = len(grp)
        for metric, col in [
            ("asr", "asr"), ("test_asr", "transfer_rate"),
            ("stealth_auc", "stealth_auc_avg"), ("stealth_tpr", "stealth_tpr_avg"),
            ("weighted_auc", "weighted_auc"), ("weighted_tpr", "weighted_tpr"),
        ]:
            vals = grp[col].dropna() if col in grp.columns else pd.Series(dtype=float)
            records.append({
                "dataset": ds, "arch": arch, "attack_type": attack, "cover_rate": cr, "n": n,
                "metric": metric,
                "mean":   round(vals.mean(),   4) if len(vals) > 0 else np.nan,
                "std":    round(vals.std(),    4) if len(vals) > 1 else np.nan,
                "median": round(vals.median(), 4) if len(vals) > 0 else np.nan,
                "min":    round(vals.min(),    4) if len(vals) > 0 else np.nan,
                "max":    round(vals.max(),    4) if len(vals) > 0 else np.nan,
            })
    return pd.DataFrame(records)


def build_wide_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {}
    for col in ["asr", "transfer_rate", "stealth_auc_avg", "stealth_tpr_avg", "weighted_auc", "weighted_tpr"]:
        if col in df.columns:
            agg_dict[col] = ["mean", "std", "count"]
    wide = df.groupby(["dataset", "arch", "attack_type", "cover_rate"]).agg(agg_dict)
    wide.columns = ["_".join(c) for c in wide.columns]
    return wide.reset_index().round(4)


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (ds, attack), grp in df.groupby(["dataset", "attack_type"]):
        covers = pd.to_numeric(grp["cover_rate"], errors="coerce")
        for metric, col in [
            ("ASR", "asr"), ("test-ASR", "transfer_rate"),
            ("stealth_AUC", "stealth_auc_avg"), ("stealth_TPR", "stealth_tpr_avg"),
            ("weighted_AUC", "weighted_auc"),
        ]:
            if col not in grp.columns:
                continue
            vals = grp[col]
            mask = covers.notna() & vals.notna()
            x, y = covers[mask].values, vals[mask].values
            if len(x) < 4:
                continue
            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
            slope, *_ = stats.linregress(x, y)
            records.append({
                "dataset": ds, "attack_type": attack, "metric": metric,
                "pearson_r": round(r_p, 4), "pearson_p": round(p_p, 4),
                "spearman_r": round(r_s, 4), "spearman_p": round(p_s, 4),
                "linear_slope": round(slope, 6), "n": int(mask.sum()),
            })
    return pd.DataFrame(records)


# ─────────────────────── 绘图工具 ───────────────────────

def _cover_pct_labels(covers):
    return [f"{v * 100:.1f}%" for v in covers]


def plot_trend_mean_ci(df: pd.DataFrame, save_dir: Path):
    """均值±SE 趋势折线图，按数据集着色，按攻击分面。"""
    metrics = [
        ("asr",             "ASR（源域攻击成功率）"),
        ("transfer_rate",   "test-ASR（目标域迁移率）"),
        ("stealth_auc_avg", "Stealth AUC（隐蔽性）"),
        ("weighted_auc",    "Weighted AUC（检测率）"),
    ]
    ds_colors = {"cifar10": "#E74C3C", "tiny_imagenet": "#3498DB", "mnistm": "#2ECC71"}

    for col, ylabel in metrics:
        if col not in df.columns:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        fig.suptitle(f"Cover Rate 对 {ylabel} 的影响（均值±SE）", fontsize=13, fontweight="bold")
        for ax, attack in zip(axes, COVER_ABLATION_ATTACKS):
            sub = df[df["attack_type"] == attack]
            for ds in ["cifar10", "tiny_imagenet", "mnistm"]:
                dsub = sub[sub["dataset"] == ds]
                agg = dsub.groupby("cover_rate")[col].agg(["mean", "sem", "count"]).reset_index()
                agg = agg[agg["count"] >= 1]
                if agg.empty:
                    continue
                ax.errorbar(
                    agg["cover_rate"], agg["mean"], yerr=agg["sem"],
                    color=ds_colors.get(ds, "gray"), marker="o",
                    linewidth=2, markersize=6, capsize=4,
                    label=DATASET_LABELS.get(ds, ds),
                )
            ax.set_title(ATTACK_LABELS.get(attack, attack), fontweight="bold")
            ax.set_xlabel("Cover Rate")
            if ax == axes[0]:
                ax.set_ylabel(ylabel)
            covers_vals = sorted(sub["cover_rate"].unique()) if not sub.empty else []
            if covers_vals:
                ax.set_xticks(covers_vals)
                ax.set_xticklabels(_cover_pct_labels(np.array(covers_vals)), rotation=30, ha="right")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        plt.tight_layout()
        fname = save_dir / f"trend_mean_ci_{col}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print(f"  图: {fname.name}")


def plot_trend_by_arch(df: pd.DataFrame, save_dir: Path):
    """各架构趋势折线，攻击分列，指标分行（3×3）。"""
    metrics_list = [
        ("asr",             "ASR"),
        ("transfer_rate",   "test-ASR"),
        ("stealth_auc_avg", "Stealth AUC"),
    ]
    arch_styles = {"resnet18": ("#E74C3C", "o"), "mobilenet": ("#3498DB", "s"), "vgg": ("#2ECC71", "^")}

    for ds in ["cifar10", "tiny_imagenet", "mnistm"]:
        sub_ds = df[df["dataset"] == ds]
        if sub_ds.empty:
            continue
        fig, axes = plt.subplots(3, 3, figsize=(14, 11))
        fig.suptitle(f"{DATASET_LABELS.get(ds, ds)}：Cover Rate 对各指标的影响（按架构）",
                     fontsize=13, fontweight="bold")
        for row_idx, (col, ylabel) in enumerate(metrics_list):
            for col_idx, attack in enumerate(COVER_ABLATION_ATTACKS):
                ax = axes[row_idx][col_idx]
                sub = sub_ds[sub_ds["attack_type"] == attack]
                if sub.empty or col not in sub.columns:
                    ax.axis("off")
                    continue
                for arch, (color, marker) in arch_styles.items():
                    asub = sub[sub["arch"] == arch]
                    agg = asub.groupby("cover_rate")[col].agg(["mean", "sem"]).reset_index()
                    if agg.empty:
                        continue
                    ax.plot(agg["cover_rate"].values, agg["mean"].values,
                            color=color, marker=marker, linewidth=2, markersize=6,
                            label=ARCH_LABELS.get(arch, arch))
                    ax.fill_between(
                        agg["cover_rate"].values,
                        (agg["mean"] - agg["sem"]).values,
                        (agg["mean"] + agg["sem"]).values,
                        color=color, alpha=0.15,
                    )
                if row_idx == 0:
                    ax.set_title(ATTACK_LABELS.get(attack, attack), fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(ylabel, fontsize=11)
                if row_idx == 2:
                    ax.set_xlabel("Cover Rate", fontsize=11)
                covers_vals = sorted(sub["cover_rate"].unique()) if not sub.empty else []
                if covers_vals:
                    ax.set_xticks(covers_vals)
                    ax.set_xticklabels(_cover_pct_labels(np.array(covers_vals)),
                                       rotation=30, ha="right", fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=9)
        plt.tight_layout()
        fname = save_dir / f"trend_by_arch_{ds}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print(f"  图: {fname.name}")


def plot_boxplot_by_cover(df: pd.DataFrame, save_dir: Path):
    """箱线图：cover_rate 分组，攻击分面。"""
    metrics = [
        ("asr",             "ASR（源域）"),
        ("transfer_rate",   "test-ASR（目标域）"),
        ("stealth_auc_avg", "Stealth AUC"),
        ("weighted_auc",    "Weighted AUC（检测率）"),
    ]
    for col, ylabel in metrics:
        if col not in df.columns:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(f"Cover Rate 对 {ylabel} 的分布（箱线图）", fontsize=13, fontweight="bold")
        for ax, attack in zip(axes, COVER_ABLATION_ATTACKS):
            sub = df[df["attack_type"] == attack].copy()
            if sub.empty:
                ax.axis("off")
                continue
            covers = sorted(sub["cover_rate"].unique())
            data_list = [sub[sub["cover_rate"] == cr][col].dropna().values for cr in covers]
            bp = ax.boxplot(data_list, patch_artist=True,
                            medianprops=dict(color="black", linewidth=2))
            colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(covers)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
            ax.set_xticks(range(1, len(covers) + 1))
            ax.set_xticklabels(_cover_pct_labels(np.array(covers)), rotation=30, ha="right")
            ax.set_title(ATTACK_LABELS.get(attack, attack), fontweight="bold")
            ax.set_xlabel("Cover Rate")
            if ax == axes[0]:
                ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fname = save_dir / f"boxplot_{col}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print(f"  图: {fname.name}")


def plot_scatter_asr_vs_transfer(df: pd.DataFrame, save_dir: Path):
    """散点图：ASR vs. test-ASR，cover_rate 着色，攻击分面。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("ASR vs. test-ASR，按 cover_rate 着色", fontsize=13, fontweight="bold")
    for ax, attack in zip(axes, COVER_ABLATION_ATTACKS):
        sub = df[(df["attack_type"] == attack) & df["asr"].notna() & df["transfer_rate"].notna()]
        if sub.empty:
            ax.axis("off")
            continue
        covers = sorted(sub["cover_rate"].unique())
        c2color = dict(zip(covers, cm.plasma(np.linspace(0.1, 0.9, len(covers)))))
        for _, row in sub.iterrows():
            ax.scatter(row["asr"], row["transfer_rate"],
                       color=c2color.get(row["cover_rate"], "gray"), alpha=0.7, s=35)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.45)
        ax.set_xlabel("ASR（源域）")
        if ax == axes[0]:
            ax.set_ylabel("test-ASR（目标域）")
        ax.set_title(ATTACK_LABELS.get(attack, attack), fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c2color[cr],
                        markersize=7, label=f"cover={cr*100:.1f}%") for cr in covers]
        ax.legend(handles=elems, fontsize=8, loc="upper left")
    plt.tight_layout()
    fname = save_dir / "scatter_asr_vs_transfer.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  图: {fname.name}")


def plot_scatter_transfer_vs_stealth(df: pd.DataFrame, save_dir: Path):
    """散点图：test-ASR vs. Stealth AUC（迁移-隐蔽权衡），cover_rate 着色。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("迁移性 vs. 隐蔽性权衡（test-ASR vs. Stealth AUC），按 cover_rate 着色",
                 fontsize=12, fontweight="bold")
    for ax, attack in zip(axes, COVER_ABLATION_ATTACKS):
        sub = df[(df["attack_type"] == attack) & df["transfer_rate"].notna() & df["stealth_auc_avg"].notna()]
        if sub.empty:
            ax.axis("off")
            continue
        covers = sorted(sub["cover_rate"].unique())
        c2color = dict(zip(covers, cm.viridis(np.linspace(0.15, 0.85, len(covers)))))
        for _, row in sub.iterrows():
            ax.scatter(row["stealth_auc_avg"], row["transfer_rate"],
                       color=c2color.get(row["cover_rate"], "gray"), alpha=0.7, s=35)
        ax.set_xlabel("Stealth AUC（隐蔽性）")
        if ax == axes[0]:
            ax.set_ylabel("test-ASR（迁移性）")
        ax.set_title(ATTACK_LABELS.get(attack, attack), fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c2color[cr],
                        markersize=7, label=f"cover={cr*100:.1f}%") for cr in covers]
        ax.legend(handles=elems, fontsize=8, loc="upper right")
    plt.tight_layout()
    fname = save_dir / "scatter_transfer_vs_stealth.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  图: {fname.name}")


def plot_comparison_all_metrics(df: pd.DataFrame, save_dir: Path):
    """3×3 综合图：攻击列 × 指标行，数据集着色。"""
    metrics_list = [
        ("asr",             "ASR"),
        ("transfer_rate",   "test-ASR"),
        ("stealth_auc_avg", "Stealth AUC"),
    ]
    ds_styles = {
        "cifar10":       ("#E74C3C", "o"),
        "tiny_imagenet": ("#3498DB", "s"),
        "mnistm":        ("#2ECC71", "^"),
    }
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle("cover_rate 对攻击性、迁移性与隐蔽性的综合影响", fontsize=14, fontweight="bold")
    for row_idx, (col, ylabel) in enumerate(metrics_list):
        for col_idx, attack in enumerate(COVER_ABLATION_ATTACKS):
            ax = axes[row_idx][col_idx]
            sub = df[df["attack_type"] == attack]
            if sub.empty or col not in sub.columns:
                ax.axis("off")
                continue
            for ds, (color, marker) in ds_styles.items():
                dsub = sub[sub["dataset"] == ds]
                agg = dsub.groupby("cover_rate")[col].agg(["mean", "sem"]).reset_index()
                if agg.empty:
                    continue
                ax.plot(agg["cover_rate"].values, agg["mean"].values,
                        color=color, marker=marker, linewidth=2, markersize=6,
                        label=DATASET_LABELS.get(ds, ds))
                ax.fill_between(
                    agg["cover_rate"].values,
                    (agg["mean"] - agg["sem"]).values,
                    (agg["mean"] + agg["sem"]).values,
                    color=color, alpha=0.15,
                )
            if row_idx == 0:
                ax.set_title(ATTACK_LABELS.get(attack, attack), fontweight="bold", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=11)
            if row_idx == 2:
                ax.set_xlabel("Cover Rate", fontsize=11)
            covers_vals = sorted(sub["cover_rate"].unique()) if not sub.empty else []
            if covers_vals:
                ax.set_xticks(covers_vals)
                ax.set_xticklabels(_cover_pct_labels(np.array(covers_vals)),
                                   rotation=30, ha="right", fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=9)
    plt.tight_layout()
    fname = save_dir / "comparison_all_metrics.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  图: {fname.name}")


def plot_cover_effect_per_dataset(df: pd.DataFrame, save_dir: Path):
    """每个数据集独立一张图，x=cover_rate，三条线代表三种攻击，y=test-ASR/stealth。"""
    for col, ylabel in [("transfer_rate", "test-ASR"), ("stealth_auc_avg", "Stealth AUC")]:
        if col not in df.columns:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        fig.suptitle(f"各数据集：cover_rate → {ylabel}（攻击对比）", fontsize=13, fontweight="bold")
        attack_colors = {"WaNet": "#E74C3C", "adaptive_blend": "#2ECC71", "adaptive_patch": "#3498DB"}
        attack_markers = {"WaNet": "o", "adaptive_blend": "s", "adaptive_patch": "^"}
        for ax, ds in zip(axes, ["cifar10", "tiny_imagenet", "mnistm"]):
            sub_ds = df[df["dataset"] == ds]
            for attack in COVER_ABLATION_ATTACKS:
                asub = sub_ds[sub_ds["attack_type"] == attack]
                agg = asub.groupby("cover_rate")[col].agg(["mean", "sem"]).reset_index()
                if agg.empty:
                    continue
                ax.errorbar(
                    agg["cover_rate"], agg["mean"], yerr=agg["sem"],
                    color=attack_colors.get(attack, "gray"),
                    marker=attack_markers.get(attack, "o"),
                    linewidth=2, markersize=6, capsize=3,
                    label=ATTACK_LABELS.get(attack, attack),
                )
            ax.set_title(DATASET_LABELS.get(ds, ds), fontweight="bold")
            ax.set_xlabel("Cover Rate")
            if ax == axes[0]:
                ax.set_ylabel(ylabel)
            covers_vals = sorted(sub_ds["cover_rate"].unique()) if not sub_ds.empty else []
            if covers_vals:
                ax.set_xticks(covers_vals)
                ax.set_xticklabels(_cover_pct_labels(np.array(covers_vals)), rotation=30, ha="right")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        plt.tight_layout()
        fname = save_dir / f"per_dataset_{col}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print(f"  图: {fname.name}")


# ─────────────────────── 主程序 ───────────────────────

def main():
    parser = argparse.ArgumentParser(description="cover-rate 独立消融实验综合分析")
    parser.add_argument("--min-asr", type=float, default=0.0,
                        help="过滤 ASR 低于此阈值的点（默认 0.0，不过滤）")
    parser.add_argument("--skip-plots", action="store_true", help="跳过图表生成")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 加载数据 ===")
    df = load_data(min_asr=args.min_asr)
    print(f"共 {len(df)} 条记录 | 攻击: {sorted(df.attack_type.unique())}")
    for at in sorted(df.attack_type.unique()):
        cr = sorted(df[df.attack_type == at]["cover_rate"].dropna().unique())
        print(f"  {at}: cover_rate 值 = {[round(v, 4) for v in cr]}")

    # ── 统计汇总表 ──
    print("\n=== 生成统计汇总表 ===")
    summary = build_cover_rate_summary(df)
    summary.to_csv(TABLES_DIR / "cover_rate_summary.csv", index=False)
    print("  表: cover_rate_summary.csv")
    wide = build_wide_summary(df)
    wide.to_csv(TABLES_DIR / "cover_rate_wide.csv", index=False)
    print("  表: cover_rate_wide.csv")

    # ── 相关性分析 ──
    print("\n=== 相关性分析 ===")
    corr = compute_correlations(df)
    corr.to_csv(TABLES_DIR / "cover_rate_correlations.csv", index=False)
    print("  表: cover_rate_correlations.csv")
    print(corr.to_string(index=False))

    # ── 核心均值统计（打印用于快速核查）──
    print("\n=== 各攻击 × cover_rate 均值（全数据集汇总）===")
    g = df.groupby(["attack_type", "cover_rate"])[["asr", "transfer_rate", "stealth_auc_avg"]].mean().round(3)
    print(g.to_string())

    # ── 图表 ──
    if not args.skip_plots:
        print("\n=== 生成图表 ===")
        plot_trend_mean_ci(df, FIGURES_DIR)
        plot_trend_by_arch(df, FIGURES_DIR)
        plot_boxplot_by_cover(df, FIGURES_DIR)
        plot_scatter_asr_vs_transfer(df, FIGURES_DIR)
        plot_scatter_transfer_vs_stealth(df, FIGURES_DIR)
        plot_comparison_all_metrics(df, FIGURES_DIR)
        plot_cover_effect_per_dataset(df, FIGURES_DIR)

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
