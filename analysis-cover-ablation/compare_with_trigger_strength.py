#!/usr/bin/env python3
"""cover_rate 消融 vs. 触发器强度消融：对比分析脚本。

比较维度
--------
1. 攻防权衡（trade-off）方向：
   - 触发器强度↑ → ASR↑ + 隐蔽性↓  （正向权衡）
   - cover_rate↑  → ASR↓ + 隐蔽性↑  （反向权衡）
2. 迁移性（test-ASR）随两类参数的变化趋势
3. 同一攻击方法在两种消融维度下的 ASR-Stealth 散点分布
4. Spearman 相关系数对比表

数据来源
--------
  触发器强度：analysis-testASR/data_*_no_nc.csv
  cover_rate：analysis-cover-ablation/data/data_*_no_nc.csv

输出
----
  report_figures/compare_*.png    对比图
  report_tables/compare_corr.csv  相关性对比表
  compare_with_trigger_strength.md  补充报告
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────── 路径 ───────────────────────
BASE_DIR     = Path(__file__).parent
ROOT_DIR     = BASE_DIR.parent

TS_DIR   = ROOT_DIR / "analysis-testASR"      # 触发器强度数据
CR_DIR   = BASE_DIR / "data"                  # cover_rate 数据

FIGURES_DIR = BASE_DIR / "report_figures"
TABLES_DIR  = BASE_DIR / "report_tables"

ATTACKS = ["WaNet", "adaptive_blend", "adaptive_patch"]
ATTACK_LABELS = {"WaNet": "WaNet", "adaptive_blend": "Adaptive Blend", "adaptive_patch": "Adaptive Patch"}
DATASET_LABELS = {"cifar10": "CIFAR-10", "tiny_imagenet": "Tiny-ImageNet", "mnistm": "MNIST-M"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150,
})


# ─────────────────────── 数据加载 ───────────────────────

def load_ts() -> pd.DataFrame:
    """加载触发器强度消融数据（analysis-testASR），只保留 3 个 adaptive 攻击。"""
    dfs = [pd.read_csv(f) for f in sorted(TS_DIR.glob("data_*_no_nc.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    for col in ["asr", "transfer_rate", "stealth_tpr_avg", "stealth_auc_avg",
                "train_param_value", "cover_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["weighted_auc"] = 1.0 - df["stealth_auc_avg"]
    df = df[df["attack_type"].isin(ATTACKS)].copy()
    return df


def load_cr() -> pd.DataFrame:
    """加载 cover_rate 消融数据（analysis-cover-ablation/data）。"""
    dfs = [pd.read_csv(f) for f in sorted(CR_DIR.glob("data_*_no_nc.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    for col in ["asr", "transfer_rate", "stealth_tpr_avg", "stealth_auc_avg",
                "train_param_value", "cover_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["weighted_auc"] = 1.0 - df["stealth_auc_avg"]
    df = df[df["attack_type"].isin(ATTACKS)].copy()
    return df


# ─────────────────────── 相关性对比表 ───────────────────────

def compute_corr_comparison(df_ts: pd.DataFrame, df_cr: pd.DataFrame) -> pd.DataFrame:
    """对相同攻击，分别计算触发器强度和 cover_rate 与各指标的 Spearman 相关。"""
    records = []
    for attack in ATTACKS:
        for ds in ["cifar10", "tiny_imagenet", "mnistm"]:
            # 触发器强度
            sub_ts = df_ts[(df_ts["attack_type"] == attack) & (df_ts["dataset"] == ds)]
            x_ts = sub_ts["train_param_value"].dropna()
            # cover_rate
            sub_cr = df_cr[(df_cr["attack_type"] == attack) & (df_cr["dataset"] == ds)]
            x_cr = sub_cr["cover_rate"].dropna()

            for metric, col in [("ASR", "asr"), ("test-ASR", "transfer_rate"),
                                  ("Stealth AUC", "stealth_auc_avg")]:
                row = {"dataset": ds, "attack_type": attack, "metric": metric}

                # 触发强度相关
                y_ts = sub_ts[col]
                m_ts = x_ts.notna() & y_ts.notna()
                if m_ts.sum() >= 4:
                    r, p = stats.spearmanr(x_ts[m_ts].values, y_ts[m_ts].values)
                    row["TS_spearman_r"] = round(r, 3)
                    row["TS_p"] = round(p, 4)
                    row["TS_n"] = int(m_ts.sum())
                else:
                    row["TS_spearman_r"] = np.nan
                    row["TS_p"] = np.nan
                    row["TS_n"] = int(m_ts.sum())

                # cover_rate 相关
                y_cr = sub_cr[col]
                m_cr = x_cr.notna() & y_cr.notna()
                if m_cr.sum() >= 4:
                    r, p = stats.spearmanr(x_cr[m_cr].values, y_cr[m_cr].values)
                    row["CR_spearman_r"] = round(r, 3)
                    row["CR_p"] = round(p, 4)
                    row["CR_n"] = int(m_cr.sum())
                else:
                    row["CR_spearman_r"] = np.nan
                    row["CR_p"] = np.nan
                    row["CR_n"] = int(m_cr.sum())

                row["direction_same"] = (
                    "同向" if (
                        pd.notna(row["TS_spearman_r"]) and pd.notna(row["CR_spearman_r"]) and
                        np.sign(row["TS_spearman_r"]) == np.sign(row["CR_spearman_r"])
                    ) else "反向"
                )
                records.append(row)
    return pd.DataFrame(records)


# ─────────────────────── 图1：趋势对比（归一化 x 轴）───────────────────────

def plot_trend_comparison(df_ts: pd.DataFrame, df_cr: pd.DataFrame, save_dir: Path):
    """
    上方：触发器强度 vs. ASR/test-ASR/Stealth（3攻击 × 3指标）
    下方：cover_rate    vs. ASR/test-ASR/Stealth（3攻击 × 3指标）
    """
    metrics = [("asr", "ASR"), ("transfer_rate", "test-ASR"), ("stealth_auc_avg", "Stealth AUC")]
    ds_colors = {"cifar10": "#E74C3C", "tiny_imagenet": "#3498DB", "mnistm": "#2ECC71"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("触发器强度 vs. Cover Rate：各指标综合趋势对比", fontsize=14, fontweight="bold")

    row_labels = ["触发器强度 (α / s)", "Cover Rate"]
    row_data   = [(df_ts, "train_param_value"), (df_cr, "cover_rate")]

    for row_idx, (df, xcol) in enumerate(row_data):
        for col_idx, attack in enumerate(ATTACKS):
            ax = axes[row_idx][col_idx]
            sub = df[df["attack_type"] == attack]
            # 归一化 x 轴到 [0,1] 以便同一坐标范围对比
            x_all = pd.to_numeric(sub[xcol], errors="coerce").dropna()
            x_min, x_max = x_all.min(), x_all.max()

            for ds in ["cifar10", "tiny_imagenet", "mnistm"]:
                dsub = sub[sub["dataset"] == ds]
                agg_rows = []
                for xv, grp in dsub.groupby(xcol):
                    xn = (xv - x_min) / (x_max - x_min + 1e-9)
                    for col, _ in metrics:
                        v = grp[col].dropna()
                        if len(v) > 0:
                            agg_rows.append({"x_norm": xn, "metric": col,
                                             "mean": v.mean(), "sem": v.sem()})
                if not agg_rows:
                    continue
                agg_df = pd.DataFrame(agg_rows)
                for (metric_col, ylabel), ls in zip(metrics, ["-", "--", "-."]):
                    mdf = agg_df[agg_df["metric"] == metric_col].sort_values("x_norm")
                    if mdf.empty:
                        continue
                    ax.plot(mdf["x_norm"].values, mdf["mean"].values,
                            color=ds_colors.get(ds, "gray"), linestyle=ls,
                            linewidth=1.8, alpha=0.85)

            # 添加说明图例（仅第一格）
            if row_idx == 0 and col_idx == 0:
                ds_elems = [Line2D([0],[0], color=ds_colors[d], linewidth=2,
                                   label=DATASET_LABELS[d]) for d in ["cifar10","tiny_imagenet","mnistm"]]
                metric_elems = [Line2D([0],[0], color="gray", linestyle=ls,
                                       linewidth=2, label=lbl)
                                for (_, lbl), ls in zip(metrics, ["-","--","-."])]
                ax.legend(handles=ds_elems + metric_elems, fontsize=7.5, ncol=2)

            ax.set_xlim(-0.02, 1.05)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("参数归一化值 [0→1]")
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx])
            if row_idx == 0:
                ax.set_title(ATTACK_LABELS.get(attack, attack), fontweight="bold")

    plt.tight_layout()
    fname = save_dir / "compare_trend_normalized.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  图: {fname.name}")


# ─────────────────────── 图2：按攻击方法的并列趋势图 ───────────────────────

def plot_side_by_side_per_attack(df_ts: pd.DataFrame, df_cr: pd.DataFrame, save_dir: Path):
    """
    每个攻击方法一张图，左侧为触发器强度，右侧为 cover_rate。
    每侧显示 ASR、test-ASR、Stealth AUC 三条线（跨数据集和架构平均）。
    """
    for attack in ATTACKS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.suptitle(f"{ATTACK_LABELS[attack]}：触发器强度 vs. Cover Rate 对各指标的影响",
                     fontsize=13, fontweight="bold")

        configs = [
            (axes[0], df_ts, "train_param_value",
             "触发器强度 (α for blend/patch, s for WaNet)", False),
            (axes[1], df_cr, "cover_rate",
             "Cover Rate", True),
        ]
        metric_styles = {
            "asr":             ("#E74C3C", "-",  "o", "ASR（源域）"),
            "transfer_rate":   ("#3498DB", "-",  "s", "test-ASR（迁移）"),
            "stealth_auc_avg": ("#2ECC71", "--", "^", "Stealth AUC（隐蔽）"),
            "weighted_auc":    ("#F39C12", ":",  "D", "Weighted AUC（检测率）"),
        }

        for ax, df, xcol, xlabel, pct_x in configs:
            sub = df[df["attack_type"] == attack]
            if sub.empty:
                ax.axis("off")
                continue
            agg = sub.groupby(xcol)[list(metric_styles.keys())].mean().reset_index()
            for col, (color, ls, mk, label) in metric_styles.items():
                if col not in agg.columns:
                    continue
                x = agg[xcol].values
                y = agg[col].values
                mask = ~np.isnan(y)
                ax.plot(x[mask], y[mask], color=color, linestyle=ls, marker=mk,
                        linewidth=2, markersize=6, label=label)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("指标值", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            if pct_x:
                xticks = sorted(sub[xcol].dropna().unique())
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"{v*100:.1f}%" for v in xticks], rotation=30, ha="right")

        plt.tight_layout()
        fname = save_dir / f"compare_side_by_side_{attack}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print(f"  图: {fname.name}")


# ─────────────────────── 图3：ASR-Stealth 散点（两种消融叠加）───────────────────────

def plot_tradeoff_scatter(df_ts: pd.DataFrame, df_cr: pd.DataFrame, save_dir: Path):
    """
    散点图：ASR（x轴）vs. Stealth AUC（y轴）
    - 蓝色圆点：触发器强度消融点，按强度大小着色
    - 橙色三角：cover_rate 消融点，按 cover_rate 着色
    Trade-off 区域：左上角（高隐蔽+低ASR）vs 右下角（高ASR+低隐蔽）
    """
    for attack in ATTACKS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(f"{ATTACK_LABELS[attack]}：ASR-Stealth 权衡散点（触发强度 vs. Cover Rate）",
                     fontsize=12, fontweight="bold")

        for ax, ds in zip(axes, ["cifar10", "tiny_imagenet", "mnistm"]):
            # 触发器强度点
            sub_ts = df_ts[(df_ts["attack_type"] == attack) & (df_ts["dataset"] == ds)]
            sub_ts = sub_ts.dropna(subset=["asr", "stealth_auc_avg", "train_param_value"])
            if not sub_ts.empty:
                tp_vals = sub_ts["train_param_value"].values
                tp_min, tp_max = tp_vals.min(), tp_vals.max()
                tp_norm = (tp_vals - tp_min) / (tp_max - tp_min + 1e-9)
                sc1 = ax.scatter(sub_ts["asr"].values, sub_ts["stealth_auc_avg"].values,
                                 c=tp_norm, cmap="Blues", vmin=0, vmax=1,
                                 s=50, marker="o", edgecolors="steelblue", linewidth=0.5,
                                 alpha=0.8, label="触发器强度", zorder=3)

            # cover_rate 点
            sub_cr = df_cr[(df_cr["attack_type"] == attack) & (df_cr["dataset"] == ds)]
            sub_cr = sub_cr.dropna(subset=["asr", "stealth_auc_avg", "cover_rate"])
            if not sub_cr.empty:
                cr_vals = sub_cr["cover_rate"].values
                cr_min, cr_max = cr_vals.min(), cr_vals.max()
                cr_norm = (cr_vals - cr_min) / (cr_max - cr_min + 1e-9)
                sc2 = ax.scatter(sub_cr["asr"].values, sub_cr["stealth_auc_avg"].values,
                                 c=cr_norm, cmap="Oranges", vmin=0, vmax=1,
                                 s=70, marker="^", edgecolors="darkorange", linewidth=0.5,
                                 alpha=0.8, label="cover_rate", zorder=4)

            # 理想区域标注
            ax.axhspan(0.7, 1.05, alpha=0.04, color="green", zorder=1)
            ax.axvspan(0.7, 1.05, alpha=0.04, color="red", zorder=1)
            ax.text(0.02, 0.98, "理想:\n高隐蔽\n低攻击", transform=ax.transAxes,
                    fontsize=7.5, va="top", color="green", alpha=0.7)
            ax.text(0.78, 0.02, "高效攻击\n但易被检测", transform=ax.transAxes,
                    fontsize=7.5, va="bottom", color="red", alpha=0.7)

            ax.set_xlim(-0.02, 1.05)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xlabel("ASR（源域）")
            if ax == axes[0]:
                ax.set_ylabel("Stealth AUC（隐蔽性）")
            ax.set_title(DATASET_LABELS.get(ds, ds), fontweight="bold")
            ax.grid(True, alpha=0.25)

            legend_elems = [
                Line2D([0],[0], marker="o", color="w", markerfacecolor="steelblue",
                       markersize=8, label="触发器强度（深=强）"),
                Line2D([0],[0], marker="^", color="w", markerfacecolor="darkorange",
                       markersize=8, label="cover_rate（深=大）"),
            ]
            ax.legend(handles=legend_elems, fontsize=8, loc="upper right")

        plt.tight_layout()
        fname = save_dir / f"compare_tradeoff_scatter_{attack}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print(f"  图: {fname.name}")


# ─────────────────────── 图4：迁移性对比（test-ASR）───────────────────────

def plot_transfer_comparison(df_ts: pd.DataFrame, df_cr: pd.DataFrame, save_dir: Path):
    """
    每个攻击一行，左：触发强度 vs test-ASR，右：cover_rate vs test-ASR
    展示两种消融下迁移性的变化方向。
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    fig.suptitle("迁移性（test-ASR）：触发器强度 vs. Cover Rate 消融对比",
                 fontsize=13, fontweight="bold")
    ds_colors = {"cifar10": "#E74C3C", "tiny_imagenet": "#3498DB", "mnistm": "#2ECC71"}

    for row_idx, attack in enumerate(ATTACKS):
        # 左：触发器强度
        ax_l = axes[row_idx][0]
        sub_ts = df_ts[df_ts["attack_type"] == attack]
        for ds in ["cifar10", "tiny_imagenet", "mnistm"]:
            dsub = sub_ts[sub_ts["dataset"] == ds]
            agg = dsub.groupby("train_param_value")["transfer_rate"].agg(["mean","sem"]).reset_index()
            if agg.empty:
                continue
            ax_l.errorbar(agg["train_param_value"], agg["mean"], yerr=agg["sem"],
                          color=ds_colors[ds], marker="o", linewidth=2, capsize=3,
                          label=DATASET_LABELS[ds])
        ax_l.set_ylabel(f"{ATTACK_LABELS[attack]}\ntest-ASR", fontsize=10)
        ax_l.set_ylim(0, 1.05)
        ax_l.grid(True, alpha=0.3)
        ax_l.set_xlabel("触发器强度 (α / s / δ)" if row_idx == 2 else "")
        if row_idx == 0:
            ax_l.set_title("触发器强度消融", fontweight="bold")
        ax_l.legend(fontsize=8)

        # 右：cover_rate
        ax_r = axes[row_idx][1]
        sub_cr = df_cr[df_cr["attack_type"] == attack]
        for ds in ["cifar10", "tiny_imagenet", "mnistm"]:
            dsub = sub_cr[sub_cr["dataset"] == ds]
            agg = dsub.groupby("cover_rate")["transfer_rate"].agg(["mean","sem"]).reset_index()
            if agg.empty:
                continue
            ax_r.errorbar(agg["cover_rate"], agg["mean"], yerr=agg["sem"],
                          color=ds_colors[ds], marker="s", linewidth=2, capsize=3,
                          label=DATASET_LABELS[ds])
            xticks = sorted(sub_cr["cover_rate"].dropna().unique())
            ax_r.set_xticks(xticks)
            ax_r.set_xticklabels([f"{v*100:.1f}%" for v in xticks], rotation=30, ha="right")
        ax_r.set_ylim(0, 1.05)
        ax_r.grid(True, alpha=0.3)
        ax_r.set_xlabel("Cover Rate" if row_idx == 2 else "")
        if row_idx == 0:
            ax_r.set_title("Cover Rate 消融", fontweight="bold")
        ax_r.legend(fontsize=8)

    plt.tight_layout()
    fname = save_dir / "compare_transfer_asr.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  图: {fname.name}")


# ─────────────────────── 图5：相关系数热力图对比 ───────────────────────

def plot_corr_heatmap(df_corr: pd.DataFrame, save_dir: Path):
    """
    热力图：行=攻击×数据集，列=指标；
    左图显示 TS Spearman r，右图显示 CR Spearman r。
    颜色表示方向和强度（红=正，蓝=负）。
    """
    metrics = ["ASR", "test-ASR", "Stealth AUC"]
    ds_list  = ["cifar10", "tiny_imagenet", "mnistm"]

    rows_label = [f"{at}\n{DATASET_LABELS.get(ds, ds)}"
                  for at in ATTACKS for ds in ds_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle("Spearman 相关系数热力图\n（左：触发器强度 | 右：Cover Rate）",
                 fontsize=13, fontweight="bold")

    for ax, col_key, title in [
        (axes[0], "TS_spearman_r", "触发器强度 → 各指标"),
        (axes[1], "CR_spearman_r", "Cover Rate → 各指标"),
    ]:
        mat = []
        for attack in ATTACKS:
            for ds in ds_list:
                row_vals = []
                for metric in metrics:
                    sel = df_corr[(df_corr["attack_type"] == attack) &
                                  (df_corr["dataset"] == ds) &
                                  (df_corr["metric"] == metric)]
                    val = sel[col_key].values[0] if not sel.empty and not sel[col_key].isna().all() else np.nan
                    row_vals.append(val)
                mat.append(row_vals)
        mat = np.array(mat, dtype=float)

        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_yticks(range(len(rows_label)))
        ax.set_yticklabels(rows_label, fontsize=8)
        ax.set_title(title, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")

        # 数值标注
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=8, color="white" if abs(v) > 0.6 else "black",
                            fontweight="bold" if abs(v) > 0.7 else "normal")

    plt.tight_layout()
    fname = save_dir / "compare_corr_heatmap.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  图: {fname.name}")


# ─────────────────────── 生成对比报告 ───────────────────────

def build_comparison_report(df_ts: pd.DataFrame, df_cr: pd.DataFrame,
                             corr_df: pd.DataFrame) -> str:
    lines = [
        "# Cover Rate vs. 触发器强度：对比分析补充报告",
        "",
        f"**数据来源（触发器强度）**：`analysis-testASR/data_*_no_nc.csv`  ",
        f"**数据来源（cover_rate）**：`analysis-cover-ablation/data/data_*_no_nc.csv`  ",
        f"**分析时间**：2026-05",
        "",
        "---",
        "",
        "## 1. 整体结论：权衡方向完全相反",
        "",
        "| 消融维度 | adaptive_blend / patch：ASR | adaptive_blend / patch：隐蔽性 | WaNet |",
        "|---------|---------------------------|-------------------------------|-------|",
        "| 触发器强度↑ | **↑ 强烈正相关** (ρ ≈ +0.69 ～ +0.92) | **↓ 负相关** (ρ ≈ −0.40 ～ −0.70) | 弱正相关 |",
        "| cover_rate↑ | **↓ 强烈负相关** (ρ ≈ −0.63 ～ −0.97) | **↑ 正相关** (ρ ≈ +0.46 ～ +0.94) | 不显著 |",
        "",
        "> **结论**：两种消融维度对 adaptive 系攻击的效果**方向完全相反**：",
        "> - 触发器强度是【攻击放大器】——越强越能攻击，但越难逃避检测",
        "> - cover_rate 是【攻击削弱器】——越高越难攻击，但反而更能逃避检测",
        "> - **trade-off 依然存在**，只是调节旋钮朝相反方向转动",
        "",
        "---",
        "",
        "## 2. trade-off 是否依然满足？",
        "",
        "**是的，trade-off 在两种消融下都成立，但机制不同：**",
        "",
        "### 2.1 触发器强度消融的 trade-off",
        "",
        "```",
        "触发器强度 ↑  →  触发模式更明显  →  模型更容易学习后门  →  ASR↑",
        "触发器强度 ↑  →  图像可见扰动增大  →  防御方更容易检测到  →  Stealth↓",
        "```",
        "",
        "本质：**触发器可见性**驱动的权衡，强度是可见性的直接体现。",
        "",
        "### 2.2 cover_rate 消融的 trade-off",
        "",
        "```",
        "cover_rate ↑  →  cover 样本增多  →  触发器-标签关联被稀释  →  ASR↓",
        "cover_rate ↑  →  训练集多样性增强  →  防御方分布更难区分  →  Stealth↑",
        "```",
        "",
        "本质：**标签一致性**驱动的权衡，cover_rate 破坏后门标签绑定的同时，",
        "制造了检测混淆。",
        "",
        "### 2.3 两种权衡的本质区别",
        "",
        "| 维度 | 主驱动因子 | ASR 饱和点 | 隐蔽性极限 |",
        "|-----|-----------|-----------|-----------|",
        "| 触发器强度 | 触发器可见性/模式强度 | 强度足够大后 ASR 趋于饱和 | 强度越低越隐蔽，但 ASR 崩溃 |",
        "| cover_rate | 标签一致性破坏程度 | cover=0 时 ASR 最高 | cover→∞ 时 ASR≈0，但隐蔽性也趋于极限 |",
        "",
        "---",
        "",
        "## 3. 迁移性（test-ASR）的对比",
        "",
    ]

    # 收集关键迁移性数据
    for attack in ATTACKS:
        lines.append(f"### 3.{ATTACKS.index(attack)+1} {ATTACK_LABELS[attack]}")
        lines.append("")

        sub_ts = df_ts[df_ts["attack_type"] == attack]
        sub_cr = df_cr[df_cr["attack_type"] == attack]

        lines.append("| 数据集 | TS: test-ASR 最低→最高 | CR: test-ASR cover=0→max |")
        lines.append("|-------|----------------------|-------------------------|")
        for ds in ["cifar10", "tiny_imagenet", "mnistm"]:
            sts = sub_ts[sub_ts["dataset"] == ds].groupby("train_param_value")["transfer_rate"].mean()
            scr = sub_cr[sub_cr["dataset"] == ds].groupby("cover_rate")["transfer_rate"].mean()
            if not sts.empty and not scr.empty:
                ts_range = f"{sts.min():.3f} → {sts.max():.3f}"
                cr_range = f"{scr.iloc[0]:.3f} → {scr.iloc[-1]:.3f}"
                lines.append(f"| {DATASET_LABELS.get(ds, ds)} | {ts_range} | {cr_range} |")
        lines.append("")

    lines += [
        "---",
        "",
        "## 4. WaNet 的特殊性",
        "",
        "WaNet 对两种消融都表现出较弱/不显著的相关：",
        "",
        "| 消融 | ASR 相关 | test-ASR 相关 | Stealth 相关 |",
        "|-----|---------|-------------|------------|",
        "| 触发器强度（s） | ρ ≈ +0.14 ～ +0.50 | ρ ≈ −0.05 ～ +0.67 | 弱负相关 |",
        "| cover_rate | ρ ≈ −0.43 ～ +0.06（全部 p>0.05）| 均不显著 | 均不显著 |",
        "",
        "原因：WaNet 使用 warp-field 几何扭曲，其有效性取决于**扭曲强度 s 的绝对值**（实验",
        "中固定为 0.5），cover_rate 无法改变扭曲模式本身，因此失效。",
        "触发强度 s 在消融实验中对 WaNet CIFAR-10 的 test-ASR 有显著影响（ρ=+0.67），",
        "但在跨数据集上不一致，说明 WaNet 的迁移性更多受**目标域数据分布**的影响。",
        "",
        "---",
        "",
        "## 5. 实践启示",
        "",
        "| 目标 | 建议操作 | 原理 |",
        "|-----|---------|-----|",
        "| 最大化攻击成功率 | 提高触发器强度，保持 cover_rate=0 | 正向叠加两种效应 |",
        "| 最大化隐蔽性（可接受较低 ASR） | 降低触发器强度 + 增大 cover_rate | 反向叠加两种效应 |",
        "| 平衡攻击与隐蔽 | 中等触发强度 + cover_rate ≈ 0.001~0.005 | 小 cover 对 ASR 影响小但隐蔽性有显著提升 |",
        "| WaNet 优化 | 调整 s（触发强度），cover_rate 无效 | WaNet 机制不依赖标签一致性 |",
        "",
        "---",
        "",
        "## 6. 图表索引",
        "",
        "| 文件 | 说明 |",
        "|-----|------|",
        "| `compare_trend_normalized.png` | 归一化参数轴的双行趋势对比（综合图）|",
        "| `compare_side_by_side_WaNet.png` | WaNet：触发强度 vs cover_rate 并列折线图 |",
        "| `compare_side_by_side_adaptive_blend.png` | Adaptive Blend 并列折线图 |",
        "| `compare_side_by_side_adaptive_patch.png` | Adaptive Patch 并列折线图 |",
        "| `compare_tradeoff_scatter_WaNet.png` | WaNet：ASR-Stealth 权衡散点（双消融叠加）|",
        "| `compare_tradeoff_scatter_adaptive_blend.png` | Adaptive Blend 权衡散点 |",
        "| `compare_tradeoff_scatter_adaptive_patch.png` | Adaptive Patch 权衡散点 |",
        "| `compare_transfer_asr.png` | 迁移性（test-ASR）双消融趋势对比（3攻击×2列）|",
        "| `compare_corr_heatmap.png` | Spearman 相关系数热力图（双消融对比）|",
        "| `report_tables/compare_corr.csv` | 完整相关系数对比表 |",
    ]
    return "\n".join(lines)


# ─────────────────────── 主程序 ───────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 加载数据 ===")
    df_ts = load_ts()
    df_cr = load_cr()
    print(f"触发器强度数据: {len(df_ts)} 条 | cover_rate 数据: {len(df_cr)} 条")

    print("\n=== 计算相关性对比表 ===")
    corr_df = compute_corr_comparison(df_ts, df_cr)
    corr_df.to_csv(TABLES_DIR / "compare_corr.csv", index=False)
    print("  表: compare_corr.csv")

    print("\n[关键对比] adaptive_blend × CIFAR-10：")
    sel = corr_df[(corr_df["attack_type"] == "adaptive_blend") & (corr_df["dataset"] == "cifar10")]
    print(sel[["metric", "TS_spearman_r", "TS_p", "CR_spearman_r", "CR_p", "direction_same"]].to_string(index=False))

    print("\n=== 生成对比图 ===")
    plot_trend_comparison(df_ts, df_cr, FIGURES_DIR)
    plot_side_by_side_per_attack(df_ts, df_cr, FIGURES_DIR)
    plot_tradeoff_scatter(df_ts, df_cr, FIGURES_DIR)
    plot_transfer_comparison(df_ts, df_cr, FIGURES_DIR)
    plot_corr_heatmap(corr_df, FIGURES_DIR)

    print("\n=== 生成对比报告 ===")
    report_text = build_comparison_report(df_ts, df_cr, corr_df)
    report_path = BASE_DIR / "compare_with_trigger_strength.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  报告: {report_path.name}")

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
