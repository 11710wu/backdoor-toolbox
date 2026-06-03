#!/usr/bin/env python3
"""按攻击方法绘制 cover_rate 折线图：每张图 9 条曲线（3 数据集 × 3 架构）。

输出
----
  report_figures/lines_test_asr_{attack}.png   横轴 cover_rate，纵轴 test-ASR
  report_figures/lines_stealth_{attack}.png      横轴 cover_rate，纵轴 Stealth AUC
  report_tables/lines_detail_{attack}.csv        各组合明细表
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "report_figures"
TABLES_DIR = BASE_DIR / "report_tables"

ATTACKS = ["WaNet", "adaptive_blend", "adaptive_patch"]
ATTACK_LABELS = {
    "WaNet": "WaNet",
    "adaptive_blend": "Adaptive Blend",
    "adaptive_patch": "Adaptive Patch",
}
DATASETS = ["cifar10", "tiny_imagenet", "mnistm"]
ARCHS = ["resnet18", "mobilenet", "vgg"]
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "tiny_imagenet": "Tiny-ImageNet",
    "mnistm": "MNIST-M",
}
ARCH_LABELS = {
    "resnet18": "ResNet-18",
    "mobilenet": "MobileNetV2",
    "vgg": "VGG-19",
}

# 9 条曲线：数据集用线型区分，架构用颜色区分
DS_STYLES = {
    "cifar10": ("-", "o"),
    "tiny_imagenet": ("--", "s"),
    "mnistm": ("-.", "^"),
}
ARCH_COLORS = {
    "resnet18": "#E74C3C",
    "mobilenet": "#3498DB",
    "vgg": "#2ECC71",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
})


def load_data() -> pd.DataFrame:
    dfs = [pd.read_csv(f) for f in sorted(DATA_DIR.glob("data_*_no_nc.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    for col in ["cover_rate", "transfer_rate", "stealth_auc_avg", "stealth_tpr_avg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["attack_type"].isin(ATTACKS)].copy()


def _cover_pct_labels(covers: np.ndarray) -> list[str]:
    return [f"{v * 100:g}%" for v in covers]


def build_detail_table(df: pd.DataFrame, attack: str) -> pd.DataFrame:
    sub = df[df["attack_type"] == attack]
    rows = []
    for ds in DATASETS:
        for arch in ARCHS:
            combo = sub[(sub["dataset"] == ds) & (sub["arch"] == arch)]
            for cr, grp in combo.groupby("cover_rate"):
                rows.append({
                    "attack_type": attack,
                    "dataset": ds,
                    "arch": arch,
                    "cover_rate": cr,
                    "test_asr": grp["transfer_rate"].iloc[0] if len(grp) == 1 else grp["transfer_rate"].mean(),
                    "stealth_auc": grp["stealth_auc_avg"].iloc[0] if len(grp) == 1 else grp["stealth_auc_avg"].mean(),
                    "stealth_tpr": grp["stealth_tpr_avg"].iloc[0] if len(grp) == 1 else grp["stealth_tpr_avg"].mean(),
                })
    out = pd.DataFrame(rows)
    return out.sort_values(["dataset", "arch", "cover_rate"]).round(4)


def plot_one_metric(df: pd.DataFrame, attack: str, ycol: str, ylabel: str, fname: str):
    sub = df[df["attack_type"] == attack].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        f"{ATTACK_LABELS[attack]}: Cover Rate vs {ylabel}\n(9 curves: 3 datasets x 3 architectures)",
        fontsize=13,
        fontweight="bold",
    )

    legend_elems: list[Line2D] = []
    for ds in DATASETS:
        ls, mk = DS_STYLES[ds]
        for arch in ARCHS:
            combo = sub[(sub["dataset"] == ds) & (sub["arch"] == arch)].sort_values("cover_rate")
            if combo.empty:
                continue
            x = combo["cover_rate"].values
            y = combo[ycol].values
            color = ARCH_COLORS[arch]
            label = f"{DATASET_LABELS[ds]} · {ARCH_LABELS[arch]}"
            ax.plot(x, y, color=color, linestyle=ls, marker=mk,
                    linewidth=2, markersize=6, label=label)
            legend_elems.append(
                Line2D([0], [0], color=color, linestyle=ls, marker=mk,
                       linewidth=2, markersize=6, label=label)
            )

    covers = sorted(sub["cover_rate"].dropna().unique())
    ax.set_xticks(covers)
    ax.set_xticklabels(_cover_pct_labels(np.array(covers)), rotation=30, ha="right")
    ax.set_xlabel("Cover Rate")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax.legend(handles=legend_elems, loc="best", ncol=2,
              title="dataset · arch (color=arch, linestyle=dataset)", framealpha=0.92)

    plt.tight_layout()
    out = FIGURES_DIR / fname
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  图: {out.name}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"加载 {len(df)} 条记录")

    for attack in ATTACKS:
        detail = build_detail_table(df, attack)
        detail.to_csv(TABLES_DIR / f"lines_detail_{attack}.csv", index=False)
        print(f"  表: lines_detail_{attack}.csv ({len(detail)} 行)")

        plot_one_metric(
            df, attack, "transfer_rate", "test-ASR",
            f"lines_test_asr_{attack}.png",
        )
        plot_one_metric(
            df, attack, "stealth_auc_avg", "Stealth AUC",
            f"lines_stealth_{attack}.png",
        )

    print("完成。")


if __name__ == "__main__":
    main()
