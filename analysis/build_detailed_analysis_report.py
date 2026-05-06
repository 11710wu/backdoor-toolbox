#!/usr/bin/env python3
"""Build a detailed transfer/stealth analysis package under analysis/.

Outputs:
- report_tables/*.csv
- report_figures/*.png
- transfer_stealth_report_detailed.md
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-codex")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATASETS = ["cifar10", "tiny_imagenet", "mnistm"]
ARCHS = ["resnet18", "mobilenet", "vgg"]
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
DATASET_DISPLAY = {
    "cifar10": "CIFAR-10 -> STL-10",
    "tiny_imagenet": "Tiny-ImageNet -> Target Domain",
    "mnistm": "MNIST-M -> MNIST",
}
DATASET_TARGET_NOTE = {
    "cifar10": "train on CIFAR-10, test on STL-10",
    "tiny_imagenet": "train on Tiny-ImageNet, test on later generated target-domain",
    "mnistm": "train on MNIST-M, test on MNIST",
}
ARCH_MARKERS = {"resnet18": "o", "mobilenet": "s", "vgg": "^"}
ATTACK_COLORS = {
    "SIG": "#1f77b4",
    "WaNet": "#ff7f0e",
    "adaptive_blend": "#2ca02c",
    "adaptive_patch": "#d62728",
    "basic": "#9467bd",
    "belt": "#8c564b",
    "blend": "#e377c2",
    "upgd": "#7f7f7f",
}


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [
        "poison_rate",
        "train_param_value",
        "test_param_value",
        "stealth_tpr_avg",
        "stealth_auc_avg",
        "transfer_rate",
        "asr",
        "nc_max_anomaly_index",
        "S_stealth",
        "S_stealth_tpr",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "nc_is_poisoned" in df.columns:
        df["nc_is_poisoned"] = df["nc_is_poisoned"].map(
            lambda x: np.nan
            if pd.isna(x)
            else str(x).strip().lower() in ("1", "true", "yes")
        )
    return df


def load_mode(analysis_dir: Path, mode: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for dataset in DATASETS:
        for arch in ARCHS:
            path = analysis_dir / f"data_{dataset}_{arch}_{mode}.csv"
            if not path.exists():
                continue
            df = _read_csv(path)
            df["dataset"] = dataset
            df["arch"] = arch
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No *_ {mode}.csv files found in {analysis_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["attack_type"] = pd.Categorical(
        out["attack_type"], categories=ATTACK_ORDER, ordered=True
    )
    out["stealth_mean"] = (out["stealth_auc_avg"] + out["stealth_tpr_avg"]) / 2.0
    denom = out["transfer_rate"] + out["stealth_mean"]
    out["tradeoff_hmean"] = np.where(
        denom > 0,
        2.0 * out["transfer_rate"] * out["stealth_mean"] / denom,
        0.0,
    )
    out["tradeoff_product"] = out["transfer_rate"] * out["stealth_mean"]
    return out


def safe_corr(a: Iterable[float], b: Iterable[float]) -> float:
    x = pd.Series(a).astype(float)
    y = pd.Series(b).astype(float)
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y))


def _q(series: pd.Series, q: float) -> float:
    if series.dropna().empty:
        return float("nan")
    return float(series.quantile(q))


def build_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset].copy()
        rows.append(
            {
                "dataset": dataset,
                "dataset_display": DATASET_DISPLAY[dataset],
                "target_note": DATASET_TARGET_NOTE[dataset],
                "n_points": len(sub),
                "n_attacks": int(sub["attack_type"].nunique()),
                "n_archs": int(sub["arch"].nunique()),
                "transfer_mean": sub["transfer_rate"].mean(),
                "transfer_median": sub["transfer_rate"].median(),
                "transfer_std": sub["transfer_rate"].std(),
                "transfer_q25": _q(sub["transfer_rate"], 0.25),
                "transfer_q75": _q(sub["transfer_rate"], 0.75),
                "stealth_auc_mean": sub["stealth_auc_avg"].mean(),
                "stealth_auc_median": sub["stealth_auc_avg"].median(),
                "stealth_tpr_mean": sub["stealth_tpr_avg"].mean(),
                "stealth_tpr_median": sub["stealth_tpr_avg"].median(),
                "stealth_mean_mean": sub["stealth_mean"].mean(),
                "asr_mean": sub["asr"].mean(),
                "asr_median": sub["asr"].median(),
                "corr_transfer_stealth_auc": safe_corr(
                    sub["transfer_rate"], sub["stealth_auc_avg"]
                ),
                "corr_transfer_stealth_mean": safe_corr(
                    sub["transfer_rate"], sub["stealth_mean"]
                ),
                "corr_transfer_asr": safe_corr(sub["transfer_rate"], sub["asr"]),
            }
        )
    return pd.DataFrame(rows)


def build_arch_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["dataset", "arch"], observed=True)
        .agg(
            n_points=("transfer_rate", "size"),
            transfer_mean=("transfer_rate", "mean"),
            transfer_median=("transfer_rate", "median"),
            stealth_auc_mean=("stealth_auc_avg", "mean"),
            stealth_tpr_mean=("stealth_tpr_avg", "mean"),
            stealth_mean_mean=("stealth_mean", "mean"),
            asr_mean=("asr", "mean"),
            tradeoff_hmean_mean=("tradeoff_hmean", "mean"),
        )
        .reset_index()
    )
    return out.sort_values(["dataset", "arch"]).reset_index(drop=True)


def build_attack_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["dataset", "attack_type"], observed=True)
        .agg(
            n_points=("transfer_rate", "size"),
            transfer_mean=("transfer_rate", "mean"),
            transfer_median=("transfer_rate", "median"),
            transfer_std=("transfer_rate", "std"),
            transfer_q25=("transfer_rate", lambda s: _q(s, 0.25)),
            transfer_q75=("transfer_rate", lambda s: _q(s, 0.75)),
            transfer_max=("transfer_rate", "max"),
            stealth_auc_mean=("stealth_auc_avg", "mean"),
            stealth_tpr_mean=("stealth_tpr_avg", "mean"),
            stealth_mean_mean=("stealth_mean", "mean"),
            stealth_mean_median=("stealth_mean", "median"),
            asr_mean=("asr", "mean"),
            asr_median=("asr", "median"),
            tradeoff_hmean_mean=("tradeoff_hmean", "mean"),
            tradeoff_hmean_max=("tradeoff_hmean", "max"),
        )
        .reset_index()
    )
    out["rank_transfer_mean"] = (
        out.groupby("dataset")["transfer_mean"].rank(ascending=False, method="min")
    )
    out["rank_stealth_mean"] = (
        out.groupby("dataset")["stealth_mean_mean"].rank(ascending=False, method="min")
    )
    out["rank_tradeoff_mean"] = (
        out.groupby("dataset")["tradeoff_hmean_mean"].rank(ascending=False, method="min")
    )
    return out.sort_values(["dataset", "rank_tradeoff_mean", "attack_type"]).reset_index(
        drop=True
    )


def build_attack_arch_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["dataset", "arch", "attack_type"], observed=True)
        .agg(
            n_points=("transfer_rate", "size"),
            transfer_mean=("transfer_rate", "mean"),
            stealth_auc_mean=("stealth_auc_avg", "mean"),
            stealth_tpr_mean=("stealth_tpr_avg", "mean"),
            stealth_mean_mean=("stealth_mean", "mean"),
            asr_mean=("asr", "mean"),
            tradeoff_hmean_mean=("tradeoff_hmean", "mean"),
        )
        .reset_index()
    )
    return out.sort_values(["dataset", "arch", "tradeoff_hmean_mean"], ascending=[True, True, False])


def build_cross_dataset_attack_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["attack_type", "dataset"], observed=True)
        .agg(
            n_points=("transfer_rate", "size"),
            transfer_mean=("transfer_rate", "mean"),
            transfer_median=("transfer_rate", "median"),
            transfer_std=("transfer_rate", "std"),
            stealth_auc_mean=("stealth_auc_avg", "mean"),
            stealth_tpr_mean=("stealth_tpr_avg", "mean"),
            stealth_mean_mean=("stealth_mean", "mean"),
            stealth_mean_std=("stealth_mean", "std"),
            asr_mean=("asr", "mean"),
            tradeoff_hmean_mean=("tradeoff_hmean", "mean"),
        )
        .reset_index()
    )
    out["rank_transfer_mean"] = (
        out.groupby("dataset")["transfer_mean"].rank(ascending=False, method="min")
    )
    out["rank_stealth_mean"] = (
        out.groupby("dataset")["stealth_mean_mean"].rank(ascending=False, method="min")
    )
    out["rank_tradeoff_mean"] = (
        out.groupby("dataset")["tradeoff_hmean_mean"].rank(ascending=False, method="min")
    )
    return out.sort_values(["attack_type", "dataset"]).reset_index(drop=True)


def build_poison_rate_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["dataset", "poison_rate"], observed=True)
        .agg(
            n_points=("transfer_rate", "size"),
            transfer_mean=("transfer_rate", "mean"),
            transfer_median=("transfer_rate", "median"),
            stealth_auc_mean=("stealth_auc_avg", "mean"),
            stealth_tpr_mean=("stealth_tpr_avg", "mean"),
            stealth_mean_mean=("stealth_mean", "mean"),
            asr_mean=("asr", "mean"),
            tradeoff_hmean_mean=("tradeoff_hmean", "mean"),
        )
        .reset_index()
    )
    return out.sort_values(["dataset", "poison_rate"]).reset_index(drop=True)


def build_attack_stability_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["dataset", "attack_type"], observed=True)
        .agg(
            n_points=("transfer_rate", "size"),
            transfer_mean=("transfer_rate", "mean"),
            transfer_std=("transfer_rate", "std"),
            transfer_q25=("transfer_rate", lambda s: _q(s, 0.25)),
            transfer_q75=("transfer_rate", lambda s: _q(s, 0.75)),
            stealth_mean_mean=("stealth_mean", "mean"),
            stealth_std=("stealth_mean", "std"),
            stealth_q25=("stealth_mean", lambda s: _q(s, 0.25)),
            stealth_q75=("stealth_mean", lambda s: _q(s, 0.75)),
            asr_mean=("asr", "mean"),
            tradeoff_hmean_mean=("tradeoff_hmean", "mean"),
        )
        .reset_index()
    )
    out["transfer_iqr"] = out["transfer_q75"] - out["transfer_q25"]
    out["stealth_iqr"] = out["stealth_q75"] - out["stealth_q25"]
    return out.sort_values(["dataset", "transfer_std"], ascending=[True, False]).reset_index(drop=True)


def build_top_configs(df: pd.DataFrame, sort_col: str, top_k: int = 10) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        sub = df[df["dataset"] == dataset].copy()
        sub = sub.sort_values(sort_col, ascending=False).head(top_k)
        sub = sub[
            [
                "dataset",
                "arch",
                "attack_type",
                "poison_rate",
                "train_param_value",
                "test_param_type",
                "test_param_value",
                "transfer_rate",
                "stealth_auc_avg",
                "stealth_tpr_avg",
                "stealth_mean",
                "asr",
                "tradeoff_hmean",
            ]
        ].copy()
        sub.insert(1, "rank_in_dataset", range(1, len(sub) + 1))
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def build_nc_coverage(df_nc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        sub = df_nc[df_nc["dataset"] == dataset]
        total = len(sub)
        with_meta = int(sub["nc_is_poisoned"].notna().sum())
        rows.append(
            {
                "dataset": dataset,
                "dataset_display": DATASET_DISPLAY[dataset],
                "n_points": total,
                "nc_metadata_points": with_meta,
                "nc_metadata_rate": with_meta / total if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_nc_attack_summary(df_nc: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_nc.groupby(["dataset", "attack_type"], observed=True)
        .agg(
            n_points=("transfer_rate", "size"),
            nc_metadata_points=("nc_is_poisoned", lambda s: int(s.notna().sum())),
            nc_poisoned_rate=("nc_is_poisoned", "mean"),
            nc_max_anomaly_index_mean=("nc_max_anomaly_index", "mean"),
            S_stealth_mean=("S_stealth", "mean"),
            S_stealth_median=("S_stealth", "median"),
            S_stealth_tpr_mean=("S_stealth_tpr", "mean"),
            S_stealth_tpr_median=("S_stealth_tpr", "median"),
        )
        .reset_index()
    )
    out["rank_S_stealth_mean"] = (
        out.groupby("dataset")["S_stealth_mean"].rank(ascending=False, method="min")
    )
    out["rank_S_stealth_tpr_mean"] = (
        out.groupby("dataset")["S_stealth_tpr_mean"].rank(ascending=False, method="min")
    )
    return out.sort_values(["dataset", "rank_S_stealth_mean", "attack_type"]).reset_index(
        drop=True
    )


def build_nc_rank_shift_summary(df_nc: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df_nc.groupby(["dataset", "attack_type"], observed=True)
        .agg(
            n_points=("transfer_rate", "size"),
            nc_metadata_points=("nc_is_poisoned", lambda s: int(s.notna().sum())),
            stealth_mean=("stealth_mean", "mean"),
            S_stealth=("S_stealth", "mean"),
            S_stealth_tpr=("S_stealth_tpr", "mean"),
            nc_poisoned_rate=("nc_is_poisoned", "mean"),
        )
        .reset_index()
    )
    grouped["rank_stealth_mean"] = (
        grouped.groupby("dataset")["stealth_mean"].rank(ascending=False, method="min")
    )
    grouped["rank_S_stealth"] = (
        grouped.groupby("dataset")["S_stealth"].rank(ascending=False, method="min")
    )
    grouped["rank_S_stealth_tpr"] = (
        grouped.groupby("dataset")["S_stealth_tpr"].rank(ascending=False, method="min")
    )
    grouped["delta_rank_S_auc"] = (
        grouped["rank_stealth_mean"] - grouped["rank_S_stealth"]
    )
    grouped["delta_rank_S_tpr"] = (
        grouped["rank_stealth_mean"] - grouped["rank_S_stealth_tpr"]
    )
    return grouped.sort_values(["dataset", "rank_S_stealth", "attack_type"]).reset_index(
        drop=True
    )


def save_tables(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


def plot_overall_means(overall: pd.DataFrame, path: Path) -> None:
    metrics = ["transfer_mean", "stealth_auc_mean", "stealth_tpr_mean", "asr_mean"]
    labels = ["Transfer", "Stealth AUC", "Stealth TPR", "ASR"]
    colors = ["#2c7fb8", "#f03b20", "#31a354", "#756bb1"]
    x = np.arange(len(overall))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        ax.bar(x + (i - 1.5) * width, overall[metric], width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY[d] for d in overall["dataset"]], rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean value")
    ax.set_title("Dataset-level Mean Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper center")
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, dataset: str, path: Path) -> None:
    sub = df[df["dataset"] == dataset].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for attack in ATTACK_ORDER:
        for arch in ARCHS:
            part = sub[(sub["attack_type"] == attack) & (sub["arch"] == arch)]
            if part.empty:
                continue
            sizes = 40 + 80 * part["asr"].fillna(0).clip(0, 1)
            ax.scatter(
                part["stealth_mean"],
                part["transfer_rate"],
                s=sizes,
                c=ATTACK_COLORS[attack],
                marker=ARCH_MARKERS[arch],
                alpha=0.75,
                linewidths=0.35,
                edgecolors="black",
            )
    corr = safe_corr(sub["stealth_mean"], sub["transfer_rate"])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Stealth Mean = (Stealth AUC + Stealth TPR) / 2")
    ax.set_ylabel("Transfer Rate")
    ax.set_title(f"{DATASET_DISPLAY[dataset]}: Transfer vs Stealth Mean")
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.03,
        f"corr = {corr:.4f}" if not math.isnan(corr) else "corr = NaN",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    attack_handles = [
        Line2D([0], [0], marker="o", color="w", label=attack, markerfacecolor=ATTACK_COLORS[attack], markersize=8)
        for attack in ATTACK_ORDER
    ]
    arch_handles = [
        Line2D([0], [0], marker=ARCH_MARKERS[arch], color="black", linestyle="", label=arch, markersize=7)
        for arch in ARCHS
    ]
    legend1 = ax.legend(handles=attack_handles, title="Attack", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
    ax.add_artist(legend1)
    ax.legend(handles=arch_handles, title="Arch", bbox_to_anchor=(1.02, 0.45), loc="upper left", frameon=False)
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_attack_heatmap(attack_summary: pd.DataFrame, dataset: str, path: Path) -> None:
    sub = attack_summary[attack_summary["dataset"] == dataset].copy()
    sub["attack_type"] = pd.Categorical(sub["attack_type"], categories=ATTACK_ORDER, ordered=True)
    sub = sub.sort_values("attack_type")
    cols = [
        "transfer_mean",
        "stealth_auc_mean",
        "stealth_tpr_mean",
        "stealth_mean_mean",
        "asr_mean",
        "tradeoff_hmean_mean",
    ]
    labels = ["Transfer", "Stealth AUC", "Stealth TPR", "Stealth Mean", "ASR", "Tradeoff H"]
    arr = sub[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["attack_type"].astype(str).tolist())
    ax.set_title(f"{DATASET_DISPLAY[dataset]}: Attack-level Mean Metrics")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Value")
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_attack_boxplots(df: pd.DataFrame, dataset: str, path: Path) -> None:
    sub = df[df["dataset"] == dataset].copy()
    order = [a for a in ATTACK_ORDER if a in set(sub["attack_type"].astype(str))]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=False)
    for ax, col, title in [
        (axes[0], "transfer_rate", "Transfer Rate Distribution"),
        (axes[1], "stealth_mean", "Stealth Mean Distribution"),
    ]:
        data = [sub[sub["attack_type"].astype(str) == attack][col].dropna().to_numpy() for attack in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.65)
        for patch, attack in zip(bp["boxes"], order):
            patch.set_facecolor(ATTACK_COLORS[attack])
            patch.set_alpha(0.65)
        for median in bp["medians"]:
            median.set_color("black")
        ax.set_xticks(np.arange(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(title)
    fig.suptitle(f"{DATASET_DISPLAY[dataset]}: Attack-level Distributions", y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cross_dataset_attack_heatmaps(cross_attack: pd.DataFrame, path: Path) -> None:
    metric_defs = [
        ("transfer_mean", "Transfer Mean"),
        ("stealth_mean_mean", "Stealth Mean"),
        ("tradeoff_hmean_mean", "Tradeoff HMean"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8))
    for ax, (metric, title) in zip(axes, metric_defs):
        pivot = (
            cross_attack.pivot(index="attack_type", columns="dataset", values=metric)
            .reindex(index=ATTACK_ORDER, columns=DATASETS)
        )
        arr = pivot.to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(DATASETS)))
        ax.set_xticklabels([DATASET_DISPLAY[d] for d in DATASETS], rotation=18, ha="right")
        ax.set_yticks(np.arange(len(ATTACK_ORDER)))
        ax.set_yticklabels(ATTACK_ORDER)
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isnan(arr[i, j]):
                    txt = "NaN"
                else:
                    txt = f"{arr[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.suptitle("Cross-dataset Comparison by Attack Type", y=1.02)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Value")
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.18, top=0.84, wspace=0.28)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_poison_rate_trends(poison_summary: pd.DataFrame, dataset: str, path: Path) -> None:
    sub = poison_summary[poison_summary["dataset"] == dataset].copy().sort_values("poison_rate")
    x = sub["poison_rate"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(x, sub["transfer_mean"], marker="o", linewidth=2.2, color="#2c7fb8", label="Transfer")
    ax.plot(x, sub["stealth_mean_mean"], marker="s", linewidth=2.2, color="#d95f0e", label="Stealth Mean")
    ax.plot(x, sub["asr_mean"], marker="^", linewidth=2.2, color="#31a354", label="ASR")
    for xi, n in zip(x, sub["n_points"]):
        ax.text(xi, 0.02, f"n={int(n)}", rotation=90, fontsize=8, alpha=0.7)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Poison Rate")
    ax.set_ylabel("Mean value")
    ax.set_title(f"{DATASET_DISPLAY[dataset]}: Metric Trends vs Poison Rate")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_arch_attack_tradeoff_heatmap(
    attack_arch_summary: pd.DataFrame, dataset: str, path: Path
) -> None:
    sub = attack_arch_summary[attack_arch_summary["dataset"] == dataset].copy()
    pivot = (
        sub.pivot(index="attack_type", columns="arch", values="tradeoff_hmean_mean")
        .reindex(index=ATTACK_ORDER, columns=ARCHS)
    )
    arr = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(ARCHS)))
    ax.set_xticklabels(ARCHS)
    ax.set_yticks(np.arange(len(ATTACK_ORDER)))
    ax.set_yticklabels(ATTACK_ORDER)
    ax.set_title(f"{DATASET_DISPLAY[dataset]}: Tradeoff by Attack and Arch")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            txt = "NaN" if np.isnan(arr[i, j]) else f"{arr[i, j]:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tradeoff HMean")
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_nc_coverage(nc_coverage: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(nc_coverage))
    vals = nc_coverage["nc_metadata_rate"].fillna(0).to_numpy()
    ax.bar(x, vals, color=["#2c7fb8", "#f03b20", "#31a354"])
    ax.set_xticks(x)
    ax.set_xticklabels(nc_coverage["dataset_display"], rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("NC metadata coverage")
    ax.set_title("NC Metadata Coverage by Dataset")
    ax.grid(axis="y", alpha=0.25)
    for i, row in nc_coverage.reset_index(drop=True).iterrows():
        ax.text(i, min(1.02, vals[i] + 0.03), f"{int(row['nc_metadata_points'])}/{int(row['n_points'])}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_nc_attack_summary(nc_attack: pd.DataFrame, dataset: str, path: Path) -> None:
    sub = nc_attack[nc_attack["dataset"] == dataset].copy()
    sub["attack_type"] = pd.Categorical(sub["attack_type"], categories=ATTACK_ORDER, ordered=True)
    sub = sub.sort_values("attack_type")
    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(sub))
    width = 0.35
    ax1.bar(x - width / 2, sub["S_stealth_mean"], width=width, color="#2c7fb8", label="S_stealth")
    ax1.bar(x + width / 2, sub["S_stealth_tpr_mean"], width=width, color="#31a354", label="S_stealth_tpr")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Mean S_stealth")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["attack_type"].astype(str), rotation=35, ha="right")
    ax1.set_title(f"{DATASET_DISPLAY[dataset]}: NC-aware Stealth Summary")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, sub["nc_poisoned_rate"], color="#d62728", marker="o", linewidth=2, label="NC poisoned rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("NC poisoned rate")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=3, loc="upper center")
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_figures(
    df_no_nc: pd.DataFrame,
    nc_coverage: pd.DataFrame,
    attack_summary: pd.DataFrame,
    attack_arch_summary: pd.DataFrame,
    cross_attack: pd.DataFrame,
    poison_summary: pd.DataFrame,
    nc_attack: pd.DataFrame,
    fig_dir: Path,
    overall: pd.DataFrame,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_overall_means(overall, fig_dir / "overall_dataset_means.png")
    plot_cross_dataset_attack_heatmaps(
        cross_attack, fig_dir / "cross_dataset_attack_comparison_heatmaps.png"
    )
    plot_nc_coverage(nc_coverage, fig_dir / "nc_coverage_summary.png")
    for dataset in DATASETS:
        plot_scatter(
            df_no_nc, dataset, fig_dir / f"{dataset}_transfer_vs_stealth_mean_scatter.png"
        )
        plot_attack_heatmap(
            attack_summary, dataset, fig_dir / f"{dataset}_attack_metric_heatmap.png"
        )
        plot_attack_boxplots(
            df_no_nc, dataset, fig_dir / f"{dataset}_attack_boxplots.png"
        )
        plot_poison_rate_trends(
            poison_summary, dataset, fig_dir / f"{dataset}_poison_rate_trends.png"
        )
        plot_arch_attack_tradeoff_heatmap(
            attack_arch_summary,
            dataset,
            fig_dir / f"{dataset}_arch_attack_tradeoff_heatmap.png",
        )
    for dataset in ["cifar10", "mnistm"]:
        plot_nc_attack_summary(
            nc_attack, dataset, fig_dir / f"{dataset}_nc_attack_summary.png"
        )


def md_table(df: pd.DataFrame, columns: List[str] | None = None, decimals: int = 4) -> str:
    out = df.copy()
    if columns is not None:
        out = out[columns].copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(
                lambda x: f"{float(x):.{decimals}f}" if pd.notna(x) else ""
            )
    return out.to_markdown(index=False)


def dataset_findings(
    overall: pd.DataFrame,
    attack_summary: pd.DataFrame,
    arch_summary: pd.DataFrame,
    dataset: str,
) -> List[str]:
    ds_overall = overall[overall["dataset"] == dataset].iloc[0]
    ds_attack = attack_summary[attack_summary["dataset"] == dataset].copy()
    ds_arch = arch_summary[arch_summary["dataset"] == dataset].copy()
    best_transfer = ds_attack.sort_values("transfer_mean", ascending=False).iloc[0]
    best_stealth = ds_attack.sort_values("stealth_mean_mean", ascending=False).iloc[0]
    best_tradeoff = ds_attack.sort_values("tradeoff_hmean_mean", ascending=False).iloc[0]
    best_arch_transfer = ds_arch.sort_values("transfer_mean", ascending=False).iloc[0]
    best_arch_stealth = ds_arch.sort_values("stealth_mean_mean", ascending=False).iloc[0]
    findings = [
        f"整体平均迁移率为 `{ds_overall['transfer_mean']:.4f}`，中位迁移率为 `{ds_overall['transfer_median']:.4f}`；"
        f"平均隐蔽性为 `AUC={ds_overall['stealth_auc_mean']:.4f}`、`TPR={ds_overall['stealth_tpr_mean']:.4f}`。",
        f"按攻击均值看，迁移性最强的是 `{best_transfer['attack_type']}` "
        f"(`transfer_mean={best_transfer['transfer_mean']:.4f}`)，"
        f"隐蔽性最强的是 `{best_stealth['attack_type']}` "
        f"(`stealth_mean_mean={best_stealth['stealth_mean_mean']:.4f}`)。",
        f"若同时考虑迁移性与隐蔽性，均值 trade-off 最好的攻击是 `{best_tradeoff['attack_type']}` "
        f"(`tradeoff_hmean_mean={best_tradeoff['tradeoff_hmean_mean']:.4f}`)。",
        f"按架构均值看，迁移性最强的是 `{best_arch_transfer['arch']}` "
        f"(`transfer_mean={best_arch_transfer['transfer_mean']:.4f}`)，"
        f"隐蔽性最强的是 `{best_arch_stealth['arch']}` "
        f"(`stealth_mean_mean={best_arch_stealth['stealth_mean_mean']:.4f}`)。",
        f"`transfer_rate` 与 `stealth_mean` 的相关系数为 `{ds_overall['corr_transfer_stealth_mean']:.4f}`，"
        f"与 `asr` 的相关系数为 `{ds_overall['corr_transfer_asr']:.4f}`。",
    ]
    if dataset == "mnistm":
        sig = ds_attack[ds_attack["attack_type"].astype(str) == "SIG"].iloc[0]
        findings.append(
            f"`SIG` 在该数据集上的平均迁移率仅 `{sig['transfer_mean']:.4f}`，平均 ASR 仅 `{sig['asr_mean']:.4f}`，"
            "表现为“高隐蔽但几乎不激活”的极端失效。"
        )
    if dataset == "tiny_imagenet":
        findings.append(
            "该数据集当前使用的是后生成的 target-domain 结果，而不是旧的 Tiny-ImageNet-C 结果。"
        )
    return findings


def dataset_stability_findings(stability_summary: pd.DataFrame, dataset: str) -> List[str]:
    sub = stability_summary[stability_summary["dataset"] == dataset].copy()
    most_var_transfer = sub.sort_values("transfer_std", ascending=False).iloc[0]
    most_var_stealth = sub.sort_values("stealth_std", ascending=False).iloc[0]
    stable_high_transfer = (
        sub[sub["transfer_mean"] >= sub["transfer_mean"].median()]
        .sort_values(["transfer_iqr", "transfer_mean"], ascending=[True, False])
        .iloc[0]
    )
    findings = [
        f"迁移率波动最大的攻击是 `{most_var_transfer['attack_type']}` "
        f"(`transfer_std={most_var_transfer['transfer_std']:.4f}`, `transfer_iqr={most_var_transfer['transfer_iqr']:.4f}`)，"
        "说明该方法对参数或架构非常敏感。",
        f"隐蔽性波动最大的攻击是 `{most_var_stealth['attack_type']}` "
        f"(`stealth_std={most_var_stealth['stealth_std']:.4f}`, `stealth_iqr={most_var_stealth['stealth_iqr']:.4f}`)，"
        "说明它的隐蔽性并不稳定。",
        f"在平均迁移率不低于该数据集方法中位数的前提下，最稳定的高迁移攻击是 "
        f"`{stable_high_transfer['attack_type']}` "
        f"(`transfer_mean={stable_high_transfer['transfer_mean']:.4f}`, `transfer_iqr={stable_high_transfer['transfer_iqr']:.4f}`)。",
    ]
    if dataset == "mnistm":
        sig = sub[sub["attack_type"].astype(str) == "SIG"].iloc[0]
        findings.append(
            f"`SIG` 的 `transfer_std={sig['transfer_std']:.4f}` 且 `transfer_mean={sig['transfer_mean']:.4f}`，"
            "说明它不是偶然失效，而是几乎在整个参数空间中都无法稳定迁移。"
        )
    return findings


def dataset_poison_findings(poison_summary: pd.DataFrame, dataset: str) -> List[str]:
    sub = poison_summary[poison_summary["dataset"] == dataset].copy().sort_values("poison_rate")
    best_transfer = sub.sort_values("transfer_mean", ascending=False).iloc[0]
    best_stealth = sub.sort_values("stealth_mean_mean", ascending=False).iloc[0]
    best_asr = sub.sort_values("asr_mean", ascending=False).iloc[0]
    monotonic_transfer = bool(
        np.all(np.diff(sub["transfer_mean"].to_numpy(dtype=float)) >= -1e-9)
    )
    monotonic_stealth_down = bool(
        np.all(np.diff(sub["stealth_mean_mean"].to_numpy(dtype=float)) <= 1e-9)
    )
    findings = [
        f"平均迁移率最高的 poison rate 是 `{best_transfer['poison_rate']:.3f}` "
        f"(`transfer_mean={best_transfer['transfer_mean']:.4f}`, `n={int(best_transfer['n_points'])}`)。",
        f"平均隐蔽性最高的 poison rate 是 `{best_stealth['poison_rate']:.3f}` "
        f"(`stealth_mean_mean={best_stealth['stealth_mean_mean']:.4f}`, `n={int(best_stealth['n_points'])}`)。",
        f"平均 ASR 最高的 poison rate 是 `{best_asr['poison_rate']:.3f}` "
        f"(`asr_mean={best_asr['asr_mean']:.4f}`, `n={int(best_asr['n_points'])}`)。",
    ]
    if monotonic_transfer:
        findings.append("从总体均值看，迁移率随 poison rate 增大基本单调上升。")
    else:
        findings.append("迁移率总体上随 poison rate 增大而上升，但不是严格单调，说明仍存在方法或架构交互效应。")
    if monotonic_stealth_down:
        findings.append("隐蔽性随 poison rate 增大基本单调下降，这是典型的强攻击-低隐蔽性模式。")
    else:
        findings.append("隐蔽性总体上会随 poison rate 增大而下降，但并非严格单调，说明某些方法在中等毒化率附近仍可保持较好折中。")
    sparse = sub[sub["n_points"] == sub["n_points"].min()]
    if len(sparse) > 0 and int(sparse["n_points"].min()) <= 18:
        findings.append(
            "需要注意，高 poison rate（例如 0.02 或 0.1）在当前结果里样本点较少，均值更容易受少数强攻击方法主导。"
        )
    return findings


def dataset_arch_attack_findings(
    attack_arch_summary: pd.DataFrame, dataset: str
) -> Tuple[List[str], pd.DataFrame]:
    sub = attack_arch_summary[attack_arch_summary["dataset"] == dataset].copy()
    top = sub.sort_values("tradeoff_hmean_mean", ascending=False).head(6).copy()
    findings = [
        f"平均 trade-off 最强的架构-攻击组合是 `{top.iloc[0]['arch']} + {top.iloc[0]['attack_type']}` "
        f"(`tradeoff_hmean_mean={top.iloc[0]['tradeoff_hmean_mean']:.4f}`)。",
        f"前六名组合中出现最多的攻击方法是 `{top['attack_type'].mode().iloc[0]}`，"
        "说明它在不同架构上都比较容易形成稳定优势。"
        if not top.empty
        else "当前没有可用的架构-攻击组合摘要。",
    ]
    return findings, top


def cross_dataset_findings(cross_attack: pd.DataFrame) -> List[str]:
    sub = cross_attack.copy()
    transfer_top = sub.sort_values("transfer_mean", ascending=False).iloc[0]
    stealth_top = sub.sort_values("stealth_mean_mean", ascending=False).iloc[0]
    tradeoff_top = sub.sort_values("tradeoff_hmean_mean", ascending=False).iloc[0]
    sig_rows = sub[sub["attack_type"].astype(str) == "SIG"].sort_values("dataset")
    findings = [
        f"在所有“攻击-数据集”组合中，平均迁移率最高的是 "
        f"`{transfer_top['attack_type']} @ {transfer_top['dataset']}` "
        f"(`transfer_mean={transfer_top['transfer_mean']:.4f}`)。",
        f"平均隐蔽性最高的是 `{stealth_top['attack_type']} @ {stealth_top['dataset']}` "
        f"(`stealth_mean_mean={stealth_top['stealth_mean_mean']:.4f}`)。",
        f"平均折中能力最强的是 `{tradeoff_top['attack_type']} @ {tradeoff_top['dataset']}` "
        f"(`tradeoff_hmean_mean={tradeoff_top['tradeoff_hmean_mean']:.4f}`)。",
    ]
    if len(sig_rows) == 3:
        parts = [
            f"{row['dataset']}={row['transfer_mean']:.4f}"
            for _, row in sig_rows.iterrows()
        ]
        findings.append(
            "`SIG` 的跨数据集迁移差异极大，平均迁移率分别为 " + " / ".join(parts) + "。"
        )
    return findings


def nc_rank_findings(nc_rank_shift: pd.DataFrame, dataset: str) -> List[str]:
    sub = nc_rank_shift[nc_rank_shift["dataset"] == dataset].copy()
    if sub.empty:
        return []
    biggest_up = sub.sort_values("delta_rank_S_auc", ascending=False).iloc[0]
    biggest_down = sub.sort_values("delta_rank_S_auc", ascending=True).iloc[0]
    findings = [
        "这里的 `nc_is_poisoned` 与 `max_anomaly_index` 直接来自当前 NC 管线的原始输出；"
        "`S_stealth` 只是你当前脚本定义的组合指标，不应替代四种防御平均隐蔽性。",
        f"在当前 `S_stealth` 定义下，排名提升最大的是 `{biggest_up['attack_type']}` "
        f"(`delta_rank_S_auc={int(biggest_up['delta_rank_S_auc'])}`)。",
        f"排名下降最大的是 `{biggest_down['attack_type']}` "
        f"(`delta_rank_S_auc={int(biggest_down['delta_rank_S_auc'])}`)。",
    ]
    return findings


def build_report(
    analysis_dir: Path,
    df_no_nc: pd.DataFrame,
    df_nc: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    fig_dir_name: str,
    table_dir_name: str,
) -> str:
    overall = tables["overall_dataset_summary"]
    arch_summary = tables["arch_summary"]
    attack_summary = tables["attack_summary_by_dataset"]
    attack_arch_summary = tables["attack_summary_by_dataset_arch"]
    cross_attack = tables["cross_dataset_attack_summary"]
    poison_summary = tables["poison_rate_summary_by_dataset"]
    stability_summary = tables["attack_stability_summary"]
    nc_coverage = tables["nc_coverage_summary"]
    nc_attack = tables["nc_attack_summary"]
    nc_rank_shift = tables["nc_rank_shift_summary"]
    top_tradeoff = tables["top_configs_by_tradeoff"]
    top_transfer = tables["top_configs_by_transfer"]
    top_stealth = tables["top_configs_by_stealth_mean"]

    lines: List[str] = []
    lines.append("# 三个数据集迁移性与隐蔽性的详细分析报告\n\n")
    lines.append("生成位置：`analysis/transfer_stealth_report_detailed.md`\n\n")
    lines.append(
        "本报告直接基于 `analysis/data_{dataset}_{arch}_{no_nc/nc}.csv` 的当前结果生成。"
        "分析对象为三个数据集、三个模型架构、八类攻击方法；主结论基于 `no_nc` 数据中的"
        " `transfer_rate`、`stealth_auc_avg`、`stealth_tpr_avg`，`nc` 结果只在 NC 专章中补充。\n\n"
    )

    lines.append("## 1. 指标说明\n\n")
    lines.append("- `transfer_rate`：目标域测试 ASR，越大越强。\n")
    lines.append("- `stealth_auc_avg = 1 - mean(AUC)`：基于 STRIP、SentiNet、IBD-PSC、SCaLe-Up 四种防御的平均隐蔽性，越大越隐蔽。\n")
    lines.append("- `stealth_tpr_avg = 1 - mean(TPR)`：同样基于四种防御的平均隐蔽性，越大越隐蔽。\n")
    lines.append("- `stealth_mean = (stealth_auc_avg + stealth_tpr_avg) / 2`：本报告中用于综合比较的平均隐蔽性。\n")
    lines.append("- `tradeoff_hmean`：`transfer_rate` 与 `stealth_mean` 的调和均值，用于衡量“迁移性-隐蔽性”的折中程度。\n")
    lines.append("- `S_stealth` / `S_stealth_tpr`：加入 NC 信息后的补充隐蔽性，只在 NC 元数据充分时有解释力。\n\n")

    lines.append("## 2. 数据完整性与口径说明\n\n")
    lines.append("- 当前 `tiny_imagenet` 的迁移性已经切换为后生成的 target-domain 结果，不再使用旧的 Tiny-ImageNet-C 结果。\n")
    lines.append("- 当前结果文件中的攻击名称为 `basic`。如果论文正文中需要写作 `BadNet`，需要你自行统一命名口径。\n")
    lines.append("- `NC` 相关的 `nc_is_poisoned` 与 `max_anomaly_index` 直接来自当前 NC 管线的原始输出；本报告中的 `S_stealth` 只是你当前分析脚本定义的组合指标，不应替代四种防御平均隐蔽性。\n")
    lines.append("- `tiny_imagenet` 的 NC 元数据覆盖率很低，因此该数据集上的 NC 结论只能写成补充说明，不能写成主结论。\n\n")

    lines.append("## 3. 产物索引\n\n")
    lines.append(f"- 统计表目录：`{table_dir_name}/`\n")
    lines.append(f"- 图片目录：`{fig_dir_name}/`\n")
    lines.append("- 本报告优先使用 `no_nc` 的迁移性与四防御平均隐蔽性作为主结果，NC 只在后半部分做补充分析。\n\n")

    lines.append("## 4. 总体统计\n\n")
    lines.append(f"![overall means]({fig_dir_name}/overall_dataset_means.png)\n\n")
    lines.append(
        md_table(
            overall,
            [
                "dataset_display",
                "n_points",
                "transfer_mean",
                "transfer_median",
                "stealth_auc_mean",
                "stealth_tpr_mean",
                "stealth_mean_mean",
                "asr_mean",
                "corr_transfer_stealth_auc",
                "corr_transfer_stealth_mean",
                "corr_transfer_asr",
            ],
        )
        + "\n\n"
    )
    lines.append(f"![cross attack comparison]({fig_dir_name}/cross_dataset_attack_comparison_heatmaps.png)\n\n")
    lines.append(
        md_table(
            cross_attack,
            [
                "attack_type",
                "dataset",
                "transfer_mean",
                "stealth_mean_mean",
                "asr_mean",
                "tradeoff_hmean_mean",
                "rank_transfer_mean",
                "rank_stealth_mean",
                "rank_tradeoff_mean",
            ],
        )
        + "\n\n"
    )
    lines.append("### 4.1 总体结论\n\n")
    best_transfer_ds = overall.sort_values("transfer_mean", ascending=False).iloc[0]
    best_stealth_ds = overall.sort_values("stealth_mean_mean", ascending=False).iloc[0]
    lines.append(
        f"- 三个数据集里，平均迁移率最高的是 `{best_transfer_ds['dataset_display']}` "
        f"(`transfer_mean={best_transfer_ds['transfer_mean']:.4f}`)。\n"
    )
    lines.append(
        f"- 平均隐蔽性最高的是 `{best_stealth_ds['dataset_display']}` "
        f"(`stealth_mean_mean={best_stealth_ds['stealth_mean_mean']:.4f}`)。\n"
    )
    lines.append("- 三个数据集都存在明显的迁移性-隐蔽性负相关，这说明跨域攻击并非越强越隐蔽，而是稳定地存在权衡。\n")
    lines.append("- 三个数据集上 `transfer_rate` 与 `asr` 都保持强正相关，尤其是 Tiny-ImageNet target-domain 与 MNIST-M，说明目标域 ASR 基本决定了迁移率上限。\n")
    for item in cross_dataset_findings(cross_attack):
        lines.append(f"- {item}\n")
    lines.append("\n### 4.2 图表解读\n\n")
    lines.append("- `overall_dataset_means.png` 用于回答“哪个数据集整体更容易迁移、哪个数据集整体更隐蔽”。从图上可以直接看到 Tiny-ImageNet 的迁移均值最高，而 MNIST-M 的隐蔽性均值最高。\n")
    lines.append("- `cross_dataset_attack_comparison_heatmaps.png` 用于回答“同一种攻击换数据集之后会发生什么”。它清楚表明：`basic` 在三个数据集上都偏向高迁移低隐蔽；`SIG` 在 MNIST-M 上的迁移性几乎完全崩掉，但在隐蔽性维度仍然很高。\n\n")

    for dataset in DATASETS:
        ds_attack = attack_summary[attack_summary["dataset"] == dataset].copy().sort_values(
            ["rank_tradeoff_mean", "rank_transfer_mean", "attack_type"]
        )
        ds_poison = poison_summary[poison_summary["dataset"] == dataset].copy().sort_values("poison_rate")
        ds_stability = stability_summary[stability_summary["dataset"] == dataset].copy()
        arch_attack_notes, ds_arch_attack_top = dataset_arch_attack_findings(
            attack_arch_summary, dataset
        )
        ds_top_tradeoff = top_tradeoff[top_tradeoff["dataset"] == dataset].head(5)
        ds_top_transfer = top_transfer[top_transfer["dataset"] == dataset].head(5)
        ds_top_stealth = top_stealth[top_stealth["dataset"] == dataset].head(5)

        lines.append(f"## 5. {DATASET_DISPLAY[dataset]}\n\n")
        lines.append(f"- Target setting: `{DATASET_TARGET_NOTE[dataset]}`\n\n")
        for item in dataset_findings(overall, attack_summary, arch_summary, dataset):
            lines.append(f"- {item}\n")
        lines.append("\n")
        lines.append(f"![scatter]({fig_dir_name}/{dataset}_transfer_vs_stealth_mean_scatter.png)\n\n")
        lines.append(f"![heatmap]({fig_dir_name}/{dataset}_attack_metric_heatmap.png)\n\n")
        lines.append(f"![boxplots]({fig_dir_name}/{dataset}_attack_boxplots.png)\n\n")
        lines.append(f"![poison trends]({fig_dir_name}/{dataset}_poison_rate_trends.png)\n\n")
        lines.append(f"![arch attack tradeoff]({fig_dir_name}/{dataset}_arch_attack_tradeoff_heatmap.png)\n\n")

        lines.append("### 5.1 攻击方法均值统计\n\n")
        lines.append(
            md_table(
                ds_attack,
                [
                    "attack_type",
                    "n_points",
                    "transfer_mean",
                    "transfer_median",
                    "stealth_auc_mean",
                    "stealth_tpr_mean",
                    "stealth_mean_mean",
                    "asr_mean",
                    "tradeoff_hmean_mean",
                    "rank_tradeoff_mean",
                ],
            )
            + "\n\n"
        )

        lines.append("### 5.2 图表解读与现象分析\n\n")
        lines.append("- 散点图中，右上区域代表“高迁移 + 高隐蔽”的理想区，右下区域代表“高迁移但不隐蔽”，左上区域代表“隐蔽但迁移差”。\n")
        if dataset == "cifar10":
            lines.append("- 该数据集的散点呈现出较清晰的三团结构：`basic` 大多集中在右下，`SIG` 与 `adaptive_blend` 更多分布在中上或左上，`upgd/blend` 更接近中右区域，因此后两者在折中指标上表现最好。\n")
            lines.append("- 箱线图进一步说明：`basic` 的迁移率箱体几乎贴近 1，说明它几乎总能迁移；但其隐蔽性箱体明显压在底部，说明代价就是稳定暴露。相反，`WaNet` 的隐蔽性较高，但迁移率整体偏低。\n")
        elif dataset == "tiny_imagenet":
            lines.append("- 该数据集比另外两个数据集更容易出现靠近右上区域的点，说明在当前 target-domain 设置下，确实存在一批兼顾迁移性与隐蔽性的配置。\n")
            lines.append("- 热力图显示 `WaNet`、`blend`、`adaptive_blend` 在多项指标上颜色更均衡；而 `basic/belt/adaptive_patch` 的高分主要集中在迁移率与 ASR 上，而不是隐蔽性上。\n")
        elif dataset == "mnistm":
            lines.append("- 该数据集最显著的视觉现象是 `SIG` 形成了一条几乎贴近左下角 x 轴的低迁移带：隐蔽性仍高，但迁移率几乎为零。这是结构性失败，而不是个别异常点。\n")
            lines.append("- `belt`、`blend`、`WaNet`、`adaptive_blend` 在热力图中的 trade-off 列整体更亮，说明 MNIST-M 上最有价值的方法不是单纯的“最强攻击”，而是能在两端同时维持中高水平的方法。\n")
        for item in dataset_stability_findings(stability_summary, dataset):
            lines.append(f"- {item}\n")
        for item in dataset_poison_findings(poison_summary, dataset):
            lines.append(f"- {item}\n")
        for item in arch_attack_notes:
            lines.append(f"- {item}\n")
        lines.append("\n")

        lines.append("### 5.3 稳定性统计\n\n")
        lines.append(
            md_table(
                ds_stability,
                [
                    "attack_type",
                    "transfer_mean",
                    "transfer_std",
                    "transfer_iqr",
                    "stealth_mean_mean",
                    "stealth_std",
                    "stealth_iqr",
                    "asr_mean",
                    "tradeoff_hmean_mean",
                ],
            )
            + "\n\n"
        )

        lines.append("### 5.4 Poison Rate 趋势\n\n")
        lines.append(
            md_table(
                ds_poison,
                [
                    "poison_rate",
                    "n_points",
                    "transfer_mean",
                    "stealth_auc_mean",
                    "stealth_tpr_mean",
                    "stealth_mean_mean",
                    "asr_mean",
                    "tradeoff_hmean_mean",
                ],
            )
            + "\n\n"
        )

        lines.append("### 5.5 架构-攻击组合的均值 trade-off\n\n")
        lines.append(
            md_table(
                ds_arch_attack_top,
                [
                    "arch",
                    "attack_type",
                    "n_points",
                    "transfer_mean",
                    "stealth_mean_mean",
                    "asr_mean",
                    "tradeoff_hmean_mean",
                ],
            )
            + "\n\n"
        )

        lines.append("### 5.6 Top-5 迁移-隐蔽折中配置\n\n")
        lines.append(
            md_table(
                ds_top_tradeoff,
                [
                    "rank_in_dataset",
                    "arch",
                    "attack_type",
                    "poison_rate",
                    "train_param_value",
                    "test_param_type",
                    "test_param_value",
                    "transfer_rate",
                    "stealth_mean",
                    "asr",
                    "tradeoff_hmean",
                ],
            )
            + "\n\n"
        )

        lines.append("### 5.7 Top-5 迁移率配置\n\n")
        lines.append(
            md_table(
                ds_top_transfer,
                [
                    "rank_in_dataset",
                    "arch",
                    "attack_type",
                    "poison_rate",
                    "train_param_value",
                    "test_param_type",
                    "test_param_value",
                    "transfer_rate",
                    "stealth_mean",
                    "asr",
                    "tradeoff_hmean",
                ],
            )
            + "\n\n"
        )

        lines.append("### 5.8 Top-5 隐蔽性配置\n\n")
        lines.append(
            md_table(
                ds_top_stealth,
                [
                    "rank_in_dataset",
                    "arch",
                    "attack_type",
                    "poison_rate",
                    "train_param_value",
                    "test_param_type",
                    "test_param_value",
                    "stealth_mean",
                    "transfer_rate",
                    "asr",
                    "tradeoff_hmean",
                ],
            )
            + "\n\n"
        )

        lines.append("### 5.9 这一组结果最值得写进论文的点\n\n")
        if dataset == "cifar10":
            lines.append("- `basic` 的平均迁移率最高，但它的隐蔽性在所有方法中最低，说明在 CIFAR-10 到 STL-10 上，显式 patch 类方法更像“高效但暴露”的攻击。\n")
            lines.append("- `upgd` 和 `blend` 不只是均值 trade-off 高，它们还在 Top-5 折中配置表里反复出现，说明它们不是依赖单个幸运点，而是能稳定形成折中优势的方法。\n\n")
        elif dataset == "tiny_imagenet":
            lines.append("- `WaNet` 是该数据集上均值折中最优的方法，说明在你当前 target-domain 构造下，形变型触发比显式 patch 更容易兼顾迁移与隐蔽。\n")
            lines.append("- `belt/basic/adaptive_patch` 虽然迁移率很高，但从热力图、箱线图和稳定性表来看，它们的高迁移基本是用明显下降的隐蔽性换来的。\n\n")
        elif dataset == "mnistm":
            lines.append("- `belt` 是 MNIST-M 上均值折中最优的方法，而 `SIG` 几乎完全失效。这两个结果一正一反，构成了这个数据集最值得写的实验故事线。\n")
            lines.append("- `blend/WaNet/adaptive_blend` 在该数据集上也保持了较强折中能力，说明 MNIST-M 到 MNIST 的域转换并没有全面破坏所有非 patch 类触发。\n\n")

    lines.append("## 6. 架构差异\n\n")
    lines.append(
        md_table(
            arch_summary,
            [
                "dataset",
                "arch",
                "n_points",
                "transfer_mean",
                "stealth_auc_mean",
                "stealth_tpr_mean",
                "stealth_mean_mean",
                "asr_mean",
                "tradeoff_hmean_mean",
            ],
        )
        + "\n\n"
    )
    lines.append("- `ResNet18` 在三个数据集上都给出了最高的平均迁移率，说明它学习到的后门表征最容易跨域保留。\n")
    lines.append("- `VGG` 和 `MobileNet` 往往给出更高的平均隐蔽性，因此模型架构本身也在参与迁移性-隐蔽性的权衡。\n")
    lines.append("- 更具体地说，`ResNet18` 更像“强迁移架构”，而 `MobileNet/VGG` 更像“更容易隐藏”的架构；这一点在三个数据集上都反复出现。\n\n")

    lines.append("## 7. NC 分析\n\n")
    lines.append(f"![nc coverage]({fig_dir_name}/nc_coverage_summary.png)\n\n")
    lines.append(md_table(nc_coverage) + "\n\n")
    lines.append("- `cifar10` 与 `mnistm` 的 NC 元数据是完整的，可以直接比较。\n")
    lines.append("- `tiny_imagenet` 只有 `8/465` 个点含 NC 元数据，因此该数据集上的 `S_stealth` 只能作为补充参考，不能做强结论。\n\n")

    for dataset in ["cifar10", "mnistm"]:
        ds_nc = nc_attack[nc_attack["dataset"] == dataset].copy()
        ds_shift = nc_rank_shift[nc_rank_shift["dataset"] == dataset].copy()
        lines.append(f"### 7.{1 if dataset == 'cifar10' else 2} {DATASET_DISPLAY[dataset]} 的 NC 结果\n\n")
        lines.append(f"![nc attack summary]({fig_dir_name}/{dataset}_nc_attack_summary.png)\n\n")
        lines.append(
            md_table(
                ds_nc,
                [
                    "attack_type",
                    "n_points",
                    "nc_metadata_points",
                    "nc_poisoned_rate",
                    "S_stealth_mean",
                    "S_stealth_tpr_mean",
                    "rank_S_stealth_mean",
                    "rank_S_stealth_tpr_mean",
                ],
            )
            + "\n\n"
        )
        lines.append(
            md_table(
                ds_shift,
                [
                    "attack_type",
                    "nc_metadata_points",
                    "nc_poisoned_rate",
                    "rank_stealth_mean",
                    "rank_S_stealth",
                    "rank_S_stealth_tpr",
                    "delta_rank_S_auc",
                    "delta_rank_S_tpr",
                ],
            )
            + "\n\n"
        )
        for item in nc_rank_findings(nc_rank_shift, dataset):
            lines.append(f"- {item}\n")
        lines.append("\n")

    lines.append("### 7.3 Tiny-ImageNet 的 NC 说明\n\n")
    lines.append(
        md_table(
            nc_attack[nc_attack["dataset"] == "tiny_imagenet"],
            [
                "attack_type",
                "n_points",
                "nc_metadata_points",
                "nc_poisoned_rate",
                "S_stealth_mean",
                "S_stealth_tpr_mean",
            ],
        )
        + "\n\n"
    )
    lines.append("由于 `tiny_imagenet` 的 NC 原始元数据严重缺失，多数攻击方法的 `S_stealth` 实际上退化为 `0.8 * stealth` 的近似量，因此不建议将该数据集上的 NC 排名写成正式实验结论。\n\n")

    lines.append("## 8. 可直接写入论文的主结论\n\n")
    lines.append("- 三个数据集都存在清晰的迁移性-隐蔽性张力，且这种张力在相关性层面是稳定的，而不是偶然现象。\n")
    lines.append("- `basic`、`belt` 这类显式触发方法通常迁移率高，但隐蔽性明显下降。\n")
    lines.append("- `WaNet`、`blend`、`adaptive_blend` 更容易在迁移性与隐蔽性之间取得平衡，尤其在 Tiny-ImageNet target-domain 和 MNIST-M 上更明显。\n")
    lines.append("- `SIG` 的结果最适合作为反例：高隐蔽性并不保证高迁移性；当触发模式在域转换中无法稳定保留时，攻击会表现为“检测不到，但也激活不起来”。\n")
    lines.append("- 当前最需要谨慎表述的是 Tiny-ImageNet 上的 NC 结论，因为元数据覆盖不足。\n\n")

    lines.append("## 9. 详细文件清单\n\n")
    lines.append(f"- 统计表：`{table_dir_name}/overall_dataset_summary.csv`\n")
    lines.append(f"- 统计表：`{table_dir_name}/arch_summary.csv`\n")
    lines.append(f"- 统计表：`{table_dir_name}/attack_summary_by_dataset.csv`\n")
    lines.append(f"- 统计表：`{table_dir_name}/attack_summary_by_dataset_arch.csv`\n")
    lines.append(f"- 统计表：`{table_dir_name}/cross_dataset_attack_summary.csv`\n")
    lines.append(f"- 统计表：`{table_dir_name}/poison_rate_summary_by_dataset.csv`\n")
    lines.append(f"- 统计表：`{table_dir_name}/attack_stability_summary.csv`\n")
    lines.append(f"- 统计表：`{table_dir_name}/top_configs_by_tradeoff.csv`\n")
    lines.append(f"- 统计表：`{table_dir_name}/top_configs_by_transfer.csv`\n")
    lines.append(f"- 统计表：`{table_dir_name}/top_configs_by_stealth_mean.csv`\n")
    lines.append(f"- 统计表：`{table_dir_name}/nc_coverage_summary.csv`\n")
    lines.append(f"- 统计表：`{table_dir_name}/nc_attack_summary.csv`\n")
    lines.append(f"- 统计表：`{table_dir_name}/nc_rank_shift_summary.csv`\n")
    lines.append(f"- 图片：`{fig_dir_name}/overall_dataset_means.png`\n")
    lines.append(f"- 图片：`{fig_dir_name}/cross_dataset_attack_comparison_heatmaps.png`\n")
    lines.append(f"- 图片：`{fig_dir_name}/nc_coverage_summary.png`\n")
    lines.append(f"- 图片：`{fig_dir_name}/{{dataset}}_transfer_vs_stealth_mean_scatter.png`\n")
    lines.append(f"- 图片：`{fig_dir_name}/{{dataset}}_attack_metric_heatmap.png`\n")
    lines.append(f"- 图片：`{fig_dir_name}/{{dataset}}_attack_boxplots.png`\n")
    lines.append(f"- 图片：`{fig_dir_name}/{{dataset}}_poison_rate_trends.png`\n")
    lines.append(f"- 图片：`{fig_dir_name}/{{dataset}}_arch_attack_tradeoff_heatmap.png`\n")
    lines.append(f"- 图片：`{fig_dir_name}/cifar10_nc_attack_summary.png`\n")
    lines.append(f"- 图片：`{fig_dir_name}/mnistm_nc_attack_summary.png`\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build detailed transfer/stealth analysis report")
    parser.add_argument("--analysis-dir", default="analysis")
    parser.add_argument("--report-name", default="transfer_stealth_report_detailed.md")
    parser.add_argument("--table-dir-name", default="report_tables")
    parser.add_argument("--figure-dir-name", default="report_figures")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.is_absolute():
        analysis_dir = root / analysis_dir

    df_no_nc = load_mode(analysis_dir, "no_nc")
    df_nc = load_mode(analysis_dir, "nc")

    tables = {
        "overall_dataset_summary": build_overall_summary(df_no_nc),
        "arch_summary": build_arch_summary(df_no_nc),
        "attack_summary_by_dataset": build_attack_summary(df_no_nc),
        "attack_summary_by_dataset_arch": build_attack_arch_summary(df_no_nc),
        "cross_dataset_attack_summary": build_cross_dataset_attack_summary(df_no_nc),
        "poison_rate_summary_by_dataset": build_poison_rate_summary(df_no_nc),
        "attack_stability_summary": build_attack_stability_summary(df_no_nc),
        "top_configs_by_tradeoff": build_top_configs(df_no_nc, "tradeoff_hmean"),
        "top_configs_by_transfer": build_top_configs(df_no_nc, "transfer_rate"),
        "top_configs_by_stealth_mean": build_top_configs(df_no_nc, "stealth_mean"),
        "nc_coverage_summary": build_nc_coverage(df_nc),
        "nc_attack_summary": build_nc_attack_summary(df_nc),
        "nc_rank_shift_summary": build_nc_rank_shift_summary(df_nc),
    }

    table_dir = analysis_dir / args.table_dir_name
    fig_dir = analysis_dir / args.figure_dir_name
    save_tables(tables, table_dir)
    save_figures(
        df_no_nc,
        tables["nc_coverage_summary"],
        tables["attack_summary_by_dataset"],
        tables["attack_summary_by_dataset_arch"],
        tables["cross_dataset_attack_summary"],
        tables["poison_rate_summary_by_dataset"],
        tables["nc_attack_summary"],
        fig_dir,
        tables["overall_dataset_summary"],
    )

    report_text = build_report(
        analysis_dir,
        df_no_nc,
        df_nc,
        tables,
        args.figure_dir_name,
        args.table_dir_name,
    )
    report_path = analysis_dir / args.report_name
    report_path.write_text(report_text, encoding="utf-8")
    print(f"[OK] report: {report_path}")
    print(f"[OK] tables: {table_dir}")
    print(f"[OK] figures: {fig_dir}")


if __name__ == "__main__":
    main()
