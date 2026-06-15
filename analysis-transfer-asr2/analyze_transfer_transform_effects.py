#!/usr/bin/env python3
"""Compare raw, centered-sigmoid, and log-sigmoid transfer scales.

This script belongs to the older analysis path and intentionally does not
modify analysis-transfer-asr2/paper_analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transfer_metric import (
    LOGSIGMOID_K,
    SIGMOID_LINEAR_K,
    TRANSFER_EPS,
    compute_transfer_logsigmoid_score,
    compute_transfer_sigmoid_score,
    score_to_contrast,
    sigmoid,
)


DEFAULT_INPUTS = [
    ("baseline_full", Path("baseline_full_analysis/baseline_full_acc_transfer_stealth_rows.csv")),
    ("arch_acc", Path("arch_acc_analysis/arch_acc_transfer_stealth_rows.csv")),
    ("combined_arch_noise", Path("arch_acc_analysis/combined_acc_effect_rows.csv")),
]

TRANSFER_COLS = [
    "transfer_rate",
    "transfer_sigmoid_contrast",
    "transfer_logsigmoid_contrast",
    "transfer_log_rate",
]


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_rows(analysis_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for source_name, rel_path in DEFAULT_INPUTS:
        path = analysis_dir / rel_path
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "transfer_rate" not in df.columns or "stealth_avg" not in df.columns:
            continue
        df = df.copy()
        df["analysis_source"] = source_name
        if "primary_main_analysis" in df.columns:
            df = df[_to_bool_series(df["primary_main_analysis"])]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    for col in ["transfer_rate", "stealth_avg", "clean_acc", "source_asr", "transfer_asr", "difficulty"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["transfer_rate", "stealth_avg"]).copy()
    out = out[out["transfer_rate"] > 0].copy()
    return out


def add_transfer_transforms(df: pd.DataFrame, linear_k: float, log_k: float) -> pd.DataFrame:
    out = df.copy()
    out["transfer_log_rate"] = np.log(np.clip(out["transfer_rate"].to_numpy(dtype=float), TRANSFER_EPS, None))
    out["transfer_sigmoid_score"] = out["transfer_rate"].map(lambda v: compute_transfer_sigmoid_score(v, linear_k))
    out["transfer_logsigmoid_score"] = out["transfer_rate"].map(lambda v: compute_transfer_logsigmoid_score(v, log_k))
    out["transfer_sigmoid_contrast"] = out["transfer_sigmoid_score"].map(score_to_contrast)
    out["transfer_logsigmoid_contrast"] = out["transfer_logsigmoid_score"].map(score_to_contrast)
    return out


def _corr_record(df: pd.DataFrame, source: str, group_type: str, group_name: str, x_col: str) -> Dict[str, object]:
    sub = df[[x_col, "stealth_avg"]].dropna()
    return {
        "analysis_source": source,
        "group_type": group_type,
        "group_name": group_name,
        "x_metric": x_col,
        "n": int(len(sub)),
        "pearson": sub[x_col].corr(sub["stealth_avg"], method="pearson") if len(sub) >= 2 else np.nan,
        "spearman": sub[x_col].corr(sub["stealth_avg"], method="spearman") if len(sub) >= 2 else np.nan,
        "kendall": sub[x_col].corr(sub["stealth_avg"], method="kendall") if len(sub) >= 2 else np.nan,
        "x_median": sub[x_col].median() if len(sub) else np.nan,
        "stealth_median": sub["stealth_avg"].median() if len(sub) else np.nan,
    }


def build_correlation_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for source, source_df in df.groupby("analysis_source", dropna=False):
        for x_col in TRANSFER_COLS:
            records.append(_corr_record(source_df, str(source), "all", "all", x_col))
        for group_col in ["dataset", "attack_type", "arch_base", "variation_source"]:
            if group_col not in source_df.columns:
                continue
            for name, group in source_df.groupby(group_col, dropna=False):
                if len(group) < 5:
                    continue
                for x_col in TRANSFER_COLS:
                    records.append(_corr_record(group, str(source), group_col, str(name), x_col))
    return pd.DataFrame(records)


def build_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for source, group in df.groupby("analysis_source", dropna=False):
        tr = group["transfer_rate"].dropna()
        if tr.empty:
            continue
        records.append(
            {
                "analysis_source": source,
                "n": int(len(tr)),
                "raw_mean": tr.mean(),
                "raw_median": tr.median(),
                "raw_q25": tr.quantile(0.25),
                "raw_q75": tr.quantile(0.75),
                "raw_q95": tr.quantile(0.95),
                "raw_max": tr.max(),
                "share_raw_0p9_1p1": ((tr >= 0.9) & (tr <= 1.1)).mean(),
                "share_raw_0p99_1p01": ((tr >= 0.99) & (tr <= 1.01)).mean(),
                "share_exact_1": (tr.round(12) == 1.0).mean(),
                "sigmoid_contrast_median": group["transfer_sigmoid_contrast"].median(),
                "logsigmoid_contrast_median": group["transfer_logsigmoid_contrast"].median(),
                "share_sigmoid_center_-0p2_0p2": ((group["transfer_sigmoid_contrast"] >= -0.2) & (group["transfer_sigmoid_contrast"] <= 0.2)).mean(),
                "share_logsigmoid_center_-0p2_0p2": ((group["transfer_logsigmoid_contrast"] >= -0.2) & (group["transfer_logsigmoid_contrast"] <= 0.2)).mean(),
            }
        )
    return pd.DataFrame(records)


def build_mapping_table(linear_k: float, log_k: float) -> pd.DataFrame:
    values = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.2, 1.5, 2.0]
    rows = []
    for value in values:
        sig_score = compute_transfer_sigmoid_score(value, linear_k)
        log_score = compute_transfer_logsigmoid_score(value, log_k)
        rows.append(
            {
                "transfer_rate": value,
                "sigmoid_score": sig_score,
                "sigmoid_contrast": score_to_contrast(sig_score),
                "logsigmoid_score": log_score,
                "logsigmoid_contrast": score_to_contrast(log_score),
            }
        )
    return pd.DataFrame(rows)


def _safe_xlim(series: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return (0.0, 1.0)
    lo = float(vals.quantile(0.01))
    hi = float(vals.quantile(0.99))
    if lo == hi:
        lo -= 0.1
        hi += 0.1
    return lo, hi


def plot_mapping_curves(out_dir: Path, linear_k: float, log_k: float) -> None:
    xs = np.linspace(0.5, 1.5, 500)
    sigmoid_scores = [compute_transfer_sigmoid_score(x, linear_k) for x in xs]
    logsigmoid_scores = [compute_transfer_logsigmoid_score(x, log_k) for x in xs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, xlim, title in [
        (axes[0], (0.5, 1.5), "Mapping over [0.5, 1.5]"),
        (axes[1], (0.85, 1.15), "Zoom around 1"),
    ]:
        ax.plot(xs, sigmoid_scores, label=f"sigmoid(k*(r-1)), k={linear_k:g}", linewidth=2)
        ax.plot(xs, logsigmoid_scores, label=f"sigmoid(k*log(r)), k={log_k:g}", linewidth=2)
        ax.axvline(1.0, color="black", linewidth=0.9, alpha=0.5)
        ax.axhline(0.5, color="black", linewidth=0.9, alpha=0.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("raw transfer_rate")
        ax.set_ylabel("transformed score")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "transfer_transform_mapping_curves.png", dpi=220)
    plt.close(fig)


def plot_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("transfer_rate", "Raw transfer_rate", None),
        ("transfer_sigmoid_contrast", "Centered sigmoid contrast", (-1.02, 1.02)),
        ("transfer_logsigmoid_contrast", "Log-sigmoid contrast", (-1.02, 1.02)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (col, title, xlim) in zip(axes, metrics):
        vals = df[col].dropna()
        if col == "transfer_rate":
            hi = min(float(vals.quantile(0.995)), 2.5) if len(vals) else 2.5
            vals = vals.clip(upper=hi)
            ax.axvline(1.0, color="black", linewidth=0.9, alpha=0.5)
            ax.set_xlim(0, hi)
        else:
            ax.axvline(0.0, color="black", linewidth=0.9, alpha=0.5)
            ax.set_xlim(*xlim)
        ax.hist(vals, bins=45, alpha=0.78, color="#4A90E2", edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "transfer_transform_histograms.png", dpi=220)
    plt.close(fig)


def plot_scatter_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        ("transfer_rate", "raw transfer_rate"),
        ("transfer_sigmoid_contrast", "sigmoid contrast"),
        ("transfer_logsigmoid_contrast", "log-sigmoid contrast"),
    ]
    for source, group in df.groupby("analysis_source", dropna=False):
        if len(group) < 5:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
        for ax, (col, label) in zip(axes, specs):
            sub = group.dropna(subset=[col, "stealth_avg"])
            color = sub["dataset"].astype("category").cat.codes if "dataset" in sub.columns else "#4A90E2"
            ax.scatter(sub[col], sub["stealth_avg"], c=color, cmap="tab10", alpha=0.68, s=25)
            ax.set_xlabel(label)
            ax.set_title(f"{label} (n={len(sub)})")
            ax.grid(True, alpha=0.25)
            if col == "transfer_rate":
                ax.set_xlim(*_safe_xlim(sub[col]))
            else:
                ax.set_xlim(-1.02, 1.02)
        axes[0].set_ylabel("stealth_avg")
        fig.suptitle(f"{source}: transfer scale comparison", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        safe_source = str(source).replace("/", "_")
        fig.savefig(out_dir / f"{safe_source}_scatter_raw_sigmoid_logsigmoid.png", dpi=220)
        plt.close(fig)


def plot_binned_trends(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for source, source_df in df.groupby("analysis_source", dropna=False):
        for col in TRANSFER_COLS:
            sub = source_df.dropna(subset=[col, "stealth_avg"]).copy()
            if len(sub) < 10 or sub[col].nunique() < 4:
                continue
            try:
                sub["bin"] = pd.qcut(sub[col].rank(method="first"), q=min(10, len(sub)), labels=False, duplicates="drop")
            except Exception:
                continue
            trend = (
                sub.groupby("bin", dropna=False)
                .agg(x_median=(col, "median"), stealth_median=("stealth_avg", "median"), n=("stealth_avg", "size"))
                .reset_index()
            )
            trend["analysis_source"] = source
            trend["x_metric"] = col
            rows.append(trend)
    if not rows:
        return
    trend_df = pd.concat(rows, ignore_index=True)
    trend_df.to_csv(out_dir / "transfer_transform_binned_trends.csv", index=False)

    for source, source_df in trend_df.groupby("analysis_source", dropna=False):
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
        for ax, col in zip(axes, TRANSFER_COLS):
            sub = source_df[source_df["x_metric"] == col]
            if sub.empty:
                continue
            ax.plot(sub["x_median"].to_numpy(), sub["stealth_median"].to_numpy(), marker="o", linewidth=2)
            ax.set_xlabel(col)
            ax.set_title(col)
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("median stealth_avg")
        fig.suptitle(f"{source}: rank-binned median trends", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        safe_source = str(source).replace("/", "_")
        fig.savefig(out_dir / f"{safe_source}_binned_trends_raw_sigmoid_logsigmoid.png", dpi=220)
        plt.close(fig)


BASELINE_TRANSFER_SCALES = [
    {
        "name": "raw",
        "column": "transfer_rate",
        "label": "transfer_rate = transfer_asr^2 / source_asr (clipped at 2.0 for display)",
        "clip": 2.0,
        "center": 1.0,
    },
    {
        "name": "sigmoid",
        "column": "transfer_sigmoid_contrast",
        "label": "centered sigmoid contrast: 2*sigmoid(k*(transfer_rate-1))-1",
        "clip": None,
        "center": 0.0,
    },
    {
        "name": "logsigmoid",
        "column": "transfer_logsigmoid_contrast",
        "label": "log-sigmoid contrast: 2*sigmoid(k*log(transfer_rate))-1",
        "clip": None,
        "center": 0.0,
    },
]


def _scale_plot_column(df: pd.DataFrame, spec: Dict[str, object]) -> pd.Series:
    vals = pd.to_numeric(df[str(spec["column"])], errors="coerce")
    clip = spec.get("clip")
    if clip is not None:
        vals = vals.clip(upper=float(clip))
    return vals


def _format_corr_title(sub: pd.DataFrame, x_col: str, dataset: Optional[str] = None) -> str:
    pair = sub[[x_col, "stealth_avg"]].dropna()
    if len(pair) < 2:
        pearson_val = float("nan")
        spearman_val = float("nan")
    else:
        pearson_val = pair[x_col].corr(pair["stealth_avg"], method="pearson")
        spearman_val = pair[x_col].corr(pair["stealth_avg"], method="spearman")
    prefix = f"{dataset}\n" if dataset else ""
    return f"{prefix}Pearson={pearson_val:.3f}, Spearman={spearman_val:.3f}, n={len(pair)}"


def _write_png(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_baseline_combined_by_dataset(baseline: pd.DataFrame, out_dir: Path) -> None:
    colors = {"cifar10": "#1f77b4", "mnistm": "#2ca02c", "tiny_imagenet": "#ff7f0e"}
    for spec in BASELINE_TRANSFER_SCALES:
        plot_df = baseline.dropna(subset=[str(spec["column"]), "stealth_avg", "dataset"]).copy()
        if plot_df.empty:
            continue
        plot_df["x_plot"] = _scale_plot_column(plot_df, spec)
        fig, ax = plt.subplots(figsize=(11, 7))
        for dataset, group in plot_df.groupby("dataset", dropna=False):
            ax.scatter(
                group["x_plot"],
                group["stealth_avg"],
                s=25,
                alpha=0.58,
                color=colors.get(str(dataset), None),
                label=str(dataset),
            )
        ax.axvline(float(spec["center"]), color="black", linewidth=0.9, alpha=0.35)
        ax.set_xlabel(str(spec["label"]))
        ax.set_ylabel("stealth_avg = mean(1 - TPR)")
        ax.set_title(f"Baseline: transfer vs stealth by dataset ({spec['name']})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        if spec.get("clip") is not None:
            ax.set_xlim(left=min(0.0, float(plot_df["x_plot"].min()) - 0.03), right=float(spec["clip"]) + 0.03)
        else:
            ax.set_xlim(-1.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        _write_png(fig, out_dir / f"baseline_by_dataset_{spec['name']}.png")


def plot_baseline_each_dataset(baseline: pd.DataFrame, out_dir: Path) -> None:
    dataset_order = ["cifar10", "mnistm", "tiny_imagenet"]
    for spec in BASELINE_TRANSFER_SCALES:
        for dataset in dataset_order:
            sub = baseline[baseline["dataset"] == dataset].dropna(subset=[str(spec["column"]), "stealth_avg"]).copy()
            if sub.empty:
                continue
            sub["x_plot"] = _scale_plot_column(sub, spec)
            fig, ax = plt.subplots(figsize=(7.5, 5.4))
            ax.scatter(sub["x_plot"], sub["stealth_avg"], s=25, alpha=0.62, color="#1f77b4")
            ax.axvline(float(spec["center"]), color="black", linewidth=0.9, alpha=0.35)
            ax.set_xlabel(str(spec["label"]))
            ax.set_ylabel("stealth_avg = mean(1 - TPR)")
            ax.set_title(_format_corr_title(sub, str(spec["column"]), dataset=dataset))
            ax.grid(True, alpha=0.25)
            if spec.get("clip") is not None:
                ax.set_xlim(left=min(0.0, float(sub["x_plot"].min()) - 0.03), right=float(spec["clip"]) + 0.03)
            else:
                ax.set_xlim(-1.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            _write_png(fig, out_dir / f"baseline_{dataset}_{spec['name']}.png")


def build_baseline_requested_correlations(baseline: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for spec in BASELINE_TRANSFER_SCALES:
        col = str(spec["column"])
        for dataset_name, group in [("all", baseline)] + list(baseline.groupby("dataset", dropna=False)):
            pair = group[[col, "stealth_avg"]].dropna()
            records.append(
                {
                    "scale": spec["name"],
                    "x_metric": col,
                    "dataset": str(dataset_name),
                    "n": int(len(pair)),
                    "pearson": pair[col].corr(pair["stealth_avg"], method="pearson") if len(pair) >= 2 else np.nan,
                    "spearman": pair[col].corr(pair["stealth_avg"], method="spearman") if len(pair) >= 2 else np.nan,
                    "kendall": pair[col].corr(pair["stealth_avg"], method="kendall") if len(pair) >= 2 else np.nan,
                    "x_median": pair[col].median() if len(pair) else np.nan,
                    "stealth_median": pair["stealth_avg"].median() if len(pair) else np.nan,
                }
            )
    return pd.DataFrame(records)


def write_baseline_requested_report(out_dir: Path, corr: pd.DataFrame, baseline: pd.DataFrame) -> None:
    def table(df: pd.DataFrame) -> str:
        if df.empty:
            return "No rows."
        return df.to_markdown(index=False)

    files = sorted(p.name for p in out_dir.glob("*.png"))
    lines = [
        "# Baseline Transfer Transform Separate Figures",
        "",
        "这组图只针对旧 baseline 图做 raw / sigmoid / log-sigmoid 对比，且每张图单独保存，方便逐张观察。",
        "",
        "## 数学定义",
        "",
        "```text",
        "raw:          r = transfer_rate = transfer_asr^2 / source_asr",
        "sigmoid:      2 * sigmoid(11 * (r - 1)) - 1",
        "log-sigmoid:  2 * sigmoid(11.5 * log(r)) - 1",
        "```",
        "",
        "raw 图仍然以 `r=1` 为中心；sigmoid 和 log-sigmoid 图以 `contrast=0` 为中心。",
        "",
        "## 数据",
        "",
        f"- baseline rows: `{len(baseline)}`",
        "- 数据来源：`baseline_full_analysis/baseline_full_acc_transfer_stealth_rows.csv`",
        "",
        "## 相关性对比",
        "",
        table(corr[["scale", "dataset", "n", "pearson", "spearman", "kendall", "x_median", "stealth_median"]]),
        "",
        "## 生成图片",
        "",
    ]
    lines.extend(f"- `{name}`" for name in files)
    (out_dir / "README_BASELINE_TRANSFER_TRANSFORM_COMPARISON_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_baseline_rows_for_requested_comparison(analysis_dir: Path, linear_k: float, log_k: float) -> pd.DataFrame:
    path = analysis_dir / "baseline_full_analysis" / "baseline_full_acc_transfer_stealth_rows.csv"
    if not path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(path)
    for col in ["transfer_rate", "stealth_avg", "clean_acc", "source_asr", "transfer_asr", "difficulty"]:
        if col in baseline.columns:
            baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
    if "include_main_analysis" in baseline.columns:
        baseline = baseline[_to_bool_series(baseline["include_main_analysis"])].copy()
    baseline = baseline.dropna(subset=["transfer_rate", "stealth_avg", "dataset"]).copy()
    baseline = baseline[baseline["transfer_rate"] > 0].copy()
    baseline["analysis_source"] = "baseline_full_requested"
    return add_transfer_transforms(baseline, linear_k, log_k)


def generate_baseline_requested_comparison(analysis_dir: Path, output_dir: Path, linear_k: float, log_k: float) -> None:
    out_dir = output_dir / "baseline_requested_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline_rows_for_requested_comparison(analysis_dir, linear_k, log_k)
    if baseline.empty:
        return
    plot_baseline_combined_by_dataset(baseline, out_dir)
    plot_baseline_each_dataset(baseline, out_dir)
    corr = build_baseline_requested_correlations(baseline)
    corr.to_csv(out_dir / "baseline_transfer_transform_correlations.csv", index=False)
    baseline.to_csv(out_dir / "baseline_transfer_transform_rows.csv", index=False)
    write_baseline_requested_report(out_dir, corr, baseline)


def write_report(
    out_dir: Path,
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    mapping: pd.DataFrame,
    corr: pd.DataFrame,
    linear_k: float,
    log_k: float,
) -> None:
    def table(df: pd.DataFrame, max_rows: int = 40) -> str:
        if df.empty:
            return "No rows."
        shown = df.head(max_rows)
        suffix = f"\n\nShowing first {max_rows} of {len(df)} rows." if len(df) > max_rows else ""
        return shown.to_markdown(index=False) + suffix

    overall_corr = corr[(corr["group_type"] == "all") & (corr["group_name"] == "all")].copy()
    overall_corr = overall_corr[
        ["analysis_source", "x_metric", "n", "pearson", "spearman", "kendall", "x_median", "stealth_median"]
    ]

    lines = [
        "# Transfer Transform Effect Report",
        "",
        "## 1. 数学定义",
        "",
        "原始迁移性仍然固定为：",
        "",
        "```text",
        "transfer_rate = transfer_asr^2 / source_asr",
        "```",
        "",
        "为了把集中在 1 附近的值视觉上分开，本报告只新增两个辅助尺度，不替换原始定义：",
        "",
        "```text",
        f"sigmoid_score = sigmoid({linear_k:g} * (transfer_rate - 1))",
        f"logsigmoid_score = sigmoid({log_k:g} * log(transfer_rate))",
        "contrast = 2 * score - 1",
        "```",
        "",
        "其中 `score=0.5` 或 `contrast=0` 对应 `transfer_rate=1`；小于 1 为负方向，大于 1 为正方向。",
        "",
        "普通 centered sigmoid 使用线性距离 `transfer_rate - 1`；log-sigmoid 使用比例距离 `log(transfer_rate)`，因此 `1.1` 和 `1/1.1` 在数学上更对称。",
        "",
        "## 2. 映射表",
        "",
        table(mapping, 30),
        "",
        "## 3. 当前旧分析表中的集中程度",
        "",
        table(summary, 20),
        "",
        "## 4. 与 stealth_avg 的相关性敏感性",
        "",
        table(overall_corr, 40),
        "",
        "Spearman / Kendall 只依赖排序，所以对严格单调变换通常不会变。Pearson 会变化，因为 sigmoid 和 log-sigmoid 改变了距离尺度。",
        "",
        "## 5. 图片",
        "",
        "- `figures/transfer_transform_mapping_curves.png`：数学映射曲线。",
        "- `figures/transfer_transform_histograms.png`：raw、sigmoid contrast、log-sigmoid contrast 的分布对比。",
        "- `figures/*_scatter_raw_sigmoid_logsigmoid.png`：旧分析数据在三种横轴下的 transfer-stealth 散点。",
        "- `figures/*_binned_trends_raw_sigmoid_logsigmoid.png`：按横轴 rank 分箱后的中位数趋势。",
        "",
        "## 6. 当前建议",
        "",
        "先把 sigmoid / log-sigmoid 当成可视化和敏感性分析辅助尺度。正式指标仍然使用原始 `transfer_rate`，不要直接把主定义改成 sigmoid。",
    ]
    (out_dir / "TRANSFER_TRANSFORM_EFFECT_REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(analysis_dir: Path, output_dir: Path, linear_k: float, log_k: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(analysis_dir)
    if rows.empty:
        raise RuntimeError(f"No usable old analysis rows found under {analysis_dir}")
    rows = add_transfer_transforms(rows, linear_k, log_k)
    rows.to_csv(output_dir / "transfer_transform_rows.csv", index=False)

    summary = build_distribution_summary(rows)
    mapping = build_mapping_table(linear_k, log_k)
    corr = build_correlation_sensitivity(rows)
    summary.to_csv(output_dir / "transfer_transform_distribution_summary.csv", index=False)
    mapping.to_csv(output_dir / "transfer_transform_mapping_table.csv", index=False)
    corr.to_csv(output_dir / "transfer_transform_correlation_sensitivity.csv", index=False)

    plot_mapping_curves(figure_dir, linear_k, log_k)
    plot_histograms(rows, figure_dir)
    plot_scatter_comparison(rows, figure_dir)
    plot_binned_trends(rows, figure_dir)
    generate_baseline_requested_comparison(analysis_dir, output_dir, linear_k, log_k)
    write_report(output_dir, rows, summary, mapping, corr, linear_k, log_k)

    print(f"[OK] rows: {output_dir / 'transfer_transform_rows.csv'}")
    print(f"[OK] report: {output_dir / 'TRANSFER_TRANSFORM_EFFECT_REPORT_CN.md'}")
    print(f"[OK] figures: {figure_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="/workspace/backdoor-toolbox-new1/analysis-transfer-asr2")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--linear-k", type=float, default=SIGMOID_LINEAR_K)
    parser.add_argument("--log-k", type=float, default=LOGSIGMOID_K)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / "transfer_transform_analysis"
    run(analysis_dir, output_dir, args.linear_k, args.log_k)


if __name__ == "__main__":
    main()
