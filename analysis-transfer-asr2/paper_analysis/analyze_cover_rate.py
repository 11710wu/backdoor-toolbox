#!/usr/bin/env python3
"""Cover-rate analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import line_plot, md_table, save_figure, simple_heatmap, write_figure_doc, write_report
from stats_utils import grouped_correlations, main_df, pairwise_adjacent_delta, save_csv_and_md, summarize_delta


def _add_parallel_quantile_band(ax, sub: pd.DataFrame, color, label: str | None = None) -> None:
    points = sub[["transfer_rate", "stealthiness"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(points) < 5 or points["transfer_rate"].nunique() < 2:
        return
    x = points["transfer_rate"].to_numpy()
    y = points["stealthiness"].to_numpy()
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return
    residual = y - (slope * x + intercept)
    lo, hi = pd.Series(residual).quantile([0.10, 0.90])
    x0, x1 = points["transfer_rate"].quantile([0.05, 0.95])
    if x1 <= x0:
        return
    xs = [x0, x1]
    lower = [slope * x0 + intercept + lo, slope * x1 + intercept + lo]
    upper = [slope * x0 + intercept + hi, slope * x1 + intercept + hi]
    ax.plot(xs, lower, color=color, linewidth=2.6, linestyle="--", label=label)
    ax.plot(xs, upper, color=color, linewidth=2.6, linestyle="--")


def _cover_rate_scatter_with_parallel_bands(cover: pd.DataFrame) -> None:
    plot_df = cover[cover["attack_type"].ne("WaNet")].dropna(subset=["transfer_rate", "stealthiness", "attack_type"]).copy()
    if plot_df.empty:
        from plot_utils import insufficient_figure

        insufficient_figure("cover_rate_metric_curves.png", "no non-WaNet cover-rate rows")
        return
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11.5, 7))
    attacks = sorted(plot_df["attack_type"].dropna().unique())
    palette = dict(zip(attacks, sns.color_palette("tab10", n_colors=len(attacks))))
    sns.scatterplot(
        data=plot_df,
        x="transfer_rate",
        y="stealthiness",
        hue="attack_type",
        style="dataset",
        palette=palette,
        s=70,
        alpha=0.62,
        linewidth=0.25,
        edgecolor="white",
        ax=ax,
    )
    for attack in attacks:
        _add_parallel_quantile_band(ax, plot_df[plot_df["attack_type"] == attack], palette[attack], f"{attack} 10-90% band")
    ax.set_title("Cover-rate Transfer-Stealth Scatter (WaNet Removed)")
    ax.set_xlabel("target-domain ASR")
    ax.set_ylabel("stealthiness")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.04, 1.04)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10, title="attack / dataset / band")
    save_figure(fig, "cover_rate_metric_curves.png")
    write_figure_doc("cover_rate_metric_curves.png")


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = main_df(df)
    cover = d[(d["result_group"] == "cover_rate") & pd.to_numeric(d["cover_rate"], errors="coerce").notna()].copy()
    group_cols = ["dataset", "transfer_dataset", "transfer_variant", "arch_base", "attack_type", "poison_rate", "strength_name", "strength_value"]
    delta = pairwise_adjacent_delta(cover, group_cols, "cover_rate", "cover_adjacent")
    summary = summarize_delta(delta, ["dataset", "attack_type"])
    relationship = grouped_correlations(cover, [])
    corr = grouped_correlations(cover, ["dataset", "attack_type"], x="cover_rate", y="transfer_rate")
    corr_stealth = grouped_correlations(cover, ["dataset", "attack_type"], x="cover_rate", y="stealthiness").rename(
        columns={"spearman": "spearman_cover_stealthiness", "pearson": "pearson_cover_stealthiness"}
    )
    corr = corr.merge(corr_stealth[["dataset", "attack_type", "spearman_cover_stealthiness", "pearson_cover_stealthiness"]], on=["dataset", "attack_type"], how="outer")
    save_csv_and_md(delta, COEFFICIENT_DIR / "cover_rate_pairwise_delta.csv", "Cover Rate Pairwise Delta")
    save_csv_and_md(corr, COEFFICIENT_DIR / "cover_rate_correlations.csv", "Cover Rate Correlations")
    save_csv_and_md(summary, TABLE_DIR / "table_5_cover_rate_summary.csv", "Cover Rate Summary")

    _cover_rate_scatter_with_parallel_bands(cover)
    path_df = (
        cover.groupby(["attack_type", "cover_rate"], dropna=False)
        .agg(transfer_rate=("transfer_rate", "median"), stealthiness=("stealthiness", "median"))
        .reset_index()
    )
    line_plot(path_df, "cover_rate_transfer_stealth_paths.png", "transfer_rate", "stealthiness", "attack_type", "Cover-rate Paths in Transfer-Stealth Plane")
    heat = summary.melt(id_vars=["dataset", "attack_type"], value_vars=[c for c in summary.columns if c in ["delta_transfer_rate_mean", "delta_stealthiness_mean"]], var_name="metric", value_name="delta") if not summary.empty else pd.DataFrame()
    simple_heatmap(heat, "cover_rate_pairwise_delta_heatmap.png", "attack_type", "metric", "delta", "Cover-rate Pairwise Delta Heatmap")

    write_report(
        REPORT_DIR / "03_cover_rate_analysis.md",
        "03 Cover-rate Analysis",
        [
            (
                "Relationship-first framing",
                "Cover-rate is treated as a gradient for observing how transferability and stealthiness move relative to each other. The main question is not whether cover_rate alone determines target-domain ASR or stealthiness; it is whether cover changes move points along the transfer-stealth trade-off plane. Cover-to-single-metric correlations are auxiliary mechanism evidence.\n\n"
                + md_table(relationship, 20),
            ),
            ("Pairwise delta summary", md_table(summary, 80)),
            ("Auxiliary cover-rate-to-single-metric correlations", md_table(corr, 80)),
            ("Interpretation draft", "Cover-rate pairwise deltas are used to observe transfer-stealth movement under matched attack configurations. A cover-rate correlation with target ASR or stealthiness should be interpreted as an explanation of that movement, not as the main result by itself."),
        ],
    )
    return {"delta": delta, "summary": summary, "correlations": corr, "relationship": relationship}


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
