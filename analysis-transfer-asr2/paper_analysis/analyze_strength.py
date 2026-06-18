#!/usr/bin/env python3
"""Attack-strength analysis."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import md_table, save_figure, simple_heatmap, write_figure_doc, write_report
from stats_utils import grouped_correlations, main_df, pairwise_adjacent_delta, save_csv_and_md, summarize_delta


def _add_parallel_quantile_band(ax, sub: pd.DataFrame) -> None:
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
    ax.plot(xs, lower, color="black", linewidth=2.2, linestyle="--")
    ax.plot(xs, upper, color="black", linewidth=2.2, linestyle="--")


def _strength_scatter_by_attack(strength: pd.DataFrame) -> None:
    plot_df = strength.dropna(subset=["transfer_rate", "stealthiness", "attack_type", "dataset"]).copy()
    if plot_df.empty:
        from plot_utils import insufficient_figure

        insufficient_figure("strength_transfer_stealth_paths_by_attack.png", "no strength rows")
        return
    sns.set_theme(style="whitegrid", context="talk")
    attacks = sorted(plot_df["attack_type"].dropna().unique())
    ncols = 4
    nrows = (len(attacks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()
    palette = dict(zip(sorted(plot_df["dataset"].dropna().unique()), sns.color_palette("Set2", n_colors=plot_df["dataset"].nunique())))
    for ax, attack in zip(axes, attacks):
        sub = plot_df[plot_df["attack_type"] == attack]
        sns.scatterplot(
            data=sub,
            x="transfer_rate",
            y="stealthiness",
            hue="dataset",
            palette=palette,
            s=34,
            alpha=0.55,
            linewidth=0.2,
            edgecolor="white",
            legend=False,
            ax=ax,
        )
        _add_parallel_quantile_band(ax, sub)
        ax.set_title(f"{attack} (n={len(sub)})", fontsize=14)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("target-domain ASR")
        ax.set_ylabel("stealthiness")
    for ax in axes[len(attacks) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=dataset, markersize=8)
        for dataset, color in palette.items()
    ]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--", linewidth=2.2, label="10-90% trend band"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5), title="dataset / band", frameon=True, fontsize=11)
    fig.suptitle("Strength Transfer-Stealth Scatter by Attack", y=1.02, fontsize=18)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    save_figure(fig, "strength_transfer_stealth_paths_by_attack.png")
    write_figure_doc("strength_transfer_stealth_paths_by_attack.png")


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = main_df(df)
    strength = d[(d["result_group"] == "baseline_strength") & pd.to_numeric(d["strength_value"], errors="coerce").notna()].copy()
    group_cols = ["dataset", "transfer_dataset", "transfer_variant", "arch_base", "attack_type", "poison_rate", "cover_rate", "label_mode"]
    delta = pairwise_adjacent_delta(strength, group_cols, "strength_value", "strength_adjacent")
    summary = summarize_delta(delta, ["dataset", "attack_type"])
    relationship = grouped_correlations(strength, [])
    save_csv_and_md(delta, COEFFICIENT_DIR / "strength_pairwise_delta.csv", "Strength Pairwise Delta")
    save_csv_and_md(summary, TABLE_DIR / "table_4_strength_summary.csv", "Strength Summary")

    slopes = []
    for keys, sub in strength.groupby(["dataset", "attack_type"], dropna=False):
        sub = sub.dropna(subset=["strength_value", "transfer_rate", "stealthiness"])
        rec = {"dataset": keys[0], "attack_type": keys[1], "n": len(sub)}
        for metric in ["transfer_rate", "stealthiness"]:
            if len(sub) >= 3 and sub["strength_value"].nunique() > 1:
                coef = pd.Series(sub[metric]).corr(pd.Series(sub["strength_value"]), method="spearman")
            else:
                coef = float("nan")
            rec[f"spearman_strength_{metric}"] = coef
        slopes.append(rec)
    slopes_df = pd.DataFrame(slopes)
    save_csv_and_md(slopes_df, COEFFICIENT_DIR / "strength_tradeoff_slope_by_attack.csv", "Strength Tradeoff Slope by Attack")

    metric_agg = (
        strength.groupby(["attack_type", "strength_value"], dropna=False)
        .agg(source_asr=("source_asr", "median"), transfer_rate=("transfer_rate", "median"), stealthiness=("stealthiness", "median"))
        .reset_index()
    )
    metric_agg["strength_order"] = metric_agg.groupby("attack_type")["strength_value"].rank(method="dense").astype(int)
    if metric_agg.empty:
        from plot_utils import insufficient_figure

        insufficient_figure("strength_metric_curves_by_attack.png", "no strength rows")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
        for ax, metric in zip(axes, ["source_asr", "transfer_rate", "stealthiness"]):
            sns.lineplot(data=metric_agg, x="strength_order", y=metric, hue="attack_type", marker="o", ci=None, ax=ax, legend=(metric == "stealthiness"))
            ax.set_title(metric)
            ax.set_xlabel("strength_order")
            ax.set_ylabel(metric)
        fig.suptitle("Strength Metric Curves by Attack (Ordered Within Attack)", y=1.03)
        save_figure(fig, "strength_metric_curves_by_attack.png")
        write_figure_doc("strength_metric_curves_by_attack.png")
    _strength_scatter_by_attack(strength)
    if not summary.empty:
        heat = summary.melt(id_vars=["dataset", "attack_type"], value_vars=[c for c in summary.columns if c in ["delta_transfer_rate_mean", "delta_stealthiness_mean"]], var_name="metric", value_name="delta")
    else:
        heat = pd.DataFrame()
    simple_heatmap(heat, "strength_pairwise_delta_heatmap.png", "attack_type", "metric", "delta", "Strength Pairwise Delta Heatmap")

    write_report(
        REPORT_DIR / "02_strength_analysis.md",
        "02 Strength Analysis",
        [
            (
                "Relationship-first framing",
                "Strength is treated as a path variable for observing the transfer-stealth relationship, not as the final object of the analysis. The primary question is whether points move in the transfer-stealth plane toward higher target-domain ASR and lower stealthiness as the strength path changes. Correlations between strength and each single metric are kept as auxiliary mechanism evidence.\n\n"
                + md_table(relationship, 20),
            ),
            ("Pairwise delta summary", md_table(summary, 80)),
            ("Auxiliary strength-to-single-metric slopes", md_table(slopes_df, 80)),
            ("Interpretation draft", "Adjacent strength deltas should be read as movement in the transfer-stealth plane. The main evidence is the direction of the transfer_rate/stealthiness movement; strength-to-transfer or strength-to-stealth correlations only explain why that relationship appears."),
        ],
    )
    return {"delta": delta, "summary": summary, "slopes": slopes_df, "relationship": relationship}


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
