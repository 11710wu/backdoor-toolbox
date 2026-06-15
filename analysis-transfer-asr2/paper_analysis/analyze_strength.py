#!/usr/bin/env python3
"""Attack-strength analysis."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import line_plot, md_table, save_figure, simple_heatmap, write_figure_doc, write_report
from stats_utils import main_df, pairwise_adjacent_delta, save_csv_and_md, summarize_delta


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = main_df(df)
    strength = d[(d["result_group"] == "baseline_strength") & pd.to_numeric(d["strength_value"], errors="coerce").notna()].copy()
    group_cols = ["dataset", "arch_base", "attack_type", "poison_rate", "cover_rate", "label_mode"]
    delta = pairwise_adjacent_delta(strength, group_cols, "strength_value", "strength_adjacent")
    summary = summarize_delta(delta, ["dataset", "attack_type"])
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
    path_df = (
        strength.groupby(["attack_type", "strength_value"], dropna=False)
        .agg(transfer_rate=("transfer_rate", "median"), stealthiness=("stealthiness", "median"))
        .reset_index()
    )
    line_plot(path_df, "strength_transfer_stealth_paths_by_attack.png", "transfer_rate", "stealthiness", "attack_type", "Strength Paths in Transfer-Stealth Plane")
    if not summary.empty:
        heat = summary.melt(id_vars=["dataset", "attack_type"], value_vars=[c for c in summary.columns if c in ["delta_transfer_rate_mean", "delta_stealthiness_mean"]], var_name="metric", value_name="delta")
    else:
        heat = pd.DataFrame()
    simple_heatmap(heat, "strength_pairwise_delta_heatmap.png", "attack_type", "metric", "delta", "Strength Pairwise Delta Heatmap")

    write_report(
        REPORT_DIR / "02_strength_analysis.md",
        "02 Strength Analysis",
        [
            ("Pairwise delta summary", md_table(summary, 80)),
            ("Strength slope by attack", md_table(slopes_df, 80)),
            ("Interpretation draft", "Adjacent strength deltas reveal whether stronger attacks move toward higher transfer_rate and lower stealthiness within matched configurations."),
        ],
    )
    return {"delta": delta, "summary": summary, "slopes": slopes_df}


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
