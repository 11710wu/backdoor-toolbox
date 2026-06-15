#!/usr/bin/env python3
"""Noise / input difficulty analysis."""

from __future__ import annotations

import pandas as pd

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import grouped_bar, line_plot, md_table, scatter_with_binned_line, simple_heatmap, write_report
from stats_utils import grouped_correlations, main_df, pairwise_adjacent_delta, save_csv_and_md, summarize_delta


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = main_df(df)
    noise = d[(d["result_group"] == "noise_acc") & (d["input_noise_type"].astype(str) != "")].copy()
    group_cols = ["attack_type", "poison_rate", "strength_name", "strength_value", "cover_rate", "input_noise_type"]
    delta = pairwise_adjacent_delta(noise, group_cols, "input_noise_level", "noise_adjacent")
    summary = summarize_delta(delta, ["attack_type", "input_noise_type"]) if not delta.empty else pd.DataFrame()
    by_level = (
        noise.groupby(["input_noise_type", "input_noise_level", "attack_type"], dropna=False)
        .agg(n=("transfer_rate", "size"), clean_acc=("clean_acc", "median"), transfer_rate=("transfer_rate", "median"), stealthiness=("stealthiness", "median"))
        .reset_index()
    )
    acc_bin_corr = grouped_correlations(noise, ["acc_bin"], x="transfer_rate", y="stealthiness") if "acc_bin" in noise.columns else pd.DataFrame()

    save_csv_and_md(delta, COEFFICIENT_DIR / "noise_pairwise_delta.csv", "Noise Pairwise Delta")
    save_csv_and_md(acc_bin_corr, COEFFICIENT_DIR / "noise_acc_bin_correlations.csv", "Noise ACC-bin Correlations")
    save_csv_and_md(by_level, COEFFICIENT_DIR / "noise_by_type_level.csv", "Noise by Type and Level")
    save_csv_and_md(by_level, TABLE_DIR / "table_8_noise_summary.csv", "Noise Summary")

    line_plot(by_level, "noise_acc_vs_level_by_type.png", "input_noise_level", "clean_acc", "input_noise_type", "Noise ACC vs Level")
    metric_long = by_level.melt(id_vars=["input_noise_type", "input_noise_level"], value_vars=["clean_acc", "transfer_rate", "stealthiness"], var_name="metric", value_name="value")
    line_plot(metric_long, "noise_metric_curves_by_noise_type.png", "input_noise_level", "value", "input_noise_type", "Noise Metric Curves by Type", style="metric")
    plot_delta = summary.melt(id_vars=["attack_type", "input_noise_type"], value_vars=[c for c in summary.columns if c in ["delta_clean_acc_mean", "delta_transfer_rate_mean", "delta_stealthiness_mean"]], var_name="metric", value_name="delta") if not summary.empty else pd.DataFrame()
    grouped_bar(plot_delta, "noise_paired_delta_by_level.png", "input_noise_type", "delta", "metric", "Noise Paired Delta by Level")
    scatter_with_binned_line(noise, "noise_transfer_vs_stealth_by_noise_type.png", "transfer_rate", "stealthiness", hue="input_noise_type", col=None, title="Noise Transfer-Stealth by Type")
    scatter_with_binned_line(noise, "noise_transfer_vs_stealth_by_acc_bin.png", "transfer_rate", "stealthiness", hue="acc_bin", col=None, title="Noise Transfer-Stealth by ACC Bin")
    simple_heatmap(by_level, "noise_attack_heatmap.png", "attack_type", "input_noise_type", "transfer_rate", "Noise Attack Heatmap", cmap="viridis", center=None)
    defense = noise.melt(id_vars=["input_noise_type"], value_vars=["stealth_sentinet", "stealth_scaleup", "stealth_strip", "stealth_ibd_psc"], var_name="defense", value_name="stealth_component")
    defense_sum = defense.groupby(["input_noise_type", "defense"], dropna=False).agg(stealth_mean=("stealth_component", "mean")).reset_index()
    grouped_bar(defense_sum, "noise_defense_breakdown.png", "input_noise_type", "stealth_mean", "defense", "Noise Defense Breakdown")

    write_report(
        REPORT_DIR / "06_noise_acc_analysis.md",
        "06 Noise / Difficulty Analysis",
        [
            ("Noise by type and level", md_table(by_level, 80)),
            ("Noise pairwise delta", md_table(summary, 80)),
            ("Noise ACC-bin correlations", md_table(acc_bin_corr, 80)),
            ("Interpretation draft", "Noise experiments are treated as input-difficulty interventions. Pending noise types or attacks remain visible through completeness tables."),
        ],
    )
    return {"delta": delta, "summary": summary, "by_level": by_level, "acc_bin_corr": acc_bin_corr}


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
