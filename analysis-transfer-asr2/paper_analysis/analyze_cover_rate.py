#!/usr/bin/env python3
"""Cover-rate analysis."""

from __future__ import annotations

import pandas as pd

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import line_plot, md_table, simple_heatmap, write_report
from stats_utils import grouped_correlations, main_df, pairwise_adjacent_delta, save_csv_and_md, summarize_delta


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = main_df(df)
    cover = d[(d["result_group"] == "cover_rate") & pd.to_numeric(d["cover_rate"], errors="coerce").notna()].copy()
    group_cols = ["dataset", "arch_base", "attack_type", "poison_rate", "strength_name", "strength_value"]
    delta = pairwise_adjacent_delta(cover, group_cols, "cover_rate", "cover_adjacent")
    summary = summarize_delta(delta, ["dataset", "attack_type"])
    corr = grouped_correlations(cover, ["dataset", "attack_type"], x="cover_rate", y="transfer_rate")
    corr_stealth = grouped_correlations(cover, ["dataset", "attack_type"], x="cover_rate", y="stealthiness").rename(
        columns={"spearman": "spearman_cover_stealthiness", "pearson": "pearson_cover_stealthiness"}
    )
    corr = corr.merge(corr_stealth[["dataset", "attack_type", "spearman_cover_stealthiness", "pearson_cover_stealthiness"]], on=["dataset", "attack_type"], how="outer")
    save_csv_and_md(delta, COEFFICIENT_DIR / "cover_rate_pairwise_delta.csv", "Cover Rate Pairwise Delta")
    save_csv_and_md(corr, COEFFICIENT_DIR / "cover_rate_correlations.csv", "Cover Rate Correlations")
    save_csv_and_md(summary, TABLE_DIR / "table_5_cover_rate_summary.csv", "Cover Rate Summary")

    metric_long = (
        cover.groupby(["attack_type", "cover_rate"], dropna=False)
        .agg(source_asr=("source_asr", "median"), transfer_rate=("transfer_rate", "median"), stealthiness=("stealthiness", "median"))
        .reset_index()
        .melt(id_vars=["attack_type", "cover_rate"], var_name="metric", value_name="value")
    )
    line_plot(metric_long, "cover_rate_metric_curves.png", "cover_rate", "value", "attack_type", "Cover-rate Metric Curves", style="metric")
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
            ("Cover-rate correlations", md_table(corr, 80)),
            ("Pairwise delta summary", md_table(summary, 80)),
            ("Interpretation draft", "Cover-rate is analyzed as an internal attack parameter. Pairwise deltas are more reliable than pooled means because they keep attack configuration closer."),
        ],
    )
    return {"delta": delta, "summary": summary, "correlations": corr}


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
