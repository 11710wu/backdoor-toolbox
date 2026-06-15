#!/usr/bin/env python3
"""RQ1: transfer_rate vs stealthiness full analysis."""

from __future__ import annotations

import pandas as pd

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import grouped_bar, md_table, scatter_with_binned_line, simple_heatmap, write_report
from stats_utils import grouped_correlations, main_df, ols_records, save_csv_and_md


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = main_df(df)
    baseline = d[d["result_group"] == "baseline_strength"].copy()
    outputs: dict[str, pd.DataFrame] = {}

    outputs["overall"] = grouped_correlations(d, [])
    outputs["by_dataset"] = grouped_correlations(d, ["dataset"])
    outputs["by_attack"] = grouped_correlations(d, ["attack_type"])
    outputs["by_dataset_attack"] = grouped_correlations(d, ["dataset", "attack_type"])
    outputs["by_arch"] = grouped_correlations(d, ["dataset", "arch_base"])

    for name, table in [
        ("rq1_overall_correlations.csv", outputs["overall"]),
        ("rq1_by_dataset_correlations.csv", outputs["by_dataset"]),
        ("rq1_by_attack_correlations.csv", outputs["by_attack"]),
        ("rq1_by_dataset_attack_correlations.csv", outputs["by_dataset_attack"]),
        ("rq1_by_arch_correlations.csv", outputs["by_arch"]),
    ]:
        save_csv_and_md(table, COEFFICIENT_DIR / name, name.replace(".csv", ""))

    sensitivity_rows = []
    for threshold in [0.05, 0.10]:
        sub = df[
            (df["complete_source"].astype(bool))
            & (df["complete_transfer"].astype(bool))
            & (df["complete_defense_results"].astype(bool))
            & (pd.to_numeric(df["source_asr"], errors="coerce") >= threshold)
        ].copy()
        rec = grouped_correlations(sub, []).iloc[0].to_dict()
        rec["source_asr_threshold"] = threshold
        sensitivity_rows.append(rec)
    sensitivity = pd.DataFrame(sensitivity_rows)
    outputs["sensitivity"] = sensitivity
    save_csv_and_md(sensitivity, COEFFICIENT_DIR / "rq1_source_asr_threshold_sensitivity.csv", "RQ1 Source ASR Threshold Sensitivity")

    ols_basic = ols_records(d, "stealthiness ~ transfer_rate", "stealthiness", ["transfer_rate"])
    ols_controls = ols_records(
        d,
        "stealthiness ~ transfer_rate + clean_acc + source_asr + C(dataset) + C(attack_type) + C(arch_base)",
        "stealthiness",
        ["transfer_rate", "clean_acc", "source_asr"],
        ["dataset", "attack_type", "arch_base"],
    )
    save_csv_and_md(ols_basic, COEFFICIENT_DIR / "rq1_ols_basic.csv", "RQ1 OLS Basic")
    save_csv_and_md(ols_controls, COEFFICIENT_DIR / "rq1_ols_with_controls.csv", "RQ1 OLS With Controls")

    summary = (
        d.groupby(["dataset", "attack_type"], dropna=False)
        .agg(
            n=("transfer_rate", "size"),
            transfer_rate_median=("transfer_rate", "median"),
            stealthiness_median=("stealthiness", "median"),
            source_asr_median=("source_asr", "median"),
        )
        .reset_index()
    )
    save_csv_and_md(summary, TABLE_DIR / "table_3_rq1_by_dataset_attack.csv", "RQ1 by Dataset and Attack")
    save_csv_and_md(outputs["overall"], TABLE_DIR / "table_2_rq1_overall_summary.csv", "RQ1 Overall Summary")

    defense = d.melt(
        id_vars=["attack_type"],
        value_vars=["stealth_sentinet", "stealth_scaleup", "stealth_strip", "stealth_ibd_psc"],
        var_name="defense",
        value_name="stealth_component",
    )
    defense_summary = (
        defense.groupby(["attack_type", "defense"], dropna=False)
        .agg(n=("stealth_component", "size"), stealth_mean=("stealth_component", "mean"))
        .reset_index()
    )
    save_csv_and_md(defense_summary, TABLE_DIR / "table_10_defense_breakdown.csv", "Defense Breakdown")

    scatter_with_binned_line(
        baseline,
        "rq1_dataset_facets_scatter_binned.png",
        "transfer_rate",
        "stealthiness",
        hue="attack_type",
        col="dataset",
        title="RQ1: Transfer Rate vs Stealthiness by Dataset",
    )

    rank_rows = []
    for dataset, sub in baseline.groupby("dataset", dropna=False):
        sub = sub.dropna(subset=["transfer_rate", "stealthiness"]).copy()
        if len(sub) < 6:
            continue
        sub["rank_bin"] = pd.qcut(sub["transfer_rate"].rank(method="first"), q=min(10, len(sub)), labels=False, duplicates="drop")
        agg = sub.groupby("rank_bin", observed=False).agg(stealthiness=("stealthiness", "median")).reset_index()
        agg["dataset"] = dataset
        rank_rows.append(agg)
    rank_df = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()
    from plot_utils import line_plot

    line_plot(rank_df, "rq1_rank_binned_trend_by_dataset.png", "rank_bin", "stealthiness", "dataset", "RQ1 Rank-binned Trend by Dataset")
    simple_heatmap(outputs["by_dataset_attack"], "rq1_attack_dataset_spearman_heatmap.png", "attack_type", "dataset", "spearman", "RQ1 Spearman by Attack and Dataset")

    metric_heat = summary.copy()
    metric_heat["transfer_rate_median_norm"] = metric_heat.groupby("dataset")["transfer_rate_median"].transform(lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0)
    metric_heat["stealthiness_median_norm"] = metric_heat.groupby("dataset")["stealthiness_median"].transform(lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0)
    metric_long = metric_heat.melt(id_vars=["attack_type"], value_vars=["transfer_rate_median_norm", "stealthiness_median_norm"], var_name="metric", value_name="value")
    simple_heatmap(metric_long, "rq1_attack_metric_heatmap.png", "attack_type", "metric", "value", "RQ1 Attack Metric Heatmap", cmap="viridis", center=None)
    grouped_bar(defense_summary, "rq1_defense_breakdown_by_attack.png", "attack_type", "stealth_mean", "defense", "RQ1 Defense Breakdown by Attack")
    grouped_bar(sensitivity, "rq1_source_asr_threshold_sensitivity.png", "source_asr_threshold", "spearman", None, "RQ1 Source-ASR Threshold Sensitivity", rotate=False)

    write_report(
        REPORT_DIR / "01_rq1_tradeoff_full_analysis.md",
        "01 RQ1 Transfer-Stealth Full Analysis",
        [
            ("Overall correlation", md_table(outputs["overall"])),
            ("Dataset correlations", md_table(outputs["by_dataset"])),
            ("Attack correlations", md_table(outputs["by_attack"], 40)),
            ("Dataset-attack correlations", md_table(outputs["by_dataset_attack"], 60)),
            ("Source-ASR threshold sensitivity", md_table(sensitivity)),
            (
                "Interpretation draft",
                "Negative Spearman values support a transfer-stealth tradeoff. Weak or positive cells should be checked against sample size, pending results, and attack-specific behavior.",
            ),
        ],
    )
    return outputs


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
