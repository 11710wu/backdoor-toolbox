#!/usr/bin/env python3
"""Architecture and ACC analysis."""

from __future__ import annotations

import itertools

import pandas as pd

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import grouped_bar, md_table, scatter_with_binned_line, simple_heatmap, write_report
from stats_utils import correlation_record, grouped_correlations, main_df, save_csv_and_md, summarize_delta


def _arch_pairwise_delta(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["result_group"].isin(["baseline_strength", "arch_acc"])].copy()
    key_cols = ["dataset", "attack_type", "poison_rate", "strength_name", "strength_value", "cover_rate", "label_mode"]
    rows = []
    for keys, group in sub.groupby(key_cols, dropna=False):
        arch_rows = list(group.iterrows())
        if len(arch_rows) < 2:
            continue
        for (_, left), (_, right) in itertools.combinations(arch_rows, 2):
            if left["arch_base"] == right["arch_base"]:
                continue
            if {left["result_group"], right["result_group"]} != {"baseline_strength", "arch_acc"}:
                continue
            # Put supplement architectures first when present.
            if right["result_group"] == "arch_acc":
                new, base = right, left
            else:
                new, base = left, right
            rec = {col: val for col, val in zip(key_cols, keys)}
            rec["comparison_group"] = f"{new['arch_base']}-{base['arch_base']}"
            rec["new_arch"] = new["arch_base"]
            rec["base_arch"] = base["arch_base"]
            for metric in ["clean_acc", "source_asr", "transfer_asr", "transfer_rate", "stealthiness"]:
                rec[f"delta_{metric}"] = new[metric] - base[metric]
            rows.append(rec)
    return pd.DataFrame(rows)


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = main_df(df)
    arch_df = d[d["result_group"].isin(["baseline_strength", "arch_acc"])].copy()
    delta = _arch_pairwise_delta(arch_df)
    summary = summarize_delta(delta, ["dataset", "comparison_group", "new_arch", "base_arch"]) if not delta.empty else pd.DataFrame()
    rel_by_arch = grouped_correlations(arch_df, ["dataset", "arch_base"])
    shift_rows = []
    if not summary.empty:
        for _, row in summary[["dataset", "comparison_group", "new_arch", "base_arch"]].drop_duplicates().iterrows():
            base_sub = arch_df[(arch_df["dataset"] == row["dataset"]) & (arch_df["arch_base"] == row["base_arch"])]
            new_sub = arch_df[(arch_df["dataset"] == row["dataset"]) & (arch_df["arch_base"] == row["new_arch"])]
            base_rec = correlation_record(base_sub, "base")
            new_rec = correlation_record(new_sub, "new")
            shift_rows.append(
                {
                    "dataset": row["dataset"],
                    "comparison_group": row["comparison_group"],
                    "new_arch": row["new_arch"],
                    "base_arch": row["base_arch"],
                    "base_n": base_rec["n"],
                    "new_n": new_rec["n"],
                    "base_spearman": base_rec["spearman"],
                    "new_spearman": new_rec["spearman"],
                    "delta_spearman": new_rec["spearman"] - base_rec["spearman"],
                    "base_pearson": base_rec["pearson"],
                    "new_pearson": new_rec["pearson"],
                    "delta_pearson": new_rec["pearson"] - base_rec["pearson"],
                }
            )
    rel = pd.DataFrame(shift_rows)
    acc_transfer = grouped_correlations(arch_df, ["dataset", "arch_base"], x="clean_acc", y="transfer_rate")
    acc_stealth = grouped_correlations(arch_df, ["dataset", "arch_base"], x="clean_acc", y="stealthiness")
    acc_shift = acc_transfer[["dataset", "arch_base", "n", "spearman", "pearson"]].rename(columns={"spearman": "spearman_acc_transfer", "pearson": "pearson_acc_transfer"})
    acc_shift = acc_shift.merge(
        acc_stealth[["dataset", "arch_base", "spearman", "pearson"]].rename(columns={"spearman": "spearman_acc_stealth", "pearson": "pearson_acc_stealth"}),
        on=["dataset", "arch_base"],
        how="outer",
    )

    save_csv_and_md(delta, COEFFICIENT_DIR / "arch_pairwise_delta.csv", "Architecture Pairwise Delta")
    save_csv_and_md(rel, COEFFICIENT_DIR / "arch_relationship_shift.csv", "Architecture Relationship Shift")
    save_csv_and_md(rel_by_arch, COEFFICIENT_DIR / "arch_relationship_by_arch.csv", "Architecture Relationship by Arch")
    save_csv_and_md(acc_shift, COEFFICIENT_DIR / "arch_acc_correlation_shift.csv", "Architecture ACC Correlation Shift")
    arch_summary = (
        arch_df.groupby(["dataset", "arch_base", "attack_type"], dropna=False)
        .agg(n=("transfer_rate", "size"), clean_acc=("clean_acc", "median"), transfer_rate=("transfer_rate", "median"), stealthiness=("stealthiness", "median"))
        .reset_index()
    )
    save_csv_and_md(arch_summary, TABLE_DIR / "table_7_arch_summary.csv", "Architecture Summary")

    metric = arch_summary.groupby(["dataset", "arch_base"], dropna=False).agg(clean_acc=("clean_acc", "median"), transfer_rate=("transfer_rate", "median"), stealthiness=("stealthiness", "median")).reset_index()
    metric["dataset_arch"] = metric["dataset"].astype(str) + ":" + metric["arch_base"].astype(str)
    metric_long = metric.melt(id_vars=["dataset_arch"], value_vars=["clean_acc", "transfer_rate", "stealthiness"], var_name="metric", value_name="value")
    grouped_bar(metric_long, "arch_metric_overview.png", "dataset_arch", "value", "metric", "Architecture Metric Overview")

    plot_delta = summary.melt(id_vars=["dataset", "comparison_group"], value_vars=[c for c in summary.columns if c in ["delta_clean_acc_mean", "delta_transfer_rate_mean", "delta_stealthiness_mean"]], var_name="metric", value_name="delta") if not summary.empty else pd.DataFrame()
    grouped_bar(plot_delta, "arch_pairwise_delta_summary.png", "comparison_group", "delta", "metric", "Architecture Pairwise Delta Summary")
    grouped_bar(rel, "arch_relationship_shift_spearman.png", "comparison_group", "delta_spearman", "dataset", "Architecture Spearman Relationship Shift")
    acc_long = acc_shift.melt(id_vars=["dataset", "arch_base"], value_vars=["spearman_acc_transfer", "spearman_acc_stealth"], var_name="metric", value_name="spearman")
    grouped_bar(acc_long, "arch_acc_vs_transfer_and_stealth.png", "arch_base", "spearman", "metric", "ACC Relationships with Transfer and Stealth")
    scatter_with_binned_line(arch_df, "arch_transfer_vs_stealth_by_arch.png", "transfer_rate", "stealthiness", hue="arch_base", col="dataset", title="Architecture Transfer-Stealth Plane")
    simple_heatmap(arch_summary, "arch_attack_heatmap.png", "attack_type", "arch_base", "transfer_rate", "Architecture Attack Heatmap", cmap="viridis", center=None)
    defense = arch_df.melt(id_vars=["arch_base"], value_vars=["stealth_sentinet", "stealth_scaleup", "stealth_strip", "stealth_ibd_psc"], var_name="defense", value_name="stealth_component")
    defense_sum = defense.groupby(["arch_base", "defense"], dropna=False).agg(stealth_mean=("stealth_component", "mean")).reset_index()
    grouped_bar(defense_sum, "arch_defense_breakdown.png", "arch_base", "stealth_mean", "defense", "Architecture Defense Breakdown")

    write_report(
        REPORT_DIR / "05_arch_acc_analysis.md",
        "05 Architecture / ACC Analysis",
        [
            ("Pairwise delta summary", md_table(summary, 80)),
            ("Transfer-stealth relationship shift", md_table(rel, 80)),
            ("Transfer-stealth relationship by architecture", md_table(rel_by_arch, 80)),
            ("ACC correlation shift", md_table(acc_shift, 80)),
            ("Interpretation draft", "Architecture results should emphasize matched deltas and relationship shifts. ACC should not be over-stated as a sole cause when correlations vary by architecture and attack."),
        ],
    )
    return {"delta": delta, "summary": summary, "relationship": rel, "relationship_by_arch": rel_by_arch, "acc_shift": acc_shift}


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
