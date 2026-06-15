#!/usr/bin/env python3
"""Label-mode / all-to-one analysis."""

from __future__ import annotations

import pandas as pd

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import grouped_bar, md_table, scatter_with_binned_line, write_report
from stats_utils import main_df, save_csv_and_md, summarize_delta


def _label_mode_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = df[df["result_group"].isin(["baseline_strength", "label_mode"])].copy()
    key_cols = ["dataset", "arch_base", "attack_type", "poison_rate", "strength_name", "strength_value"]
    for keys, group in sub.groupby(key_cols, dropna=False):
        if group["label_mode"].nunique(dropna=False) < 2:
            continue
        base = group[group["label_mode"].isin(["default", "clean"])].head(1)
        if base.empty:
            base = group.head(1)
        base_row = base.iloc[0]
        for _, row in group.iterrows():
            if row["result_dir"] == base_row["result_dir"]:
                continue
            rec = {col: val for col, val in zip(key_cols, keys)}
            rec["comparison_group"] = f"{base_row['label_mode']}->{row['label_mode']}"
            for metric in ["clean_acc", "source_asr", "transfer_asr", "transfer_rate", "stealthiness"]:
                rec[f"delta_{metric}"] = row[metric] - base_row[metric]
            rows.append(rec)
    return pd.DataFrame(rows)


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = main_df(df)
    label_related = d[d["result_group"].isin(["baseline_strength", "label_mode"])].copy()
    delta = _label_mode_deltas(label_related)
    summary = summarize_delta(delta, ["dataset", "attack_type", "comparison_group"]) if not delta.empty else pd.DataFrame()
    completeness = (
        df[df["result_group"] == "label_mode"]
        .groupby(["dataset", "arch_base", "attack_type", "label_mode", "analysis_status"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    save_csv_and_md(delta, COEFFICIENT_DIR / "label_mode_pairwise_delta.csv", "Label Mode Pairwise Delta")
    save_csv_and_md(summary, TABLE_DIR / "table_6_label_mode_summary.csv", "Label Mode Summary")
    save_csv_and_md(completeness, TABLE_DIR / "label_mode_completeness.csv", "Label Mode Completeness")

    scatter_with_binned_line(
        label_related,
        "label_mode_transfer_vs_stealth.png",
        "transfer_rate",
        "stealthiness",
        hue="label_mode",
        col="dataset",
        title="Label Mode in Transfer-Stealth Plane",
    )
    plot_delta = summary.melt(
        id_vars=["dataset", "attack_type", "comparison_group"],
        value_vars=[c for c in summary.columns if c in ["delta_transfer_rate_mean", "delta_stealthiness_mean"]],
        var_name="metric",
        value_name="delta",
    ) if not summary.empty else pd.DataFrame()
    grouped_bar(plot_delta, "label_mode_pairwise_delta.png", "comparison_group", "delta", "metric", "Label Mode Pairwise Delta")
    completeness_plot = completeness.groupby(["attack_type", "analysis_status"], dropna=False)["n"].sum().reset_index()
    grouped_bar(completeness_plot, "label_mode_completeness.png", "attack_type", "n", "analysis_status", "Label Mode Completeness")

    write_report(
        REPORT_DIR / "04_label_mode_analysis.md",
        "04 Label-mode Analysis",
        [
            ("Completeness", md_table(completeness, 80)),
            ("Pairwise delta summary", md_table(summary, 80)),
            ("Interpretation draft", "Label-mode conclusions should only be treated as strong when matched configurations exist and completeness is high. Pending UPGD/all-to-one rows remain visible in the completeness table."),
        ],
    )
    return {"delta": delta, "summary": summary, "completeness": completeness}


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
