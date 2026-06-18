#!/usr/bin/env python3
"""Completeness tables and report."""

from __future__ import annotations

import pandas as pd

from config import OUTPUT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import md_table, write_report
from stats_utils import save_csv_and_md


def build_completeness_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    status_cols = ["result_group", "dataset", "transfer_dataset", "arch_base", "attack_type", "analysis_status"]
    comp = (
        df.groupby(status_cols, dropna=False)
        .size()
        .reset_index(name="n_dirs")
        .sort_values(["result_group", "dataset", "arch_base", "attack_type", "analysis_status"])
    )
    missing_rows = []
    for _, row in df.iterrows():
        items = str(row.get("missing_items", "")).split(";") if row.get("missing_items", "") else []
        if not items:
            items = ["none"]
        for item in items:
            missing_rows.append(
                {
                    "result_group": row.get("result_group", ""),
                    "dataset": row.get("dataset", ""),
                    "transfer_dataset": row.get("transfer_dataset", ""),
                    "arch_base": row.get("arch_base", ""),
                    "attack_type": row.get("attack_type", ""),
                    "missing_item": item,
                    "analysis_status": row.get("analysis_status", ""),
                    "result_dir": row.get("result_dir", ""),
                }
            )
    missing = pd.DataFrame(missing_rows)
    missing_summary = (
        missing.groupby(["result_group", "dataset", "transfer_dataset", "arch_base", "attack_type", "missing_item"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["result_group", "dataset", "arch_base", "attack_type", "missing_item"])
    )

    save_csv_and_md(comp, OUTPUT_DIR / "completeness_report.csv", "Completeness Report")
    save_csv_and_md(missing_summary, OUTPUT_DIR / "missing_by_group_attack.csv", "Missing Items by Group and Attack")
    save_csv_and_md(comp, TABLE_DIR / "table_1_experiment_coverage.csv", "Experiment Coverage")

    status_summary = df.groupby(["result_group", "analysis_status"], dropna=False).size().reset_index(name="n")
    write_report(
        REPORT_DIR / "00_data_completeness_report.md",
        "00 Data Completeness Report",
        [
            ("Status by result group", md_table(status_summary, 60)),
            ("Coverage by dataset / arch / attack", md_table(comp, 80)),
            ("Missing item summary", md_table(missing_summary, 80)),
            (
                "Interpretation Guardrail",
                "Only rows with analysis_status=complete enter main analysis. Partial and pending rows are kept so incomplete experiments are visible instead of silently dropped.",
            ),
        ],
    )
    return comp, missing_summary


if __name__ == "__main__":
    df_in = pd.read_csv(OUTPUT_DIR / "master_results.csv")
    build_completeness_tables(df_in)
