#!/usr/bin/env python3
"""Run the complete current-results analysis pipeline."""

from __future__ import annotations

import pandas as pd

from analyze_arch_acc import run as run_arch
from analyze_cover_rate import run as run_cover
from analyze_label_mode import run as run_label
from analyze_noise_acc import run as run_noise
from analyze_overall_synthesis import run as run_overall
from analyze_rq1_tradeoff import run as run_rq1
from analyze_rq2_acc_moderation import run as run_rq2
from analyze_strength import run as run_strength
from analysis_stage_plan import run as run_stage_plan
from build_completeness_tables import build_completeness_tables
from build_master_table import build_master_table
from check_outputs import run as run_checks
from config import OUTPUT_DIR, ensure_output_dirs
from make_coefficients import run as run_extra_coefficients
from make_teacher_report_selection import run as run_teacher


def main() -> None:
    ensure_output_dirs()
    df = build_master_table()
    build_completeness_tables(df)
    outputs = {}
    outputs["stage_plan"] = run_stage_plan(df)
    outputs["rq1"] = run_rq1(df)
    outputs["strength"] = run_strength(df)
    outputs["cover"] = run_cover(df)
    outputs["label"] = run_label(df)
    outputs["arch"] = run_arch(df)
    outputs["noise"] = run_noise(df)
    outputs["rq2"] = run_rq2(df, arch_outputs=outputs["arch"], noise_outputs=outputs["noise"])
    outputs["bootstrap"] = run_extra_coefficients(df, outputs)
    run_overall(df, outputs)
    run_teacher()
    quality = run_checks()
    print(f"Analysis complete: {OUTPUT_DIR}")
    print(f"Rows in master_results.csv: {len(df)}")
    print(f"Quality checks: {int((quality['ok'] != True).sum())} failures out of {len(quality)} checks")


if __name__ == "__main__":
    main()
