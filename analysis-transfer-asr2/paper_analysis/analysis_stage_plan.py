#!/usr/bin/env python3
"""Generate current-stage analysis planning reports.

These reports document the analysis workflow and figure intent before any
image-based interpretation is performed.
"""

from __future__ import annotations

import pandas as pd

from config import OUTPUT_DIR, REPORT_DIR, TABLE_DIR, ensure_output_dirs
from figure_specs import FIGURE_SPECS
from plot_utils import md_table, write_report


TOOL_SELECTION = [
    {
        "priority": 1,
        "tool": "ARIS result-to-claim",
        "type": "result-to-claim skill",
        "keep": "yes",
        "current_use": "Turn result evidence into supported / partial / unsupported claims after figures and coefficients are generated.",
        "planned_output": "Future claims audit, not run in this planning stage.",
        "source": "https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/result-to-claim/SKILL.md",
    },
    {
        "priority": 2,
        "tool": "ARIS experiment-audit",
        "type": "experiment audit skill",
        "keep": "yes",
        "current_use": "Check whether result files, metrics, scope, missing data, and normalization support the analysis.",
        "planned_output": "Risk audit for missing data, metric distortion, and scope mismatch.",
        "source": "https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/main/skills/experiment-audit/SKILL.md",
    },
    {
        "priority": 3,
        "tool": "Initial Data Analysis",
        "type": "data-quality principle",
        "keep": "yes",
        "current_use": "Force completeness and anomaly checks before answering RQ1/RQ2.",
        "planned_output": "Completeness-first workflow gate.",
        "source": "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009819",
    },
    {
        "priority": 4,
        "tool": "Frontiers AI Playbook",
        "type": "robustness and overclaim guidance",
        "keep": "yes",
        "current_use": "Use silent-failure and anti-overclaim checks for ACC/difficulty interpretation.",
        "planned_output": "Risk checks before final synthesis.",
        "source": "https://www.frontiersin.org/ai-playbook/research-stages/analysis-code-statistics",
    },
    {
        "priority": 5,
        "tool": "Statistical analysis skill",
        "type": "statistical evidence workflow",
        "keep": "yes",
        "current_use": "Prioritize matched-pair delta, Spearman/rank trend, bootstrap CI, and then OLS/Pearson.",
        "planned_output": "Coefficient tables with n, effect size, CI, and status.",
        "source": "https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/scientific-skills/statistical-analysis/SKILL.md",
    },
    {
        "priority": 6,
        "tool": "Better Figures",
        "type": "figure audit principle",
        "keep": "yes",
        "current_use": "Require every figure to answer one analysis question and expose risks.",
        "planned_output": "Figure purpose and later figure audit.",
        "source": "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833",
    },
]


WORKFLOW_STAGES = [
    {
        "stage_order": 1,
        "stage": "Data inventory",
        "tool_basis": "Initial Data Analysis + experiment-audit",
        "purpose": "Collect all current result rows and mark complete / partial / pending before any trend claim.",
        "outputs": "master_results.csv, completeness_report.csv, missing_by_group_attack.csv",
        "gate": "No conclusion can ignore partial/pending status.",
    },
    {
        "stage_order": 2,
        "stage": "Metric sanity",
        "tool_basis": "experiment-audit + statistical analysis skill",
        "purpose": "Check fixed metric formulas and denominator sensitivity.",
        "outputs": "transfer_rate, stealthiness, difficulty, source_asr threshold sensitivity",
        "gate": "Rows with unstable source_asr are not main evidence.",
    },
    {
        "stage_order": 3,
        "stage": "RQ1 relationship analysis",
        "tool_basis": "statistical analysis skill",
        "purpose": "Estimate transfer_rate vs stealthiness overall and by dataset / attack / architecture.",
        "outputs": "RQ1 correlations, heatmaps, rank trends, defense breakdown",
        "gate": "Report n and heterogeneity; do not claim universality from pooled trends.",
    },
    {
        "stage_order": 4,
        "stage": "Five experiment blocks",
        "tool_basis": "AgentSociety-style single-to-batch workflow",
        "purpose": "Analyze strength, cover-rate, label-mode, architecture, and noise as separate evidence sources.",
        "outputs": "group-specific delta tables, curves, heatmaps, reports",
        "gate": "Within-group matched comparisons are stronger than pooled averages.",
    },
    {
        "stage_order": 5,
        "stage": "RQ2 ACC/difficulty moderation",
        "tool_basis": "Frontiers robustness + statistical analysis skill",
        "purpose": "Test whether ACC/difficulty changes the transfer-stealth relation.",
        "outputs": "ACC-bin correlations, interaction regressions, arch-vs-noise comparison",
        "gate": "Phrase ACC as a moderator unless causal evidence is explicit.",
    },
    {
        "stage_order": 6,
        "stage": "Figure purpose planning",
        "tool_basis": "Better Figures",
        "purpose": "Document why each figure exists before reading conclusions from it.",
        "outputs": "table_0_figure_plan.csv, 12_figure_plan_and_purpose.md",
        "gate": "Every figure must map to an analysis question and coefficient table.",
    },
    {
        "stage_order": 7,
        "stage": "Later image-based audit",
        "tool_basis": "Better Figures + result-to-claim",
        "purpose": "After data are complete, inspect generated figures and decide main / backup / redraw.",
        "outputs": "Future figure audit and claims audit",
        "gate": "Not performed in this current planning step.",
    },
]


def _analysis_area(filename: str) -> str:
    if filename.startswith("rq1_"):
        return "RQ1 transfer-stealth"
    if filename.startswith("strength_"):
        return "Attack strength"
    if filename.startswith("cover_rate_"):
        return "Cover-rate"
    if filename.startswith("label_mode_"):
        return "Label mode"
    if filename.startswith("arch_"):
        return "Architecture / ACC"
    if filename.startswith("noise_"):
        return "Noise / difficulty"
    if filename.startswith("rq2_"):
        return "RQ2 ACC moderation"
    return "Other"


def _figure_role(recommend: str) -> str:
    if recommend == "yes":
        return "main_candidate"
    if recommend == "conditional":
        return "completeness_or_caveat"
    return "backup_or_diagnostic"


def _build_figure_plan() -> pd.DataFrame:
    rows = []
    for filename, spec in sorted(FIGURE_SPECS.items()):
        rows.append(
            {
                "figure": filename,
                "analysis_area": _analysis_area(filename),
                "role": _figure_role(spec.recommend),
                "purpose": spec.purpose,
                "focus": spec.focus,
                "how_to_read": spec.how_to_read,
                "coefficient_files": spec.coefficient_files,
                "interpretation_boundary": "Planning only: do not infer final conclusions until figure audit and claim audit are run.",
            }
        )
    return pd.DataFrame(rows)


def _status_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "result_group" not in df or "analysis_status" not in df:
        return pd.DataFrame(columns=["result_group", "analysis_status", "n"])
    return df.groupby(["result_group", "analysis_status"], dropna=False).size().reset_index(name="n")


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    tool_df = pd.DataFrame(TOOL_SELECTION)
    workflow_df = pd.DataFrame(WORKFLOW_STAGES)
    figure_df = _build_figure_plan()
    status_df = _status_summary(df)

    tool_df.to_csv(TABLE_DIR / "table_0_analysis_tool_selection.csv", index=False)
    workflow_df.to_csv(TABLE_DIR / "table_0_analysis_workflow.csv", index=False)
    figure_df.to_csv(TABLE_DIR / "table_0_figure_plan.csv", index=False)

    write_report(
        REPORT_DIR / "10_analysis_toolkit_selection.md",
        "10 Analysis Toolkit Selection",
        [
            (
                "Current Stage",
                "This project is in the current-results analysis stage. The goal is to audit evidence, missing data, metric stability, and planned figures before making final image-based conclusions.",
            ),
            ("Selected Tools", md_table(tool_df[["priority", "tool", "type", "current_use", "planned_output"]], 20)),
            (
                "Tools Not Used As Core",
                "Generic data-science prompts, automatic paper-writing prompts, dashboards, and broad literature-review tools are not core tools here because they do not directly audit whether current experiment results support RQ1/RQ2.",
            ),
        ],
    )

    write_report(
        REPORT_DIR / "11_analysis_workflow_plan.md",
        "11 Analysis Workflow Plan",
        [
            ("Workflow Stages", md_table(workflow_df, 20)),
            ("Current Data Status", md_table(status_df, 80)),
            (
                "Interpretation Boundary",
                "This workflow plan only defines the order of analysis and required evidence gates. It does not inspect generated images or turn visual patterns into final claims.",
            ),
        ],
    )

    main_figs = figure_df[figure_df["role"] == "main_candidate"]
    diagnostics = figure_df[figure_df["role"] != "main_candidate"]
    write_report(
        REPORT_DIR / "12_figure_plan_and_purpose.md",
        "12 Figure Plan And Purpose",
        [
            (
                "Purpose",
                "This report records why each planned figure exists, what it should help judge, and which coefficient/table output it should be paired with. It is not a post-hoc interpretation of image content.",
            ),
            ("Main Candidate Figures", md_table(main_figs[["figure", "analysis_area", "purpose", "focus", "coefficient_files"]], 80)),
            ("Diagnostic / Backup Figures", md_table(diagnostics[["figure", "analysis_area", "role", "purpose", "focus", "coefficient_files"]], 120)),
            (
                "Use Rule",
                "A figure should enter the final analysis only if its corresponding coefficient table exists, its sample size is adequate, and later figure/claim audits do not flag missing-data or grouping-confound risks.",
            ),
        ],
    )

    return {"tools": tool_df, "workflow": workflow_df, "figures": figure_df, "status": status_df}


def main() -> None:
    master_path = OUTPUT_DIR / "master_results.csv"
    df = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame()
    run(df)
    print(f"Wrote analysis-stage planning reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
