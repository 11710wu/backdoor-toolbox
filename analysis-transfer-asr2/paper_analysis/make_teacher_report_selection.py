#!/usr/bin/env python3
"""Select teacher-report material after full analysis outputs exist."""

from __future__ import annotations

from pathlib import Path

from config import FIGURE_DIR, OUTPUT_DIR, RECOMMENDED_FIGURES, REPORT_DIR


def run() -> None:
    lines = [
        "# 09 Teacher Report Selection",
        "",
        "This file is generated after the full analysis. Recommended items are selected only if the corresponding files exist.",
        "",
        "## Recommended Figures",
        "",
    ]
    for fig in RECOMMENDED_FIGURES:
        path = FIGURE_DIR / fig
        status = "available" if path.exists() else "missing"
        lines.append(f"- `{fig}`: {status}")
    lines.extend(
        [
            "",
            "## Recommended Coefficient Tables",
            "",
            "- `coefficients/rq1_overall_correlations.csv`",
            "- `coefficients/rq1_by_dataset_attack_correlations.csv`",
            "- `coefficients/strength_pairwise_delta.csv`",
            "- `coefficients/cover_rate_pairwise_delta.csv`",
            "- `coefficients/arch_pairwise_delta.csv`",
            "- `coefficients/rq2_interaction_regression.csv`",
            "- `coefficients/bootstrap_ci_correlations.csv`",
            "- `coefficients/bootstrap_ci_pairwise_delta.csv`",
            "",
            "## Suggested Flow",
            "",
            "1. Show current completeness first.",
            "2. Present RQ1 overall tradeoff.",
            "3. Show heterogeneity by attack and dataset.",
            "4. Use strength and cover-rate as within-attack mechanism evidence.",
            "5. Use architecture and noise to discuss ACC/difficulty moderation.",
            "6. End with pending results and next experiments to finish.",
            "",
        ]
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "09_teacher_report_selection.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT_DIR / "teacher_report_pack.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
