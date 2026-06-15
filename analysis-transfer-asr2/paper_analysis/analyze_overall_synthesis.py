#!/usr/bin/env python3
"""Overall synthesis report."""

from __future__ import annotations

import pandas as pd

from config import COEFFICIENT_DIR, REPORT_DIR
from plot_utils import md_table, write_report


def _fmt(value, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _key_findings(outputs: dict[str, dict]) -> str:
    lines = []
    rq1 = outputs.get("rq1", {})
    overall = rq1.get("overall", pd.DataFrame())
    if not overall.empty:
        row = overall.iloc[0]
        lines.append(
            f"- RQ1 overall: n={int(row.get('n', 0))}, Spearman={_fmt(row.get('spearman'))}, "
            f"Pearson={_fmt(row.get('pearson'))}, median transfer_rate={_fmt(row.get('transfer_rate_median'))}, "
            f"median stealthiness={_fmt(row.get('stealthiness_median'))}."
        )
    sens = rq1.get("sensitivity", pd.DataFrame())
    if not sens.empty:
        bits = [f"source_asr>={_fmt(r.source_asr_threshold, 2)}: Spearman={_fmt(r.spearman)}" for r in sens.itertuples()]
        lines.append("- Source-ASR sensitivity: " + "; ".join(bits) + ".")
    by_da = rq1.get("by_dataset_attack", pd.DataFrame())
    if not by_da.empty:
        valid = by_da[pd.to_numeric(by_da["n"], errors="coerce") >= 5].dropna(subset=["spearman"])
        if not valid.empty:
            neg = valid.sort_values("spearman").head(5)[["dataset", "attack_type", "n", "spearman"]]
            pos = valid.sort_values("spearman", ascending=False).head(5)[["dataset", "attack_type", "n", "spearman"]]
            lines.append("- Strongest negative dataset-attack cells:\n\n" + md_table(neg, 5))
            lines.append("- Strongest positive or weakest tradeoff dataset-attack cells:\n\n" + md_table(pos, 5))
    rq2 = outputs.get("rq2", {})
    inter = rq2.get("interaction", pd.DataFrame())
    if not inter.empty:
        txa = inter[inter["term"].isin(["transfer_x_acc", "transfer_x_difficulty"])]
        if not txa.empty:
            lines.append("- RQ2 interaction terms:\n\n" + md_table(txa[["formula", "term", "coef", "p_value", "r2", "n"]], 10))
    return "\n\n".join(lines) if lines else "No key findings available."


def run(df: pd.DataFrame, outputs: dict[str, dict]) -> None:
    status = df.groupby(["result_group", "analysis_status"], dropna=False).size().reset_index(name="n")
    rq1_overall = outputs.get("rq1", {}).get("overall", pd.DataFrame())
    strength = outputs.get("strength", {}).get("summary", pd.DataFrame())
    cover = outputs.get("cover", {}).get("summary", pd.DataFrame())
    label = outputs.get("label", {}).get("completeness", pd.DataFrame())
    arch = outputs.get("arch", {}).get("summary", pd.DataFrame())
    noise = outputs.get("noise", {}).get("summary", pd.DataFrame())
    rq2 = outputs.get("rq2", {}).get("interaction", pd.DataFrame())
    pending = df[df["analysis_status"] != "complete"].groupby(["result_group", "dataset", "analysis_status"], dropna=False).size().reset_index(name="n")
    bootstrap_corr = outputs.get("bootstrap", {}).get("bootstrap_correlations", pd.DataFrame())
    bootstrap_delta = outputs.get("bootstrap", {}).get("bootstrap_delta", pd.DataFrame())

    write_report(
        REPORT_DIR / "08_overall_synthesis.md",
        "08 Overall Synthesis",
        [
            ("Current data completeness", md_table(status, 80)),
            ("Pending / partial rows to keep provisional", md_table(pending, 80)),
            ("Automatically extracted key findings", _key_findings(outputs)),
            ("RQ1 strongest current evidence", md_table(rq1_overall, 20)),
            ("Bootstrap CI for correlations", md_table(bootstrap_corr, 60)),
            ("Strength evidence", md_table(strength, 40)),
            ("Cover-rate evidence", md_table(cover, 40)),
            ("Label-mode completeness caveat", md_table(label, 40)),
            ("Architecture / ACC evidence", md_table(arch, 40)),
            ("Noise / difficulty evidence", md_table(noise, 40)),
            ("RQ2 moderation evidence", md_table(rq2, 80)),
            ("Bootstrap CI for pairwise deltas", md_table(bootstrap_delta, 60)),
            (
                "Recommended caution",
                "Treat conclusions with low sample size or pending status as provisional. Prefer matched deltas, Spearman/rank trends, and interaction summaries over raw pooled means.",
            ),
        ],
    )


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"), {})
