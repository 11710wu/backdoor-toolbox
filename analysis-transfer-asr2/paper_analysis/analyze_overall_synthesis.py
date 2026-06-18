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
            f"Pearson={_fmt(row.get('pearson'))}, median target-domain ASR={_fmt(row.get('transfer_rate_median'))}, "
            f"median stealthiness={_fmt(row.get('stealthiness_median'))}."
        )
    sens = rq1.get("sensitivity", pd.DataFrame())
    if not sens.empty:
        bits = [f"source_asr>={_fmt(r.source_asr_threshold, 2)}: Spearman={_fmt(r.spearman)}" for r in sens.itertuples()]
        lines.append("- Source-ASR sensitivity: " + "; ".join(bits) + ".")
    metric_sens = rq1.get("metric_sensitivity", pd.DataFrame())
    if not metric_sens.empty:
        lines.append("- Transfer-metric sensitivity:\n\n" + md_table(metric_sens[["transfer_metric", "n", "spearman", "pearson", "transfer_rate_median", "stealthiness_median"]], 20))
    by_group = rq1.get("by_result_group", pd.DataFrame())
    if not by_group.empty:
        cols = ["result_group", "n", "spearman", "pearson", "transfer_rate_median", "stealthiness_median"]
        lines.append("- Five result-group transfer-stealth correlations. These are the main five-group evidence; strength/cover/label/architecture/noise variables are windows for observing this relationship, not standalone conclusions:\n\n" + md_table(by_group[[c for c in cols if c in by_group.columns]], 10))
    by_da = rq1.get("by_dataset_attack", pd.DataFrame())
    if not by_da.empty:
        valid = by_da[pd.to_numeric(by_da["n"], errors="coerce") >= 5].dropna(subset=["spearman"])
        if not valid.empty:
            neg = valid.sort_values("spearman").head(5)[["dataset", "attack_type", "n", "spearman"]]
            pos = valid.sort_values("spearman", ascending=False).head(5)[["dataset", "attack_type", "n", "spearman"]]
            lines.append("- Strongest negative dataset-attack cells:\n\n" + md_table(neg, 5))
            lines.append("- Strongest positive or weakest tradeoff dataset-attack cells:\n\n" + md_table(pos, 5))
    rq2 = outputs.get("rq2", {})
    intervention = rq2.get("arch_noise", pd.DataFrame())
    if not intervention.empty:
        lines.append("- RQ2 ACC/difficulty/domain windows for transfer-stealth movement:\n\n" + md_table(intervention, 10))
    inter = rq2.get("interaction", pd.DataFrame())
    if not inter.empty:
        txa = inter[inter["term"].isin(["transfer_x_acc", "transfer_x_difficulty"])]
        if not txa.empty:
            lines.append("- RQ2 diagnostic interaction terms:\n\n" + md_table(txa[["formula", "term", "coef", "p_value", "r2", "n"]], 10))
    return "\n\n".join(lines) if lines else "No key findings available."


def run(df: pd.DataFrame, outputs: dict[str, dict]) -> None:
    status = df.groupby(["result_group", "analysis_status"], dropna=False).size().reset_index(name="n")
    rq1_overall = outputs.get("rq1", {}).get("overall", pd.DataFrame())
    rq1_by_result_group = outputs.get("rq1", {}).get("by_result_group", pd.DataFrame())
    strength = outputs.get("strength", {}).get("summary", pd.DataFrame())
    cover = outputs.get("cover", {}).get("summary", pd.DataFrame())
    label_summary = outputs.get("label", {}).get("summary", pd.DataFrame())
    label = outputs.get("label", {}).get("completeness", pd.DataFrame())
    arch = outputs.get("arch", {}).get("primary_summary", pd.DataFrame())
    if arch.empty:
        arch = outputs.get("arch", {}).get("summary", pd.DataFrame())
    noise = outputs.get("noise", {}).get("summary", pd.DataFrame())
    noise_baseline = outputs.get("noise", {}).get("baseline_overall", pd.DataFrame())
    noise_baseline_attack = outputs.get("noise", {}).get("baseline_by_attack", pd.DataFrame())
    target_domain = outputs.get("target_domain", {}).get("domain_overall", pd.DataFrame())
    target_domain_delta = outputs.get("target_domain", {}).get("paired_overall", pd.DataFrame())
    rq2_intervention = outputs.get("rq2", {}).get("arch_noise", pd.DataFrame())
    rq2_interaction = outputs.get("rq2", {}).get("interaction", pd.DataFrame())
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
            ("RQ1 result-group transfer-stealth correlations", md_table(rq1_by_result_group, 20)),
            ("Bootstrap CI for correlations", md_table(bootstrap_corr, 60)),
            ("Strength window: transfer-stealth movement evidence", md_table(strength, 40)),
            ("Cover-rate window: transfer-stealth movement evidence", md_table(cover, 40)),
            ("Label-mode clean/all-to-one window: transfer-stealth movement evidence", md_table(label_summary, 40)),
            ("Label-mode completeness caveat", md_table(label, 40)),
            ("Primary architecture / ACC window evidence", md_table(arch, 40)),
            ("Noise-vs-baseline transfer-stealth movement evidence", md_table(noise_baseline, 20)),
            ("Noise-vs-baseline by attack", md_table(noise_baseline_attack, 40)),
            ("Within-noise difficulty window evidence", md_table(noise, 40)),
            ("Target-domain ACC window evidence", md_table(target_domain, 40)),
            ("Target-domain paired movement evidence", md_table(target_domain_delta, 20)),
            ("RQ2 moderation windows evidence", md_table(rq2_intervention, 20)),
            ("RQ2 diagnostic interaction evidence", md_table(rq2_interaction, 80)),
            ("Bootstrap CI for pairwise deltas", md_table(bootstrap_delta, 60)),
            (
                "Recommended caution",
                "Treat conclusions with low sample size or pending status as provisional. Prefer designed contrasts, matched deltas, Spearman/rank trends, and interaction summaries over raw pooled means. For the five major groups, keep the transfer-stealth relationship as the main object and use variables only as observation windows or mechanism explanations.",
            ),
        ],
    )


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"), {})
