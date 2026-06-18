#!/usr/bin/env python3
"""Target-domain ACC analysis for Tiny-ImageNet transfer datasets."""

from __future__ import annotations

import pandas as pd

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import grouped_bar, md_table, scatter_with_binned_line, simple_heatmap, write_report
from stats_utils import analysis_df, grouped_correlations, ols_records, save_csv_and_md


TINY_TRANSFER_DATASETS = ["imagenetv2_tiny", "qwen"]


def _target_domain_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = analysis_df(df, main_transfer_only=False)
    out = d[(d["dataset"] == "tiny_imagenet") & (d["transfer_dataset"].isin(TINY_TRANSFER_DATASETS))].copy()
    for col in ["transfer_acc", "transfer_asr", "transfer_rate", "stealthiness", "source_asr", "clean_acc"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if len(out) >= 3 and out["transfer_acc"].nunique(dropna=True) >= 3:
        try:
            out["target_acc_bin"] = pd.qcut(out["transfer_acc"], q=3, labels=["low_target_acc", "mid_target_acc", "high_target_acc"], duplicates="drop")
            out["target_acc_bin"] = out["target_acc_bin"].astype(str).replace("nan", "unknown")
        except Exception:
            out["target_acc_bin"] = "unknown"
    else:
        out["target_acc_bin"] = "unknown"
    return out


def _paired_delta(tiny: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "result_group",
        "dataset",
        "arch",
        "arch_base",
        "attack_type",
        "poison_rate",
        "strength_name",
        "strength_value",
        "cover_rate",
        "mask_rate",
        "label_mode",
        "transfer_variant",
        "result_dir",
    ]
    key_cols = [col for col in key_cols if col in tiny.columns]
    rows = []
    for keys, group in tiny.groupby(key_cols, dropna=False):
        iv2 = group[group["transfer_dataset"] == "imagenetv2_tiny"].head(1)
        qwen = group[group["transfer_dataset"] == "qwen"].head(1)
        if iv2.empty or qwen.empty:
            continue
        iv2_row = iv2.iloc[0]
        qwen_row = qwen.iloc[0]
        rec = {col: val for col, val in zip(key_cols, keys)}
        for metric in [
            "transfer_acc",
            "transfer_asr",
            "transfer_asr_chance_adjusted",
            "transfer_rate",
            "legacy_transfer_rate",
            "transfer_retention_rate",
            "transfer_gap",
            "joint_transfer",
            "stealthiness",
            "source_asr",
            "clean_acc",
        ]:
            rec[f"iv2_{metric}"] = iv2_row.get(metric)
            rec[f"qwen_{metric}"] = qwen_row.get(metric)
            rec[f"delta_{metric}"] = qwen_row.get(metric) - iv2_row.get(metric)
        rows.append(rec)
    return pd.DataFrame(rows)


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tiny = _target_domain_rows(df)
    paired = _paired_delta(tiny)

    domain_summary = (
        tiny.groupby(["transfer_dataset", "result_group"], dropna=False)
        .agg(
            n=("transfer_rate", "size"),
            transfer_acc_mean=("transfer_acc", "mean"),
            transfer_acc_median=("transfer_acc", "median"),
            transfer_asr_mean=("transfer_asr", "mean"),
            transfer_asr_median=("transfer_asr", "median"),
            stealthiness_mean=("stealthiness", "mean"),
            stealthiness_median=("stealthiness", "median"),
            clean_acc_median=("clean_acc", "median"),
            source_asr_median=("source_asr", "median"),
        )
        .reset_index()
    )
    domain_overall = (
        tiny.groupby(["transfer_dataset"], dropna=False)
        .agg(
            n=("transfer_rate", "size"),
            transfer_acc_mean=("transfer_acc", "mean"),
            transfer_acc_median=("transfer_acc", "median"),
            transfer_asr_mean=("transfer_asr", "mean"),
            transfer_asr_median=("transfer_asr", "median"),
            stealthiness_mean=("stealthiness", "mean"),
            stealthiness_median=("stealthiness", "median"),
            clean_acc_median=("clean_acc", "median"),
            source_asr_median=("source_asr", "median"),
        )
        .reset_index()
    )

    acc_transfer_corr = grouped_correlations(tiny, ["transfer_dataset"], x="transfer_acc", y="transfer_rate")
    acc_stealth_corr = grouped_correlations(tiny, ["transfer_dataset"], x="transfer_acc", y="stealthiness")
    acc_bin_corr = grouped_correlations(tiny, ["transfer_dataset", "target_acc_bin"], x="transfer_rate", y="stealthiness")
    attack_corr = grouped_correlations(tiny, ["transfer_dataset", "attack_type"], x="transfer_acc", y="transfer_rate")

    paired_summary = pd.DataFrame()
    if not paired.empty:
        paired_summary = (
            paired.groupby(["result_group", "attack_type"], dropna=False)
            .agg(
                n_pairs=("delta_transfer_asr", "size"),
                delta_transfer_acc_mean=("delta_transfer_acc", "mean"),
                delta_transfer_acc_median=("delta_transfer_acc", "median"),
                delta_transfer_asr_mean=("delta_transfer_asr", "mean"),
                delta_transfer_asr_median=("delta_transfer_asr", "median"),
                delta_stealthiness_mean=("delta_stealthiness", "mean"),
                delta_legacy_transfer_rate_mean=("delta_legacy_transfer_rate", "mean"),
                delta_joint_transfer_mean=("delta_joint_transfer", "mean"),
            )
            .reset_index()
        )
    paired_overall = pd.DataFrame()
    if not paired.empty:
        paired_overall = pd.DataFrame(
            [
                {
                    "n_pairs": len(paired),
                    "delta_transfer_acc_mean": paired["delta_transfer_acc"].mean(),
                    "delta_transfer_acc_median": paired["delta_transfer_acc"].median(),
                    "delta_transfer_asr_mean": paired["delta_transfer_asr"].mean(),
                    "delta_transfer_asr_median": paired["delta_transfer_asr"].median(),
                    "delta_stealthiness_mean": paired["delta_stealthiness"].mean(),
                    "delta_legacy_transfer_rate_mean": paired["delta_legacy_transfer_rate"].mean(),
                    "delta_joint_transfer_mean": paired["delta_joint_transfer"].mean(),
                    "spearman_delta_acc_delta_asr": paired["delta_transfer_acc"].corr(paired["delta_transfer_asr"], method="spearman"),
                }
            ]
        )

    reg_parts = [
        ols_records(tiny, "transfer_asr ~ transfer_acc", "transfer_asr", ["transfer_acc"]),
        ols_records(tiny, "transfer_asr ~ transfer_acc + source_asr + clean_acc + C(attack_type) + C(arch_base) + C(transfer_dataset)", "transfer_asr", ["transfer_acc", "source_asr", "clean_acc"], ["attack_type", "arch_base", "transfer_dataset"]),
        ols_records(tiny, "stealthiness ~ transfer_acc", "stealthiness", ["transfer_acc"]),
        ols_records(tiny, "stealthiness ~ transfer_acc + transfer_asr + source_asr + C(attack_type) + C(arch_base) + C(transfer_dataset)", "stealthiness", ["transfer_acc", "transfer_asr", "source_asr"], ["attack_type", "arch_base", "transfer_dataset"]),
    ]
    regressions = pd.concat(reg_parts, ignore_index=True)

    save_csv_and_md(domain_summary, TABLE_DIR / "table_11_target_domain_summary.csv", "Target-domain Summary")
    save_csv_and_md(domain_overall, COEFFICIENT_DIR / "target_domain_overall_summary.csv", "Target-domain Overall Summary")
    save_csv_and_md(paired, COEFFICIENT_DIR / "target_domain_paired_delta_rows.csv", "Target-domain Paired Delta Rows")
    save_csv_and_md(paired_summary, COEFFICIENT_DIR / "target_domain_paired_delta_summary.csv", "Target-domain Paired Delta Summary")
    save_csv_and_md(paired_overall, COEFFICIENT_DIR / "target_domain_paired_delta_overall.csv", "Target-domain Paired Delta Overall")
    save_csv_and_md(acc_transfer_corr, COEFFICIENT_DIR / "target_domain_acc_transfer_correlations.csv", "Target-domain ACC Transfer Correlations")
    save_csv_and_md(acc_stealth_corr, COEFFICIENT_DIR / "target_domain_acc_stealth_correlations.csv", "Target-domain ACC Stealth Correlations")
    save_csv_and_md(acc_bin_corr, COEFFICIENT_DIR / "target_domain_acc_bin_tradeoff_correlations.csv", "Target-domain ACC-bin Tradeoff Correlations")
    save_csv_and_md(attack_corr, COEFFICIENT_DIR / "target_domain_attackwise_acc_transfer_correlations.csv", "Target-domain Attack-wise ACC Transfer Correlations")
    save_csv_and_md(regressions, COEFFICIENT_DIR / "target_domain_acc_regressions.csv", "Target-domain ACC Regressions")

    overview = domain_overall.melt(
        id_vars=["transfer_dataset"],
        value_vars=["transfer_acc_mean", "transfer_asr_mean", "stealthiness_mean"],
        var_name="metric",
        value_name="value",
    )
    grouped_bar(overview, "target_domain_acc_transfer_overview.png", "transfer_dataset", "value", "metric", "Target-domain ACC / Transfer / Stealth", rotate=False)

    if not paired_overall.empty:
        delta_plot = paired_overall.melt(
            value_vars=["delta_transfer_acc_mean", "delta_transfer_asr_mean", "delta_stealthiness_mean"],
            var_name="metric",
            value_name="delta",
        )
    else:
        delta_plot = pd.DataFrame()
    grouped_bar(delta_plot, "target_domain_paired_delta.png", "metric", "delta", None, "Qwen minus ImageNetV2 Paired Delta")
    scatter_with_binned_line(tiny, "target_domain_acc_vs_transfer.png", "transfer_acc", "transfer_rate", hue="transfer_dataset", col=None, title="Target ACC vs Target-domain ASR")
    scatter_with_binned_line(tiny, "target_domain_transfer_stealth_by_domain.png", "transfer_rate", "stealthiness", hue="transfer_dataset", col=None, title="Transfer-Stealth by Target Domain")
    heat = paired_summary.melt(
        id_vars=["attack_type"],
        value_vars=[c for c in ["delta_transfer_acc_mean", "delta_transfer_asr_mean", "delta_stealthiness_mean"] if c in paired_summary.columns],
        var_name="metric",
        value_name="delta",
    ) if not paired_summary.empty else pd.DataFrame()
    simple_heatmap(heat, "target_domain_attack_delta_heatmap.png", "attack_type", "metric", "delta", "Target-domain Paired Delta by Attack")

    write_report(
        REPORT_DIR / "13_target_domain_acc_analysis.md",
        "13 Target-domain ACC Analysis",
        [
            (
                "Relationship-first framing",
                "ImageNetV2-tiny vs Qwen is a target-domain ACC window. Its role is to observe how target-domain ASR changes when the target-domain clean ACC changes under matched source configurations. Stealthiness is still measured from source-domain defenses, so target-domain ACC should not be written as directly changing stealthiness.",
            ),
            ("Target-domain overall summary", md_table(domain_overall, 40)),
            ("Target-domain summary by result group", md_table(domain_summary, 80)),
            ("Qwen minus ImageNetV2 paired delta overall", md_table(paired_overall, 20)),
            ("Qwen minus ImageNetV2 paired delta by attack", md_table(paired_summary, 80)),
            ("Target ACC vs transfer correlations", md_table(acc_transfer_corr, 40)),
            ("Target ACC vs stealth correlations", md_table(acc_stealth_corr, 40)),
            ("Target ACC-bin transfer-stealth correlations", md_table(acc_bin_corr, 80)),
            ("Attack-wise target ACC vs transfer correlations", md_table(attack_corr, 80)),
            ("Target ACC regressions", md_table(regressions, 80)),
            (
                "Interpretation guardrail",
                "Transfer ACC is measured on the target transfer dataset, while stealthiness is computed from the four source-domain defense TPRs. Therefore the target-domain ACC window is mainly about how target-domain ASR moves under a target-domain distribution change. It can supplement the broader transfer-stealth analysis, but it should not be interpreted as a direct stealthiness intervention unless the paired rows or conditioned correlations show a consistent association.",
            ),
        ],
    )
    return {
        "target_rows": tiny,
        "domain_summary": domain_summary,
        "domain_overall": domain_overall,
        "paired": paired,
        "paired_summary": paired_summary,
        "paired_overall": paired_overall,
        "acc_transfer_corr": acc_transfer_corr,
        "acc_stealth_corr": acc_stealth_corr,
        "acc_bin_corr": acc_bin_corr,
        "attack_corr": attack_corr,
        "regressions": regressions,
    }


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
