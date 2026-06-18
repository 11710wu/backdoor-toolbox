#!/usr/bin/env python3
"""RQ2: ACC/difficulty moderation analysis."""

from __future__ import annotations

import pandas as pd

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import grouped_bar, md_table, write_report
from stats_utils import grouped_correlations, main_df, ols_records, save_csv_and_md


def _target_domain_key_regressions(target_domain_outputs: dict | None) -> pd.DataFrame:
    if not target_domain_outputs:
        return pd.DataFrame()
    regressions = target_domain_outputs.get("regressions", pd.DataFrame())
    if regressions is None or regressions.empty or "term" not in regressions.columns:
        return pd.DataFrame()
    keep_terms = ["transfer_acc", "source_asr", "transfer_asr", "transfer_dataset_qwen"]
    cols = ["formula", "term", "coef", "std_err", "p_value", "r2", "n"]
    out = regressions[regressions["term"].isin(keep_terms)].copy()
    return out[[c for c in cols if c in out.columns]]


def run(
    df: pd.DataFrame,
    arch_outputs: dict | None = None,
    noise_outputs: dict | None = None,
    target_domain_outputs: dict | None = None,
) -> dict[str, pd.DataFrame]:
    d = main_df(df)
    acc_transfer = grouped_correlations(d, ["result_group"], x="clean_acc", y="transfer_rate")
    acc_stealth = grouped_correlations(d, ["result_group"], x="clean_acc", y="stealthiness")
    acc_corr = acc_transfer[["result_group", "n", "spearman", "pearson"]].rename(columns={"spearman": "spearman_acc_transfer", "pearson": "pearson_acc_transfer"})
    acc_corr = acc_corr.merge(
        acc_stealth[["result_group", "spearman", "pearson"]].rename(columns={"spearman": "spearman_acc_stealth", "pearson": "pearson_acc_stealth"}),
        on="result_group",
        how="outer",
    )
    acc_bin = grouped_correlations(d, ["result_group", "acc_bin"], x="transfer_rate", y="stealthiness") if "acc_bin" in d.columns else pd.DataFrame()

    reg_parts = [
        ols_records(d, "stealthiness ~ transfer_rate", "stealthiness", ["transfer_rate"]),
        ols_records(d, "stealthiness ~ transfer_rate + clean_acc + source_asr", "stealthiness", ["transfer_rate", "clean_acc", "source_asr"]),
    ]
    inter = d.copy()
    inter["transfer_x_acc"] = pd.to_numeric(inter["transfer_rate"], errors="coerce") * pd.to_numeric(inter["clean_acc"], errors="coerce")
    inter["transfer_x_difficulty"] = pd.to_numeric(inter["transfer_rate"], errors="coerce") * pd.to_numeric(inter["difficulty"], errors="coerce")
    acc_bin_x_cols = ["transfer_rate"]
    if "acc_bin" in inter.columns:
        acc_dummies = pd.get_dummies(inter["acc_bin"].astype(str), prefix="acc_bin", drop_first=True, dtype=float)
        for col in acc_dummies.columns:
            inter[col] = acc_dummies[col]
            interaction_col = f"transfer_x_{col}"
            inter[interaction_col] = pd.to_numeric(inter["transfer_rate"], errors="coerce") * inter[col]
            acc_bin_x_cols.extend([col, interaction_col])
    reg_parts.extend(
        [
            ols_records(inter, "stealthiness ~ transfer_rate + clean_acc + transfer_rate:clean_acc", "stealthiness", ["transfer_rate", "clean_acc", "transfer_x_acc"]),
            ols_records(inter, "stealthiness ~ transfer_rate + difficulty + transfer_rate:difficulty", "stealthiness", ["transfer_rate", "difficulty", "transfer_x_difficulty"]),
            ols_records(inter, "stealthiness ~ transfer_rate * C(acc_bin)", "stealthiness", acc_bin_x_cols),
        ]
    )
    interaction = pd.concat(reg_parts, ignore_index=True)

    attack_effect = grouped_correlations(d, ["attack_type"], x="clean_acc", y="transfer_rate")
    attack_stealth = grouped_correlations(d, ["attack_type"], x="clean_acc", y="stealthiness")
    attack_effect = attack_effect[["attack_type", "n", "spearman", "pearson"]].rename(columns={"spearman": "spearman_acc_transfer", "pearson": "pearson_acc_transfer"})
    attack_effect = attack_effect.merge(
        attack_stealth[["attack_type", "spearman", "pearson"]].rename(columns={"spearman": "spearman_acc_stealth", "pearson": "pearson_acc_stealth"}),
        on="attack_type",
        how="outer",
    )

    arch_delta = arch_outputs.get("summary", pd.DataFrame()) if arch_outputs else pd.DataFrame()
    arch_primary_delta = arch_outputs.get("primary_summary", pd.DataFrame()) if arch_outputs else pd.DataFrame()
    noise_delta = noise_outputs.get("summary", pd.DataFrame()) if noise_outputs else pd.DataFrame()
    noise_vs_baseline_overall = noise_outputs.get("baseline_overall", pd.DataFrame()) if noise_outputs else pd.DataFrame()
    noise_vs_baseline_by_attack = noise_outputs.get("baseline_by_attack", pd.DataFrame()) if noise_outputs else pd.DataFrame()
    noise_vs_baseline_by_level = noise_outputs.get("baseline_by_level", pd.DataFrame()) if noise_outputs else pd.DataFrame()
    target_domain_overall = target_domain_outputs.get("domain_overall", pd.DataFrame()) if target_domain_outputs else pd.DataFrame()
    target_domain_paired = target_domain_outputs.get("paired_overall", pd.DataFrame()) if target_domain_outputs else pd.DataFrame()
    target_domain_paired_summary = target_domain_outputs.get("paired_summary", pd.DataFrame()) if target_domain_outputs else pd.DataFrame()
    target_domain_acc_transfer = target_domain_outputs.get("acc_transfer_corr", pd.DataFrame()) if target_domain_outputs else pd.DataFrame()
    target_domain_acc_stealth = target_domain_outputs.get("acc_stealth_corr", pd.DataFrame()) if target_domain_outputs else pd.DataFrame()
    target_domain_regressions = _target_domain_key_regressions(target_domain_outputs)
    comp_rows = []
    arch_table = arch_primary_delta if arch_primary_delta is not None and not arch_primary_delta.empty else arch_delta
    if arch_table is not None and not arch_table.empty:
        for _, row in arch_table.iterrows():
            comp_rows.append(
                {
                    "variation_source": f"architecture_{row.get('comparison_group')}",
                    "acc_measure": "source_clean_acc",
                    "comparison_type": "primary_matched_architecture_delta",
                    "dataset": row.get("dataset"),
                    "transfer_dataset": row.get("transfer_dataset"),
                    "comparison_group": row.get("comparison_group"),
                    "n_groups": row.get("n_pairs"),
                    "delta_clean_acc_mean": row.get("delta_clean_acc_mean"),
                    "delta_transfer_rate_mean": row.get("delta_transfer_rate_mean"),
                    "delta_stealthiness_mean": row.get("delta_stealthiness_mean"),
                }
            )
    if noise_vs_baseline_overall is not None and not noise_vs_baseline_overall.empty:
        row = noise_vs_baseline_overall.iloc[0]
        comp_rows.append(
            {
                "variation_source": "noise_vs_baseline",
                "acc_measure": "source_clean_acc",
                "comparison_type": "configuration_controlled_group_comparison",
                "dataset": "cifar10",
                "transfer_dataset": "stl10",
                "comparison_group": row.get("comparison_group"),
                "n_groups": row.get("n_pairs"),
                "delta_clean_acc_mean": row.get("delta_clean_acc_mean"),
                "delta_transfer_rate_mean": row.get("delta_transfer_rate_mean"),
                "delta_stealthiness_mean": row.get("delta_stealthiness_mean"),
            }
        )
    elif noise_delta is not None and not noise_delta.empty:
        rec = {"variation_source": "noise_adjacent_levels", "acc_measure": "source_clean_acc", "comparison_type": "within_noise_adjacent_delta", "dataset": "cifar10", "transfer_dataset": "stl10", "comparison_group": "noise_adjacent_levels", "n_groups": len(noise_delta)}
        for col in ["delta_clean_acc_mean", "delta_transfer_rate_mean", "delta_stealthiness_mean"]:
            rec[col] = pd.to_numeric(noise_delta.get(col, pd.Series(dtype=float)), errors="coerce").mean()
        comp_rows.append(rec)
    if target_domain_paired is not None and not target_domain_paired.empty:
        row = target_domain_paired.iloc[0]
        comp_rows.append(
            {
                "variation_source": "target_domain_qwen_minus_imagenetv2",
                "acc_measure": "target_transfer_acc",
                "comparison_type": "matched_target_domain_delta",
                "dataset": "tiny_imagenet",
                "transfer_dataset": "qwen_minus_imagenetv2_tiny",
                "comparison_group": "qwen_minus_imagenetv2_tiny",
                "n_groups": row.get("n_pairs"),
                "delta_clean_acc_mean": row.get("delta_transfer_acc_mean"),
                "delta_transfer_rate_mean": row.get("delta_transfer_asr_mean"),
                "delta_stealthiness_mean": row.get("delta_stealthiness_mean"),
            }
        )
    arch_noise = pd.DataFrame(comp_rows)

    save_csv_and_md(acc_corr, COEFFICIENT_DIR / "rq2_acc_transfer_stealth_correlations.csv", "RQ2 ACC Transfer/Stealth Correlations")
    save_csv_and_md(acc_bin, COEFFICIENT_DIR / "rq2_acc_bin_correlations.csv", "RQ2 ACC-bin Correlations")
    save_csv_and_md(interaction, COEFFICIENT_DIR / "rq2_interaction_regression.csv", "RQ2 Interaction Regression")
    save_csv_and_md(arch_noise, COEFFICIENT_DIR / "rq2_arch_noise_comparison.csv", "RQ2 Architecture vs Noise Comparison")
    save_csv_and_md(arch_noise, COEFFICIENT_DIR / "rq2_acc_intervention_comparison.csv", "RQ2 ACC Intervention Comparison")
    save_csv_and_md(attack_effect, COEFFICIENT_DIR / "rq2_attack_conditioned_acc_effect.csv", "RQ2 Attack-conditioned ACC Effect")
    save_csv_and_md(acc_bin, TABLE_DIR / "table_9_acc_moderation_summary.csv", "ACC Moderation Summary")

    grouped_bar(acc_bin, "rq2_acc_bin_spearman.png", "acc_bin", "spearman", "result_group", "RQ2 ACC-bin Spearman")
    comp_long = arch_noise.melt(id_vars=["variation_source"], value_vars=[c for c in arch_noise.columns if c.startswith("delta_")], var_name="metric", value_name="delta") if not arch_noise.empty else pd.DataFrame()
    grouped_bar(comp_long, "rq2_intervention_delta_summary.png", "variation_source", "delta", "metric", "RQ2 Intervention Delta Summary")
    grouped_bar(comp_long, "rq2_arch_vs_noise_comparison.png", "variation_source", "delta", "metric", "RQ2 ACC Intervention Comparison")
    attack_long = attack_effect.melt(id_vars=["attack_type"], value_vars=["spearman_acc_transfer", "spearman_acc_stealth"], var_name="metric", value_name="spearman") if not attack_effect.empty else pd.DataFrame()
    grouped_bar(attack_long, "rq2_attack_conditioned_acc_effect.png", "attack_type", "spearman", "metric", "RQ2 Attack-conditioned ACC Effect")

    write_report(
        REPORT_DIR / "07_rq2_acc_moderation_full_analysis.md",
        "07 RQ2 ACC Moderation Full Analysis",
        [
            (
                "RQ2 framing",
                "RQ2 treats ACC/difficulty/domain shift as moderators of the transfer-stealth relationship, not as standalone explanations for transferability or stealthiness separately. The three comparison designs are used as observation windows into that relationship: (1) primary architecture/ACC contrasts only, namely CIFAR-10 SmallCNN minus original ResNet18 and Tiny-ImageNet ResNet34 minus original ResNet18, (2) noise results compared with matched baseline CIFAR ResNet18 results, and (3) Tiny-ImageNet target-domain ACC comparison between ImageNetV2-tiny and Qwen. The intervention table has four contrast rows because the architecture window has two intended contrasts. ACC-bin and interaction regressions are supporting diagnostics.",
            ),
            ("ACC/difficulty/domain windows: transfer-stealth movement summary", md_table(arch_noise, 80)),
            (
                "Noise vs baseline transfer-stealth movement",
                "This is the main noise window for RQ2. It compares CIFAR noise rows with matched baseline-strength ResNet18 configurations when attack, poison rate, strength, cover rate, label mode, BELT mask_rate when present, and transfer target match. Because the noise group uses SmallCNN while the baseline is original ResNet18, this comparison should be interpreted as a configuration-controlled group comparison rather than a strict same-architecture paired test. Read delta transfer_rate and delta stealthiness together as transfer-stealth movement; delta clean_acc is difficulty context.\n\n"
                + md_table(noise_vs_baseline_overall, 20),
            ),
            ("Noise vs baseline by attack", md_table(noise_vs_baseline_by_attack, 80)),
            ("Noise vs baseline by level", md_table(noise_vs_baseline_by_level, 80)),
            ("Auxiliary ACC-to-single-metric correlations (diagnostic)", md_table(acc_corr, 80)),
            ("ACC-bin transfer-stealth correlations (diagnostic)", md_table(acc_bin, 80)),
            ("Interaction regressions (diagnostic)", md_table(interaction, 80)),
            ("Attack-conditioned ACC effect", md_table(attack_effect, 80)),
            (
                "Target-domain ACC moderation: ImageNetV2-tiny vs Qwen",
                "This is part of RQ2 because it directly compares two target domains with different target-domain ACC while keeping the source model and source-domain defense measurements fixed. It mainly observes how target-domain ASR moves under the target-domain ACC window; stealthiness should not be interpreted as a direct target-domain effect because it is computed from source-domain defenses. It is kept as a detailed companion report in `13_target_domain_acc_analysis.md`, but its key evidence is summarized here.",
            ),
            ("Target-domain overall summary", md_table(target_domain_overall, 40)),
            ("Target-domain paired delta overall", md_table(target_domain_paired, 20)),
            ("Target-domain paired delta by attack", md_table(target_domain_paired_summary, 80)),
            ("Target ACC vs target-domain ASR correlations", md_table(target_domain_acc_transfer, 40)),
            ("Target ACC vs stealthiness correlations", md_table(target_domain_acc_stealth, 40)),
            ("Target-domain ACC key regressions", md_table(target_domain_regressions, 80)),
            (
                "Interpretation draft",
                "ACC moderation is evaluated through three observation windows: primary architecture contrasts, noise-vs-ResNet18-baseline differences, and ImageNetV2-tiny vs Qwen target-domain ACC differences. These windows show how ACC/difficulty/domain shift can move or reshape the transfer-stealth relationship. They should not be collapsed into one monotonic causal mechanism, and ACC-to-single-metric correlations should remain diagnostic rather than the main result. The Qwen comparison is especially important because higher target-domain ACC coincides with lower target-domain ASR, separating target-domain classification accuracy from backdoor transfer vulnerability.",
            ),
        ],
    )
    return {
        "acc_corr": acc_corr,
        "acc_bin": acc_bin,
        "interaction": interaction,
        "arch_noise": arch_noise,
        "noise_vs_baseline_overall": noise_vs_baseline_overall,
        "noise_vs_baseline_by_attack": noise_vs_baseline_by_attack,
        "noise_vs_baseline_by_level": noise_vs_baseline_by_level,
        "attack_effect": attack_effect,
        "target_domain_overall": target_domain_overall,
        "target_domain_paired": target_domain_paired,
        "target_domain_paired_summary": target_domain_paired_summary,
        "target_domain_acc_transfer": target_domain_acc_transfer,
        "target_domain_acc_stealth": target_domain_acc_stealth,
        "target_domain_regressions": target_domain_regressions,
    }


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
