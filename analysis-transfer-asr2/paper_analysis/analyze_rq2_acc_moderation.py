#!/usr/bin/env python3
"""RQ2: ACC/difficulty moderation analysis."""

from __future__ import annotations

import pandas as pd

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import grouped_bar, md_table, write_report
from stats_utils import grouped_correlations, main_df, ols_records, save_csv_and_md


def run(df: pd.DataFrame, arch_outputs: dict | None = None, noise_outputs: dict | None = None) -> dict[str, pd.DataFrame]:
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
    reg_parts.extend(
        [
            ols_records(inter, "stealthiness ~ transfer_rate + clean_acc + transfer_rate:clean_acc", "stealthiness", ["transfer_rate", "clean_acc", "transfer_x_acc"]),
            ols_records(inter, "stealthiness ~ transfer_rate + difficulty + transfer_rate:difficulty", "stealthiness", ["transfer_rate", "difficulty", "transfer_x_difficulty"]),
            ols_records(inter, "stealthiness ~ transfer_rate + C(acc_bin) + transfer_rate:C(acc_bin)", "stealthiness", ["transfer_rate"], ["acc_bin"]),
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
    noise_delta = noise_outputs.get("summary", pd.DataFrame()) if noise_outputs else pd.DataFrame()
    comp_rows = []
    for source_name, table in [("architecture", arch_delta), ("noise", noise_delta)]:
        if table is None or table.empty:
            continue
        rec = {"variation_source": source_name, "n_groups": len(table)}
        for col in ["delta_clean_acc_mean", "delta_transfer_rate_mean", "delta_stealthiness_mean"]:
            rec[col] = pd.to_numeric(table.get(col, pd.Series(dtype=float)), errors="coerce").mean()
        comp_rows.append(rec)
    arch_noise = pd.DataFrame(comp_rows)

    save_csv_and_md(acc_corr, COEFFICIENT_DIR / "rq2_acc_transfer_stealth_correlations.csv", "RQ2 ACC Transfer/Stealth Correlations")
    save_csv_and_md(acc_bin, COEFFICIENT_DIR / "rq2_acc_bin_correlations.csv", "RQ2 ACC-bin Correlations")
    save_csv_and_md(interaction, COEFFICIENT_DIR / "rq2_interaction_regression.csv", "RQ2 Interaction Regression")
    save_csv_and_md(arch_noise, COEFFICIENT_DIR / "rq2_arch_noise_comparison.csv", "RQ2 Architecture vs Noise Comparison")
    save_csv_and_md(attack_effect, COEFFICIENT_DIR / "rq2_attack_conditioned_acc_effect.csv", "RQ2 Attack-conditioned ACC Effect")
    save_csv_and_md(acc_bin, TABLE_DIR / "table_9_acc_moderation_summary.csv", "ACC Moderation Summary")

    grouped_bar(acc_bin, "rq2_acc_bin_spearman.png", "acc_bin", "spearman", "result_group", "RQ2 ACC-bin Spearman")
    comp_long = arch_noise.melt(id_vars=["variation_source"], value_vars=[c for c in arch_noise.columns if c.startswith("delta_")], var_name="metric", value_name="delta") if not arch_noise.empty else pd.DataFrame()
    grouped_bar(comp_long, "rq2_intervention_delta_summary.png", "variation_source", "delta", "metric", "RQ2 Intervention Delta Summary")
    grouped_bar(comp_long, "rq2_arch_vs_noise_comparison.png", "variation_source", "delta", "metric", "RQ2 Architecture vs Noise Comparison")
    attack_long = attack_effect.melt(id_vars=["attack_type"], value_vars=["spearman_acc_transfer", "spearman_acc_stealth"], var_name="metric", value_name="spearman") if not attack_effect.empty else pd.DataFrame()
    grouped_bar(attack_long, "rq2_attack_conditioned_acc_effect.png", "attack_type", "spearman", "metric", "RQ2 Attack-conditioned ACC Effect")

    write_report(
        REPORT_DIR / "07_rq2_acc_moderation_full_analysis.md",
        "07 RQ2 ACC Moderation Full Analysis",
        [
            ("ACC vs transfer / stealth correlations", md_table(acc_corr, 80)),
            ("ACC-bin transfer-stealth correlations", md_table(acc_bin, 80)),
            ("Interaction regressions", md_table(interaction, 80)),
            ("Architecture vs noise comparison", md_table(arch_noise, 80)),
            ("Attack-conditioned ACC effect", md_table(attack_effect, 80)),
            ("Interpretation draft", "ACC should be interpreted as a moderating factor if acc-bin correlations or interaction terms differ across groups. Architecture and noise are different interventions and should not be collapsed into one causal mechanism."),
        ],
    )
    return {
        "acc_corr": acc_corr,
        "acc_bin": acc_bin,
        "interaction": interaction,
        "arch_noise": arch_noise,
        "attack_effect": attack_effect,
    }


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
