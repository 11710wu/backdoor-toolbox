#!/usr/bin/env python3
"""Architecture and ACC analysis."""

from __future__ import annotations

import itertools

import pandas as pd

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import grouped_bar, md_table, scatter_with_binned_line, simple_heatmap, write_report
from stats_utils import correlation_record, grouped_correlations, main_df, save_csv_and_md, strict_config_key_columns, summarize_delta


PRIMARY_ARCH_COMPARISONS = {
    ("cifar10", "stl10", "SmallCNN-ResNet18"),
    ("tiny_imagenet", "imagenetv2_tiny", "ResNet34-ResNet18"),
}


def _arch_pairwise_delta(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["result_group"].isin(["baseline_strength", "arch_acc"])].copy()
    key_cols = strict_config_key_columns(sub)
    rows = []
    for keys, group in sub.groupby(key_cols, dropna=False):
        arch_rows = list(group.iterrows())
        if len(arch_rows) < 2:
            continue
        for (_, left), (_, right) in itertools.combinations(arch_rows, 2):
            if left["arch_base"] == right["arch_base"]:
                continue
            if {left["result_group"], right["result_group"]} != {"baseline_strength", "arch_acc"}:
                continue
            # Put supplement architectures first when present.
            if right["result_group"] == "arch_acc":
                new, base = right, left
            else:
                new, base = left, right
            rec = {col: val for col, val in zip(key_cols, keys)}
            rec["comparison_group"] = f"{new['arch_base']}-{base['arch_base']}"
            rec["new_arch"] = new["arch_base"]
            rec["base_arch"] = base["arch_base"]
            for metric in ["clean_acc", "source_asr", "transfer_asr", "transfer_rate", "stealthiness"]:
                rec[f"delta_{metric}"] = new[metric] - base[metric]
            rows.append(rec)
    return pd.DataFrame(rows)


def _arch_vs_resnet18_baseline_rows(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = d[
        (d["result_group"] == "baseline_strength")
        & (d["arch_base"] == "ResNet18")
    ].copy()
    arch = d[d["result_group"] == "arch_acc"].copy()
    if baseline.empty or arch.empty:
        return pd.DataFrame(), arch

    arch["_arch_row_id"] = arch.index
    keys = strict_config_key_columns(d)
    baseline_keys = baseline[keys].drop_duplicates()
    matched = arch.merge(baseline_keys, on=keys, how="inner")
    matched_ids = set(matched["_arch_row_id"])
    unmatched = arch[~arch["_arch_row_id"].isin(matched_ids)].copy()
    return matched, unmatched


def _write_arch_unmatched_detail(d: pd.DataFrame, matched: pd.DataFrame, unmatched: pd.DataFrame) -> pd.DataFrame:
    arch = d[d["result_group"] == "arch_acc"].copy()
    matched_ids = set(matched["_arch_row_id"]) if "_arch_row_id" in matched.columns else set()
    summary = pd.DataFrame(
        [
            {
                "arch_rows_total": len(arch),
                "matched_arch_rows": len(matched_ids),
                "unmatched_arch_rows": len(unmatched),
                "matching_rule": "strict ResNet18 baseline config match",
            }
        ]
    )
    save_csv_and_md(summary, COEFFICIENT_DIR / "arch_strict_match_summary.csv", "Architecture Strict Match Summary")
    save_csv_and_md(unmatched.drop(columns=["_arch_row_id"], errors="ignore"), COEFFICIENT_DIR / "arch_unmatched_rows.csv", "Architecture Unmatched Rows")

    note = (
        "Architecture matching uses the original ResNet18 baseline as the reference and requires exact agreement on "
        "dataset, target domain, transfer variant, attack_type, poison_rate, strength_name/value, cover_rate, "
        "label_mode, and mask_rate when present. Rows without such a baseline counterpart are excluded from the "
        "main architecture analysis. This excludes old architecture `badnet` rows, old BELT rows that varied mask "
        "instead of alpha, and any trigger strengths outside the original baseline grid.\n\n"
    )
    sections = [("Strict match summary", note + md_table(summary, 20))]
    if not unmatched.empty:
        counts = (
            unmatched.groupby(["dataset", "transfer_dataset", "arch_base", "attack_type"], dropna=False)
            .size()
            .reset_index(name="unmatched_rows")
            .sort_values(["dataset", "transfer_dataset", "arch_base", "attack_type"])
        )
        sections.append(("Unmatched rows by dataset / architecture / attack", md_table(counts, 80)))
        cols = [
            "dataset",
            "transfer_dataset",
            "arch_base",
            "attack_type",
            "poison_rate",
            "strength_name",
            "strength_value",
            "cover_rate",
            "mask_rate",
            "label_mode",
            "transfer_variant",
            "clean_acc",
            "transfer_rate",
            "stealthiness",
            "folder_name",
        ]
        cols = [c for c in cols if c in unmatched.columns]
        sections.append(("Unmatched row detail", md_table(unmatched[cols].sort_values(cols[:10]), 200)))
    write_report(REPORT_DIR / "arch_unmatched_rows_detail.md", "Architecture Unmatched Rows Detail", sections)
    return summary


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d_all = main_df(df, include_unmatched_arch=True)
    matched_rows, unmatched_rows = _arch_vs_resnet18_baseline_rows(d_all)
    strict_match_summary = _write_arch_unmatched_detail(d_all, matched_rows, unmatched_rows)
    d = main_df(df)
    arch_df = d[d["result_group"].isin(["baseline_strength", "arch_acc"])].copy()
    delta = _arch_pairwise_delta(arch_df)
    summary = summarize_delta(delta, ["dataset", "transfer_dataset", "comparison_group", "new_arch", "base_arch"]) if not delta.empty else pd.DataFrame()
    if not delta.empty:
        primary_mask = delta.apply(
            lambda r: (r["dataset"], r["transfer_dataset"], r["comparison_group"]) in PRIMARY_ARCH_COMPARISONS,
            axis=1,
        )
        primary_delta = delta[primary_mask].copy()
    else:
        primary_delta = pd.DataFrame()
    primary_summary = summarize_delta(primary_delta, ["dataset", "transfer_dataset", "comparison_group", "new_arch", "base_arch"]) if not primary_delta.empty else pd.DataFrame()
    rel_by_arch = grouped_correlations(arch_df, ["dataset", "transfer_dataset", "arch_base"])
    shift_rows = []
    if not summary.empty:
        for _, row in summary[["dataset", "transfer_dataset", "comparison_group", "new_arch", "base_arch"]].drop_duplicates().iterrows():
            base_sub = arch_df[
                (arch_df["dataset"] == row["dataset"])
                & (arch_df["transfer_dataset"] == row["transfer_dataset"])
                & (arch_df["arch_base"] == row["base_arch"])
            ]
            new_sub = arch_df[
                (arch_df["dataset"] == row["dataset"])
                & (arch_df["transfer_dataset"] == row["transfer_dataset"])
                & (arch_df["arch_base"] == row["new_arch"])
            ]
            base_rec = correlation_record(base_sub, "base")
            new_rec = correlation_record(new_sub, "new")
            shift_rows.append(
                {
                    "dataset": row["dataset"],
                    "transfer_dataset": row["transfer_dataset"],
                    "comparison_group": row["comparison_group"],
                    "new_arch": row["new_arch"],
                    "base_arch": row["base_arch"],
                    "base_n": base_rec["n"],
                    "new_n": new_rec["n"],
                    "base_spearman": base_rec["spearman"],
                    "new_spearman": new_rec["spearman"],
                    "delta_spearman": new_rec["spearman"] - base_rec["spearman"],
                    "base_pearson": base_rec["pearson"],
                    "new_pearson": new_rec["pearson"],
                    "delta_pearson": new_rec["pearson"] - base_rec["pearson"],
                }
            )
    rel = pd.DataFrame(shift_rows)
    primary_rel = (
        rel[
            rel.apply(
                lambda r: (r["dataset"], r["transfer_dataset"], r["comparison_group"]) in PRIMARY_ARCH_COMPARISONS,
                axis=1,
            )
        ].copy()
        if not rel.empty
        else pd.DataFrame()
    )
    acc_transfer = grouped_correlations(arch_df, ["dataset", "transfer_dataset", "arch_base"], x="clean_acc", y="transfer_rate")
    acc_stealth = grouped_correlations(arch_df, ["dataset", "transfer_dataset", "arch_base"], x="clean_acc", y="stealthiness")
    acc_shift = acc_transfer[["dataset", "transfer_dataset", "arch_base", "n", "spearman", "pearson"]].rename(columns={"spearman": "spearman_acc_transfer", "pearson": "pearson_acc_transfer"})
    acc_shift = acc_shift.merge(
        acc_stealth[["dataset", "transfer_dataset", "arch_base", "spearman", "pearson"]].rename(columns={"spearman": "spearman_acc_stealth", "pearson": "pearson_acc_stealth"}),
        on=["dataset", "transfer_dataset", "arch_base"],
        how="outer",
    )

    save_csv_and_md(delta, COEFFICIENT_DIR / "arch_pairwise_delta.csv", "Architecture Pairwise Delta")
    save_csv_and_md(primary_delta, COEFFICIENT_DIR / "arch_primary_pairwise_delta.csv", "Architecture Primary Pairwise Delta")
    save_csv_and_md(primary_summary, COEFFICIENT_DIR / "arch_primary_pairwise_summary.csv", "Architecture Primary Pairwise Summary")
    save_csv_and_md(rel, COEFFICIENT_DIR / "arch_relationship_shift.csv", "Architecture Relationship Shift")
    save_csv_and_md(primary_rel, COEFFICIENT_DIR / "arch_primary_relationship_shift.csv", "Architecture Primary Relationship Shift")
    save_csv_and_md(rel_by_arch, COEFFICIENT_DIR / "arch_relationship_by_arch.csv", "Architecture Relationship by Arch")
    save_csv_and_md(acc_shift, COEFFICIENT_DIR / "arch_acc_correlation_shift.csv", "Architecture ACC Correlation Shift")
    arch_summary = (
        arch_df.groupby(["dataset", "transfer_dataset", "arch_base", "attack_type"], dropna=False)
        .agg(n=("transfer_rate", "size"), clean_acc=("clean_acc", "median"), transfer_rate=("transfer_rate", "median"), stealthiness=("stealthiness", "median"))
        .reset_index()
    )
    save_csv_and_md(arch_summary, TABLE_DIR / "table_7_arch_summary.csv", "Architecture Summary")

    metric = arch_summary.groupby(["dataset", "transfer_dataset", "arch_base"], dropna=False).agg(clean_acc=("clean_acc", "median"), transfer_rate=("transfer_rate", "median"), stealthiness=("stealthiness", "median")).reset_index()
    metric["dataset_arch"] = metric["dataset"].astype(str) + ":" + metric["transfer_dataset"].astype(str) + ":" + metric["arch_base"].astype(str)
    metric_long = metric.melt(id_vars=["dataset_arch"], value_vars=["clean_acc", "transfer_rate", "stealthiness"], var_name="metric", value_name="value")
    grouped_bar(metric_long, "arch_metric_overview.png", "dataset_arch", "value", "metric", "Architecture Metric Overview")

    plot_delta_source = primary_summary if not primary_summary.empty else summary
    plot_delta = plot_delta_source.melt(id_vars=["dataset", "transfer_dataset", "comparison_group"], value_vars=[c for c in plot_delta_source.columns if c in ["delta_clean_acc_mean", "delta_transfer_rate_mean", "delta_stealthiness_mean"]], var_name="metric", value_name="delta") if not plot_delta_source.empty else pd.DataFrame()
    grouped_bar(plot_delta, "arch_pairwise_delta_summary.png", "comparison_group", "delta", "metric", "Architecture Pairwise Delta Summary")
    rel_plot = primary_rel if not primary_rel.empty else rel
    grouped_bar(rel_plot, "arch_relationship_shift_spearman.png", "comparison_group", "delta_spearman", "dataset", "Architecture Spearman Relationship Shift")
    acc_long = acc_shift.melt(id_vars=["dataset", "transfer_dataset", "arch_base"], value_vars=["spearman_acc_transfer", "spearman_acc_stealth"], var_name="metric", value_name="spearman")
    grouped_bar(acc_long, "arch_acc_vs_transfer_and_stealth.png", "arch_base", "spearman", "metric", "ACC Relationships with Transfer and Stealth")
    scatter_with_binned_line(arch_df, "arch_transfer_vs_stealth_by_arch.png", "transfer_rate", "stealthiness", hue="arch_base", col="dataset", title="Architecture Transfer-Stealth Plane")
    simple_heatmap(arch_summary, "arch_attack_heatmap.png", "attack_type", "arch_base", "transfer_rate", "Architecture Attack Heatmap", cmap="viridis", center=None)
    defense = arch_df.melt(id_vars=["arch_base"], value_vars=["stealth_sentinet", "stealth_scaleup", "stealth_strip", "stealth_ibd_psc"], var_name="defense", value_name="stealth_component")
    defense_sum = defense.groupby(["arch_base", "defense"], dropna=False).agg(stealth_mean=("stealth_component", "mean")).reset_index()
    grouped_bar(defense_sum, "arch_defense_breakdown.png", "arch_base", "stealth_mean", "defense", "Architecture Defense Breakdown")

    write_report(
        REPORT_DIR / "05_arch_acc_analysis.md",
        "05 Architecture / ACC Analysis",
        [
            (
                "Relationship-first framing",
                "Architecture/ACC is treated as a contrast window for observing whether the transfer-stealth relationship changes position or slope. The main question is not whether architecture or ACC separately predicts target-domain ASR or stealthiness; it is whether the intended architecture contrasts change the relationship between those two quantities.\n\n"
                + md_table(grouped_correlations(arch_df[arch_df["result_group"] == "arch_acc"], []), 20),
            ),
            (
                "Primary architecture contrasts",
                "The main architecture evidence is not an average over all architectures. It uses two intended strict matched contrasts as windows into the transfer-stealth plane: CIFAR-10 SmallCNN minus original ResNet18, and Tiny-ImageNet ResNet34 minus original ResNet18. Matching requires the same attack type, poison rate, strength, cover rate, label mode, and BELT mask rate where present. Other baseline architectures are kept only as diagnostics. Read delta transfer_rate and delta stealthiness together as movement in that plane.\n\n"
                + md_table(primary_summary, 40),
            ),
            ("Strict architecture match summary", md_table(strict_match_summary, 20)),
            ("Primary transfer-stealth relationship shift", md_table(primary_rel, 40)),
            ("All architecture pairwise comparisons (diagnostic)", md_table(summary, 80)),
            ("All transfer-stealth relationship shifts (diagnostic)", md_table(rel, 80)),
            ("Transfer-stealth relationship by architecture", md_table(rel_by_arch, 80)),
            ("Auxiliary ACC-to-single-metric correlations", md_table(acc_shift, 80)),
            ("Interpretation draft", "Architecture results should emphasize how the intended primary contrasts move or reshape the transfer-stealth relationship. Clean-ACC correlations with transfer or stealth are supporting explanations, not the main conclusion. Comparisons against MobileNetV2 or VGG19 are diagnostic and should not be merged into the main architecture conclusion."),
        ],
    )
    return {
        "delta": delta,
        "summary": summary,
        "primary_delta": primary_delta,
        "primary_summary": primary_summary,
        "relationship": rel,
        "primary_relationship": primary_rel,
        "relationship_by_arch": rel_by_arch,
        "acc_shift": acc_shift,
        "strict_match_summary": strict_match_summary,
        "unmatched_rows": unmatched_rows,
    }


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
