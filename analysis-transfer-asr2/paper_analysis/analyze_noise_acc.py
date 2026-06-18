#!/usr/bin/env python3
"""Noise / input difficulty analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import add_binned_line, grouped_bar, insufficient_figure, line_plot, md_table, save_figure, scatter_with_binned_line, simple_heatmap, write_figure_doc, write_report
from stats_utils import grouped_correlations, main_df, pairwise_adjacent_delta, save_csv_and_md, summarize_delta


def _spearman_text(df: pd.DataFrame, x: str, y: str) -> str:
    sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 3 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
        return "Spearman: n/a"
    return f"Spearman={sub[x].corr(sub[y], method='spearman'):.3f}, n={len(sub)}"


def _noise_clean_acc_metric_scatter(noise: pd.DataFrame) -> None:
    plot_df = noise.dropna(subset=["clean_acc", "transfer_rate", "stealthiness", "attack_type", "input_noise_type"]).copy()
    if plot_df.empty:
        insufficient_figure("noise_clean_acc_metric_scatter.png", "no valid noise rows")
        return
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6), sharex=True)
    palette = dict(zip(sorted(plot_df["attack_type"].dropna().unique()), sns.color_palette("tab10", n_colors=plot_df["attack_type"].nunique())))
    panels = [("transfer_rate", "target-domain ASR"), ("stealthiness", "stealthiness")]
    for ax, (metric, label) in zip(axes, panels):
        sns.scatterplot(
            data=plot_df,
            x="clean_acc",
            y=metric,
            hue="attack_type",
            style="input_noise_type",
            palette=palette,
            s=55,
            alpha=0.62,
            linewidth=0.2,
            edgecolor="white",
            ax=ax,
            legend=(metric == "stealthiness"),
        )
        add_binned_line(ax, plot_df, "clean_acc", metric)
        ax.text(0.02, 0.05, _spearman_text(plot_df, "clean_acc", metric), transform=ax.transAxes, fontsize=10, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
        ax.set_title(f"clean ACC vs {label}")
        ax.set_xlabel("clean ACC under noise")
        ax.set_ylabel(label)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.04, 1.04)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9, title="attack / noise")
    fig.suptitle("Noise ACC Effect on Transferability and Stealthiness", y=1.02, fontsize=18)
    fig.tight_layout()
    save_figure(fig, "noise_clean_acc_metric_scatter.png")
    write_figure_doc("noise_clean_acc_metric_scatter.png")


def _noise_vs_baseline_delta_acc_effect(rows: pd.DataFrame) -> None:
    plot_df = rows.dropna(subset=["delta_clean_acc", "delta_transfer_rate", "delta_stealthiness", "attack_type", "input_noise_type"]).copy()
    if plot_df.empty:
        insufficient_figure("noise_vs_baseline_delta_acc_effect.png", "no valid noise-vs-baseline rows")
        return
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6), sharex=True)
    palette = dict(zip(sorted(plot_df["attack_type"].dropna().unique()), sns.color_palette("tab10", n_colors=plot_df["attack_type"].nunique())))
    panels = [("delta_transfer_rate", "delta target-domain ASR"), ("delta_stealthiness", "delta stealthiness")]
    for ax, (metric, label) in zip(axes, panels):
        sns.scatterplot(
            data=plot_df,
            x="delta_clean_acc",
            y=metric,
            hue="attack_type",
            style="input_noise_type",
            palette=palette,
            s=62,
            alpha=0.65,
            linewidth=0.2,
            edgecolor="white",
            ax=ax,
            legend=(metric == "delta_stealthiness"),
        )
        sns.regplot(data=plot_df, x="delta_clean_acc", y=metric, scatter=False, color="black", line_kws={"linewidth": 2.6}, ax=ax)
        ax.axhline(0, color="0.35", linewidth=1.2, linestyle=":")
        ax.axvline(0, color="0.35", linewidth=1.2, linestyle=":")
        ax.text(0.02, 0.05, _spearman_text(plot_df, "delta_clean_acc", metric), transform=ax.transAxes, fontsize=10, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
        ax.set_title(f"ACC drop vs {label}")
        ax.set_xlabel("delta clean ACC (noise - ResNet18 baseline)")
        ax.set_ylabel(label)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9, title="attack / noise")
    fig.suptitle("Noise-vs-Baseline ACC Delta Effect", y=1.02, fontsize=18)
    fig.tight_layout()
    save_figure(fig, "noise_vs_baseline_delta_acc_effect.png")
    write_figure_doc("noise_vs_baseline_delta_acc_effect.png")


def _noise_acc_bin_metric_summary(noise: pd.DataFrame) -> None:
    if "acc_bin" not in noise.columns:
        insufficient_figure("noise_acc_bin_metric_summary.png", "missing acc_bin")
        return
    plot_df = (
        noise.dropna(subset=["acc_bin", "clean_acc", "transfer_rate", "stealthiness"])
        .groupby(["acc_bin"], dropna=False)
        .agg(clean_acc=("clean_acc", "median"), transfer_rate=("transfer_rate", "median"), stealthiness=("stealthiness", "median"), n=("transfer_rate", "size"))
        .reset_index()
    )
    if plot_df.empty:
        insufficient_figure("noise_acc_bin_metric_summary.png", "no valid acc-bin rows")
        return
    order = [v for v in ["low_acc", "mid_acc", "high_acc"] if v in set(plot_df["acc_bin"])]
    long = plot_df.melt(id_vars=["acc_bin", "n"], value_vars=["clean_acc", "transfer_rate", "stealthiness"], var_name="metric", value_name="median")
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10.5, 6))
    sns.barplot(data=long, x="acc_bin", y="median", hue="metric", order=order or None, ci=None, ax=ax)
    for i, row in plot_df.set_index("acc_bin").loc[order or plot_df["acc_bin"].tolist()].reset_index().iterrows():
        ax.text(i, 1.02, f"n={int(row['n'])}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("clean ACC bin under noise")
    ax.set_ylabel("median value")
    ax.set_title("Noise ACC Bin Metric Summary")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    save_figure(fig, "noise_acc_bin_metric_summary.png")
    write_figure_doc("noise_acc_bin_metric_summary.png")


def _noise_attack_conditioned_acc_correlations(noise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attack, sub in noise.groupby("attack_type", dropna=False):
        rec = {"attack_type": attack, "n": len(sub)}
        for metric in ["transfer_rate", "stealthiness"]:
            valid = sub[["clean_acc", metric]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(valid) >= 3 and valid["clean_acc"].nunique() > 1 and valid[metric].nunique() > 1:
                rec[f"spearman_clean_acc_{metric}"] = valid["clean_acc"].corr(valid[metric], method="spearman")
            else:
                rec[f"spearman_clean_acc_{metric}"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("spearman_clean_acc_transfer_rate", ascending=False)


def _noise_attack_conditioned_acc_plot(corr: pd.DataFrame) -> None:
    if corr.empty:
        insufficient_figure("noise_attack_conditioned_acc_effect.png", "no attack-conditioned ACC correlations")
        return
    long = corr.melt(
        id_vars=["attack_type", "n"],
        value_vars=["spearman_clean_acc_transfer_rate", "spearman_clean_acc_stealthiness"],
        var_name="metric",
        value_name="spearman",
    ).dropna(subset=["spearman"])
    if long.empty:
        insufficient_figure("noise_attack_conditioned_acc_effect.png", "no valid attack-conditioned ACC correlations")
        return
    long["metric"] = long["metric"].map(
        {
            "spearman_clean_acc_transfer_rate": "clean ACC vs target-domain ASR",
            "spearman_clean_acc_stealthiness": "clean ACC vs stealthiness",
        }
    )
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=long, x="attack_type", y="spearman", hue="metric", ci=None, ax=ax)
    ax.axhline(0, color="0.3", linewidth=1.1, linestyle=":")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("attack type")
    ax.set_ylabel("Spearman")
    ax.set_title("Noise Attack-conditioned ACC Effect")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    save_figure(fig, "noise_attack_conditioned_acc_effect.png")
    write_figure_doc("noise_attack_conditioned_acc_effect.png")


def _with_attack_match_type(df: pd.DataFrame) -> pd.DataFrame:
    """Use strict attack names for controlled baseline matching."""
    out = df.copy()
    out["attack_match_type"] = out["attack_type"]
    return out


def _write_unmatched_detail(noise: pd.DataFrame, matched_rows: pd.DataFrame) -> None:
    if noise.empty:
        return
    matched_ids = set(matched_rows["_noise_row_id"]) if "_noise_row_id" in matched_rows.columns else set()
    unmatched = noise[~noise["_noise_row_id"].isin(matched_ids)].copy() if "_noise_row_id" in noise.columns else noise.copy()
    sections = []
    note = (
        "Noise-vs-baseline matching uses dataset, transfer target, transfer variant, strict attack_type, "
        "poison_rate, strength_name/value, cover_rate, label_mode, and mask_rate when present. Noise rows whose attack name or strength does not "
        "exactly match the original baseline are excluded from the current analysis and should be rerun with "
        "the matched baseline configuration. In particular, old noise `badnet` rows are not merged into "
        "baseline `basic`; they are treated as unmatched until rerun as `basic`.\n\n"
    )
    if unmatched.empty:
        sections.append(("Unmatched summary", note + "All noise rows have a matched baseline row."))
    else:
        counts = unmatched["attack_type"].value_counts(dropna=False).rename_axis("attack_type").reset_index(name="unmatched_rows")
        sections.append(("Unmatched summary", note + md_table(counts, 30)))
        cols = [
            "attack_type",
            "attack_match_type",
            "poison_rate",
            "strength_name",
            "strength_value",
            "cover_rate",
            "mask_rate",
            "label_mode",
            "transfer_variant",
            "input_noise_type",
            "input_noise_level",
            "clean_acc",
            "transfer_rate",
            "stealthiness",
            "folder_name",
        ]
        cols = [c for c in cols if c in unmatched.columns]
        for attack, sub in unmatched.groupby("attack_type", dropna=False):
            sort_cols = [c for c in cols[:8] if c in sub.columns]
            sections.append((f"{attack} unmatched rows", md_table(sub[cols].sort_values(sort_cols), 120)))
    write_report(REPORT_DIR / "noise_unmatched_rows_detail.md", "Noise Unmatched Rows Detail", sections)


def _noise_vs_baseline(d: pd.DataFrame, noise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare CIFAR noise rows with matched baseline-strength configurations.

    The noise experiments use SmallCNN while the matched baseline CIFAR rows
    are restricted to the original ResNet18 baseline, so this is a
    configuration-controlled group comparison rather than a strict
    same-architecture paired test.
    """
    baseline = d[
        (d["result_group"] == "baseline_strength")
        & (d["dataset"] == "cifar10")
        & (d["transfer_dataset"] == "stl10")
        & (d["arch_base"] == "ResNet18")
    ].copy()
    if baseline.empty or noise.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    baseline = _with_attack_match_type(baseline)
    noise = _with_attack_match_type(noise)
    noise["_noise_row_id"] = noise.index
    keys = [
        "dataset",
        "transfer_dataset",
        "transfer_variant",
        "attack_match_type",
        "poison_rate",
        "strength_name",
        "strength_value",
        "cover_rate",
        "label_mode",
    ]
    if "mask_rate" in baseline.columns and "mask_rate" in noise.columns:
        keys.append("mask_rate")
    baseline_agg = (
        baseline.groupby(keys, dropna=False)
        .agg(
            baseline_attack_types=("attack_type", lambda s: ",".join(sorted(set(map(str, s))))),
            baseline_n=("transfer_rate", "size"),
            baseline_clean_acc_mean=("clean_acc", "mean"),
            baseline_source_asr_mean=("source_asr", "mean"),
            baseline_transfer_rate_mean=("transfer_rate", "mean"),
            baseline_stealthiness_mean=("stealthiness", "mean"),
        )
        .reset_index()
    )
    rows = noise.merge(baseline_agg, on=keys, how="inner")
    _write_unmatched_detail(noise, rows)
    if rows.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    rows = rows.drop(columns=["_noise_row_id"], errors="ignore")

    rows["comparison_group"] = "noise_minus_resnet18_baseline"
    rows["baseline_arch"] = "ResNet18"
    metric_pairs = {
        "clean_acc": "baseline_clean_acc_mean",
        "source_asr": "baseline_source_asr_mean",
        "transfer_rate": "baseline_transfer_rate_mean",
        "stealthiness": "baseline_stealthiness_mean",
    }
    for metric, base_metric in metric_pairs.items():
        rows[f"delta_{metric}"] = pd.to_numeric(rows[metric], errors="coerce") - pd.to_numeric(rows[base_metric], errors="coerce")

    overall = pd.DataFrame(
        [
            {
                "comparison_group": "noise_minus_resnet18_baseline",
                "baseline_arch": "ResNet18",
                "attack_matching": "strict_attack_type_match_old_badnet_excluded",
                "n_pairs": len(rows),
                "matched_noise_rows": len(rows),
                "unmatched_noise_rows": max(len(noise) - len(rows), 0),
                "delta_clean_acc_mean": rows["delta_clean_acc"].mean(),
                "delta_clean_acc_median": rows["delta_clean_acc"].median(),
                "delta_source_asr_mean": rows["delta_source_asr"].mean(),
                "delta_transfer_rate_mean": rows["delta_transfer_rate"].mean(),
                "delta_transfer_rate_median": rows["delta_transfer_rate"].median(),
                "delta_stealthiness_mean": rows["delta_stealthiness"].mean(),
                "delta_stealthiness_median": rows["delta_stealthiness"].median(),
                "spearman_delta_acc_delta_transfer": rows["delta_clean_acc"].corr(rows["delta_transfer_rate"], method="spearman"),
                "spearman_delta_acc_delta_stealth": rows["delta_clean_acc"].corr(rows["delta_stealthiness"], method="spearman"),
            }
        ]
    )
    by_attack = (
        rows.groupby(["attack_type"], dropna=False)
        .agg(
            n=("transfer_rate", "size"),
            delta_clean_acc_mean=("delta_clean_acc", "mean"),
            delta_transfer_rate_mean=("delta_transfer_rate", "mean"),
            delta_stealthiness_mean=("delta_stealthiness", "mean"),
        )
        .reset_index()
        .sort_values("delta_transfer_rate_mean", ascending=False)
    )
    by_level = (
        rows.groupby(["input_noise_type", "input_noise_level"], dropna=False)
        .agg(
            n=("transfer_rate", "size"),
            delta_clean_acc_mean=("delta_clean_acc", "mean"),
            delta_transfer_rate_mean=("delta_transfer_rate", "mean"),
            delta_stealthiness_mean=("delta_stealthiness", "mean"),
        )
        .reset_index()
        .sort_values(["input_noise_type", "input_noise_level"])
    )
    return rows, overall, by_attack, by_level


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = main_df(df, include_unmatched_noise=True)
    noise_all = d[(d["result_group"] == "noise_acc") & (d["input_noise_type"].astype(str) != "")].copy()
    baseline_rows, baseline_overall, baseline_by_attack, baseline_by_level = _noise_vs_baseline(d, noise_all)
    noise = baseline_rows.copy()
    group_cols = [
        "transfer_dataset",
        "transfer_variant",
        "attack_type",
        "poison_rate",
        "strength_name",
        "strength_value",
        "cover_rate",
        "label_mode",
        "input_noise_type",
    ]
    if "mask_rate" in noise.columns:
        group_cols.insert(7, "mask_rate")
    delta = pairwise_adjacent_delta(noise, group_cols, "input_noise_level", "noise_adjacent")
    summary = summarize_delta(delta, ["attack_type", "input_noise_type"]) if not delta.empty else pd.DataFrame()
    by_level = (
        noise.groupby(["input_noise_type", "input_noise_level", "attack_type"], dropna=False)
        .agg(n=("transfer_rate", "size"), clean_acc=("clean_acc", "median"), transfer_rate=("transfer_rate", "median"), stealthiness=("stealthiness", "median"))
        .reset_index()
    )
    relationship = grouped_correlations(noise, [])
    acc_bin_corr = grouped_correlations(noise, ["acc_bin"], x="transfer_rate", y="stealthiness") if "acc_bin" in noise.columns else pd.DataFrame()
    attack_acc_corr = _noise_attack_conditioned_acc_correlations(noise)

    save_csv_and_md(delta, COEFFICIENT_DIR / "noise_pairwise_delta.csv", "Noise Pairwise Delta")
    save_csv_and_md(acc_bin_corr, COEFFICIENT_DIR / "noise_acc_bin_correlations.csv", "Noise ACC-bin Correlations")
    save_csv_and_md(by_level, COEFFICIENT_DIR / "noise_by_type_level.csv", "Noise by Type and Level")
    save_csv_and_md(baseline_rows, COEFFICIENT_DIR / "noise_vs_baseline_rows.csv", "Noise vs Baseline Rows")
    save_csv_and_md(baseline_overall, COEFFICIENT_DIR / "noise_vs_baseline_overall.csv", "Noise vs Baseline Overall")
    save_csv_and_md(baseline_by_attack, COEFFICIENT_DIR / "noise_vs_baseline_by_attack.csv", "Noise vs Baseline by Attack")
    save_csv_and_md(baseline_by_level, COEFFICIENT_DIR / "noise_vs_baseline_by_level.csv", "Noise vs Baseline by Level")
    save_csv_and_md(attack_acc_corr, COEFFICIENT_DIR / "noise_attack_conditioned_acc_correlations.csv", "Noise Attack-conditioned ACC Correlations")
    save_csv_and_md(by_level, TABLE_DIR / "table_8_noise_summary.csv", "Noise Summary")

    line_plot(by_level, "noise_acc_vs_level_by_type.png", "input_noise_level", "clean_acc", "input_noise_type", "Noise ACC vs Level")
    metric_long = by_level.melt(id_vars=["input_noise_type", "input_noise_level"], value_vars=["clean_acc", "transfer_rate", "stealthiness"], var_name="metric", value_name="value")
    line_plot(metric_long, "noise_metric_curves_by_noise_type.png", "input_noise_level", "value", "input_noise_type", "Noise Metric Curves by Type", style="metric")
    plot_delta = summary.melt(id_vars=["attack_type", "input_noise_type"], value_vars=[c for c in summary.columns if c in ["delta_clean_acc_mean", "delta_transfer_rate_mean", "delta_stealthiness_mean"]], var_name="metric", value_name="delta") if not summary.empty else pd.DataFrame()
    grouped_bar(plot_delta, "noise_paired_delta_by_level.png", "input_noise_type", "delta", "metric", "Noise Paired Delta by Level")
    scatter_with_binned_line(noise, "noise_transfer_vs_stealth_by_noise_type.png", "transfer_rate", "stealthiness", hue="input_noise_type", col=None, title="Noise Transfer-Stealth by Type")
    scatter_with_binned_line(noise, "noise_transfer_vs_stealth_by_acc_bin.png", "transfer_rate", "stealthiness", hue="acc_bin", col=None, title="Noise Transfer-Stealth by ACC Bin")
    _noise_clean_acc_metric_scatter(noise)
    _noise_vs_baseline_delta_acc_effect(baseline_rows)
    _noise_acc_bin_metric_summary(noise)
    _noise_attack_conditioned_acc_plot(attack_acc_corr)
    simple_heatmap(by_level, "noise_attack_heatmap.png", "attack_type", "input_noise_type", "transfer_rate", "Noise Attack Heatmap", cmap="viridis", center=None)
    defense = noise.melt(id_vars=["input_noise_type"], value_vars=["stealth_sentinet", "stealth_scaleup", "stealth_strip", "stealth_ibd_psc"], var_name="defense", value_name="stealth_component")
    defense_sum = defense.groupby(["input_noise_type", "defense"], dropna=False).agg(stealth_mean=("stealth_component", "mean")).reset_index()
    grouped_bar(defense_sum, "noise_defense_breakdown.png", "input_noise_type", "stealth_mean", "defense", "Noise Defense Breakdown")

    write_report(
        REPORT_DIR / "06_noise_acc_analysis.md",
        "06 Noise / Difficulty Analysis",
        [
            (
                "Relationship-first framing",
                "Noise/difficulty is treated as a condition for observing whether the transfer-stealth relationship becomes steeper or moves to a different region. The main question is not whether noise or ACC separately determines one metric; it is whether target-domain ASR and stealthiness remain inversely related under noise and relative to the matched ResNet18 baseline.\n\n"
                + md_table(relationship, 20),
            ),
            ("Noise by type and level", md_table(by_level, 80)),
            (
                "Noise vs baseline movement overall",
                "This is the RQ2 noise-vs-baseline comparison. It compares CIFAR noise rows with matched baseline-strength ResNet18 configurations when strict attack type, poison rate, strength, cover rate, label mode, mask rate when present, and transfer target match. Old noise rows whose attack name or strength does not exactly match the baseline are excluded from the analysis; old `badnet` rows are therefore not merged into baseline `basic` and should be rerun as `basic`. Because the current noise group uses SmallCNN while the baseline is original ResNet18, this is a configuration-controlled group comparison rather than a strict same-architecture paired test. Read delta transfer_rate and delta stealthiness together as movement in the transfer-stealth plane; clean-ACC deltas are supporting difficulty context.\n\n"
                + md_table(baseline_overall, 20),
            ),
            ("Noise vs baseline by attack", md_table(baseline_by_attack, 80)),
            ("Noise vs baseline by level", md_table(baseline_by_level, 80)),
            ("Noise pairwise delta", md_table(summary, 80)),
            ("Noise ACC-bin correlations", md_table(acc_bin_corr, 80)),
            ("Noise attack-conditioned ACC correlations", md_table(attack_acc_corr, 80)),
            ("Interpretation draft", "Noise experiments are treated in two ways: the main evidence is how noise-vs-baseline rows move in the transfer-stealth plane, while adjacent noise-level deltas are a supplementary within-noise window. Noise/ACC effects on single metrics are mechanism context and should not be written as a single monotonic causal rule."),
        ],
    )
    return {
        "delta": delta,
        "summary": summary,
        "by_level": by_level,
        "relationship": relationship,
        "acc_bin_corr": acc_bin_corr,
        "attack_acc_corr": attack_acc_corr,
        "baseline_rows": baseline_rows,
        "baseline_overall": baseline_overall,
        "baseline_by_attack": baseline_by_attack,
        "baseline_by_level": baseline_by_level,
    }


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
