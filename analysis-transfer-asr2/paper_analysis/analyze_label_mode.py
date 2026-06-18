#!/usr/bin/env python3
"""Label-mode / all-to-one analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import COEFFICIENT_DIR, REPORT_DIR, TABLE_DIR
from plot_utils import grouped_bar, insufficient_figure, md_table, save_figure, scatter_with_binned_line, write_figure_doc, write_report
from stats_utils import grouped_correlations, main_df, save_csv_and_md, summarize_delta


def _label_mode_arrow_plot(label_related: pd.DataFrame) -> None:
    plot_df = label_related.dropna(subset=["dataset", "attack_type", "label_mode", "transfer_rate", "stealthiness"]).copy()
    plot_df = plot_df[plot_df["label_mode"].isin(["clean", "all2one"])]
    if plot_df.empty:
        insufficient_figure("label_mode_clean_to_dirty_arrows.png", "no matched clean/all2one rows")
        return
    agg = (
        plot_df.groupby(["dataset", "attack_type", "label_mode"], dropna=False)
        .agg(transfer_rate=("transfer_rate", "mean"), stealthiness=("stealthiness", "mean"), n=("transfer_rate", "size"))
        .reset_index()
    )
    wide = agg.pivot_table(index=["dataset", "attack_type"], columns="label_mode", values=["transfer_rate", "stealthiness", "n"], aggfunc="first").reset_index()
    wide.columns = [f"{metric}_{mode}" if mode else str(metric) for metric, mode in wide.columns.to_flat_index()]
    required = ["transfer_rate_clean", "transfer_rate_all2one", "stealthiness_clean", "stealthiness_all2one"]
    if wide.empty or any(col not in wide.columns for col in required):
        insufficient_figure("label_mode_clean_to_dirty_arrows.png", "no matched clean/all2one group means")
        return
    wide = wide.dropna(subset=required)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9.5, 7))
    palette = dict(zip(sorted(wide["dataset"].dropna().unique()), sns.color_palette("Set2", n_colors=wide["dataset"].nunique())))
    for _, row in wide.iterrows():
        color = palette[row["dataset"]]
        x0 = row["transfer_rate_clean"]
        y0 = row["stealthiness_clean"]
        x1 = row["transfer_rate_all2one"]
        y1 = row["stealthiness_all2one"]
        ax.scatter([x0], [y0], s=95, facecolors="white", edgecolors=color, linewidths=2.2, marker="o")
        ax.scatter([x1], [y1], s=105, color=color, edgecolors="white", linewidths=0.8, marker="o")
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": color, "lw": 2.3, "alpha": 0.9})
        ax.text(x1 + 0.01, y1, f"{row['dataset']} / {row['attack_type']}", fontsize=9, color="0.2")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=dataset, markersize=8)
        for dataset, color in palette.items()
    ]
    handles.extend(
        [
            plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black", color="black", label="clean baseline", markersize=8),
            plt.Line2D([0], [0], marker="o", linestyle="", color="black", label="dirty/all-to-one", markersize=8),
        ]
    )
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("target-domain ASR")
    ax.set_ylabel("stealthiness")
    ax.set_title("Clean-label to Dirty-label Movement")
    fig.tight_layout()
    save_figure(fig, "label_mode_clean_to_dirty_arrows.png")
    write_figure_doc("label_mode_clean_to_dirty_arrows.png")


def _label_mode_clean_dirty_metric_summary(label_related: pd.DataFrame) -> None:
    plot_df = label_related.dropna(subset=["dataset", "attack_type", "label_mode", "transfer_rate", "stealthiness"]).copy()
    plot_df = plot_df[plot_df["label_mode"].isin(["clean", "all2one"])]
    if plot_df.empty:
        insufficient_figure("label_mode_clean_dirty_metric_summary.png", "no matched clean/all2one rows")
        return
    agg = (
        plot_df.groupby(["dataset", "attack_type", "label_mode"], dropna=False)
        .agg(transfer_rate=("transfer_rate", "mean"), stealthiness=("stealthiness", "mean"), n=("transfer_rate", "size"))
        .reset_index()
    )
    agg["group"] = agg["dataset"] + " / " + agg["attack_type"]
    long = agg.melt(id_vars=["group", "label_mode", "n"], value_vars=["transfer_rate", "stealthiness"], var_name="metric", value_name="value")
    sns.set_theme(style="whitegrid", context="talk")
    g = sns.catplot(
        data=long,
        x="group",
        y="value",
        hue="label_mode",
        col="metric",
        kind="bar",
        ci=None,
        height=5.2,
        aspect=1.15,
        sharey=True,
        palette={"clean": "#4C78A8", "all2one": "#F58518"},
    )
    g.set_axis_labels("", "mean value")
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0, 1.05)
    g.fig.suptitle("Clean-label vs Dirty-label Matched Metric Summary", y=1.05)
    save_figure(g.fig, "label_mode_clean_dirty_metric_summary.png")
    write_figure_doc("label_mode_clean_dirty_metric_summary.png")


def _label_mode_delta_quadrant(delta: pd.DataFrame) -> None:
    plot_df = delta.dropna(subset=["delta_transfer_rate", "delta_stealthiness", "dataset", "attack_type"]).copy()
    if plot_df.empty:
        insufficient_figure("label_mode_delta_quadrant.png", "no matched delta rows")
        return
    good = (plot_df["delta_transfer_rate"] > 0) & (plot_df["delta_stealthiness"] < 0)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9.5, 7))
    sns.scatterplot(
        data=plot_df,
        x="delta_transfer_rate",
        y="delta_stealthiness",
        hue="dataset",
        style="attack_type",
        s=58,
        alpha=0.68,
        linewidth=0.2,
        edgecolor="white",
        ax=ax,
    )
    ax.axhline(0, color="0.35", linestyle=":", linewidth=1.2)
    ax.axvline(0, color="0.35", linestyle=":", linewidth=1.2)
    ax.text(0.98, 0.05, f"+transfer / -stealth: {good.sum()}/{len(plot_df)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"})
    ax.set_xlabel("delta target-domain ASR (dirty/all-to-one - clean)")
    ax.set_ylabel("delta stealthiness (dirty/all-to-one - clean)")
    ax.set_title("Clean-label to Dirty-label Matched Delta Quadrant")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "label_mode_delta_quadrant.png")
    write_figure_doc("label_mode_delta_quadrant.png")


def _label_mode_deltas(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["dataset", "transfer_dataset", "transfer_variant", "arch_base", "attack_type", "poison_rate", "strength_name", "strength_value"]
    metric_cols = ["clean_acc", "source_asr", "transfer_asr", "transfer_rate", "stealthiness"]
    keep_cols = key_cols + ["result_group", "label_mode", "result_dir"] + metric_cols

    baseline = df[
        (df["result_group"] == "baseline_strength")
        & (df["label_mode"] == "clean")
        & (df["attack_type"].isin(["SIG", "upgd"]))
    ][keep_cols].copy()
    all2one = df[
        (df["result_group"] == "label_mode")
        & (df["label_mode"] == "all2one")
        & (df["attack_type"].isin(["SIG", "upgd"]))
    ][keep_cols].copy()
    if baseline.empty or all2one.empty:
        return pd.DataFrame()

    rows = all2one.merge(baseline, on=key_cols, how="inner", suffixes=("_all2one", "_baseline"))
    if rows.empty:
        return pd.DataFrame()
    rows["comparison_group"] = rows["attack_type"].map(lambda a: f"baseline_clean_{a}->label_all2one_{a}")
    rows["base_result_group"] = rows["result_group_baseline"]
    rows["new_result_group"] = rows["result_group_all2one"]
    rows["base_label_mode"] = rows["label_mode_baseline"]
    rows["new_label_mode"] = rows["label_mode_all2one"]
    rows["base_result_dir"] = rows["result_dir_baseline"]
    rows["new_result_dir"] = rows["result_dir_all2one"]
    for metric in metric_cols:
        rows[f"delta_{metric}"] = rows[f"{metric}_all2one"] - rows[f"{metric}_baseline"]
    out_cols = (
        key_cols
        + [
            "comparison_group",
            "base_result_group",
            "new_result_group",
            "base_label_mode",
            "new_label_mode",
            "base_result_dir",
            "new_result_dir",
        ]
        + [f"delta_{metric}" for metric in metric_cols]
    )
    return rows[out_cols]


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = main_df(df)
    label_main = d[d["result_group"] == "label_mode"].copy()
    delta = _label_mode_deltas(d)
    summary = summarize_delta(delta, ["dataset", "attack_type", "comparison_group"]) if not delta.empty else pd.DataFrame()
    if not delta.empty:
        matched_dirs = set(delta["base_result_dir"].dropna()) | set(delta["new_result_dir"].dropna())
        label_related = d[d["result_dir"].isin(matched_dirs)].copy()
    else:
        label_related = d.iloc[0:0].copy()
    completeness = (
        df[df["result_group"] == "label_mode"]
        .groupby(["dataset", "arch_base", "attack_type", "label_mode", "analysis_status"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    relationship = grouped_correlations(label_main, [])
    save_csv_and_md(delta, COEFFICIENT_DIR / "label_mode_pairwise_delta.csv", "Label Mode Pairwise Delta")
    save_csv_and_md(summary, TABLE_DIR / "table_6_label_mode_summary.csv", "Label Mode Summary")
    save_csv_and_md(completeness, TABLE_DIR / "label_mode_completeness.csv", "Label Mode Completeness")

    scatter_with_binned_line(
        label_related,
        "label_mode_transfer_vs_stealth.png",
        "transfer_rate",
        "stealthiness",
        hue="label_mode",
        col="dataset",
        title="Label Mode in Transfer-Stealth Plane",
    )
    plot_delta = summary.melt(
        id_vars=["dataset", "attack_type", "comparison_group"],
        value_vars=[c for c in summary.columns if c in ["delta_transfer_rate_mean", "delta_stealthiness_mean"]],
        var_name="metric",
        value_name="delta",
    ) if not summary.empty else pd.DataFrame()
    grouped_bar(plot_delta, "label_mode_pairwise_delta.png", "comparison_group", "delta", "metric", "Label Mode Pairwise Delta")
    _label_mode_arrow_plot(label_related)
    _label_mode_clean_dirty_metric_summary(label_related)
    _label_mode_delta_quadrant(delta)
    completeness_plot = completeness.groupby(["attack_type", "analysis_status"], dropna=False)["n"].sum().reset_index()
    grouped_bar(completeness_plot, "label_mode_completeness.png", "attack_type", "n", "analysis_status", "Label Mode Completeness")

    write_report(
        REPORT_DIR / "04_label_mode_analysis.md",
        "04 Label-mode Analysis",
        [
            ("Completeness", md_table(completeness, 80)),
            (
                "Relationship-first framing",
                "Label mode is used as a matched contrast window for observing transfer-stealth movement. The main question is whether clean-to-all-to-one matched rows move in the transfer-stealth plane toward higher target-domain ASR and lower stealthiness. The label-mode-to-single-metric changes are auxiliary explanations of that movement, not the final conclusion by themselves.\n\n"
                + md_table(relationship, 20),
            ),
            (
                "Matched clean-to-all-to-one movement summary",
                "The label-mode comparison is not made against the whole baseline pool. It matches label-mode all-to-one rows to baseline-strength clean rows with the same dataset, target domain, architecture, attack, poison rate, strength name, and strength value. Current main label-mode contrasts are SIG all-to-one vs baseline clean SIG, and UPGD all-to-one vs baseline clean UPGD. Read the deltas as movement in the transfer-stealth plane: delta target-domain ASR and delta stealthiness are the two coordinates of that movement.\n\n"
                + md_table(summary, 80),
            ),
            ("Matched row-level deltas", md_table(delta, 40)),
            ("Interpretation draft", "Label-mode conclusions should only be treated as strong when a same-attack baseline-clean configuration exists. The intended conclusion is about transfer-stealth movement under the clean-to-all-to-one window; label-mode effects on a single metric are supporting mechanism details."),
        ],
    )
    return {"delta": delta, "summary": summary, "completeness": completeness, "relationship": relationship}


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"))
