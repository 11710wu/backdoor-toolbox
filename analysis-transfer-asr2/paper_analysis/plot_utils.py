#!/usr/bin/env python3
"""Plot and markdown helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas._config.config import OptionError, register_option

from config import FIGURE_DIR, FIGURE_DOC_DIR
from figure_specs import FIGURE_SPECS, FigureSpec


try:
    pd.get_option("mode.use_inf_as_null")
except OptionError:
    register_option("mode.use_inf_as_null", False, "Compatibility option for older seaborn versions.")

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save_figure(fig: plt.Figure, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def write_figure_doc(filename: str, status: str = "generated", extra_note: str = "") -> None:
    FIGURE_DOC_DIR.mkdir(parents=True, exist_ok=True)
    spec = FIGURE_SPECS.get(
        filename,
        FigureSpec(
            filename=filename,
            purpose="Generated analysis figure.",
            data_source="master_results.csv",
            how_to_read="See axis labels and legends.",
            focus="Inspect the plotted metric trend.",
            conclusion_if_clear="Use with the corresponding coefficient tables.",
            weak_trend_note="Check completeness and sample size before interpreting.",
            coefficient_files="",
            recommend="backup",
        ),
    )
    path = FIGURE_DOC_DIR / filename.replace(".png", ".md")
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {filename}\n\n")
        f.write(f"## Purpose\n{spec.purpose}\n\n")
        f.write(f"## Data Source\n{spec.data_source}\n\n")
        f.write("## Filters\nComplete rows are used unless this figure is explicitly a completeness figure. Main analysis uses source_asr >= 0.05.\n\n")
        f.write(f"## How To Read\n{spec.how_to_read}\n\n")
        f.write(f"## Focus\n{spec.focus}\n\n")
        f.write(f"## Corresponding Coefficients\n{spec.coefficient_files or 'None'}\n\n")
        f.write(f"## If Trend Is Clear\n{spec.conclusion_if_clear}\n\n")
        f.write(f"## If Trend Is Weak\n{spec.weak_trend_note}\n\n")
        f.write(f"## Current Data Status\n{status}\n\n")
        f.write(f"## Recommended For Teacher Report\n{spec.recommend}\n")
        if extra_note:
            f.write(f"\n## Extra Note\n{extra_note}\n")


def insufficient_figure(filename: str, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.5, f"Insufficient data\n{reason}", ha="center", va="center", fontsize=13)
    save_figure(fig, filename)
    write_figure_doc(filename, status=f"insufficient data: {reason}")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    return all(col in df.columns for col in columns)


def scatter_with_binned_line(
    df: pd.DataFrame,
    filename: str,
    x: str,
    y: str,
    hue: Optional[str] = None,
    col: Optional[str] = None,
    title: str = "",
) -> None:
    needed = [x, y] + ([hue] if hue else []) + ([col] if col else [])
    if df.empty or not require_columns(df, needed):
        insufficient_figure(filename, "missing required columns or rows")
        return
    plot_df = df.dropna(subset=[x, y]).copy()
    if len(plot_df) < 3:
        insufficient_figure(filename, "fewer than 3 valid rows")
        return
    if col:
        col_values = list(plot_df[col].dropna().unique())
        if not col_values:
            insufficient_figure(filename, "no facet values")
            return
        n = len(col_values)
        fig, axes = plt.subplots(1, n, figsize=(max(6 * n, 8), 5.5), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, value in zip(axes, col_values):
            sub = plot_df[plot_df[col] == value]
            sns.scatterplot(data=sub, x=x, y=y, hue=hue, alpha=0.45, s=24, ax=ax, legend=False)
            add_binned_line(ax, sub, x, y)
            ax.set_title(f"{col} = {value}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        if hue:
            handles, labels = axes[0].get_legend_handles_labels()
            # Build a stable legend from the full dataset because per-axis legends are disabled.
            tmp_fig, tmp_ax = plt.subplots()
            sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, ax=tmp_ax)
            handles, labels = tmp_ax.get_legend_handles_labels()
            plt.close(tmp_fig)
            fig.legend(handles, labels, title=hue, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(title, y=1.02)
        save_figure(fig, filename)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, alpha=0.45, s=24, ax=ax)
        add_binned_line(ax, plot_df, x, y)
        ax.set_title(title)
        save_figure(fig, filename)
    write_figure_doc(filename)


def add_binned_line(ax, df: pd.DataFrame, x: str, y: str, bins: int = 8) -> None:
    sub = df[[x, y]].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna().sort_values(x)
    if len(sub) < 6 or sub[x].nunique() < 2:
        return
    try:
        sub["_bin"] = pd.qcut(sub[x].rank(method="first"), q=min(bins, max(2, len(sub) // 8)), duplicates="drop")
        agg = sub.groupby("_bin", observed=False).agg({x: "median", y: "median"}).dropna()
        if len(agg) >= 2:
            ax.plot(agg[x].to_numpy(), agg[y].to_numpy(), color="black", linewidth=3.0, marker="o", markersize=4, zorder=5)
    except Exception as exc:
        return


def simple_heatmap(df: pd.DataFrame, filename: str, index: str, columns: str, values: str, title: str, cmap: str = "vlag", center=0) -> None:
    if df.empty or not require_columns(df, [index, columns, values]):
        insufficient_figure(filename, "missing required columns or rows")
        return
    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    if pivot.empty:
        insufficient_figure(filename, "empty pivot table")
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pivot.columns)), max(5, 0.45 * len(pivot))))
    sns.heatmap(pivot, cmap=cmap, center=center, annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title(title)
    save_figure(fig, filename)
    write_figure_doc(filename)


def grouped_bar(df: pd.DataFrame, filename: str, x: str, y: str, hue: Optional[str], title: str, rotate: bool = True) -> None:
    needed = [x, y] + ([hue] if hue else [])
    if df.empty or not require_columns(df, needed):
        insufficient_figure(filename, "missing required columns or rows")
        return
    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        insufficient_figure(filename, "no valid metric rows")
        return
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * plot_df[x].nunique()), 6))
    sns.barplot(data=plot_df, x=x, y=y, hue=hue, ci=None, ax=ax)
    ax.set_title(title)
    if rotate:
        ax.tick_params(axis="x", rotation=35)
    save_figure(fig, filename)
    write_figure_doc(filename)


def line_plot(df: pd.DataFrame, filename: str, x: str, y: str, hue: Optional[str], title: str, style: Optional[str] = None) -> None:
    needed = [x, y] + ([hue] if hue else []) + ([style] if style else [])
    if df.empty or not require_columns(df, needed):
        insufficient_figure(filename, "missing required columns or rows")
        return
    plot_df = df.dropna(subset=[x, y]).copy()
    if len(plot_df) < 2:
        insufficient_figure(filename, "fewer than 2 valid rows")
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=plot_df, x=x, y=y, hue=hue, style=style, marker="o", ci=None, ax=ax)
    ax.set_title(title)
    save_figure(fig, filename)
    write_figure_doc(filename)


def write_report(path: Path, title: str, sections: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for heading, body in sections:
            f.write(f"## {heading}\n{body}\n\n")


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "No rows available."
    shown = df.head(max_rows)
    suffix = f"\n\nShowing first {max_rows} of {len(df)} rows." if len(df) > max_rows else ""
    return shown.to_markdown(index=False) + suffix
