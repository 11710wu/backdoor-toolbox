#!/usr/bin/env python3
"""Analyze whether CIFAR-10 noise difficulty changes transfer/stealth relation.

This script scans noisy CIFAR-10 SmallCNN result folders directly instead of
depending on extract_all_results.py. The goal is to keep the analysis tied to
the experiment definition:

  difficulty      = 1 - clean_acc
  transfer_rate   = transfer_asr ** 2 / source_asr
  stealth_avg     = mean(1 - defense_tpr) over SentiNet/STRIP/ScaleUp/IBD-PSC

All rates are normalized to [0, 1] before analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ATTACKS = [
    "adaptive_blend",
    "adaptive_patch",
    "badnet",
    "blend",
    "SIG",
    "WaNet",
    "belt",
    "upgd",
]

DEFENSE_FILES = {
    "sentinet": "sentinet_defense_results.json",
    "strip": "strip_defense_results.json",
    "scaleup": "scaleup_defense_results.json",
    "ibd_psc": "ibd_psc_defense_results.json",
}

STEALTH_COLS = [
    "stealth_sentinet",
    "stealth_strip",
    "stealth_scaleup",
    "stealth_ibd_psc",
]


@dataclass
class ParsedFolder:
    attack_type: str
    poison_rate: float
    strength_name: Optional[str]
    strength_value: Optional[float]
    cover_rate: Optional[float]
    input_noise_type: str
    input_noise_level: float
    arch: Optional[str]


def normalize_rate(value: Any) -> float:
    """Normalize a scalar rate to [0, 1] when it is stored as a percent."""
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if out > 1.0:
        out /= 100.0
    return out


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_folder_name(name: str) -> Optional[ParsedFolder]:
    attack_type = None
    for attack in sorted(ATTACKS, key=len, reverse=True):
        if name.startswith(f"{attack}_"):
            attack_type = attack
            break
    if attack_type is None:
        return None

    rest = name[len(attack_type) + 1 :]
    rate_token = rest.split("_", 1)[0]
    poison_rate = safe_float(rate_token)
    if poison_rate is None:
        return None

    strength_name = None
    strength_value = None
    for key, pat in [
        ("alpha", r"_alpha=([0-9.]+)"),
        ("delta", r"_delta=([0-9.]+)"),
        ("s", r"_s=([0-9.]+)"),
        ("mask_rate", r"_mask=([0-9.]+)"),
        ("eps", r"_eps=([0-9.]+)"),
    ]:
        m = re.search(pat, name)
        if m:
            strength_name = key
            strength_value = float(m.group(1))
            break

    cover_rate = None
    if m := re.search(r"_cover=([0-9.]+)", name):
        cover_rate = float(m.group(1))

    noise_match = re.search(r"_noise=(gaussian|uniform|salt_pepper|speckle)_level=([0-9.]+)_", name)
    if not noise_match:
        return None
    input_noise_type = noise_match.group(1)
    input_noise_level = float(noise_match.group(2))

    arch = None
    if m := re.search(r"_arch=([^/]+)$", name):
        arch = m.group(1)

    return ParsedFolder(
        attack_type=attack_type,
        poison_rate=poison_rate,
        strength_name=strength_name,
        strength_value=strength_value,
        cover_rate=cover_rate,
        input_noise_type=input_noise_type,
        input_noise_level=input_noise_level,
        arch=arch,
    )


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def choose_source_result_file(folder: Path) -> Optional[Path]:
    base = folder / "test_results_seed=2333.json"
    if base.exists():
        return base
    files = sorted(folder.glob("test_results_seed=2333*.json"))
    return files[0] if files else None


def choose_transfer_file(folder: Path) -> Optional[Path]:
    base = folder / "test_stl10_results.txt"
    if base.exists():
        return base
    files = sorted(folder.glob("test_stl10_results*.txt"))
    return files[0] if files else None


def parse_transfer_text(path: Path) -> Tuple[float, float]:
    """Return (transfer_acc, transfer_asr) parsed from STL10 result text."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return float("nan"), float("nan")

    acc = float("nan")
    asr = float("nan")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("准确率") or line.lower().startswith("accuracy"):
            if m := re.search(r"[:：]\s*([0-9.]+)", line):
                acc = normalize_rate(m.group(1))
        if "攻击成功率" in line or ("ASR" in line and ":" in line):
            if m := re.search(r"[:：]\s*([0-9.]+)", line):
                asr = normalize_rate(m.group(1))
    return acc, asr


def parse_defense(path: Path) -> Dict[str, float]:
    data = load_json(path)
    if not data:
        return {"tpr": float("nan"), "auc": float("nan"), "fpr": float("nan"), "threshold": float("nan")}
    return {
        "tpr": normalize_rate(data.get("tpr")),
        "auc": normalize_rate(data.get("auc")),
        "fpr": normalize_rate(data.get("fpr")),
        "threshold": float(data.get("threshold", data.get("threshold_low", float("nan"))) or float("nan")),
    }


def parse_result_folder(folder: Path) -> Dict[str, Any]:
    parsed = parse_folder_name(folder.name)
    if parsed is None:
        raise ValueError(f"not a noisy result folder: {folder.name}")

    row: Dict[str, Any] = {
        "result_dir": str(folder),
        "folder_name": folder.name,
        "attack_type": parsed.attack_type,
        "poison_rate": parsed.poison_rate,
        "strength_name": parsed.strength_name,
        "strength_value": parsed.strength_value,
        "cover_rate": parsed.cover_rate,
        "input_noise_type": parsed.input_noise_type,
        "input_noise_level": parsed.input_noise_level,
        "arch": parsed.arch,
    }

    missing: List[str] = []

    source_file = choose_source_result_file(folder)
    row["source_result_file"] = str(source_file) if source_file else ""
    if source_file is None:
        missing.append("source_test")
        source_data = None
    else:
        source_data = load_json(source_file)
        if not source_data:
            missing.append("source_test_json_parse")

    if source_data:
        row["clean_acc"] = normalize_rate(source_data.get("clean_acc"))
        row["source_asr"] = normalize_rate(source_data.get("asr"))
    else:
        row["clean_acc"] = float("nan")
        row["source_asr"] = float("nan")
    row["difficulty"] = 1.0 - row["clean_acc"] if pd.notna(row["clean_acc"]) else float("nan")

    transfer_file = choose_transfer_file(folder)
    row["transfer_result_file"] = str(transfer_file) if transfer_file else ""
    if transfer_file is None:
        missing.append("transfer_stl10")
        row["transfer_acc"] = float("nan")
        row["transfer_asr"] = float("nan")
    else:
        transfer_acc, transfer_asr = parse_transfer_text(transfer_file)
        row["transfer_acc"] = transfer_acc
        row["transfer_asr"] = transfer_asr
        if pd.isna(transfer_asr):
            missing.append("transfer_stl10_parse")

    source_asr = row["source_asr"]
    transfer_asr = row["transfer_asr"]
    valid_transfer_rate = pd.notna(source_asr) and source_asr > 0 and pd.notna(transfer_asr)
    row["valid_transfer_rate"] = bool(valid_transfer_rate)
    row["transfer_rate"] = (transfer_asr ** 2 / source_asr) if valid_transfer_rate else float("nan")

    complete_defenses = True
    for defense, filename in DEFENSE_FILES.items():
        path = folder / filename
        row[f"{defense}_result_file"] = str(path) if path.exists() else ""
        if not path.exists():
            missing.append(defense)
            complete_defenses = False
            rec = {"tpr": float("nan"), "auc": float("nan"), "fpr": float("nan"), "threshold": float("nan")}
        else:
            rec = parse_defense(path)
            if pd.isna(rec["tpr"]):
                missing.append(f"{defense}_parse")
                complete_defenses = False
        row[f"{defense}_tpr"] = rec["tpr"]
        row[f"{defense}_auc"] = rec["auc"]
        row[f"{defense}_fpr"] = rec["fpr"]
        row[f"{defense}_threshold"] = rec["threshold"]
        row[f"stealth_{defense}"] = 1.0 - rec["tpr"] if pd.notna(rec["tpr"]) else float("nan")

    row["complete_defense_results"] = bool(complete_defenses)
    stealth_values = [row[col] for col in STEALTH_COLS if pd.notna(row.get(col))]
    row["stealth_avg"] = float(np.mean(stealth_values)) if len(stealth_values) == len(STEALTH_COLS) else float("nan")
    row["missing_items"] = ",".join(missing)
    return row


def discover_rows(root: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    folders = sorted([p for p in root.iterdir() if p.is_dir() and "_noise=" in p.name])
    if max_rows is not None:
        folders = folders[:max_rows]

    rows: List[Dict[str, Any]] = []
    skipped: List[Tuple[str, str]] = []
    for folder in folders:
        try:
            rows.append(parse_result_folder(folder))
        except Exception as exc:
            skipped.append((str(folder), str(exc)))

    df = pd.DataFrame(rows)
    if skipped:
        skipped_df = pd.DataFrame(skipped, columns=["folder", "reason"])
        df.attrs["skipped"] = skipped_df
    return df


def add_analysis_flags(df: pd.DataFrame, source_asr_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["valid_source_asr_for_main"] = out["source_asr"] >= source_asr_threshold
    out["include_main_analysis"] = (
        out["valid_transfer_rate"]
        & out["complete_defense_results"]
        & out["valid_source_asr_for_main"]
        & out["stealth_avg"].notna()
        & out["clean_acc"].notna()
    )
    try:
        out["acc_bin"] = pd.qcut(out.loc[out["include_main_analysis"], "clean_acc"], 3, labels=["low_acc", "mid_acc", "high_acc"])
        all_bins = pd.Series(index=out.index, dtype="object")
        all_bins.loc[out["include_main_analysis"]] = out["acc_bin"].astype("object")
        out["acc_bin"] = all_bins
    except Exception:
        out["acc_bin"] = None
    return out


def pearson(x: pd.Series, y: pd.Series) -> float:
    sub = pd.concat([x, y], axis=1).dropna()
    if len(sub) < 2:
        return float("nan")
    return float(sub.iloc[:, 0].corr(sub.iloc[:, 1], method="pearson"))


def spearman(x: pd.Series, y: pd.Series) -> float:
    sub = pd.concat([x, y], axis=1).dropna()
    if len(sub) < 2:
        return float("nan")
    return float(sub.iloc[:, 0].corr(sub.iloc[:, 1], method="spearman"))


def corr_record(df: pd.DataFrame, group_type: str, group_name: str, source_asr_filter: str) -> Dict[str, Any]:
    return {
        "group_type": group_type,
        "group_name": group_name,
        "source_asr_filter": source_asr_filter,
        "n_rows": int(len(df)),
        "pearson_clean_acc_transfer_rate": pearson(df["clean_acc"], df["transfer_rate"]),
        "spearman_clean_acc_transfer_rate": spearman(df["clean_acc"], df["transfer_rate"]),
        "pearson_difficulty_transfer_rate": pearson(df["difficulty"], df["transfer_rate"]),
        "spearman_difficulty_transfer_rate": spearman(df["difficulty"], df["transfer_rate"]),
        "pearson_clean_acc_stealth_avg": pearson(df["clean_acc"], df["stealth_avg"]),
        "spearman_clean_acc_stealth_avg": spearman(df["clean_acc"], df["stealth_avg"]),
        "pearson_difficulty_stealth_avg": pearson(df["difficulty"], df["stealth_avg"]),
        "spearman_difficulty_stealth_avg": spearman(df["difficulty"], df["stealth_avg"]),
        "pearson_transfer_rate_stealth_avg": pearson(df["transfer_rate"], df["stealth_avg"]),
        "spearman_transfer_rate_stealth_avg": spearman(df["transfer_rate"], df["stealth_avg"]),
    }


def build_correlations(df: pd.DataFrame, thresholds: Sequence[Optional[float]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    base = df[df["valid_transfer_rate"] & df["complete_defense_results"] & df["stealth_avg"].notna()].copy()
    for threshold in thresholds:
        if threshold is None:
            sub = base.copy()
            label = "all"
        else:
            sub = base[base["source_asr"] >= threshold].copy()
            label = f"source_asr>={threshold:g}"
        if sub.empty:
            continue
        records.append(corr_record(sub, "all", "all", label))
        for noise_type, group in sub.groupby("input_noise_type"):
            records.append(corr_record(group, "input_noise_type", str(noise_type), label))
        for attack, group in sub.groupby("attack_type"):
            records.append(corr_record(group, "attack_type", str(attack), label))
        try:
            acc_bins = pd.qcut(sub["clean_acc"], 3, labels=["low_acc", "mid_acc", "high_acc"])
            sub = sub.assign(acc_bin_tmp=acc_bins)
            for acc_bin, group in sub.groupby("acc_bin_tmp", observed=False):
                records.append(corr_record(group, "acc_bin", str(acc_bin), label))
        except Exception:
            pass
    return pd.DataFrame(records)


def build_summary_by_noise(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["valid_transfer_rate"] & df["complete_defense_results"]].copy()
    agg = sub.groupby(["input_noise_type", "input_noise_level"], dropna=False).agg(
        n_rows=("folder_name", "count"),
        clean_acc_mean=("clean_acc", "mean"),
        clean_acc_std=("clean_acc", "std"),
        clean_acc_median=("clean_acc", "median"),
        difficulty_mean=("difficulty", "mean"),
        source_asr_mean=("source_asr", "mean"),
        source_asr_median=("source_asr", "median"),
        transfer_asr_mean=("transfer_asr", "mean"),
        transfer_asr_median=("transfer_asr", "median"),
        transfer_rate_mean=("transfer_rate", "mean"),
        transfer_rate_median=("transfer_rate", "median"),
        stealth_avg_mean=("stealth_avg", "mean"),
        stealth_avg_median=("stealth_avg", "median"),
    )
    return agg.reset_index().sort_values(["input_noise_type", "input_noise_level"])


def build_summary_by_attack(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["valid_transfer_rate"] & df["complete_defense_results"]].copy()
    agg = sub.groupby(["attack_type", "input_noise_type", "input_noise_level"], dropna=False).agg(
        n_rows=("folder_name", "count"),
        clean_acc_mean=("clean_acc", "mean"),
        source_asr_mean=("source_asr", "mean"),
        transfer_asr_mean=("transfer_asr", "mean"),
        transfer_rate_mean=("transfer_rate", "mean"),
        stealth_avg_mean=("stealth_avg", "mean"),
    )
    return agg.reset_index().sort_values(["attack_type", "input_noise_type", "input_noise_level"])


def build_acc_bin_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["include_main_analysis"] & df["acc_bin"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for acc_bin, group in sub.groupby("acc_bin"):
        records.append(
            {
                "acc_bin": acc_bin,
                "n_rows": int(len(group)),
                "clean_acc_min": group["clean_acc"].min(),
                "clean_acc_max": group["clean_acc"].max(),
                "clean_acc_mean": group["clean_acc"].mean(),
                "transfer_rate_mean": group["transfer_rate"].mean(),
                "stealth_avg_mean": group["stealth_avg"].mean(),
                "pearson_transfer_stealth": pearson(group["transfer_rate"], group["stealth_avg"]),
                "spearman_transfer_stealth": spearman(group["transfer_rate"], group["stealth_avg"]),
            }
        )
    return pd.DataFrame(records)


def build_paired_delta(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["valid_transfer_rate"] & df["complete_defense_results"]].copy()
    keys = ["attack_type", "poison_rate", "strength_name", "strength_value", "input_noise_type"]
    records: List[Dict[str, Any]] = []
    for key, group in sub.groupby(keys, dropna=False):
        ref = group[np.isclose(group["input_noise_level"], 0.03)]
        if ref.empty:
            continue
        ref_row = ref.iloc[0]
        for _, row in group.iterrows():
            if np.isclose(row["input_noise_level"], 0.03):
                continue
            rec = {k: v for k, v in zip(keys, key)}
            rec.update(
                {
                    "reference_noise_level": 0.03,
                    "input_noise_level": row["input_noise_level"],
                    "delta_clean_acc": row["clean_acc"] - ref_row["clean_acc"],
                    "delta_difficulty": row["difficulty"] - ref_row["difficulty"],
                    "delta_transfer_rate": row["transfer_rate"] - ref_row["transfer_rate"],
                    "delta_stealth_avg": row["stealth_avg"] - ref_row["stealth_avg"],
                    "reference_result_dir": ref_row["result_dir"],
                    "result_dir": row["result_dir"],
                }
            )
            records.append(rec)
    return pd.DataFrame(records)


def build_defense_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    sub = df[df["valid_transfer_rate"] & df["complete_defense_results"]].copy()
    for defense in ["sentinet", "strip", "scaleup", "ibd_psc"]:
        tmp = sub[
            [
                "input_noise_type",
                "input_noise_level",
                f"{defense}_tpr",
                f"{defense}_auc",
                f"stealth_{defense}",
            ]
        ].copy()
        tmp["defense"] = defense
        tmp = tmp.rename(
            columns={
                f"{defense}_tpr": "tpr",
                f"{defense}_auc": "auc",
                f"stealth_{defense}": "stealth",
            }
        )
        records.append(tmp)
    long_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if long_df.empty:
        return long_df
    return (
        long_df.groupby(["defense", "input_noise_type", "input_noise_level"], dropna=False)
        .agg(
            n_rows=("defense", "count"),
            tpr_mean=("tpr", "mean"),
            tpr_median=("tpr", "median"),
            stealth_mean=("stealth", "mean"),
            stealth_median=("stealth", "median"),
            auc_mean=("auc", "mean"),
            auc_median=("auc", "median"),
        )
        .reset_index()
        .sort_values(["defense", "input_noise_type", "input_noise_level"])
    )


def write_missing_report(df: pd.DataFrame, path: Path) -> None:
    total = len(df)
    lines = [
        "# Noise analysis missing/data-quality report",
        "",
        f"total_rows: {total}",
        f"complete_defense_results: {int(df['complete_defense_results'].sum()) if total else 0}",
        f"valid_transfer_rate: {int(df['valid_transfer_rate'].sum()) if total else 0}",
        f"include_main_analysis: {int(df['include_main_analysis'].sum()) if 'include_main_analysis' in df else 0}",
        "",
        "## Missing item counts",
    ]
    counts: Dict[str, int] = {}
    for item_str in df.get("missing_items", pd.Series(dtype=str)).fillna(""):
        for item in [x for x in item_str.split(",") if x]:
            counts[item] = counts.get(item, 0) + 1
    if counts:
        for key in sorted(counts):
            lines.append(f"- {key}: {counts[key]}")
    else:
        lines.append("- none")

    lines.extend(["", "## Rows with missing items"])
    missing_rows = df[df["missing_items"].fillna("") != ""]
    if missing_rows.empty:
        lines.append("none")
    else:
        for _, row in missing_rows.iterrows():
            lines.append(f"- {row['folder_name']}: {row['missing_items']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_regression_report(df: pd.DataFrame, path: Path) -> None:
    sub = df[df["include_main_analysis"]].copy()
    lines = [
        "# Noise ACC transfer stealth regression",
        "",
        "Model:",
        "stealth_avg ~ transfer_rate * difficulty + C(attack_type) + C(poison_rate) + C(input_noise_type)",
        "",
        f"n_rows: {len(sub)}",
    ]
    if len(sub) < 10:
        lines.append("Not enough rows for regression.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    try:
        import statsmodels.formula.api as smf

        model = smf.ols(
            "stealth_avg ~ transfer_rate * difficulty + C(attack_type) + C(poison_rate) + C(input_noise_type)",
            data=sub,
        ).fit()
        lines.extend(["", str(model.summary()), ""])
        coef = model.params.get("transfer_rate:difficulty", float("nan"))
        pval = model.pvalues.get("transfer_rate:difficulty", float("nan"))
        lines.extend(
            [
                "## Auto interpretation",
                f"transfer_rate:difficulty coefficient: {coef}",
                f"transfer_rate:difficulty p-value: {pval}",
            ]
        )
        if pd.notna(coef) and pd.notna(pval):
            if pval < 0.05:
                if coef < 0:
                    lines.append(
                        "The interaction is negative and statistically notable: higher difficulty tends to make the transfer-stealth relation more negative."
                    )
                else:
                    lines.append(
                        "The interaction is positive and statistically notable: higher difficulty tends to weaken or reverse a negative transfer-stealth relation."
                    )
            else:
                lines.append(
                    "The interaction is not statistically notable at 0.05; current noisy results do not strongly prove moderation by ACC/difficulty."
                )
    except Exception as exc:
        lines.extend(
            [
                "",
                f"Regression failed: {exc}",
                "Install/check statsmodels if a full regression table is required.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_acc_vs_level(df: pd.DataFrame, path: Path) -> None:
    sub = df[df["valid_transfer_rate"]].copy()
    attacks = sorted(sub["attack_type"].dropna().unique())
    ncols = 4
    nrows = math.ceil(len(attacks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax, attack in zip(axes.ravel(), attacks):
        g = sub[sub["attack_type"] == attack]
        for noise_type, ng in g.groupby("input_noise_type"):
            means = ng.groupby("input_noise_level")["clean_acc"].mean().sort_index()
            ax.plot(means.index.to_numpy(dtype=float), means.to_numpy(dtype=float), marker="o", label=noise_type)
        ax.set_title(attack)
        ax.set_xlabel("noise level")
        ax.set_ylabel("clean ACC")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(attacks) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def scatter_by_attack(df: pd.DataFrame, x: str, y: str, path: Path, ylabel: str) -> None:
    sub = df[df["include_main_analysis"]].copy()
    attacks = sorted(sub["attack_type"].dropna().unique())
    ncols = 4
    nrows = math.ceil(len(attacks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    colors = {"gaussian": "#1f77b4", "uniform": "#ff7f0e", "salt_pepper": "#2ca02c", "speckle": "#d62728"}
    for ax, attack in zip(axes.ravel(), attacks):
        g = sub[sub["attack_type"] == attack]
        for noise_type, ng in g.groupby("input_noise_type"):
            ax.scatter(ng[x], ng[y], s=20, alpha=0.75, label=noise_type, color=colors.get(noise_type))
        ax.set_title(attack)
        ax.set_xlabel(x)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(attacks) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_transfer_vs_stealth_by_acc_bin(df: pd.DataFrame, path: Path) -> None:
    sub = df[df["include_main_analysis"] & df["acc_bin"].notna()].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {"low_acc": "#d62728", "mid_acc": "#ff7f0e", "high_acc": "#2ca02c"}
    for acc_bin, group in sub.groupby("acc_bin"):
        ax.scatter(group["transfer_rate"], group["stealth_avg"], s=24, alpha=0.7, label=str(acc_bin), color=colors.get(str(acc_bin)))
    ax.set_xlabel("transfer_rate = transfer_asr^2 / source_asr")
    ax.set_ylabel("stealth_avg = mean(1 - TPR)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_defense_breakdown(df: pd.DataFrame, path: Path) -> None:
    sub = df[df["include_main_analysis"]].copy()
    long_parts = []
    for defense in ["sentinet", "strip", "scaleup", "ibd_psc"]:
        tmp = sub[["clean_acc", "input_noise_type", f"stealth_{defense}"]].copy()
        tmp["defense"] = defense
        tmp = tmp.rename(columns={f"stealth_{defense}": "stealth"})
        long_parts.append(tmp)
    long_df = pd.concat(long_parts, ignore_index=True)
    defenses = ["sentinet", "strip", "scaleup", "ibd_psc"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    colors = {"gaussian": "#1f77b4", "uniform": "#ff7f0e"}
    for ax, defense in zip(axes.ravel(), defenses):
        g = long_df[long_df["defense"] == defense]
        for noise_type, ng in g.groupby("input_noise_type"):
            ax.scatter(ng["clean_acc"], ng["stealth"], s=18, alpha=0.65, label=noise_type, color=colors.get(noise_type))
        ax.set_title(defense)
        ax.set_xlabel("clean ACC")
        ax.set_ylabel("1 - TPR")
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_corr_heatmap(corr_df: pd.DataFrame, path: Path) -> None:
    sub = corr_df[(corr_df["group_type"] == "attack_type") & (corr_df["source_asr_filter"] == "source_asr>=0.05")]
    if sub.empty:
        return
    data = sub.set_index("group_name")[["spearman_transfer_rate_stealth_avg"]].sort_index()
    fig, ax = plt.subplots(figsize=(5, max(4, 0.45 * len(data))))
    im = ax.imshow(data.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_xticks([0])
    ax.set_xticklabels(["Spearman transfer vs stealth"])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_paired_delta(delta_df: pd.DataFrame, path: Path) -> None:
    if delta_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, y, title in [
        (axes[0], "delta_transfer_rate", "Delta transfer_rate vs delta clean ACC"),
        (axes[1], "delta_stealth_avg", "Delta stealth_avg vs delta clean ACC"),
    ]:
        for attack, group in delta_df.groupby("attack_type"):
            ax.scatter(group["delta_clean_acc"], group[y], s=22, alpha=0.7, label=attack)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("delta clean ACC")
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(path: Path) -> None:
    text = """# CIFAR-10 噪声难度实验：ACC、迁移性与隐蔽性分析说明

本目录由 `analyze_noise_acc_transfer_stealth.py` 生成，用于回答：

> CIFAR-10 训练图加噪声导致 clean ACC 改变后，分类任务难度是否会调节迁移性与隐蔽性的关系？

核心定义：

- `difficulty = 1 - clean_acc`
- `transfer_rate = transfer_asr^2 / source_asr`
- `stealth_avg = mean(1 - TPR)`，TPR 来自 SentiNet、STRIP、ScaleUp、IBD-PSC 四个原始域检测方法。

## 文件作用

- `noise_acc_transfer_stealth_rows.csv`：逐实验配置明细表，是所有后续分析的来源。
- `noise_missing_report.txt`：数据完整性报告，列出缺失 source test、STL10 transfer 或 defense JSON 的目录。
- `noise_acc_transfer_stealth_summary_by_noise.csv`：按噪声类型和强度汇总，用于确认噪声是否制造 clean ACC 梯度。
- `noise_acc_transfer_stealth_summary_by_attack.csv`：按攻击、噪声类型、噪声强度汇总，用于判断趋势是否 attack-dependent。
- `noise_acc_transfer_stealth_correlations.csv`：相关性表，包含 ACC、difficulty、transfer_rate、stealth_avg 之间的 Pearson/Spearman 相关性。
- `noise_acc_bin_transfer_stealth.csv`：按 clean ACC 三分位分层后，统计每层 transfer_rate 与 stealth_avg 的关系。
- `noise_acc_transfer_stealth_regression.txt`：交互项回归结果，重点看 `transfer_rate:difficulty`。**公式、虚拟变量与 OLS 计算见上级目录 [INTERACTION_REGRESSION_EXPLAINED_CN.md](../INTERACTION_REGRESSION_EXPLAINED_CN.md)。**
- `noise_paired_delta_by_level.csv`：同一攻击配置内，以 noise level 0.030 为 reference，比较更高噪声强度造成的变化。
- `noise_defense_breakdown_summary.csv`：拆开四个 defense，看平均隐蔽性主要由哪些检测方法驱动。
- `figures/`：核心可视化图表。

## 解读顺序

1. 先看 `noise_missing_report.txt`，确认数据完整性。
2. 看 `summary_by_noise.csv` 和 `figures/noise_acc_vs_level.png`，确认噪声是否降低 clean ACC。
3. 看 `figures/noise_acc_vs_transfer_rate.png`，判断 ACC 是否影响迁移性。
4. 看 `figures/noise_acc_vs_stealth.png`，判断 ACC 是否影响隐蔽性。
5. 看 `noise_acc_bin_transfer_stealth.csv` 和 `figures/noise_transfer_vs_stealth_by_acc_bin.png`，判断不同 ACC 难度层下 transfer-stealth 关系是否变化。
6. 看 `noise_acc_transfer_stealth_regression.txt`，用交互项判断 difficulty 是否调节 transfer_rate 与 stealth_avg 的关系。
7. 看 `noise_defense_breakdown_summary.csv` 和 `figures/noise_defense_stealth_breakdown.png`，确认平均隐蔽性是否由单个 defense 主导。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CIFAR-10 noise ACC, transferability, and stealth.")
    parser.add_argument("--root", type=Path, default=Path("poisoned_train_set/cifar10"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis-transfer-asr2/noise_analysis"))
    parser.add_argument("--source-asr-threshold", type=float, default=0.05)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"result root not found: {args.root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = discover_rows(args.root, args.max_rows)
    if df.empty:
        raise RuntimeError(f"no noisy result folders found under {args.root}")
    df = add_analysis_flags(df, args.source_asr_threshold)

    rows_path = args.output_dir / "noise_acc_transfer_stealth_rows.csv"
    df.to_csv(rows_path, index=False)

    missing_path = args.output_dir / "noise_missing_report.txt"
    write_missing_report(df, missing_path)

    summary_by_noise = build_summary_by_noise(df)
    summary_by_noise.to_csv(args.output_dir / "noise_acc_transfer_stealth_summary_by_noise.csv", index=False)

    summary_by_attack = build_summary_by_attack(df)
    summary_by_attack.to_csv(args.output_dir / "noise_acc_transfer_stealth_summary_by_attack.csv", index=False)

    corr_df = build_correlations(df, thresholds=[None, 0.01, 0.05, 0.10])
    corr_df.to_csv(args.output_dir / "noise_acc_transfer_stealth_correlations.csv", index=False)

    acc_bin_df = build_acc_bin_table(df)
    acc_bin_df.to_csv(args.output_dir / "noise_acc_bin_transfer_stealth.csv", index=False)

    delta_df = build_paired_delta(df)
    delta_df.to_csv(args.output_dir / "noise_paired_delta_by_level.csv", index=False)

    defense_df = build_defense_breakdown(df)
    defense_df.to_csv(args.output_dir / "noise_defense_breakdown_summary.csv", index=False)

    write_regression_report(df, args.output_dir / "noise_acc_transfer_stealth_regression.txt")
    write_readme(args.output_dir / "README_CN.md")

    plot_acc_vs_level(df, figures_dir / "noise_acc_vs_level.png")
    scatter_by_attack(df, "clean_acc", "transfer_rate", figures_dir / "noise_acc_vs_transfer_rate.png", "transfer_rate")
    scatter_by_attack(df, "clean_acc", "stealth_avg", figures_dir / "noise_acc_vs_stealth.png", "stealth_avg")
    plot_transfer_vs_stealth_by_acc_bin(df, figures_dir / "noise_transfer_vs_stealth_by_acc_bin.png")
    plot_defense_breakdown(df, figures_dir / "noise_defense_stealth_breakdown.png")
    plot_corr_heatmap(corr_df, figures_dir / "noise_corr_heatmap_by_attack.png")
    plot_paired_delta(delta_df, figures_dir / "noise_paired_delta.png")

    print(f"rows: {len(df)} -> {rows_path}")
    print(f"include_main_analysis: {int(df['include_main_analysis'].sum())}")
    print(f"output dir: {args.output_dir}")
    if not summary_by_noise.empty:
        print("\nSummary by noise:")
        print(summary_by_noise.to_string(index=False))
    if not acc_bin_df.empty:
        print("\nACC-bin transfer/stealth:")
        print(acc_bin_df.to_string(index=False))


if __name__ == "__main__":
    main()
