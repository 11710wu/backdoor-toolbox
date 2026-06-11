#!/usr/bin/env python3
"""Analyze ACC/difficulty, transferability, and stealth across architectures.

This script reads existing result folders only. It does not train, test, or
modify experiment result directories.

Core definitions:
  difficulty    = 1 - clean_acc
  transfer_rate = transfer_asr^2 / source_asr
  stealth_avg   = mean(1 - TPR) over SentiNet, STRIP, ScaleUp, IBD-PSC
"""

from __future__ import annotations

import argparse
import html
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
    "basic",
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

PRIMARY_DATASETS = ["cifar10", "tiny_imagenet"]
ARCH_EXCLUDED_ATTACKS = {"SIG", "upgd"}
NOISE_DEFAULT = Path("/workspace/backdoor-toolbox-noise/analysis-transfer-asr2/noise_analysis/noise_acc_transfer_stealth_rows.csv")


@dataclass
class ParsedFolder:
    dataset: str
    attack_type: str
    attack_family: str
    poison_rate: float
    strength_name: Optional[str]
    strength_value: Optional[float]
    cover_rate: Optional[float]
    arch: str
    arch_base: str
    dataset_from_arch: str


def normalize_rate(value: Any) -> float:
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


def _attack_family(attack_type: str) -> str:
    # Historical baseline folders often use "basic" for BadNet-style patch.
    return "badnet" if attack_type == "basic" else attack_type


def _arch_base(arch: str, dataset: str) -> str:
    suffix = f"_{dataset}"
    return arch[: -len(suffix)] if arch.endswith(suffix) else arch


def parse_folder_name(dataset: str, folder_name: str) -> Optional[ParsedFolder]:
    attack_type = None
    for attack in sorted(ATTACKS, key=len, reverse=True):
        if folder_name.startswith(f"{attack}_"):
            attack_type = attack
            break
    if attack_type is None:
        return None

    rest = folder_name[len(attack_type) + 1 :]
    rate_token = rest.split("_", 1)[0]
    poison_rate = safe_float(rate_token)
    if poison_rate is None:
        return None

    strength_name = None
    strength_value = None
    for key, pattern in [
        ("alpha", r"_alpha=([0-9.]+)"),
        ("delta", r"_delta=([0-9.]+)"),
        ("s", r"_s=([0-9.]+)"),
        ("mask_rate", r"_mask=([0-9.]+)"),
        ("eps", r"_eps=([0-9.]+)"),
    ]:
        match = re.search(pattern, folder_name)
        if match:
            strength_name = key
            strength_value = float(match.group(1))
            break

    cover_rate = None
    if match := re.search(r"_cover=([0-9.]+)", folder_name):
        cover_rate = float(match.group(1))

    arch_match = re.search(r"_arch=([^/]+)$", folder_name)
    if not arch_match:
        return None
    arch = arch_match.group(1)

    dataset_match = re.search(r"_arch=[\w]+_(cifar10|tiny_imagenet|mnistm)$", folder_name)
    dataset_from_arch = dataset_match.group(1) if dataset_match else dataset

    return ParsedFolder(
        dataset=dataset,
        attack_type=attack_type,
        attack_family=_attack_family(attack_type),
        poison_rate=poison_rate,
        strength_name=strength_name,
        strength_value=strength_value,
        cover_rate=cover_rate,
        arch=arch,
        arch_base=_arch_base(arch, dataset_from_arch),
        dataset_from_arch=dataset_from_arch,
    )


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def choose_source_result_file(folder: Path) -> Optional[Path]:
    base = folder / "test_results_seed=2333.json"
    if base.exists():
        return base
    files = sorted(folder.glob("test_results_seed=2333*.json"))
    return files[0] if files else None


def choose_transfer_file(folder: Path, dataset: str) -> Optional[Path]:
    if dataset == "cifar10":
        patterns = ["test_stl10_results.txt", "test_stl10_results*.txt"]
    elif dataset == "tiny_imagenet":
        patterns = ["test_tiny_target_domain_results.txt", "test_tiny_target_domain_results*.txt"]
    elif dataset == "mnistm":
        patterns = [
            "test_mnistm_results.txt",
            "test_mnistm_results*.txt",
            "test_mnist_cross_results.txt",
            "test_mnist_cross_results*.txt",
        ]
    else:
        patterns = []

    for pattern in patterns:
        if "*" not in pattern:
            candidate = folder / pattern
            if candidate.exists():
                return candidate
        else:
            files = sorted(folder.glob(pattern))
            if files:
                return files[0]
    return None


def parse_transfer_text(path: Path) -> Tuple[float, float]:
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
            if match := re.search(r"[:：]\s*([0-9.]+)", line):
                acc = normalize_rate(match.group(1))
        if "攻击成功率" in line or ("ASR" in line and ":" in line):
            if match := re.search(r"[:：]\s*([0-9.]+)", line):
                asr = normalize_rate(match.group(1))
    return acc, asr


def parse_defense(path: Path) -> Dict[str, float]:
    data = load_json(path)
    if not data:
        return {"tpr": float("nan"), "auc": float("nan"), "fpr": float("nan"), "threshold": float("nan")}
    threshold = data.get("threshold", data.get("threshold_low", float("nan")))
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = float("nan")
    return {
        "tpr": normalize_rate(data.get("tpr")),
        "auc": normalize_rate(data.get("auc")),
        "fpr": normalize_rate(data.get("fpr")),
        "threshold": threshold,
    }


def parse_result_folder(folder: Path, result_root_name: str, result_group: str, dataset: str) -> Optional[Dict[str, Any]]:
    parsed = parse_folder_name(dataset, folder.name)
    if parsed is None:
        return None

    row: Dict[str, Any] = {
        "result_root": result_root_name,
        "result_group": result_group,
        "dataset": dataset,
        "arch": parsed.arch,
        "arch_base": parsed.arch_base,
        "dataset_arch": f"{dataset}:{parsed.arch_base}",
        "folder_name": folder.name,
        "result_dir": str(folder),
        "attack_type": parsed.attack_type,
        "attack_family": parsed.attack_family,
        "poison_rate": parsed.poison_rate,
        "strength_name": parsed.strength_name,
        "strength_value": parsed.strength_value,
        "cover_rate": parsed.cover_rate,
    }
    missing: List[str] = []

    source_file = choose_source_result_file(folder)
    row["source_result_file"] = str(source_file) if source_file else ""
    source_data = load_json(source_file) if source_file else None
    if source_file is None:
        missing.append("source_test")
    elif not source_data:
        missing.append("source_test_parse")

    row["clean_acc"] = normalize_rate(source_data.get("clean_acc")) if source_data else float("nan")
    row["source_asr"] = normalize_rate(source_data.get("asr")) if source_data else float("nan")
    row["difficulty"] = 1.0 - row["clean_acc"] if pd.notna(row["clean_acc"]) else float("nan")

    transfer_file = choose_transfer_file(folder, dataset)
    row["transfer_result_file"] = str(transfer_file) if transfer_file else ""
    if transfer_file is None:
        missing.append("transfer_test")
        row["transfer_acc"] = float("nan")
        row["transfer_asr"] = float("nan")
    else:
        transfer_acc, transfer_asr = parse_transfer_text(transfer_file)
        row["transfer_acc"] = transfer_acc
        row["transfer_asr"] = transfer_asr
        if pd.isna(transfer_asr):
            missing.append("transfer_test_parse")

    valid_transfer_rate = pd.notna(row["source_asr"]) and row["source_asr"] > 0 and pd.notna(row["transfer_asr"])
    row["valid_transfer_rate"] = bool(valid_transfer_rate)
    row["transfer_rate"] = (row["transfer_asr"] ** 2 / row["source_asr"]) if valid_transfer_rate else float("nan")

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


def discover_rows(roots: Sequence[Tuple[Path, str]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for root, group_name in roots:
        if not root.exists():
            continue
        for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            dataset = dataset_dir.name
            for folder in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
                row = parse_result_folder(folder, root.name, group_name, dataset)
                if row is not None:
                    rows.append(row)
    return pd.DataFrame(rows)


def add_analysis_flags(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["valid_source_asr_for_main"] = out["source_asr"] >= threshold
    out["include_main_analysis"] = (
        out["valid_transfer_rate"]
        & out["complete_defense_results"]
        & out["valid_source_asr_for_main"]
        & out["stealth_avg"].notna()
        & out["clean_acc"].notna()
    )
    out["main_focus"] = (
        ((out["dataset"] == "cifar10") & (out["arch_base"].isin(["ResNet18", "mobilenetv2", "vgg19_bn", "SmallCNN"])))
        | (
            (out["dataset"] == "tiny_imagenet")
            & (out["arch_base"].isin(["ResNet18", "mobilenetv2", "vgg19_bn", "ResNet34"]))
        )
    )
    out["primary_main_analysis"] = out["include_main_analysis"] & out["main_focus"]

    all_bins = pd.Series(index=out.index, dtype="object")
    primary = out[out["primary_main_analysis"]]
    if len(primary) >= 3:
        try:
            bins = pd.qcut(primary["clean_acc"], 3, labels=["low_acc", "mid_acc", "high_acc"], duplicates="drop")
            all_bins.loc[primary.index] = bins.astype("object")
        except Exception:
            pass
    out["acc_bin"] = all_bins

    dataset_bins = pd.Series(index=out.index, dtype="object")
    for dataset, group in out[out["primary_main_analysis"]].groupby("dataset"):
        if len(group) < 3:
            continue
        try:
            bins = pd.qcut(group["clean_acc"], 3, labels=["low_acc", "mid_acc", "high_acc"], duplicates="drop")
            dataset_bins.loc[group.index] = bins.astype("object")
        except Exception:
            pass
    out["dataset_acc_bin"] = dataset_bins
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
        "clean_acc_mean": df["clean_acc"].mean(),
        "transfer_rate_mean": df["transfer_rate"].mean(),
        "stealth_avg_mean": df["stealth_avg"].mean(),
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


def filtered_base(df: pd.DataFrame, threshold: Optional[float], primary_only: bool = True) -> pd.DataFrame:
    mask = df["valid_transfer_rate"] & df["complete_defense_results"] & df["stealth_avg"].notna() & df["clean_acc"].notna()
    if primary_only:
        mask &= df["main_focus"]
    if threshold is not None:
        mask &= df["source_asr"] >= threshold
    return df[mask].copy()


def exclude_arch_attacks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "attack_family" not in df.columns:
        return df.copy()
    return df[~df["attack_family"].isin(ARCH_EXCLUDED_ATTACKS)].copy()


def build_correlations(df: pd.DataFrame, thresholds: Sequence[Optional[float]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for threshold in thresholds:
        sub = exclude_arch_attacks(filtered_base(df, threshold))
        label = "all" if threshold is None else f"source_asr>={threshold:g}"
        if sub.empty:
            continue
        records.append(corr_record(sub, "all", "all", label))
        for col in ["dataset", "arch_base", "dataset_arch", "attack_family", "result_group"]:
            for name, group in sub.groupby(col, dropna=False):
                if len(group) >= 2:
                    records.append(corr_record(group, col, str(name), label))
        try:
            bins = pd.qcut(sub["clean_acc"], 3, labels=["low_acc", "mid_acc", "high_acc"], duplicates="drop")
            for bin_name, group in sub.assign(acc_bin_tmp=bins).groupby("acc_bin_tmp", observed=False):
                if len(group) >= 2:
                    records.append(corr_record(group, "acc_bin", str(bin_name), label))
        except Exception:
            pass
    return pd.DataFrame(records)


def build_summary_by_dataset_arch(df: pd.DataFrame) -> pd.DataFrame:
    sub = exclude_arch_attacks(filtered_base(df, 0.05, primary_only=False))
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["dataset", "arch_base", "dataset_arch", "result_group"], dropna=False)
        .agg(
            n_rows=("folder_name", "count"),
            clean_acc_mean=("clean_acc", "mean"),
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
        .reset_index()
        .sort_values(["dataset", "arch_base", "result_group"])
    )


def build_summary_by_attack(df: pd.DataFrame) -> pd.DataFrame:
    sub = exclude_arch_attacks(filtered_base(df, 0.05))
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["dataset", "arch_base", "attack_family"], dropna=False)
        .agg(
            n_rows=("folder_name", "count"),
            clean_acc_mean=("clean_acc", "mean"),
            source_asr_mean=("source_asr", "mean"),
            transfer_asr_mean=("transfer_asr", "mean"),
            transfer_rate_mean=("transfer_rate", "mean"),
            stealth_avg_mean=("stealth_avg", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "attack_family", "arch_base"])
    )


def build_acc_bin_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = exclude_arch_attacks(df[df["primary_main_analysis"] & df["acc_bin"].notna()].copy())
    records: List[Dict[str, Any]] = []
    for bin_col, group_type in [("acc_bin", "global_acc_bin"), ("dataset_acc_bin", "dataset_acc_bin")]:
        sub2 = exclude_arch_attacks(df[df["primary_main_analysis"] & df[bin_col].notna()].copy())
        groups: Iterable[Tuple[Any, pd.DataFrame]]
        if group_type == "dataset_acc_bin":
            groups = sub2.groupby(["dataset", bin_col], dropna=False)
        else:
            groups = sub2.groupby(bin_col, dropna=False)
        for key, group in groups:
            dataset = key[0] if isinstance(key, tuple) else "all"
            acc_bin = key[1] if isinstance(key, tuple) else key
            records.append(
                {
                    "group_type": group_type,
                    "dataset": dataset,
                    "acc_bin": acc_bin,
                    "n_rows": int(len(group)),
                    "clean_acc_min": group["clean_acc"].min(),
                    "clean_acc_max": group["clean_acc"].max(),
                    "clean_acc_mean": group["clean_acc"].mean(),
                    "transfer_rate_mean": group["transfer_rate"].mean(),
                    "transfer_rate_median": group["transfer_rate"].median(),
                    "stealth_avg_mean": group["stealth_avg"].mean(),
                    "stealth_avg_median": group["stealth_avg"].median(),
                    "pearson_transfer_stealth": pearson(group["transfer_rate"], group["stealth_avg"]),
                    "spearman_transfer_stealth": spearman(group["transfer_rate"], group["stealth_avg"]),
                }
            )
    return pd.DataFrame(records)


def _add_zscore_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        std = out[col].std(ddof=0)
        if pd.isna(std) or std == 0:
            out[f"{col}_z"] = float("nan")
        else:
            out[f"{col}_z"] = (out[col] - out[col].mean()) / std
    return out


def _fit_acc_moderation_models(data: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "raw_interaction_coef": float("nan"),
        "raw_interaction_pvalue": float("nan"),
        "std_transfer_coef": float("nan"),
        "std_transfer_pvalue": float("nan"),
        "std_difficulty_coef": float("nan"),
        "std_difficulty_pvalue": float("nan"),
        "std_interaction_coef": float("nan"),
        "std_interaction_pvalue": float("nan"),
        "std_r_squared": float("nan"),
        "model_error": "",
    }
    sub = data.dropna(subset=["clean_acc", "difficulty", "transfer_rate", "stealth_avg"]).copy()
    if len(sub) < 20:
        out["model_error"] = "not_enough_rows"
        return out
    try:
        import statsmodels.formula.api as smf

        formula_raw = "stealth_avg ~ transfer_rate * difficulty + C(dataset) + C(arch_base) + C(attack_family) + C(poison_rate)"
        raw_model = smf.ols(formula_raw, data=sub).fit()
        out["raw_interaction_coef"] = raw_model.params.get("transfer_rate:difficulty", float("nan"))
        out["raw_interaction_pvalue"] = raw_model.pvalues.get("transfer_rate:difficulty", float("nan"))

        zsub = _add_zscore_columns(sub, ["transfer_rate", "difficulty", "stealth_avg"])
        formula_z = "stealth_avg_z ~ transfer_rate_z * difficulty_z + C(dataset) + C(arch_base) + C(attack_family) + C(poison_rate)"
        z_model = smf.ols(formula_z, data=zsub).fit(cov_type="HC3")
        out["std_transfer_coef"] = z_model.params.get("transfer_rate_z", float("nan"))
        out["std_transfer_pvalue"] = z_model.pvalues.get("transfer_rate_z", float("nan"))
        out["std_difficulty_coef"] = z_model.params.get("difficulty_z", float("nan"))
        out["std_difficulty_pvalue"] = z_model.pvalues.get("difficulty_z", float("nan"))
        out["std_interaction_coef"] = z_model.params.get("transfer_rate_z:difficulty_z", float("nan"))
        out["std_interaction_pvalue"] = z_model.pvalues.get("transfer_rate_z:difficulty_z", float("nan"))
        out["std_r_squared"] = z_model.rsquared
    except Exception as exc:
        out["model_error"] = str(exc)
    return out


def build_acc_moderation_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = exclude_arch_attacks(df)
    cohorts = [
        ("primary_main_source_asr>=0.05", df["primary_main_analysis"]),
        ("primary_main_source_asr>=0.10", df["primary_main_analysis"] & (df["source_asr"] >= 0.10)),
        ("include_main_source_asr>=0.05", df["include_main_analysis"]),
        ("include_main_source_asr>=0.10", df["include_main_analysis"] & (df["source_asr"] >= 0.10)),
    ]
    records: List[Dict[str, Any]] = []
    for cohort, mask in cohorts:
        sub = df[mask].dropna(subset=["clean_acc", "difficulty", "transfer_rate", "stealth_avg"]).copy()
        if sub.empty:
            continue
        rec: Dict[str, Any] = {
            "cohort": cohort,
            "n_rows": int(len(sub)),
            "clean_acc_mean": sub["clean_acc"].mean(),
            "difficulty_mean": sub["difficulty"].mean(),
            "transfer_rate_mean": sub["transfer_rate"].mean(),
            "transfer_rate_median": sub["transfer_rate"].median(),
            "stealth_avg_mean": sub["stealth_avg"].mean(),
            "stealth_avg_median": sub["stealth_avg"].median(),
            "pearson_transfer_stealth": pearson(sub["transfer_rate"], sub["stealth_avg"]),
            "spearman_transfer_stealth": spearman(sub["transfer_rate"], sub["stealth_avg"]),
            "pearson_clean_acc_transfer": pearson(sub["clean_acc"], sub["transfer_rate"]),
            "spearman_clean_acc_transfer": spearman(sub["clean_acc"], sub["transfer_rate"]),
            "pearson_clean_acc_stealth": pearson(sub["clean_acc"], sub["stealth_avg"]),
            "spearman_clean_acc_stealth": spearman(sub["clean_acc"], sub["stealth_avg"]),
        }
        rec.update(_fit_acc_moderation_models(sub))
        records.append(rec)
    return pd.DataFrame(records)


def build_acc_bin_slope_summary(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for bin_col, group_type in [("acc_bin", "global_acc_bin"), ("dataset_acc_bin", "dataset_acc_bin")]:
        sub = exclude_arch_attacks(df[df["primary_main_analysis"] & df[bin_col].notna()].copy())
        if sub.empty:
            continue
        if group_type == "dataset_acc_bin":
            groups = sub.groupby(["dataset", bin_col], dropna=False)
        else:
            groups = sub.groupby(bin_col, dropna=False)
        for key, group in groups:
            dataset = key[0] if isinstance(key, tuple) else "all"
            acc_bin = key[1] if isinstance(key, tuple) else key
            records.append(
                {
                    "group_type": group_type,
                    "dataset": dataset,
                    "acc_bin": acc_bin,
                    "n_rows": int(len(group)),
                    "clean_acc_mean": group["clean_acc"].mean(),
                    "difficulty_mean": group["difficulty"].mean(),
                    "transfer_rate_mean": group["transfer_rate"].mean(),
                    "transfer_rate_median": group["transfer_rate"].median(),
                    "stealth_avg_mean": group["stealth_avg"].mean(),
                    "stealth_avg_median": group["stealth_avg"].median(),
                    "pearson_transfer_stealth": pearson(group["transfer_rate"], group["stealth_avg"]),
                    "spearman_transfer_stealth": spearman(group["transfer_rate"], group["stealth_avg"]),
                }
            )
    return pd.DataFrame(records)


def match_key_columns() -> List[str]:
    return ["dataset", "attack_family", "poison_rate", "strength_name", "strength_value", "cover_rate"]


def build_pairwise_delta(df: pd.DataFrame) -> pd.DataFrame:
    sub = exclude_arch_attacks(filtered_base(df, 0.05))
    keys = match_key_columns()
    candidates = [
        ("cifar10", "SmallCNN", "ResNet18"),
        ("cifar10", "SmallCNN", "mobilenetv2"),
        ("cifar10", "SmallCNN", "vgg19_bn"),
        ("tiny_imagenet", "ResNet34", "ResNet18"),
        ("tiny_imagenet", "ResNet34", "mobilenetv2"),
        ("tiny_imagenet", "ResNet34", "vgg19_bn"),
    ]
    records: List[Dict[str, Any]] = []
    for dataset, new_arch, base_arch in candidates:
        new_df = sub[(sub["dataset"] == dataset) & (sub["arch_base"] == new_arch)].copy()
        base_df = sub[(sub["dataset"] == dataset) & (sub["arch_base"] == base_arch)].copy()
        if new_df.empty or base_df.empty:
            continue
        merged = new_df.merge(base_df, on=keys, suffixes=("_new", "_base"))
        for _, row in merged.iterrows():
            rec = {key: row[key] for key in keys}
            rec.update(
                {
                    "new_arch": new_arch,
                    "base_arch": base_arch,
                    "new_result_dir": row["result_dir_new"],
                    "base_result_dir": row["result_dir_base"],
                }
            )
            for metric in ["clean_acc", "difficulty", "source_asr", "transfer_asr", "transfer_rate", "stealth_avg"]:
                rec[f"{metric}_new"] = row[f"{metric}_new"]
                rec[f"{metric}_base"] = row[f"{metric}_base"]
                rec[f"delta_{metric}"] = row[f"{metric}_new"] - row[f"{metric}_base"]
            records.append(rec)
    return pd.DataFrame(records)


def build_defense_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    sub = exclude_arch_attacks(filtered_base(df, 0.05))
    records: List[pd.DataFrame] = []
    for defense in ["sentinet", "strip", "scaleup", "ibd_psc"]:
        cols = ["dataset", "arch_base", f"{defense}_tpr", f"{defense}_auc", f"stealth_{defense}"]
        tmp = sub[cols].copy()
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
        long_df.groupby(["dataset", "arch_base", "defense"], dropna=False)
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
        .sort_values(["dataset", "arch_base", "defense"])
    )


def write_missing_report(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Architecture ACC analysis missing/data-quality report",
        "",
        f"total_rows: {len(df)}",
        f"complete_defense_results: {int(df['complete_defense_results'].sum()) if not df.empty else 0}",
        f"valid_transfer_rate: {int(df['valid_transfer_rate'].sum()) if not df.empty else 0}",
        f"include_main_analysis: {int(df['include_main_analysis'].sum()) if 'include_main_analysis' in df else 0}",
        f"primary_main_analysis: {int(df['primary_main_analysis'].sum()) if 'primary_main_analysis' in df else 0}",
        "",
        "## Rows by root/dataset/arch",
    ]
    if not df.empty:
        counts = (
            df.groupby(["result_root", "dataset", "arch_base"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["result_root", "dataset", "arch_base"])
        )
        for _, row in counts.iterrows():
            lines.append(f"- {row['result_root']} / {row['dataset']} / {row['arch_base']}: {int(row['n'])}")

    lines.extend(["", "## Missing item counts"])
    missing_counts: Dict[str, int] = {}
    for item_str in df.get("missing_items", pd.Series(dtype=str)).fillna(""):
        for item in [x for x in item_str.split(",") if x]:
            missing_counts[item] = missing_counts.get(item, 0) + 1
    if missing_counts:
        for item in sorted(missing_counts):
            lines.append(f"- {item}: {missing_counts[item]}")
    else:
        lines.append("- none")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_regression_report(df: pd.DataFrame, path: Path) -> None:
    sub = exclude_arch_attacks(df[df["primary_main_analysis"]].copy())
    lines = [
        "# Architecture ACC transfer stealth regression",
        "",
        "Main architecture-analysis attacks excluded: SIG, upgd.",
        "",
        "Primary model:",
        "stealth_avg ~ transfer_rate * difficulty + C(dataset) + C(arch_base) + C(attack_family) + C(poison_rate)",
        "",
        f"n_rows: {len(sub)}",
    ]
    if len(sub) < 20:
        lines.append("Not enough rows for regression.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    try:
        import statsmodels.formula.api as smf

        model = smf.ols(
            "stealth_avg ~ transfer_rate * difficulty + C(dataset) + C(arch_base) + C(attack_family) + C(poison_rate)",
            data=sub,
        ).fit()
        lines.extend(["", str(model.summary()), ""])
        coef = model.params.get("transfer_rate:difficulty", float("nan"))
        pval = model.pvalues.get("transfer_rate:difficulty", float("nan"))
        lines.extend(
            [
                "## Interaction interpretation",
                f"transfer_rate:difficulty coefficient: {coef}",
                f"transfer_rate:difficulty p-value: {pval}",
            ]
        )

        lines.extend(["", "## Per-dataset slope models"])
        for dataset, group in sub.groupby("dataset"):
            if len(group) < 20 or group["arch_base"].nunique() < 2:
                lines.append(f"### {dataset}: skipped, insufficient rows or arch diversity")
                continue
            dmodel = smf.ols(
                "stealth_avg ~ transfer_rate * clean_acc + C(arch_base) + C(attack_family) + C(poison_rate)",
                data=group,
            ).fit()
            lines.extend([f"### {dataset}", str(dmodel.summary()), ""])

        if sub["dataset_arch"].nunique() >= 2:
            lines.extend(["", "## Dataset-arch slope model"])
            amodel = smf.ols(
                "stealth_avg ~ transfer_rate * C(dataset_arch) + clean_acc + C(attack_family) + C(poison_rate)",
                data=sub,
            ).fit()
            lines.extend([str(amodel.summary()), ""])
    except Exception as exc:
        lines.extend(["", f"Regression failed: {exc}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_scatter(df: pd.DataFrame, x: str, y: str, color_col: str, path: Path, title: str) -> None:
    sub = exclude_arch_attacks(df[df["primary_main_analysis"]].dropna(subset=[x, y]).copy())
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, group in sub.groupby(color_col):
        ax.scatter(group[x], group[y], alpha=0.65, s=28, label=str(name))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_transfer_stealth_by_acc_bin(df: pd.DataFrame, path: Path) -> None:
    sub = exclude_arch_attacks(df[df["primary_main_analysis"] & df["acc_bin"].notna()].dropna(subset=["transfer_rate", "stealth_avg"]).copy())
    if sub.empty:
        return
    bins = ["low_acc", "mid_acc", "high_acc"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, bin_name in zip(axes, bins):
        group = sub[sub["acc_bin"] == bin_name]
        for dataset_arch, dg in group.groupby("dataset_arch"):
            ax.scatter(dg["transfer_rate"], dg["stealth_avg"], alpha=0.7, s=26, label=dataset_arch)
        ax.set_title(f"{bin_name} (n={len(group)})")
        ax.set_xlabel("transfer_rate")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("stealth_avg")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pairwise_delta(pairwise: pd.DataFrame, path: Path) -> None:
    if pairwise.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, group in pairwise.groupby(["dataset", "new_arch", "base_arch"]):
        label = "/".join(map(str, name))
        axes[0].scatter(group["delta_clean_acc"], group["delta_transfer_rate"], alpha=0.65, s=26, label=label)
        axes[1].scatter(group["delta_clean_acc"], group["delta_stealth_avg"], alpha=0.65, s=26, label=label)
    axes[0].set_xlabel("delta_clean_acc")
    axes[0].set_ylabel("delta_transfer_rate")
    axes[1].set_xlabel("delta_clean_acc")
    axes[1].set_ylabel("delta_stealth_avg")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_defense_breakdown(breakdown: pd.DataFrame, path: Path) -> None:
    if breakdown.empty:
        return
    pivot = breakdown.pivot_table(index=["dataset", "arch_base"], columns="defense", values="stealth_mean", aggfunc="mean")
    if pivot.empty:
        return
    ax = pivot.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_ylabel("mean stealth = 1 - TPR")
    ax.set_title("Defense-specific stealth by dataset/arch")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def build_pairwise_delta_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    return (
        pairwise.groupby(["dataset", "new_arch", "base_arch"], dropna=False)
        .agg(
            n_rows=("dataset", "count"),
            delta_clean_acc_mean=("delta_clean_acc", "mean"),
            delta_transfer_rate_mean=("delta_transfer_rate", "mean"),
            delta_stealth_avg_mean=("delta_stealth_avg", "mean"),
            delta_source_asr_mean=("delta_source_asr", "mean"),
            delta_transfer_asr_mean=("delta_transfer_asr", "mean"),
            corr_delta_acc_transfer=("delta_clean_acc", lambda s: pearson(s, pairwise.loc[s.index, "delta_transfer_rate"])),
            corr_delta_acc_stealth=("delta_clean_acc", lambda s: pearson(s, pairwise.loc[s.index, "delta_stealth_avg"])),
        )
        .reset_index()
        .sort_values(["dataset", "new_arch", "base_arch"])
    )


def build_arch_relationship_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = exclude_arch_attacks(filtered_base(df, 0.05))
    if sub.empty:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for (dataset, arch), group in sub.groupby(["dataset", "arch_base"], dropna=False):
        records.append(
            {
                "dataset": dataset,
                "arch_base": arch,
                "n_rows": int(len(group)),
                "clean_acc_mean": group["clean_acc"].mean(),
                "transfer_rate_mean": group["transfer_rate"].mean(),
                "stealth_avg_mean": group["stealth_avg"].mean(),
                "pearson_transfer_stealth": pearson(group["transfer_rate"], group["stealth_avg"]),
                "spearman_transfer_stealth": spearman(group["transfer_rate"], group["stealth_avg"]),
                "pearson_acc_transfer": pearson(group["clean_acc"], group["transfer_rate"]),
                "spearman_acc_transfer": spearman(group["clean_acc"], group["transfer_rate"]),
                "pearson_acc_stealth": pearson(group["clean_acc"], group["stealth_avg"]),
                "spearman_acc_stealth": spearman(group["clean_acc"], group["stealth_avg"]),
            }
        )
    return pd.DataFrame(records).sort_values(["dataset", "arch_base"])


def build_pairwise_relationship_shift(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for (dataset, new_arch, base_arch), group in pairwise.groupby(["dataset", "new_arch", "base_arch"], dropna=False):
        new_spearman_ts = spearman(group["transfer_rate_new"], group["stealth_avg_new"])
        base_spearman_ts = spearman(group["transfer_rate_base"], group["stealth_avg_base"])
        new_pearson_ts = pearson(group["transfer_rate_new"], group["stealth_avg_new"])
        base_pearson_ts = pearson(group["transfer_rate_base"], group["stealth_avg_base"])

        new_spearman_acc_transfer = spearman(group["clean_acc_new"], group["transfer_rate_new"])
        base_spearman_acc_transfer = spearman(group["clean_acc_base"], group["transfer_rate_base"])
        new_spearman_acc_stealth = spearman(group["clean_acc_new"], group["stealth_avg_new"])
        base_spearman_acc_stealth = spearman(group["clean_acc_base"], group["stealth_avg_base"])

        new_pearson_acc_transfer = pearson(group["clean_acc_new"], group["transfer_rate_new"])
        base_pearson_acc_transfer = pearson(group["clean_acc_base"], group["transfer_rate_base"])
        new_pearson_acc_stealth = pearson(group["clean_acc_new"], group["stealth_avg_new"])
        base_pearson_acc_stealth = pearson(group["clean_acc_base"], group["stealth_avg_base"])

        records.append(
            {
                "dataset": dataset,
                "new_arch": new_arch,
                "base_arch": base_arch,
                "n_rows": int(len(group)),
                "new_clean_acc_mean": group["clean_acc_new"].mean(),
                "base_clean_acc_mean": group["clean_acc_base"].mean(),
                "delta_clean_acc_mean": group["delta_clean_acc"].mean(),
                "new_transfer_rate_mean": group["transfer_rate_new"].mean(),
                "base_transfer_rate_mean": group["transfer_rate_base"].mean(),
                "delta_transfer_rate_mean": group["delta_transfer_rate"].mean(),
                "new_stealth_avg_mean": group["stealth_avg_new"].mean(),
                "base_stealth_avg_mean": group["stealth_avg_base"].mean(),
                "delta_stealth_avg_mean": group["delta_stealth_avg"].mean(),
                "new_spearman_transfer_stealth": new_spearman_ts,
                "base_spearman_transfer_stealth": base_spearman_ts,
                "delta_spearman_transfer_stealth": new_spearman_ts - base_spearman_ts,
                "new_pearson_transfer_stealth": new_pearson_ts,
                "base_pearson_transfer_stealth": base_pearson_ts,
                "delta_pearson_transfer_stealth": new_pearson_ts - base_pearson_ts,
                "new_spearman_acc_transfer": new_spearman_acc_transfer,
                "base_spearman_acc_transfer": base_spearman_acc_transfer,
                "delta_spearman_acc_transfer": new_spearman_acc_transfer - base_spearman_acc_transfer,
                "new_spearman_acc_stealth": new_spearman_acc_stealth,
                "base_spearman_acc_stealth": base_spearman_acc_stealth,
                "delta_spearman_acc_stealth": new_spearman_acc_stealth - base_spearman_acc_stealth,
                "new_pearson_acc_transfer": new_pearson_acc_transfer,
                "base_pearson_acc_transfer": base_pearson_acc_transfer,
                "delta_pearson_acc_transfer": new_pearson_acc_transfer - base_pearson_acc_transfer,
                "new_pearson_acc_stealth": new_pearson_acc_stealth,
                "base_pearson_acc_stealth": base_pearson_acc_stealth,
                "delta_pearson_acc_stealth": new_pearson_acc_stealth - base_pearson_acc_stealth,
                "spearman_delta_acc_transfer": spearman(group["delta_clean_acc"], group["delta_transfer_rate"]),
                "spearman_delta_acc_stealth": spearman(group["delta_clean_acc"], group["delta_stealth_avg"]),
                "pearson_delta_acc_transfer": pearson(group["delta_clean_acc"], group["delta_transfer_rate"]),
                "pearson_delta_acc_stealth": pearson(group["delta_clean_acc"], group["delta_stealth_avg"]),
            }
        )
    return pd.DataFrame(records).sort_values(["dataset", "new_arch", "base_arch"])


def arch_plot_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    data = exclude_arch_attacks(df[df["primary_main_analysis"]].dropna(subset=["transfer_rate", "stealth_avg", "clean_acc"]).copy())
    if data.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    groups: List[Tuple[str, pd.DataFrame]] = [("all", data)]
    groups.extend([(str(name), group) for name, group in data.groupby("dataset", dropna=False)])
    groups.extend([(str(name), group) for name, group in data.groupby("dataset_arch", dropna=False)])
    for name, group in groups:
        rows.append(
            {
                "group": name,
                "n_rows": int(len(group)),
                "clean_acc_mean": group["clean_acc"].mean(),
                "transfer_rate_p25": group["transfer_rate"].quantile(0.25),
                "transfer_rate_median": group["transfer_rate"].median(),
                "transfer_rate_p75": group["transfer_rate"].quantile(0.75),
                "transfer_rate_p95": group["transfer_rate"].quantile(0.95),
                "transfer_rate_max": group["transfer_rate"].max(),
                "share_transfer_0p9_1p1": float(((group["transfer_rate"] >= 0.9) & (group["transfer_rate"] <= 1.1)).mean()),
                "share_transfer_gt_1p5": float((group["transfer_rate"] > 1.5).mean()),
                "pearson_transfer_stealth": pearson(group["transfer_rate"], group["stealth_avg"]),
                "spearman_transfer_stealth": spearman(group["transfer_rate"], group["stealth_avg"]),
            }
        )
    return pd.DataFrame(rows)


def arch_binned_trend(df: pd.DataFrame, by: str = "dataset_arch", bins: int = 6) -> pd.DataFrame:
    data = exclude_arch_attacks(df[df["primary_main_analysis"]].dropna(subset=["transfer_rate", "stealth_avg", "clean_acc"]).copy())
    records: List[Dict[str, Any]] = []
    if data.empty:
        return pd.DataFrame()
    for name, group in data.groupby(by, dropna=False):
        if len(group) < bins:
            continue
        try:
            qbins = pd.qcut(group["transfer_rate"], bins, duplicates="drop")
        except Exception:
            continue
        for bin_id, (_, bg) in enumerate(group.assign(transfer_bin=qbins).groupby("transfer_bin", observed=False), start=1):
            if bg.empty:
                continue
            records.append(
                {
                    by: str(name),
                    "bin_id": bin_id,
                    "n_rows": int(len(bg)),
                    "transfer_rate_median": bg["transfer_rate"].median(),
                    "transfer_rate_mean": bg["transfer_rate"].mean(),
                    "stealth_avg_median": bg["stealth_avg"].median(),
                    "stealth_avg_mean": bg["stealth_avg"].mean(),
                    "stealth_avg_q25": bg["stealth_avg"].quantile(0.25),
                    "stealth_avg_q75": bg["stealth_avg"].quantile(0.75),
                    "clean_acc_mean": bg["clean_acc"].mean(),
                }
            )
    return pd.DataFrame(records)


def arch_rank_trend(df: pd.DataFrame, by: str = "dataset_arch", bins: int = 8) -> pd.DataFrame:
    data = exclude_arch_attacks(df[df["primary_main_analysis"]].dropna(subset=["transfer_rate", "stealth_avg"]).copy())
    if data.empty:
        return pd.DataFrame()
    data["transfer_rank_pct"] = data.groupby(by)["transfer_rate"].rank(method="average", pct=True)
    data["stealth_rank_pct"] = data.groupby(by)["stealth_avg"].rank(method="average", pct=True)
    records: List[Dict[str, Any]] = []
    for name, group in data.groupby(by, dropna=False):
        if len(group) < bins:
            continue
        try:
            qbins = pd.qcut(group["transfer_rank_pct"], bins, duplicates="drop")
        except Exception:
            continue
        for bin_id, (_, bg) in enumerate(group.assign(rank_bin=qbins).groupby("rank_bin", observed=False), start=1):
            if bg.empty:
                continue
            records.append(
                {
                    by: str(name),
                    "bin_id": bin_id,
                    "n_rows": int(len(bg)),
                    "transfer_rank_median": bg["transfer_rank_pct"].median(),
                    "stealth_rank_median": bg["stealth_rank_pct"].median(),
                }
            )
    return pd.DataFrame(records)


def plot_arch_metric_overview(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    sub = summary[summary["dataset"].isin(PRIMARY_DATASETS)].copy()
    if sub.empty:
        return
    sub["label"] = sub["dataset"] + ":" + sub["arch_base"]
    sub = sub.sort_values(["dataset", "clean_acc_mean"])
    metrics = [
        ("clean_acc_mean", "Mean clean ACC", 1.0),
        ("transfer_rate_mean", "Mean transfer_rate", max(1.1, float(sub["transfer_rate_mean"].max()) * 1.08)),
        ("stealth_avg_mean", "Mean stealth_avg", 1.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    y = np.arange(len(sub))
    colors = ["#4c78a8" if ds == "cifar10" else "#f58518" for ds in sub["dataset"]]
    for ax, (col, title, xmax) in zip(axes, metrics):
        values = sub[col].to_numpy()
        ax.barh(y, values, color=colors, alpha=0.86)
        ax.set_title(title)
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", alpha=0.22)
        for idx, value in enumerate(values):
            ax.text(value + xmax * 0.01, idx, f"{value:.3f}", va="center", fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub["label"])
    fig.suptitle("Architecture overview: ACC, transferability, stealth")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_arch_transfer_stealth_facets(df: pd.DataFrame, path: Path) -> None:
    sub = exclude_arch_attacks(df[df["primary_main_analysis"]].dropna(subset=["transfer_rate", "stealth_avg"]).copy())
    if sub.empty:
        return
    sub["transfer_rate_plot"] = sub["transfer_rate"].clip(upper=2.0)
    datasets = [d for d in PRIMARY_DATASETS if d in set(sub["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.2 * len(datasets), 5.4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    colors = {
        "ResNet18": "#4c78a8",
        "mobilenetv2": "#54a24b",
        "vgg19_bn": "#b279a2",
        "SmallCNN": "#e45756",
        "ResNet34": "#f58518",
    }
    for ax, dataset in zip(axes, datasets):
        group = sub[sub["dataset"] == dataset].copy()
        for arch, ag in group.groupby("arch_base", dropna=False):
            ax.scatter(
                ag["transfer_rate_plot"],
                ag["stealth_avg"],
                s=24,
                alpha=0.58,
                label=str(arch),
                color=colors.get(str(arch)),
            )
        sp = spearman(group["transfer_rate"], group["stealth_avg"])
        pe = pearson(group["transfer_rate"], group["stealth_avg"])
        ax.set_title(f"{dataset}\nPearson={pe:.3f}, Spearman={sp:.3f}")
        ax.set_xlabel("transfer_rate (clipped at 2.0 for display)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("stealth_avg = mean(1 - TPR)")
    fig.suptitle("Transfer-stealth scatter by dataset and architecture")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_arch_binned_trend(trend: pd.DataFrame, path: Path) -> None:
    if trend.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    colors = {
        "ResNet18": "#4c78a8",
        "mobilenetv2": "#54a24b",
        "vgg19_bn": "#b279a2",
        "SmallCNN": "#e45756",
        "ResNet34": "#f58518",
    }
    for ax, dataset in zip(axes, PRIMARY_DATASETS):
        dtrend = trend[trend["dataset_arch"].str.startswith(f"{dataset}:")].copy()
        if dtrend.empty:
            ax.set_visible(False)
            continue
        for dataset_arch, group in dtrend.groupby("dataset_arch", dropna=False):
            arch = str(dataset_arch).split(":", 1)[-1]
            group = group.sort_values("transfer_rate_median")
            x = group["transfer_rate_median"].clip(upper=2.0).to_numpy()
            y = group["stealth_avg_median"].to_numpy()
            y25 = group["stealth_avg_q25"].to_numpy()
            y75 = group["stealth_avg_q75"].to_numpy()
            ax.plot(x, y, marker="o", linewidth=1.7, label=arch, color=colors.get(arch))
            ax.fill_between(x, y25, y75, alpha=0.10, color=colors.get(arch))
        ax.set_title(dataset)
        ax.set_xlabel("median transfer_rate in bin (clipped at 2.0)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("median stealth_avg with IQR band")
    fig.suptitle("Binned median trend: transfer_rate vs stealth_avg")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_arch_rank_trend(rank_trend: pd.DataFrame, path: Path) -> None:
    if rank_trend.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    colors = {
        "ResNet18": "#4c78a8",
        "mobilenetv2": "#54a24b",
        "vgg19_bn": "#b279a2",
        "SmallCNN": "#e45756",
        "ResNet34": "#f58518",
    }
    for ax, dataset in zip(axes, PRIMARY_DATASETS):
        dtrend = rank_trend[rank_trend["dataset_arch"].str.startswith(f"{dataset}:")].copy()
        if dtrend.empty:
            ax.set_visible(False)
            continue
        for dataset_arch, group in dtrend.groupby("dataset_arch", dropna=False):
            arch = str(dataset_arch).split(":", 1)[-1]
            group = group.sort_values("transfer_rank_median")
            ax.plot(
                group["transfer_rank_median"].to_numpy(),
                group["stealth_rank_median"].to_numpy(),
                marker="o",
                linewidth=1.7,
                label=arch,
                color=colors.get(arch),
            )
        ax.set_title(dataset)
        ax.set_xlabel("within-arch transfer_rate rank percentile")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("within-arch stealth_avg rank percentile")
    fig.suptitle("Rank-binned trend: Spearman-style view, excluding SIG/upgd")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_arch_acc_moderation_summary(df: pd.DataFrame, acc_bin_slope: pd.DataFrame, path: Path) -> None:
    sub = exclude_arch_attacks(df[df["primary_main_analysis"]].dropna(subset=["clean_acc", "transfer_rate", "stealth_avg", "acc_bin"]).copy())
    if sub.empty:
        return
    sub["transfer_rate_plot"] = sub["transfer_rate"].clip(upper=2.0)
    colors = {"low_acc": "#b23b3b", "mid_acc": "#a66a00", "high_acc": "#287d55"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))

    for acc_bin, group in sub.groupby("acc_bin", dropna=False):
        label = str(acc_bin)
        axes[0].scatter(group["clean_acc"], group["transfer_rate_plot"], s=20, alpha=0.48, color=colors.get(label), label=label)
        axes[1].scatter(group["clean_acc"], group["stealth_avg"], s=20, alpha=0.48, color=colors.get(label), label=label)
        axes[2].scatter(group["transfer_rate_plot"], group["stealth_avg"], s=20, alpha=0.48, color=colors.get(label), label=label)

    axes[0].set_title("ACC -> transfer_rate")
    axes[0].set_xlabel("clean_acc")
    axes[0].set_ylabel("transfer_rate (clipped at 2.0)")
    axes[0].text(
        0.03,
        0.95,
        f"Spearman={spearman(sub['clean_acc'], sub['transfer_rate']):.3f}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#d8dee8"},
    )

    axes[1].set_title("ACC -> stealth_avg")
    axes[1].set_xlabel("clean_acc")
    axes[1].set_ylabel("stealth_avg")
    axes[1].text(
        0.03,
        0.95,
        f"Spearman={spearman(sub['clean_acc'], sub['stealth_avg']):.3f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#d8dee8"},
    )

    axes[2].set_title("transfer_rate -> stealth_avg by ACC bin")
    axes[2].set_xlabel("transfer_rate (clipped at 2.0)")
    axes[2].set_ylabel("stealth_avg")
    for acc_bin, group in sub.groupby("acc_bin", dropna=False):
        label = str(acc_bin)
        if len(group) < 3:
            continue
        x = group["transfer_rate_plot"].to_numpy(dtype=float)
        y = group["stealth_avg"].to_numpy(dtype=float)
        try:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
            axes[2].plot(xs, slope * xs + intercept, color=colors.get(label), linewidth=2.0)
        except Exception:
            pass
    if not acc_bin_slope.empty:
        global_slopes = acc_bin_slope[acc_bin_slope["group_type"] == "global_acc_bin"].copy()
        note_parts = []
        for bin_name in ["high_acc", "mid_acc", "low_acc"]:
            row = first_row(global_slopes, dataset="all", acc_bin=bin_name)
            if row is not None:
                note_parts.append(f"{bin_name}: {metric(row, 'spearman_transfer_stealth')}")
        if note_parts:
            axes[2].text(
                0.03,
                0.95,
                "Spearman\n" + "\n".join(note_parts),
                transform=axes[2].transAxes,
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.86, "edgecolor": "#d8dee8"},
            )

    for ax in axes:
        ax.grid(alpha=0.25)
    axes[2].legend(fontsize=8, title="ACC bin")
    fig.suptitle("ACC has uneven marginal effects and moderates the transfer-stealth relationship")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_arch_interaction_slope_by_acc(df: pd.DataFrame, path: Path) -> None:
    sub = exclude_arch_attacks(df[df["primary_main_analysis"]].dropna(subset=["difficulty", "transfer_rate", "stealth_avg"]).copy())
    if len(sub) < 20:
        return
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return

    zsub = _add_zscore_columns(sub, ["transfer_rate", "difficulty", "stealth_avg"])
    try:
        model = smf.ols(
            "stealth_avg_z ~ transfer_rate_z * difficulty_z + C(dataset) + C(arch_base) + C(attack_family) + C(poison_rate)",
            data=zsub,
        ).fit(cov_type="HC3")
    except Exception:
        return

    b_transfer = model.params.get("transfer_rate_z", float("nan"))
    b_interaction = model.params.get("transfer_rate_z:difficulty_z", float("nan"))
    if not np.isfinite(b_transfer) or not np.isfinite(b_interaction):
        return

    diff_quantiles = sub["difficulty"].quantile([0.2, 0.5, 0.8])
    diff_mean = sub["difficulty"].mean()
    diff_std = sub["difficulty"].std(ddof=0)
    transfer_z = np.linspace(-2.0, 2.0, 120)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    labels = [("High ACC / low difficulty", diff_quantiles.loc[0.2]), ("Median ACC", diff_quantiles.loc[0.5]), ("Low ACC / high difficulty", diff_quantiles.loc[0.8])]
    colors = ["#287d55", "#a66a00", "#b23b3b"]
    for (label, diff_value), color in zip(labels, colors):
        diff_z = (diff_value - diff_mean) / diff_std if diff_std else 0.0
        slope = b_transfer + b_interaction * diff_z
        y = slope * transfer_z
        axes[0].plot(transfer_z, y, color=color, linewidth=2.4, label=f"{label}: slope={slope:.3f}")
        axes[1].bar(label, slope, color=color, alpha=0.86)

    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.45)
    axes[0].axvline(0, color="black", linewidth=0.8, alpha=0.45)
    axes[0].set_xlabel("transfer_rate z-score")
    axes[0].set_ylabel("predicted stealth_avg z-score contribution")
    axes[0].set_title("Predicted transfer-stealth slope changes with difficulty")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.45)
    axes[1].set_ylabel("conditional slope")
    axes[1].set_title("More negative slope means stronger tradeoff")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.25)

    pval = model.pvalues.get("transfer_rate_z:difficulty_z", float("nan"))
    fig.suptitle(f"Controlled interaction: transfer_rate_z:difficulty_z = {b_interaction:.4f}, p = {pval:.2e}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_arch_dataset_acc_bin_spearman(acc_bin_slope: pd.DataFrame, path: Path) -> None:
    if acc_bin_slope.empty:
        return
    sub = acc_bin_slope[acc_bin_slope["group_type"] == "dataset_acc_bin"].copy()
    if sub.empty:
        return
    sub["acc_bin"] = pd.Categorical(sub["acc_bin"], categories=["low_acc", "mid_acc", "high_acc"], ordered=True)
    pivot = sub.pivot_table(index="dataset", columns="acc_bin", values="spearman_transfer_stealth", aggfunc="mean", observed=False)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.24
    colors = {"low_acc": "#b23b3b", "mid_acc": "#a66a00", "high_acc": "#287d55"}
    for idx, bin_name in enumerate(["low_acc", "mid_acc", "high_acc"]):
        values = pivot[bin_name].to_numpy(dtype=float) if bin_name in pivot else np.full(len(pivot.index), np.nan)
        pos = x + (idx - 1) * width
        ax.bar(pos, values, width=width, label=bin_name, color=colors[bin_name], alpha=0.86)
        for px, value in zip(pos, values):
            if np.isfinite(value):
                ax.text(px, value + (0.025 if value >= 0 else -0.035), f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Spearman(transfer_rate, stealth_avg)")
    ax.set_title("Dataset-specific ACC bins: transfer-stealth strength changes by ACC range")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pairwise_delta_summary(pairwise_summary: pd.DataFrame, path: Path) -> None:
    if pairwise_summary.empty:
        return
    sub = pairwise_summary.copy()
    sub["label"] = sub["dataset"] + "\n" + sub["new_arch"] + " - " + sub["base_arch"]
    metrics = [
        ("delta_clean_acc_mean", "Delta clean ACC"),
        ("delta_transfer_rate_mean", "Delta transfer_rate"),
        ("delta_stealth_avg_mean", "Delta stealth_avg"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    y = np.arange(len(sub))
    for ax, (col, title) in zip(axes, metrics):
        values = sub[col].to_numpy()
        colors = np.where(values >= 0, "#54a24b", "#e45756")
        ax.barh(y, values, color=colors, alpha=0.86)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.55)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.22)
        for idx, value in enumerate(values):
            ha = "left" if value >= 0 else "right"
            offset = 0.006 if value >= 0 else -0.006
            ax.text(value + offset, idx, f"{value:+.3f}", va="center", ha=ha, fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub["label"])
    fig.suptitle("Same-configuration delta excluding SIG/upgd: new architecture minus baseline architecture")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pairwise_relationship_shift(relationship_shift: pd.DataFrame, path: Path) -> None:
    if relationship_shift.empty:
        return
    sub = relationship_shift.copy()
    sub["label"] = sub["dataset"] + "\n" + sub["new_arch"] + " - " + sub["base_arch"]
    y = np.arange(len(sub))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, base_col, new_col, delta_col, title in [
        (
            axes[0],
            "base_spearman_transfer_stealth",
            "new_spearman_transfer_stealth",
            "delta_spearman_transfer_stealth",
            "Spearman transfer_rate vs stealth_avg",
        ),
        (
            axes[1],
            "base_pearson_transfer_stealth",
            "new_pearson_transfer_stealth",
            "delta_pearson_transfer_stealth",
            "Pearson transfer_rate vs stealth_avg",
        ),
    ]:
        ax.barh(y - 0.18, sub[base_col].to_numpy(), height=0.34, color="#9aa7b5", alpha=0.90, label="baseline")
        ax.barh(y + 0.18, sub[new_col].to_numpy(), height=0.34, color="#2457a6", alpha=0.88, label="new model")
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.55)
        ax.set_xlim(-1.0, 0.25)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.22)
        for idx, (_, row) in enumerate(sub.iterrows()):
            value = row[delta_col]
            if np.isfinite(value):
                ax.text(0.24, idx, f"Δ={value:+.3f}", va="center", ha="right", fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub["label"])
    axes[1].legend(fontsize=9, loc="lower right")
    fig.suptitle("Transfer-stealth correlation shift after replacing baseline with new architecture")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_acc_correlation_shift(relationship_shift: pd.DataFrame, path: Path) -> None:
    if relationship_shift.empty:
        return
    sub = relationship_shift.copy()
    sub["label"] = sub["dataset"] + "\n" + sub["new_arch"] + " - " + sub["base_arch"]
    y = np.arange(len(sub))
    metrics = [
        ("delta_spearman_acc_transfer", "Δ Spearman(ACC, transfer_rate)"),
        ("delta_spearman_acc_stealth", "Δ Spearman(ACC, stealth_avg)"),
        ("spearman_delta_acc_transfer", "Spearman(ΔACC, Δtransfer_rate)"),
        ("spearman_delta_acc_stealth", "Spearman(ΔACC, Δstealth_avg)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        values = sub[col].to_numpy()
        colors = np.where(values >= 0, "#54a24b", "#e45756")
        ax.barh(y, values, color=colors, alpha=0.86)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.55)
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.22)
        for idx, value in enumerate(values):
            if np.isfinite(value):
                ha = "left" if value >= 0 else "right"
                offset = 0.025 if value >= 0 else -0.025
                ax.text(value + offset, idx, f"{value:+.2f}", va="center", ha=ha, fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub["label"])
    fig.suptitle("How ACC relates to transferability and stealth under matched architecture replacement")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _heatmap(ax: Any, matrix: pd.DataFrame, title: str, cmap: str = "viridis") -> None:
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=np.nanmin(values), vmax=np.nanmax(values))
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if values[i, j] > np.nanmean(values) else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_attack_heatmaps(attack_summary: pd.DataFrame, path: Path) -> None:
    if attack_summary.empty:
        return
    sub = attack_summary[attack_summary["dataset"].isin(PRIMARY_DATASETS)].copy()
    if sub.empty:
        return
    sub["dataset_arch"] = sub["dataset"] + ":" + sub["arch_base"]
    transfer = sub.pivot_table(index="attack_family", columns="dataset_arch", values="transfer_rate_mean", aggfunc="mean")
    stealth = sub.pivot_table(index="attack_family", columns="dataset_arch", values="stealth_avg_mean", aggfunc="mean")
    if transfer.empty or stealth.empty:
        return
    cols = sorted(transfer.columns, key=lambda x: (x.split(":")[0], x.split(":")[1]))
    idx = sorted(transfer.index)
    transfer = transfer.reindex(index=idx, columns=cols)
    stealth = stealth.reindex(index=idx, columns=cols)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    _heatmap(axes[0], transfer, "Mean transfer_rate by attack and architecture", cmap="magma")
    _heatmap(axes[1], stealth, "Mean stealth_avg by attack and architecture", cmap="viridis")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_defense_heatmap(defense_breakdown: pd.DataFrame, path: Path) -> None:
    if defense_breakdown.empty:
        return
    sub = defense_breakdown[defense_breakdown["dataset"].isin(PRIMARY_DATASETS)].copy()
    if sub.empty:
        return
    sub["dataset_arch"] = sub["dataset"] + ":" + sub["arch_base"]
    matrix = sub.pivot_table(index="defense", columns="dataset_arch", values="stealth_mean", aggfunc="mean")
    if matrix.empty:
        return
    cols = sorted(matrix.columns, key=lambda x: (x.split(":")[0], x.split(":")[1]))
    matrix = matrix.reindex(index=["sentinet", "strip", "scaleup", "ibd_psc"], columns=cols)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _heatmap(ax, matrix, "Defense-specific mean stealth = 1 - TPR", cmap="viridis")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_combined_with_noise(df: pd.DataFrame, noise_csv: Path) -> pd.DataFrame:
    arch = df[df["primary_main_analysis"]].copy()
    arch["variation_source"] = "architecture"
    arch_cols = [
        "variation_source",
        "dataset",
        "arch_base",
        "attack_family",
        "clean_acc",
        "difficulty",
        "source_asr",
        "transfer_asr",
        "transfer_rate",
        "stealth_avg",
    ]
    arch = arch[arch_cols].rename(columns={"arch_base": "arch", "attack_family": "attack_type"})

    if not noise_csv.exists():
        return arch

    noise = pd.read_csv(noise_csv)
    if "include_main_analysis" in noise.columns:
        noise = noise[noise["include_main_analysis"].astype(bool)].copy()
    noise["variation_source"] = "noise"
    if "arch" not in noise.columns:
        noise["arch"] = "SmallCNN_cifar10"
    noise_cols = [
        "variation_source",
        "attack_type",
        "clean_acc",
        "difficulty",
        "source_asr",
        "transfer_asr",
        "transfer_rate",
        "stealth_avg",
    ]
    noise = noise[[c for c in noise_cols if c in noise.columns]].copy()
    noise["dataset"] = "cifar10"
    noise["arch"] = noise.get("arch", "SmallCNN_cifar10")
    noise = noise[["variation_source", "dataset", "arch", "attack_type", "clean_acc", "difficulty", "source_asr", "transfer_asr", "transfer_rate", "stealth_avg"]]
    return pd.concat([arch, noise], ignore_index=True)


def combined_correlation_table(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    records = []
    records.append(corr_record(combined, "all", "combined", "combined"))
    for source, group in combined.groupby("variation_source", dropna=False):
        records.append(corr_record(group, "variation_source", str(source), "combined"))
    for (source, attack), group in combined.groupby(["variation_source", "attack_type"], dropna=False):
        if len(group) >= 3:
            records.append(corr_record(group, "source_attack", f"{source}:{attack}", "combined"))
    return pd.DataFrame(records)


def combined_acc_bin_table(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for source, group in combined.groupby("variation_source", dropna=False):
        group = group.dropna(subset=["clean_acc", "transfer_rate", "stealth_avg"]).copy()
        if len(group) < 3:
            continue
        try:
            bins = pd.qcut(group["clean_acc"], 3, labels=["low_acc", "mid_acc", "high_acc"], duplicates="drop")
        except Exception:
            continue
        for bin_name, bg in group.assign(acc_bin=bins).groupby("acc_bin", observed=False):
            if bg.empty:
                continue
            records.append(
                {
                    "variation_source": str(source),
                    "acc_bin": str(bin_name),
                    "n_rows": int(len(bg)),
                    "clean_acc_min": bg["clean_acc"].min(),
                    "clean_acc_max": bg["clean_acc"].max(),
                    "clean_acc_mean": bg["clean_acc"].mean(),
                    "difficulty_mean": bg["difficulty"].mean(),
                    "transfer_rate_mean": bg["transfer_rate"].mean(),
                    "transfer_rate_median": bg["transfer_rate"].median(),
                    "stealth_avg_mean": bg["stealth_avg"].mean(),
                    "stealth_avg_median": bg["stealth_avg"].median(),
                    "pearson_transfer_stealth": pearson(bg["transfer_rate"], bg["stealth_avg"]),
                    "spearman_transfer_stealth": spearman(bg["transfer_rate"], bg["stealth_avg"]),
                }
            )
    return pd.DataFrame(records)


def combined_attack_summary(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    return (
        combined.groupby(["variation_source", "attack_type"], dropna=False)
        .agg(
            n_rows=("variation_source", "count"),
            clean_acc_mean=("clean_acc", "mean"),
            difficulty_mean=("difficulty", "mean"),
            source_asr_mean=("source_asr", "mean"),
            transfer_asr_mean=("transfer_asr", "mean"),
            transfer_rate_mean=("transfer_rate", "mean"),
            transfer_rate_median=("transfer_rate", "median"),
            stealth_avg_mean=("stealth_avg", "mean"),
            stealth_avg_median=("stealth_avg", "median"),
        )
        .reset_index()
        .sort_values(["attack_type", "variation_source"])
    )


def combined_plot_diagnostics(combined: pd.DataFrame) -> pd.DataFrame:
    data = combined.dropna(subset=["clean_acc", "transfer_rate", "stealth_avg"]).copy()
    if data.empty:
        return pd.DataFrame()
    rows = []
    for name, group in [("all", data), *list(data.groupby("variation_source", dropna=False))]:
        label = str(name)
        rows.append(
            {
                "group": label,
                "n_rows": int(len(group)),
                "clean_acc_mean": group["clean_acc"].mean(),
                "difficulty_mean": group["difficulty"].mean(),
                "transfer_rate_p25": group["transfer_rate"].quantile(0.25),
                "transfer_rate_median": group["transfer_rate"].median(),
                "transfer_rate_p75": group["transfer_rate"].quantile(0.75),
                "transfer_rate_p95": group["transfer_rate"].quantile(0.95),
                "transfer_rate_max": group["transfer_rate"].max(),
                "share_transfer_0p9_1p1": float(((group["transfer_rate"] >= 0.9) & (group["transfer_rate"] <= 1.1)).mean()),
                "share_transfer_gt_1p5": float((group["transfer_rate"] > 1.5).mean()),
                "pearson_transfer_stealth": pearson(group["transfer_rate"], group["stealth_avg"]),
                "spearman_transfer_stealth": spearman(group["transfer_rate"], group["stealth_avg"]),
            }
        )
    return pd.DataFrame(rows)


def combined_binned_trend(combined: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    data = combined.dropna(subset=["transfer_rate", "stealth_avg", "clean_acc"]).copy()
    records: List[Dict[str, Any]] = []
    if data.empty:
        return pd.DataFrame()
    for source, group in data.groupby("variation_source", dropna=False):
        if len(group) < bins:
            continue
        try:
            qbins = pd.qcut(group["transfer_rate"], bins, duplicates="drop")
        except Exception:
            continue
        for bin_id, (_, bg) in enumerate(group.assign(transfer_bin=qbins).groupby("transfer_bin", observed=False), start=1):
            if bg.empty:
                continue
            records.append(
                {
                    "variation_source": str(source),
                    "bin_id": bin_id,
                    "n_rows": int(len(bg)),
                    "transfer_rate_median": bg["transfer_rate"].median(),
                    "transfer_rate_mean": bg["transfer_rate"].mean(),
                    "stealth_avg_median": bg["stealth_avg"].median(),
                    "stealth_avg_mean": bg["stealth_avg"].mean(),
                    "stealth_avg_q25": bg["stealth_avg"].quantile(0.25),
                    "stealth_avg_q75": bg["stealth_avg"].quantile(0.75),
                    "clean_acc_mean": bg["clean_acc"].mean(),
                    "difficulty_mean": bg["difficulty"].mean(),
                }
            )
    return pd.DataFrame(records)


def combined_rank_trend(combined: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    data = combined.dropna(subset=["transfer_rate", "stealth_avg"]).copy()
    if data.empty:
        return pd.DataFrame()
    data["transfer_rank_pct"] = data.groupby("variation_source")["transfer_rate"].rank(method="average", pct=True)
    data["stealth_rank_pct"] = data.groupby("variation_source")["stealth_avg"].rank(method="average", pct=True)
    records: List[Dict[str, Any]] = []
    for source, group in data.groupby("variation_source", dropna=False):
        if len(group) < bins:
            continue
        try:
            qbins = pd.qcut(group["transfer_rank_pct"], bins, duplicates="drop")
        except Exception:
            continue
        for bin_id, (_, bg) in enumerate(group.assign(rank_bin=qbins).groupby("rank_bin", observed=False), start=1):
            if bg.empty:
                continue
            records.append(
                {
                    "variation_source": str(source),
                    "bin_id": bin_id,
                    "n_rows": int(len(bg)),
                    "transfer_rank_median": bg["transfer_rank_pct"].median(),
                    "stealth_rank_median": bg["stealth_rank_pct"].median(),
                }
            )
    return pd.DataFrame(records)


def plot_combined_metric_overview(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    sub = summary.copy().sort_values("variation_source")
    metrics = [
        ("clean_acc_mean", "Mean clean ACC", 1.0),
        ("difficulty_mean", "Mean difficulty", max(0.35, float(sub["difficulty_mean"].max()) * 1.25)),
        ("transfer_rate_mean", "Mean transfer_rate", max(1.0, float(sub["transfer_rate_mean"].max()) * 1.15)),
        ("stealth_avg_mean", "Mean stealth_avg", 1.0),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), sharey=True)
    y = np.arange(len(sub))
    colors = {"architecture": "#4c78a8", "noise": "#e45756"}
    bar_colors = [colors.get(str(v), "#777777") for v in sub["variation_source"]]
    for ax, (col, title, xmax) in zip(axes, metrics):
        values = sub[col].to_numpy()
        ax.barh(y, values, color=bar_colors, alpha=0.88)
        ax.set_title(title)
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", alpha=0.22)
        for idx, value in enumerate(values):
            ax.text(value + xmax * 0.012, idx, f"{value:.3f}", va="center", fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub["variation_source"])
    fig.suptitle("Combined evidence overview: architecture vs noise")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_combined_transfer_stealth_facets(combined: pd.DataFrame, path: Path) -> None:
    data = combined.dropna(subset=["transfer_rate", "stealth_avg"]).copy()
    if data.empty:
        return
    data["transfer_rate_plot"] = data["transfer_rate"].clip(upper=2.0)
    sources = ["architecture", "noise"]
    attacks = sorted(data["attack_type"].dropna().astype(str).unique())
    cmap = plt.get_cmap("tab10")
    colors = {attack: cmap(i % 10) for i, attack in enumerate(attacks)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for ax, source in zip(axes, sources):
        group = data[data["variation_source"] == source].copy()
        if group.empty:
            ax.set_visible(False)
            continue
        for attack, ag in group.groupby("attack_type", dropna=False):
            ax.scatter(
                ag["transfer_rate_plot"],
                ag["stealth_avg"],
                s=22,
                alpha=0.55,
                color=colors.get(str(attack)),
                label=str(attack),
            )
        sp = spearman(group["transfer_rate"], group["stealth_avg"])
        pe = pearson(group["transfer_rate"], group["stealth_avg"])
        ax.set_title(f"{source}\nPearson={pe:.3f}, Spearman={sp:.3f}")
        ax.set_xlabel("transfer_rate (clipped at 2.0 for display)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("stealth_avg")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=8)
    fig.suptitle("Transfer-stealth scatter by evidence source")
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_combined_difficulty_relationships(combined: pd.DataFrame, path: Path) -> None:
    data = combined.dropna(subset=["difficulty", "transfer_rate", "stealth_avg"]).copy()
    if data.empty:
        return
    data["transfer_rate_plot"] = data["transfer_rate"].clip(upper=2.0)
    colors = {"architecture": "#4c78a8", "noise": "#e45756"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    specs = [
        ("transfer_rate_plot", "transfer_rate (clipped at 2.0)"),
        ("stealth_avg", "stealth_avg"),
    ]
    for ax, (ycol, ylabel) in zip(axes, specs):
        for source, group in data.groupby("variation_source", dropna=False):
            ax.scatter(group["difficulty"], group[ycol], s=18, alpha=0.48, label=str(source), color=colors.get(str(source)))
            if len(group) >= 3:
                x = group["difficulty"].to_numpy()
                y = group[ycol].to_numpy()
                finite = np.isfinite(x) & np.isfinite(y)
                if finite.sum() >= 3:
                    coef = np.polyfit(x[finite], y[finite], 1)
                    xs = np.linspace(x[finite].min(), x[finite].max(), 80)
                    ax.plot(xs, coef[0] * xs + coef[1], color=colors.get(str(source)), linewidth=2)
        ax.set_xlabel("difficulty = 1 - clean_acc")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Difficulty relationships differ between architecture and noise")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_combined_binned_trend(trend: pd.DataFrame, path: Path) -> None:
    if trend.empty:
        return
    colors = {"architecture": "#4c78a8", "noise": "#e45756"}
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for source, group in trend.groupby("variation_source", dropna=False):
        group = group.sort_values("transfer_rate_median")
        x = group["transfer_rate_median"].clip(upper=2.0).to_numpy()
        y = group["stealth_avg_median"].to_numpy()
        y25 = group["stealth_avg_q25"].to_numpy()
        y75 = group["stealth_avg_q75"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=1.9, label=str(source), color=colors.get(str(source)))
        ax.fill_between(x, y25, y75, alpha=0.12, color=colors.get(str(source)))
    ax.set_xlabel("median transfer_rate in bin (clipped at 2.0)")
    ax.set_ylabel("median stealth_avg with IQR band")
    ax.set_title("Combined binned median trend")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_combined_rank_trend(rank_trend: pd.DataFrame, path: Path) -> None:
    if rank_trend.empty:
        return
    colors = {"architecture": "#4c78a8", "noise": "#e45756"}
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for source, group in rank_trend.groupby("variation_source", dropna=False):
        group = group.sort_values("transfer_rank_median")
        ax.plot(
            group["transfer_rank_median"].to_numpy(),
            group["stealth_rank_median"].to_numpy(),
            marker="o",
            linewidth=1.9,
            label=str(source),
            color=colors.get(str(source)),
        )
    ax.set_xlabel("within-source transfer_rate rank percentile")
    ax.set_ylabel("within-source stealth_avg rank percentile")
    ax.set_title("Combined rank-binned trend")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_combined_acc_bin_spearman(acc_bins: pd.DataFrame, path: Path) -> None:
    if acc_bins.empty:
        return
    order = ["low_acc", "mid_acc", "high_acc"]
    pivot = acc_bins.pivot_table(index="acc_bin", columns="variation_source", values="spearman_transfer_stealth", aggfunc="mean")
    pivot = pivot.reindex(order)
    ax = pivot.plot(kind="bar", figsize=(8.6, 5.4), color=["#4c78a8", "#e45756"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Spearman transfer_rate vs stealth_avg")
    ax.set_title("ACC-bin Spearman by evidence source")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticklabels([str(x) for x in pivot.index], rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_combined_attack_heatmap(attack_summary: pd.DataFrame, path: Path) -> None:
    if attack_summary.empty:
        return
    transfer = attack_summary.pivot_table(index="attack_type", columns="variation_source", values="transfer_rate_mean", aggfunc="mean")
    stealth = attack_summary.pivot_table(index="attack_type", columns="variation_source", values="stealth_avg_mean", aggfunc="mean")
    if transfer.empty or stealth.empty:
        return
    idx = sorted(transfer.index)
    cols = ["architecture", "noise"]
    transfer = transfer.reindex(index=idx, columns=cols)
    stealth = stealth.reindex(index=idx, columns=cols)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 6.2))
    _heatmap(axes[0], transfer, "Mean transfer_rate by attack", cmap="magma")
    _heatmap(axes[1], stealth, "Mean stealth_avg by attack", cmap="viridis")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_combined_html_dashboard(
    summary: pd.DataFrame,
    corr_df: pd.DataFrame,
    acc_bins: pd.DataFrame,
    attack_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    path: Path,
) -> None:
    arch_summary = first_row(summary, variation_source="architecture")
    noise_summary = first_row(summary, variation_source="noise")
    arch_corr = first_row(corr_df, group_type="variation_source", group_name="architecture")
    noise_corr = first_row(corr_df, group_type="variation_source", group_name="noise")
    diag_all = first_row(diagnostics, group="all")
    key_summary_cols = [
        "variation_source",
        "n_rows",
        "clean_acc_mean",
        "difficulty_mean",
        "transfer_rate_mean",
        "transfer_rate_median",
        "stealth_avg_mean",
        "stealth_avg_median",
    ]
    key_corr_cols = [
        "group_name",
        "n_rows",
        "pearson_clean_acc_transfer_rate",
        "pearson_transfer_rate_stealth_avg",
        "spearman_transfer_rate_stealth_avg",
    ]
    key_acc_cols = [
        "variation_source",
        "acc_bin",
        "n_rows",
        "clean_acc_mean",
        "difficulty_mean",
        "transfer_rate_median",
        "stealth_avg_median",
        "spearman_transfer_stealth",
    ]
    key_attack_cols = [
        "variation_source",
        "attack_type",
        "n_rows",
        "clean_acc_mean",
        "transfer_rate_mean",
        "stealth_avg_mean",
    ]
    figure_cards = [
        image_card("arch_acc_analysis/figures/combined_metric_overview.png", "1. 两条证据链总览", "比较 architecture 和 noise 的 ACC、difficulty、transfer_rate、stealth_avg 均值。"),
        image_card("arch_acc_analysis/figures/combined_transfer_stealth_facets.png", "2. Transfer-Stealth 分面散点", "两条证据链都呈负相关，但点云结构不同。"),
        image_card("arch_acc_analysis/figures/combined_difficulty_relationships.png", "3. Difficulty 关系", "展示 architecture 和 noise 中 difficulty 与迁移/隐蔽的方向不完全一致。"),
        image_card("arch_acc_analysis/figures/combined_binned_median_trend.png", "4. 分箱中位数趋势", "减少普通散点重叠后展示趋势。"),
        image_card("arch_acc_analysis/figures/combined_rank_binned_trend.png", "5. Rank 趋势", "对应 Spearman 排序相关。"),
        image_card("arch_acc_analysis/figures/combined_acc_bin_spearman.png", "6. ACC 分层 Spearman", "看不同 ACC 区间内 transfer-stealth 强度如何变化。"),
        image_card("arch_acc_analysis/figures/combined_attack_heatmap.png", "7. 攻击类型热力图", "说明两条证据链都受 attack type 强烈影响。"),
    ]
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ACC 难度联合分析展示</title>
  <style>
    :root {{ --ink:#17202a; --muted:#5f6b7a; --line:#d8dee8; --band:#f5f7fb; --blue:#2457a6; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; background:white; }}
    header {{ padding:28px 36px 20px; border-bottom:1px solid var(--line); background:linear-gradient(180deg,#f8fafc 0%,#fff 100%); }}
    main {{ padding:0 36px 42px; max-width:1380px; }}
    h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
    h2 {{ margin:30px 0 12px; font-size:20px; }}
    h3 {{ margin:0 0 8px; font-size:16px; }}
    p {{ margin:0 0 10px; }}
    .summary-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-top:18px; }}
    .metric {{ border:1px solid var(--line); background:#fff; padding:14px 16px; border-radius:8px; }}
    .metric b {{ display:block; font-size:24px; margin-bottom:2px; }}
    .metric span {{ color:var(--muted); font-size:13px; }}
    .callout {{ border-left:4px solid var(--blue); background:var(--band); padding:14px 16px; margin:18px 0; }}
    .figure-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .figure-card {{ border:1px solid var(--line); border-radius:8px; padding:14px; background:#fff; }}
    .figure-card img {{ width:100%; height:auto; display:block; border:1px solid var(--line); background:#fff; }}
    .figure-card p {{ color:var(--muted); margin-top:8px; }}
    table {{ width:100%; border-collapse:collapse; margin:10px 0 20px; font-size:13px; }}
    th,td {{ border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }}
    th {{ background:#eef2f7; }}
    code {{ background:#eef2f7; padding:1px 4px; border-radius:4px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
    ol,ul {{ padding-left:22px; }}
    @media (max-width:980px) {{ main,header {{ padding-left:18px; padding-right:18px; }} .summary-grid,.figure-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>ACC 难度联合分析：噪声实验与架构实验</h1>
    <p>静态 dashboard，数据来自 <code>combined_acc_effect_rows.csv</code>。目标是把 architecture 和 noise 两条证据链放在同一解释框架里。</p>
    <div class="summary-grid">
      <div class="metric"><b>{metric(arch_corr, 'spearman_transfer_rate_stealth_avg')}</b><span>architecture transfer-stealth Spearman</span></div>
      <div class="metric"><b>{metric(noise_corr, 'spearman_transfer_rate_stealth_avg')}</b><span>noise transfer-stealth Spearman</span></div>
      <div class="metric"><b>{metric(arch_summary, 'stealth_avg_mean')}</b><span>architecture mean stealth</span></div>
      <div class="metric"><b>{metric(noise_summary, 'stealth_avg_mean')}</b><span>noise mean stealth</span></div>
    </div>
  </header>
  <main>
    <section class="callout">
      <p><b>一句话结论：</b>两条证据链都显示 transfer-stealth 负相关，但 architecture 和 noise 的 clean_acc 与 transfer_rate 方向不同，因此联合分析不能直接写成“ACC 单因果解释”。</p>
      <p><b>散点注意：</b>全体 transfer_rate 中位数为 <code>{metric(diag_all, 'transfer_rate_median')}</code>，落在 [0.9, 1.1] 的比例为 <code>{metric(diag_all, 'share_transfer_0p9_1p1')}</code>，所以建议看分箱趋势和 ACC-bin 结果。</p>
    </section>
    <h2>建议展示顺序</h2>
    <ol>
      <li>先讲两条证据链总览：architecture 是模型/数据集生态对照，noise 更接近同模型 difficulty intervention。</li>
      <li>再讲 transfer-stealth 共同负相关，但 difficulty/ACC 对 transfer 的方向不同。</li>
      <li>用分箱趋势和 ACC-bin 图解释“关系强度会随难度区间变化”。</li>
      <li>最后用攻击热力图提醒：attack type 是强混杂因素。</li>
    </ol>
    <h2>核心图表</h2>
    <div class="figure-grid">{''.join(figure_cards)}</div>
    <h2>总体汇总</h2>
    {html_table(summary, key_summary_cols, 10, ["variation_source"])}
    <h2>相关性对比</h2>
    {html_table(corr_df[corr_df["group_type"] == "variation_source"], key_corr_cols, 10, ["group_name"])}
    <h2>ACC 分层</h2>
    {html_table(acc_bins, key_acc_cols, 20, ["variation_source", "acc_bin"])}
    <h2>攻击类型分层</h2>
    {html_table(attack_summary, key_attack_cols, 80, ["attack_type", "variation_source"])}
  </main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_combined_report(combined: pd.DataFrame, report_path: Path, summary_path: Path, regression_path: Path) -> None:
    if combined.empty:
        report_path.write_text("# ACC 难度联合分析报告\n\n没有可用数据。\n", encoding="utf-8")
        return

    output_dir = summary_path.parent
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        combined.groupby("variation_source", dropna=False)
        .agg(
            n_rows=("variation_source", "count"),
            clean_acc_mean=("clean_acc", "mean"),
            clean_acc_median=("clean_acc", "median"),
            difficulty_mean=("difficulty", "mean"),
            transfer_rate_mean=("transfer_rate", "mean"),
            transfer_rate_median=("transfer_rate", "median"),
            stealth_avg_mean=("stealth_avg", "mean"),
            stealth_avg_median=("stealth_avg", "median"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    corr_df = combined_correlation_table(combined)
    acc_bins = combined_acc_bin_table(combined)
    attack_summary = combined_attack_summary(combined)
    diagnostics = combined_plot_diagnostics(combined)
    trend = combined_binned_trend(combined)
    rank_trend = combined_rank_trend(combined)

    corr_df.to_csv(output_dir / "combined_acc_effect_correlations.csv", index=False)
    acc_bins.to_csv(output_dir / "combined_acc_effect_acc_bins.csv", index=False)
    attack_summary.to_csv(output_dir / "combined_acc_effect_attack_summary.csv", index=False)
    diagnostics.to_csv(output_dir / "combined_acc_effect_plot_diagnostics.csv", index=False)
    trend.to_csv(output_dir / "combined_acc_effect_binned_median_trend.csv", index=False)
    rank_trend.to_csv(output_dir / "combined_acc_effect_rank_binned_trend.csv", index=False)

    plot_combined_metric_overview(summary, figures_dir / "combined_metric_overview.png")
    plot_combined_transfer_stealth_facets(combined, figures_dir / "combined_transfer_stealth_facets.png")
    plot_combined_difficulty_relationships(combined, figures_dir / "combined_difficulty_relationships.png")
    plot_combined_binned_trend(trend, figures_dir / "combined_binned_median_trend.png")
    plot_combined_rank_trend(rank_trend, figures_dir / "combined_rank_binned_trend.png")
    plot_combined_acc_bin_spearman(acc_bins, figures_dir / "combined_acc_bin_spearman.png")
    plot_combined_attack_heatmap(attack_summary, figures_dir / "combined_attack_heatmap.png")

    arch_summary = first_row(summary, variation_source="architecture")
    noise_summary = first_row(summary, variation_source="noise")
    arch_corr = first_row(corr_df, group_type="variation_source", group_name="architecture")
    noise_corr = first_row(corr_df, group_type="variation_source", group_name="noise")
    diag_all = first_row(diagnostics, group="all")
    arch_low = first_row(acc_bins, variation_source="architecture", acc_bin="low_acc")
    arch_mid = first_row(acc_bins, variation_source="architecture", acc_bin="mid_acc")
    arch_high = first_row(acc_bins, variation_source="architecture", acc_bin="high_acc")
    noise_low = first_row(acc_bins, variation_source="noise", acc_bin="low_acc")
    noise_mid = first_row(acc_bins, variation_source="noise", acc_bin="mid_acc")
    noise_high = first_row(acc_bins, variation_source="noise", acc_bin="high_acc")

    key_summary_cols = [
        "variation_source",
        "n_rows",
        "clean_acc_mean",
        "clean_acc_median",
        "difficulty_mean",
        "transfer_rate_mean",
        "transfer_rate_median",
        "stealth_avg_mean",
        "stealth_avg_median",
    ]
    key_corr_cols = [
        "group_type",
        "group_name",
        "n_rows",
        "pearson_clean_acc_transfer_rate",
        "spearman_clean_acc_transfer_rate",
        "pearson_transfer_rate_stealth_avg",
        "spearman_transfer_rate_stealth_avg",
    ]
    key_acc_cols = [
        "variation_source",
        "acc_bin",
        "n_rows",
        "clean_acc_mean",
        "difficulty_mean",
        "transfer_rate_median",
        "stealth_avg_median",
        "spearman_transfer_stealth",
    ]
    key_attack_cols = [
        "variation_source",
        "attack_type",
        "n_rows",
        "clean_acc_mean",
        "difficulty_mean",
        "transfer_rate_mean",
        "transfer_rate_median",
        "stealth_avg_mean",
        "stealth_avg_median",
    ]

    lines = [
        "# ACC 难度联合分析报告：噪声实验与架构实验",
        "",
        "## 0. 先看结论",
        "",
        "这份报告把两条证据链放在一起：",
        "",
        "- `architecture`：完整旧基线 + SmallCNN/ResNet34 架构补充，回答“不同模型/数据集生态下现象是否仍存在”。",
        "- `noise`：CIFAR-10 + SmallCNN 加噪实验，回答“同模型同数据集下主动改变输入难度后关系如何变化”。",
        "",
        "最稳妥的联合结论是：",
        "",
        f"- 两条证据链都显示 `transfer_rate` 与 `stealth_avg` 为负相关：architecture Spearman=`{metric(arch_corr, 'spearman_transfer_rate_stealth_avg')}`，noise Spearman=`{metric(noise_corr, 'spearman_transfer_rate_stealth_avg')}`。",
        f"- 两条证据链的均值水平不同：architecture 平均 stealth_avg=`{metric(arch_summary, 'stealth_avg_mean')}`，noise 平均 stealth_avg=`{metric(noise_summary, 'stealth_avg_mean')}`；noise 的隐蔽性均值明显更低。",
        f"- 两条证据链里 clean_acc 与 transfer_rate 的方向不同：architecture Pearson=`{metric(arch_corr, 'pearson_clean_acc_transfer_rate')}`，noise Pearson=`{metric(noise_corr, 'pearson_clean_acc_transfer_rate')}`。",
        "- 因此，联合分析不能直接写成“ACC 是唯一原因”。更准确的说法是：difficulty/ACC 会参与调节 transfer-stealth 关系，但 noise 和 architecture 是两种不同 intervention。",
        "",
        "## 1. 建议优先看的图",
        "",
        "| 优先级 | 图 | 用途 | 汇报时怎么说 |",
        "|---:|---|---|---|",
        "| 1 | `combined_metric_overview.png` | 比较 architecture/noise 的均值水平 | 先说明两条证据链不是同一分布 |",
        "| 2 | `combined_transfer_stealth_facets.png` | 看两条证据链各自的 transfer-stealth 点云 | 两边都负相关，但点云结构不同 |",
        "| 3 | `combined_difficulty_relationships.png` | 看 difficulty 与 transfer/stealth 的方向 | architecture 和 noise 的 difficulty effect 不等价 |",
        "| 4 | `combined_binned_median_trend.png` | 用分箱趋势展示关系 | 普通散点不明显时看这张 |",
        "| 5 | `combined_acc_bin_spearman.png` | 看 ACC 分层相关性 | 说明不同难度区间 tradeoff 强度不同 |",
        "| 6 | `combined_attack_heatmap.png` | 看 attack-dependent 差异 | 说明不能只报 overall |",
        "",
        md_image("arch_acc_analysis/figures/combined_metric_overview.png", "Combined metric overview"),
        "",
        md_image("arch_acc_analysis/figures/combined_transfer_stealth_facets.png", "Combined transfer-stealth facets"),
        "",
        md_image("arch_acc_analysis/figures/combined_difficulty_relationships.png", "Combined difficulty relationships"),
        "",
        md_image("arch_acc_analysis/figures/combined_binned_median_trend.png", "Combined binned median trend"),
        "",
        md_image("arch_acc_analysis/figures/combined_acc_bin_spearman.png", "Combined ACC-bin Spearman"),
        "",
        md_image("arch_acc_analysis/figures/combined_attack_heatmap.png", "Combined attack heatmap"),
        "",
        "HTML 展示页：`analysis-transfer-asr2/ACC_DIFFICULTY_NOISE_ARCH_COMBINED_DASHBOARD_CN.html`。",
        "",
        "## 2. 数据来源与总体汇总",
        "",
        limited_table(summary, key_summary_cols, 10, ["variation_source"]),
        "",
        "## 3. 关键发现",
        "",
        f"- 架构/模型实验共有 `{metric(arch_summary, 'n_rows', 0)}` 条主分析记录，平均 clean ACC=`{metric(arch_summary, 'clean_acc_mean')}`，平均 transfer_rate=`{metric(arch_summary, 'transfer_rate_mean')}`，平均 stealth_avg=`{metric(arch_summary, 'stealth_avg_mean')}`。",
        f"- 噪声实验共有 `{metric(noise_summary, 'n_rows', 0)}` 条主分析记录，平均 clean ACC=`{metric(noise_summary, 'clean_acc_mean')}`，平均 transfer_rate=`{metric(noise_summary, 'transfer_rate_mean')}`，平均 stealth_avg=`{metric(noise_summary, 'stealth_avg_mean')}`。",
        f"- 全体 transfer_rate 中位数为 `{metric(diag_all, 'transfer_rate_median')}`，落在 `[0.9, 1.1]` 的比例为 `{metric(diag_all, 'share_transfer_0p9_1p1')}`。所以普通散点图容易显得趋势不明显，分箱趋势更适合汇报。",
        "",
        "## 4. 相关性对比",
        "",
        limited_table(corr_df[corr_df["group_type"].isin(["all", "variation_source"])], key_corr_cols, 10, ["group_type", "group_name"]),
        "",
        "需要重点解释的一点：architecture 中 clean_acc 与 transfer_rate 是负相关，noise 中是正相关。这说明“换模型/换数据集”和“给同一模型加噪声”不是同一种 difficulty intervention。",
        "",
        "## 5. ACC 分层结果",
        "",
        limited_table(acc_bins, key_acc_cols, 20, ["variation_source", "acc_bin"]),
        "",
        "最适合汇报的三行对照：",
        "",
        "| source | low_acc | mid_acc | high_acc |",
        "|---|---:|---:|---:|",
        f"| architecture | {metric(arch_low, 'spearman_transfer_stealth')} | {metric(arch_mid, 'spearman_transfer_stealth')} | {metric(arch_high, 'spearman_transfer_stealth')} |",
        f"| noise | {metric(noise_low, 'spearman_transfer_stealth')} | {metric(noise_mid, 'spearman_transfer_stealth')} | {metric(noise_high, 'spearman_transfer_stealth')} |",
        "",
        "如果 noise 的低 ACC 层变弱或接近消失，而 high ACC 层仍强，汇报时可以说：在控制模型/数据集后，输入难度确实会改变 tradeoff 强度；但方向不是简单线性。",
        "",
        "## 6. 攻击类型分层",
        "",
        limited_table(attack_summary, key_attack_cols, 80, ["attack_type", "variation_source"]),
        "",
        "攻击类型是强混杂因素。联合报告里不能只说 architecture vs noise 的总体均值，还要说明 badnet/adaptive_patch/WaNet/SIG/upgd 等攻击的落点不同。",
        "",
        "## 7. 分箱趋势和 rank 趋势",
        "",
        "- `combined_acc_effect_binned_median_trend.csv`：对应 `combined_binned_median_trend.png`。",
        "- `combined_acc_effect_rank_binned_trend.csv`：对应 `combined_rank_binned_trend.png`。",
        "",
        "分箱趋势预览：",
        "",
        limited_table(trend, ["variation_source", "bin_id", "n_rows", "transfer_rate_median", "stealth_avg_median", "difficulty_mean"], 30, ["variation_source", "bin_id"]),
        "",
        "Rank 趋势预览：",
        "",
        limited_table(rank_trend, ["variation_source", "bin_id", "n_rows", "transfer_rank_median", "stealth_rank_median"], 30, ["variation_source", "bin_id"]),
        "",
    ]

    try:
        import statsmodels.formula.api as smf

        model = smf.ols("stealth_avg ~ transfer_rate * difficulty + C(variation_source) + C(attack_type)", data=combined).fit()
        regression_path.write_text(str(model.summary()) + "\n", encoding="utf-8")
        lines.extend(
            [
                "## 回归结果摘要",
                "",
                f"- `transfer_rate:difficulty` 系数：`{model.params.get('transfer_rate:difficulty', float('nan'))}`",
                f"- `transfer_rate:difficulty` p 值：`{model.pvalues.get('transfer_rate:difficulty', float('nan'))}`",
                "- 在这个联合回归中，交互项为正表示 difficulty 增大时，transfer_rate 对 stealth_avg 的负向斜率会被削弱。但这个结果混合了 architecture 与 noise 两类变化来源，不能单独解释为 ACC 的纯因果效应。",
                "",
                "完整回归表见 `arch_acc_analysis/combined_acc_effect_regression.txt`。",
            ]
        )
    except Exception as exc:
        regression_path.write_text(f"Regression failed: {exc}\n", encoding="utf-8")
        lines.extend(["## 回归结果摘要", "", f"回归失败：`{exc}`"])

    lines.extend(
        [
            "",
            "## 9. 解读原则",
            "",
            "- 噪声实验更接近同模型同数据集下的 difficulty intervention。",
            "- 架构实验更接近不同模型和数据集设置下的生态对照。",
            "- 两者趋势一致时，可以加强“ACC/难度影响 transfer-stealth 关系”的证据链。",
            "- 两者趋势不一致时，应优先检查 attack type、source ASR 长尾和 defense-specific 差异。",
            "",
            "## 10. 汇报建议",
            "",
            "建议按 5 页讲：",
            "",
            "1. **研究设计**：先说明两条证据链不同。architecture 是模型生态对照，noise 是更接近控制变量的 difficulty intervention。",
            "2. **共同现象**：两条证据链都显示 transfer-stealth 负相关，但强度不同。",
            "3. **关键差异**：difficulty/ACC 对 transfer_rate 的方向不同，不能把两批数据混成一个简单因果结论。",
            "4. **难度分层**：用 ACC-bin Spearman 图说明 tradeoff 强度会随难度区间变化。",
            "5. **攻击混杂**：用 attack heatmap 说明不同攻击机制决定了点落在哪个区域。",
            "",
            "可以直接使用的汇报话术：",
            "",
            "```text",
            "联合分析不是为了把噪声实验和架构实验简单混成一个大样本，",
            "而是把它们作为两条互补证据链：",
            "噪声实验更接近同模型同数据集下的难度干预，",
            "架构实验更接近不同模型和数据集生态下的外部验证。",
            "两条证据链都显示迁移性和隐蔽性整体负相关，",
            "但 clean ACC 与 transfer_rate 的方向并不一致，说明 difficulty effect 不是单一线性因果。",
            "因此更稳妥的结论是：ACC/任务难度会调节 transfer-stealth 关系，",
            "但该调节受到攻击类型、模型结构和检测器响应共同影响。",
            "```",
            "",
            "## 11. 每个输出文件的作用",
            "",
            "- `combined_acc_effect_summary.csv`：看 architecture/noise 两条证据链的总体均值差异。",
            "- `combined_acc_effect_rows.csv`：保存逐实验明细，可用于后续重新画图或做更复杂回归。",
            "- `combined_acc_effect_correlations.csv`：整体、证据链、攻击类型下的相关性。",
            "- `combined_acc_effect_acc_bins.csv`：按 clean ACC 分层后的 transfer-stealth 相关性。",
            "- `combined_acc_effect_attack_summary.csv`：按攻击类型拆解两条证据链。",
            "- `combined_acc_effect_plot_diagnostics.csv`：解释普通散点图为什么不明显。",
            "- `combined_acc_effect_binned_median_trend.csv`：分箱中位数趋势。",
            "- `combined_acc_effect_rank_binned_trend.csv`：rank 分箱趋势。",
            "- `combined_acc_effect_regression.txt`：保存联合回归完整统计表，主要查看 `transfer_rate:difficulty` 和 `C(variation_source)`。",
            "- `ACC_DIFFICULTY_NOISE_ARCH_COMBINED_DASHBOARD_CN.html`：联合分析静态展示页。",
            "- `ARCH_ACC_TRANSFER_STEALTH_RESULT_REPORT_CN.md`：只讨论架构/模型实验，是和 baseline 结果对照时的主报告。",
            "- `ACC_DIFFICULTY_NOISE_ARCH_COMBINED_REPORT_CN.md`：把噪声实验与架构实验放在同一个解释框架中，主要用于写论文讨论和后续实验设计。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_combined_html_dashboard(
        summary,
        corr_df,
        acc_bins,
        attack_summary,
        diagnostics,
        Path("analysis-transfer-asr2/ACC_DIFFICULTY_NOISE_ARCH_COMBINED_DASHBOARD_CN.html"),
    )


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "nan"


def top_table(df: pd.DataFrame, n: int = 12) -> str:
    if df.empty:
        return "无数据"
    return df.head(n).to_markdown(index=False)


def first_row(df: pd.DataFrame, **filters: str) -> Optional[pd.Series]:
    subset = df
    for key, value in filters.items():
        if key not in subset.columns:
            return None
        subset = subset[subset[key] == value]
    if subset.empty:
        return None
    return subset.iloc[0]


def metric(row: Optional[pd.Series], column: str, digits: int = 4) -> str:
    if row is None or column not in row:
        return "nan"
    return fmt_float(row[column], digits)


def signed_metric(row: Optional[pd.Series], column: str, digits: int = 4) -> str:
    if row is None or column not in row or pd.isna(row[column]):
        return "nan"
    return f"{float(row[column]):+.{digits}f}"


def sci_metric(row: Optional[pd.Series], column: str, digits: int = 2) -> str:
    if row is None or column not in row or pd.isna(row[column]):
        return "nan"
    return f"{float(row[column]):.{digits}e}"


def regression_interaction_note(primary: pd.DataFrame) -> List[str]:
    try:
        import statsmodels.formula.api as smf

        model = smf.ols(
            "stealth_avg ~ transfer_rate * difficulty + C(dataset) + C(arch_base) + C(attack_family) + C(poison_rate)",
            data=primary,
        ).fit()
        coef = model.params.get("transfer_rate:difficulty", float("nan"))
        pvalue = model.pvalues.get("transfer_rate:difficulty", float("nan"))
        direction = "更负" if coef < 0 else "更弱或更正向"
        return [
            f"- 主回归中 `transfer_rate:difficulty` 系数为 `{fmt_float(coef)}`，p 值为 `{pvalue:.3e}`。在这批架构/模型结果里，这表示 difficulty 越高，transfer-stealth 的线性斜率越{direction}。",
            "- 但这个回归是 pooled 分析，仍同时包含 dataset、arch、attack、poison_rate 差异；因此它是“调节证据”，不是单独证明 ACC 是唯一原因。",
        ]
    except Exception as exc:
        return [f"- 回归摘要生成失败：`{exc}`。完整回归文件仍可查看生成日志。"]


def md_image(path: str, title: str) -> str:
    return f"![{title}]({path})"


def limited_table(df: pd.DataFrame, columns: Sequence[str], n: int = 20, sort_by: Optional[Sequence[str]] = None) -> str:
    if df.empty:
        return "无数据"
    cols = [c for c in columns if c in df.columns]
    out = df.copy()
    if sort_by:
        keys = [c for c in sort_by if c in out.columns]
        if keys:
            out = out.sort_values(keys)
    return out[cols].head(n).to_markdown(index=False)


def html_table(df: pd.DataFrame, columns: Sequence[str], n: int = 20, sort_by: Optional[Sequence[str]] = None) -> str:
    if df.empty:
        return "<p>无数据</p>"
    cols = [c for c in columns if c in df.columns]
    out = df.copy()
    if sort_by:
        keys = [c for c in sort_by if c in out.columns]
        if keys:
            out = out.sort_values(keys)
    out = out[cols].head(n).copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    header = "".join(f"<th>{html.escape(str(col))}</th>" for col in out.columns)
    rows = []
    for _, row in out.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def image_card(path: str, title: str, text: str) -> str:
    return (
        '<article class="figure-card">'
        f'<h3>{html.escape(title)}</h3>'
        f'<img src="{html.escape(path)}" alt="{html.escape(title)}">'
        f'<p>{html.escape(text)}</p>'
        '</article>'
    )


def baseline_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["result_group"] == "baseline_full") & df["include_main_analysis"]].copy()


def baseline_summary_by_dataset_arch(sub: pd.DataFrame) -> pd.DataFrame:
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["dataset", "arch_base", "dataset_arch"], dropna=False)
        .agg(
            n_rows=("folder_name", "count"),
            clean_acc_mean=("clean_acc", "mean"),
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
        .reset_index()
        .sort_values(["dataset", "arch_base"])
    )


def baseline_summary_by_attack(sub: pd.DataFrame) -> pd.DataFrame:
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["dataset", "attack_family"], dropna=False)
        .agg(
            n_rows=("folder_name", "count"),
            clean_acc_mean=("clean_acc", "mean"),
            source_asr_mean=("source_asr", "mean"),
            transfer_asr_mean=("transfer_asr", "mean"),
            transfer_rate_mean=("transfer_rate", "mean"),
            stealth_avg_mean=("stealth_avg", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "attack_family"])
    )


def baseline_correlations(sub: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for threshold in [0.05, 0.10]:
        cur = sub[sub["source_asr"] >= threshold].copy()
        if cur.empty:
            continue
        label = f"source_asr>={threshold:g}"
        records.append(corr_record(cur, "all", "all", label))
        for col in ["dataset", "arch_base", "dataset_arch", "attack_family"]:
            for name, group in cur.groupby(col, dropna=False):
                if len(group) >= 2:
                    records.append(corr_record(group, col, str(name), label))
    return pd.DataFrame(records)


def baseline_acc_bins(sub: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    def add_records(group_type: str, dataset: str, data: pd.DataFrame) -> None:
        if len(data) < 3:
            return
        try:
            bins = pd.qcut(data["clean_acc"], 3, labels=["low_acc", "mid_acc", "high_acc"], duplicates="drop")
        except Exception:
            return
        for bin_name, group in data.assign(acc_bin=bins).groupby("acc_bin", observed=False):
            records.append(
                {
                    "group_type": group_type,
                    "dataset": dataset,
                    "acc_bin": str(bin_name),
                    "n_rows": int(len(group)),
                    "clean_acc_min": group["clean_acc"].min(),
                    "clean_acc_max": group["clean_acc"].max(),
                    "clean_acc_mean": group["clean_acc"].mean(),
                    "transfer_rate_mean": group["transfer_rate"].mean(),
                    "transfer_rate_median": group["transfer_rate"].median(),
                    "stealth_avg_mean": group["stealth_avg"].mean(),
                    "stealth_avg_median": group["stealth_avg"].median(),
                    "pearson_transfer_stealth": pearson(group["transfer_rate"], group["stealth_avg"]),
                    "spearman_transfer_stealth": spearman(group["transfer_rate"], group["stealth_avg"]),
                }
            )

    add_records("global_acc_bin", "all", sub)
    for dataset, group in sub.groupby("dataset", dropna=False):
        add_records("dataset_acc_bin", str(dataset), group)
    return pd.DataFrame(records)


def baseline_defense_breakdown(sub: pd.DataFrame) -> pd.DataFrame:
    records: List[pd.DataFrame] = []
    for defense in ["sentinet", "strip", "scaleup", "ibd_psc"]:
        cols = ["dataset", "arch_base", f"{defense}_tpr", f"{defense}_auc", f"stealth_{defense}"]
        tmp = sub[cols].copy()
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
        long_df.groupby(["dataset", "arch_base", "defense"], dropna=False)
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
        .sort_values(["dataset", "arch_base", "defense"])
    )


def add_baseline_plot_bins(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub.copy()
    if len(out) >= 3:
        try:
            out["global_acc_bin"] = pd.qcut(out["clean_acc"], 3, labels=["low_acc", "mid_acc", "high_acc"], duplicates="drop")
        except Exception:
            out["global_acc_bin"] = None
    else:
        out["global_acc_bin"] = None
    return out


def plot_baseline_scatter_clipped(sub: pd.DataFrame, figures_dir: Path) -> None:
    if sub.empty:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    data = add_baseline_plot_bins(sub.dropna(subset=["transfer_rate", "stealth_avg"]).copy())
    if data.empty:
        return

    data["transfer_rate_plot"] = data["transfer_rate"].clip(upper=2.0)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    colors = {"cifar10": "#1f77b4", "mnistm": "#2ca02c", "tiny_imagenet": "#ff7f0e"}
    for dataset, group in data.groupby("dataset", dropna=False):
        ax.scatter(
            group["transfer_rate_plot"],
            group["stealth_avg"],
            s=15,
            alpha=0.45,
            label=str(dataset),
            color=colors.get(str(dataset)),
        )
    ax.set_xlabel("transfer_rate = transfer_asr^2 / source_asr (clipped at 2.0 for display)")
    ax.set_ylabel("stealth_avg = mean(1 - TPR)")
    ax.set_title("Baseline: transfer vs stealth by dataset")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_transfer_vs_stealth_by_dataset_clipped.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    colors = {"low_acc": "#d62728", "mid_acc": "#ff7f0e", "high_acc": "#2ca02c"}
    for acc_bin, group in data.groupby("global_acc_bin", observed=False):
        ax.scatter(
            group["transfer_rate_plot"],
            group["stealth_avg"],
            s=15,
            alpha=0.45,
            label=str(acc_bin),
            color=colors.get(str(acc_bin)),
        )
    ax.set_xlabel("transfer_rate = transfer_asr^2 / source_asr (clipped at 2.0 for display)")
    ax.set_ylabel("stealth_avg = mean(1 - TPR)")
    ax.set_title("Baseline: transfer vs stealth by ACC bin")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_transfer_vs_stealth_by_acc_bin_clipped.png", dpi=180)
    plt.close(fig)


def plot_baseline_facets_by_dataset(sub: pd.DataFrame, figures_dir: Path) -> None:
    data = sub.dropna(subset=["transfer_rate", "stealth_avg"]).copy()
    if data.empty:
        return
    data["transfer_rate_plot"] = data["transfer_rate"].clip(upper=2.0)
    datasets = ["cifar10", "mnistm", "tiny_imagenet"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, dataset in zip(axes, datasets):
        group = data[data["dataset"] == dataset].copy()
        if group.empty:
            ax.set_visible(False)
            continue
        ax.scatter(group["transfer_rate_plot"], group["stealth_avg"], s=14, alpha=0.5, color="#1f77b4")
        sp = spearman(group["transfer_rate"], group["stealth_avg"])
        pe = pearson(group["transfer_rate"], group["stealth_avg"])
        ax.set_title(f"{dataset}\nPearson={pe:.3f}, Spearman={sp:.3f}")
        ax.set_xlabel("transfer_rate (clipped at 2.0)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("stealth_avg = mean(1 - TPR)")
    fig.suptitle("Baseline: dataset facets reduce cross-domain mixing")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figures_dir / "baseline_transfer_vs_stealth_dataset_facets_clipped.png", dpi=180)
    plt.close(fig)


def baseline_binned_trend(sub: pd.DataFrame, by: str = "dataset", bins: int = 8) -> pd.DataFrame:
    data = sub.dropna(subset=["transfer_rate", "stealth_avg"]).copy()
    records: List[Dict[str, Any]] = []
    groups = [("all", data)] if by == "all" else data.groupby(by, dropna=False)
    for name, group in groups:
        if len(group) < bins:
            continue
        try:
            qbins = pd.qcut(group["transfer_rate"], bins, duplicates="drop")
        except Exception:
            continue
        for bin_id, (_, bg) in enumerate(group.assign(transfer_bin=qbins).groupby("transfer_bin", observed=False), start=1):
            if bg.empty:
                continue
            records.append(
                {
                    by: str(name),
                    "bin_id": bin_id,
                    "n_rows": int(len(bg)),
                    "transfer_rate_median": bg["transfer_rate"].median(),
                    "transfer_rate_mean": bg["transfer_rate"].mean(),
                    "stealth_avg_median": bg["stealth_avg"].median(),
                    "stealth_avg_mean": bg["stealth_avg"].mean(),
                    "stealth_avg_q25": bg["stealth_avg"].quantile(0.25),
                    "stealth_avg_q75": bg["stealth_avg"].quantile(0.75),
                }
            )
    return pd.DataFrame(records)


def plot_baseline_binned_trend(trend: pd.DataFrame, figures_dir: Path) -> None:
    if trend.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    colors = {"cifar10": "#1f77b4", "mnistm": "#2ca02c", "tiny_imagenet": "#ff7f0e"}
    for dataset, group in trend.groupby("dataset", dropna=False):
        group = group.sort_values("transfer_rate_median")
        x = group["transfer_rate_median"].clip(upper=2.0).to_numpy()
        y = group["stealth_avg_median"].to_numpy()
        y25 = group["stealth_avg_q25"].to_numpy()
        y75 = group["stealth_avg_q75"].to_numpy()
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            label=str(dataset),
            color=colors.get(str(dataset)),
        )
        ax.fill_between(
            x,
            y25,
            y75,
            alpha=0.12,
            color=colors.get(str(dataset)),
        )
    ax.set_xlabel("median transfer_rate in quantile bin (clipped at 2.0 for display)")
    ax.set_ylabel("median stealth_avg with IQR band")
    ax.set_title("Baseline: binned median trend by dataset")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_binned_median_trend_by_dataset.png", dpi=180)
    plt.close(fig)


def baseline_rank_trend(sub: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    data = sub.dropna(subset=["transfer_rate", "stealth_avg"]).copy()
    if data.empty:
        return pd.DataFrame()
    data["transfer_rank_pct"] = data["transfer_rate"].rank(method="average", pct=True)
    data["stealth_rank_pct"] = data["stealth_avg"].rank(method="average", pct=True)
    records: List[Dict[str, Any]] = []
    for dataset, group in data.groupby("dataset", dropna=False):
        if len(group) < bins:
            continue
        try:
            qbins = pd.qcut(group["transfer_rank_pct"], bins, duplicates="drop")
        except Exception:
            continue
        for bin_id, (_, bg) in enumerate(group.assign(rank_bin=qbins).groupby("rank_bin", observed=False), start=1):
            records.append(
                {
                    "dataset": str(dataset),
                    "bin_id": bin_id,
                    "n_rows": int(len(bg)),
                    "transfer_rank_median": bg["transfer_rank_pct"].median(),
                    "stealth_rank_median": bg["stealth_rank_pct"].median(),
                }
            )
    return pd.DataFrame(records)


def plot_baseline_rank_trend(rank_trend: pd.DataFrame, figures_dir: Path) -> None:
    if rank_trend.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    colors = {"cifar10": "#1f77b4", "mnistm": "#2ca02c", "tiny_imagenet": "#ff7f0e"}
    for dataset, group in rank_trend.groupby("dataset", dropna=False):
        group = group.sort_values("transfer_rank_median")
        ax.plot(
            group["transfer_rank_median"].to_numpy(),
            group["stealth_rank_median"].to_numpy(),
            marker="o",
            linewidth=1.8,
            label=str(dataset),
            color=colors.get(str(dataset)),
        )
    ax.set_xlabel("rank percentile of transfer_rate")
    ax.set_ylabel("median rank percentile of stealth_avg")
    ax.set_title("Baseline: rank-binned transfer vs stealth trend")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_rank_binned_trend_by_dataset.png", dpi=180)
    plt.close(fig)


def baseline_plot_diagnostics(sub: pd.DataFrame) -> pd.DataFrame:
    data = sub.dropna(subset=["transfer_rate", "stealth_avg", "clean_acc"]).copy()
    if data.empty:
        return pd.DataFrame()
    rows = []
    for name, group in [("all", data), *list(data.groupby("dataset", dropna=False))]:
        label = str(name)
        rows.append(
            {
                "group": label,
                "n_rows": int(len(group)),
                "transfer_rate_p25": group["transfer_rate"].quantile(0.25),
                "transfer_rate_median": group["transfer_rate"].median(),
                "transfer_rate_p75": group["transfer_rate"].quantile(0.75),
                "transfer_rate_p95": group["transfer_rate"].quantile(0.95),
                "transfer_rate_max": group["transfer_rate"].max(),
                "share_transfer_0p9_1p1": float(((group["transfer_rate"] >= 0.9) & (group["transfer_rate"] <= 1.1)).mean()),
                "share_transfer_gt_1p5": float((group["transfer_rate"] > 1.5).mean()),
                "pearson_transfer_stealth": pearson(group["transfer_rate"], group["stealth_avg"]),
                "spearman_transfer_stealth": spearman(group["transfer_rate"], group["stealth_avg"]),
            }
        )
    return pd.DataFrame(rows)


def write_baseline_figure_analysis(
    sub: pd.DataFrame,
    diagnostics: pd.DataFrame,
    trend: pd.DataFrame,
    rank_trend: pd.DataFrame,
    report_path: Path,
) -> None:
    all_diag = first_row(diagnostics, group="all")
    lines = [
        "# 完整旧基线图表分析说明",
        "",
        "## 1. 为什么普通散点图看不出很清楚的趋势",
        "",
        "完整旧基线的 `transfer_rate` 大量集中在 1 附近，同时少数极端值会拉长横轴。因此普通散点图会出现一条很厚的竖带，看起来不像一条清楚的斜线。",
        "",
        "核心分布诊断：",
        "",
        f"- 主分析样本数：`{metric(all_diag, 'n_rows', 0)}`",
        f"- transfer_rate 中位数：`{metric(all_diag, 'transfer_rate_median')}`",
        f"- transfer_rate 25%/75% 分位：`{metric(all_diag, 'transfer_rate_p25')}` / `{metric(all_diag, 'transfer_rate_p75')}`",
        f"- transfer_rate 95% 分位和最大值：`{metric(all_diag, 'transfer_rate_p95')}` / `{metric(all_diag, 'transfer_rate_max')}`",
        f"- transfer_rate 落在 `[0.9, 1.1]` 的比例：`{metric(all_diag, 'share_transfer_0p9_1p1')}`",
        f"- transfer_rate 大于 `1.5` 的比例：`{metric(all_diag, 'share_transfer_gt_1p5')}`",
        "",
        "这说明：视觉上看不清，不等于没有统计关系。更合适的读法是结合分组相关性、分面图和分箱中位数趋势图。",
        "",
        "## 2. 新生成的图",
        "",
        "- `figures/baseline_transfer_vs_stealth_by_dataset_clipped.png`：按 dataset 上色的裁剪散点图，用于看完整点云分布。",
        "- `figures/baseline_transfer_vs_stealth_by_acc_bin_clipped.png`：按 ACC bin 上色的裁剪散点图，用于展示不同 ACC 区间混在一起的情况。",
        "- `figures/baseline_transfer_vs_stealth_dataset_facets_clipped.png`：按 dataset 分面，减少跨数据集混杂。",
        "- `figures/baseline_binned_median_trend_by_dataset.png`：按 transfer_rate 分箱后的中位数趋势，是最适合汇报趋势的图。",
        "- `figures/baseline_rank_binned_trend_by_dataset.png`：按 transfer_rate 排名分箱后的趋势，用于解释 Spearman 排序相关。",
        "",
        "## 3. 图表结论",
        "",
        "- 普通散点图只适合说明数据分布，不适合单独证明趋势。",
        "- 分面图显示 CIFAR-10、Tiny-ImageNet、MNIST-M 的点云结构不同，混在一起会弱化肉眼趋势。",
        "- 分箱中位数图更适合汇报：它减少了点云重叠和极端值影响，能更清楚展示不同 dataset 中 transfer_rate 与 stealth_avg 的整体方向。",
        "- rank 分箱图对应 Spearman 相关，适合解释“这更像排序/分组关系，而不是强线性关系”。",
        "",
        "## 4. 汇报时建议采用的表述",
        "",
        "```text",
        "完整旧基线的普通散点图视觉趋势不强，主要是因为 transfer_rate 大量集中在 1 附近，且少数极端值拉长横轴；同时数据混合了多个 dataset、arch 和 attack。",
        "因此我们不把普通散点图作为唯一证据，而是结合分组相关性、ACC 分层结果和分箱趋势图。",
        "完整基线中整体 Spearman 为负，各 dataset 内也为负，说明 transfer-stealth 关系更像分组/排序型负相关，而不是简单线性斜线。",
        "```",
        "",
        "## 5. 诊断表",
        "",
        diagnostics.to_markdown(index=False) if not diagnostics.empty else "无数据",
        "",
        "## 6. 分箱中位数趋势表",
        "",
        trend.to_markdown(index=False) if not trend.empty else "无数据",
        "",
        "## 7. Rank 分箱趋势表",
        "",
        rank_trend.to_markdown(index=False) if not rank_trend.empty else "无数据",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_baseline_figures_and_analysis(sub: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_baseline_scatter_clipped(sub, figures_dir)
    plot_baseline_facets_by_dataset(sub, figures_dir)
    trend = baseline_binned_trend(sub, by="dataset", bins=8)
    rank_trend = baseline_rank_trend(sub, bins=10)
    diagnostics = baseline_plot_diagnostics(sub)
    trend.to_csv(output_dir / "baseline_full_binned_median_trend_by_dataset.csv", index=False)
    rank_trend.to_csv(output_dir / "baseline_full_rank_binned_trend_by_dataset.csv", index=False)
    diagnostics.to_csv(output_dir / "baseline_full_plot_diagnostics.csv", index=False)
    plot_baseline_binned_trend(trend, figures_dir)
    plot_baseline_rank_trend(rank_trend, figures_dir)
    write_baseline_figure_analysis(
        sub,
        diagnostics,
        trend,
        rank_trend,
        Path("analysis-transfer-asr2/BASELINE_FULL_FIGURE_ANALYSIS_CN.md"),
    )


def write_baseline_regression(sub: pd.DataFrame, path: Path) -> List[str]:
    try:
        import statsmodels.formula.api as smf

        model = smf.ols(
            "stealth_avg ~ transfer_rate * difficulty + C(dataset) + C(arch_base) + C(attack_family) + C(poison_rate)",
            data=sub,
        ).fit()
        path.write_text(str(model.summary()) + "\n", encoding="utf-8")
        coef = model.params.get("transfer_rate:difficulty", float("nan"))
        pvalue = model.pvalues.get("transfer_rate:difficulty", float("nan"))
        direction = "更负" if coef < 0 else "更弱或更正向"
        return [
            f"- `transfer_rate:difficulty` 系数：`{fmt_float(coef)}`，p 值：`{pvalue:.3e}`。",
            f"- 在完整旧 baseline 数据里，该交互项表示 difficulty 越高时，transfer-stealth 线性斜率越{direction}。",
        ]
    except Exception as exc:
        path.write_text(f"Regression failed: {exc}\n", encoding="utf-8")
        return [f"- 回归失败：`{exc}`。"]


def write_baseline_full_report(df: pd.DataFrame, output_dir: Path, report_path: Path) -> None:
    sub = baseline_subset(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_dir / "baseline_full_acc_transfer_stealth_rows.csv", index=False)

    summary = baseline_summary_by_dataset_arch(sub)
    attack_summary = baseline_summary_by_attack(sub)
    corr = baseline_correlations(sub)
    acc_bins = baseline_acc_bins(sub)
    defense = baseline_defense_breakdown(sub)

    summary.to_csv(output_dir / "baseline_full_summary_by_dataset_arch.csv", index=False)
    attack_summary.to_csv(output_dir / "baseline_full_summary_by_attack.csv", index=False)
    corr.to_csv(output_dir / "baseline_full_correlations.csv", index=False)
    acc_bins.to_csv(output_dir / "baseline_full_acc_bins.csv", index=False)
    defense.to_csv(output_dir / "baseline_full_defense_breakdown.csv", index=False)
    regression_note = write_baseline_regression(sub, output_dir / "baseline_full_regression.txt")
    generate_baseline_figures_and_analysis(sub, output_dir)

    all_corr = first_row(corr, group_type="all", group_name="all", source_asr_filter="source_asr>=0.05")
    all_corr_010 = first_row(corr, group_type="all", group_name="all", source_asr_filter="source_asr>=0.1")
    cifar_corr = first_row(corr, group_type="dataset", group_name="cifar10", source_asr_filter="source_asr>=0.05")
    tiny_corr = first_row(corr, group_type="dataset", group_name="tiny_imagenet", source_asr_filter="source_asr>=0.05")
    mnist_corr = first_row(corr, group_type="dataset", group_name="mnistm", source_asr_filter="source_asr>=0.05")
    high_acc = first_row(acc_bins, group_type="global_acc_bin", dataset="all", acc_bin="high_acc")
    mid_acc = first_row(acc_bins, group_type="global_acc_bin", dataset="all", acc_bin="mid_acc")
    low_acc = first_row(acc_bins, group_type="global_acc_bin", dataset="all", acc_bin="low_acc")

    total_baseline_rows = int((df["result_group"] == "baseline_full").sum())
    complete_rows = int(((df["result_group"] == "baseline_full") & df["complete_defense_results"]).sum())
    valid_rows = int(((df["result_group"] == "baseline_full") & df["valid_transfer_rate"]).sum())
    diagnostics = pd.read_csv(output_dir / "baseline_full_plot_diagnostics.csv")
    plot_diag_all = first_row(diagnostics, group="all")

    lines = [
        "# 完整旧基线 ACC-迁移性-隐蔽性结果总结",
        "",
        "本文档总结 `/workspace/backdoor-toolbox-new1/poisoned_train_set1` 中已有完整旧基线结果。它的作用类似噪声分析文档中的“完整旧基线参照”：先回答原有大规模实验中 transfer-stealth 关系是否存在，再为后续噪声实验和新模型架构实验提供参照背景。",
        "",
        "字段、图表和逐实验明细主要保存在：",
        "",
        "```text",
        "analysis-transfer-asr2/baseline_full_analysis/",
        "```",
        "",
        "## 1. 研究问题",
        "",
        "这份完整旧基线不是为了单独回答某一个新模型或某一种噪声的效果，而是回答三个基础问题：",
        "",
        "```text",
        "1. 在原有多数据集、多模型、多攻击实验里，迁移性和隐蔽性是否整体存在 tradeoff？",
        "2. 这种 tradeoff 是否会随数据集、ACC 区间、攻击方法和检测器改变？",
        "3. 后续噪声实验和 SmallCNN/ResNet34 架构补充，应该和什么基线结果对照？",
        "```",
        "",
        "使用的定义和当前论文问题保持一致：",
        "",
        "```text",
        "difficulty = 1 - clean_acc",
        "transfer_rate = transfer_asr^2 / source_asr",
        "stealth_avg = mean(1 - TPR)",
        "```",
        "",
        "其中：",
        "",
        "- `clean_acc` 是源域测试准确率；",
        "- `source_asr` 是源域 ASR；",
        "- `transfer_asr` 是迁移域 ASR；",
        "- `transfer_rate` 是当前固定使用的迁移性指标；",
        "- `stealth_avg` 来自源域四个检测方法：`SentiNet / STRIP / ScaleUp / IBD_PSC`；",
        "- `stealth_avg` 越大，说明四个检测方法平均越不容易检出，隐蔽性越强。",
        "",
        "注意：`transfer_rate` 不是单纯的迁移 ASR，它会受到 `source_asr` 分母影响。因此后续所有结论都要同时注意 source ASR 过滤和 transfer_rate 长尾。",
        "",
        "## 2. 当前数据规模与完整性",
        "",
        f"- 解析到 baseline 目录行数：`{total_baseline_rows}`",
        f"- 完整四防御结果行数：`{complete_rows}`",
        f"- 有效 transfer_rate 行数：`{valid_rows}`",
        f"- 主分析行数：`{len(sub)}`",
        "",
        "主分析过滤条件：来自 `poisoned_train_set1`，四个防御完整，`transfer_rate` 可计算，`source_asr >= 0.05`，并且 `clean_acc / stealth_avg` 非空。",
        "",
        "这说明完整旧基线的数据完整性较好：全部 baseline 目录都有四个源域检测防御结果，也都能计算 transfer_rate。主分析从 1491 行中过滤到 1299 行，主要是为了排除 source ASR 太低时 `transfer_rate` 被分母异常放大的配置。",
        "",
        "## 3. 实验覆盖内容",
        "",
        "完整旧基线覆盖三个源域：",
        "",
        "```text",
        "CIFAR-10",
        "MNIST-M",
        "Tiny-ImageNet",
        "```",
        "",
        "覆盖的主要模型体系是：",
        "",
        "```text",
        "ResNet18",
        "MobileNetV2",
        "VGG19-BN",
        "```",
        "",
        "攻击方法覆盖 8 类：",
        "",
        "```text",
        "badnet/basic",
        "blend",
        "SIG",
        "WaNet",
        "adaptive_patch",
        "adaptive_blend",
        "belt",
        "upgd",
        "```",
        "",
        "隐蔽性检测方法覆盖 4 类：",
        "",
        "```text",
        "SentiNet",
        "STRIP",
        "ScaleUp",
        "IBD_PSC",
        "```",
        "",
        "当前代码里的 `SIG` 和 `UPGD` 仍然是 all-to-one dirty-label / label-flipping 版本，不是 clean-label。后续如果改成 clean-label，需要单独标注和重新分析。",
        "",
        "## 4. 总体结论",
        "",
        f"- 在完整旧 baseline 数据上，`transfer_rate` 与 `stealth_avg` 的整体相关为：Pearson=`{metric(all_corr, 'pearson_transfer_rate_stealth_avg')}`，Spearman=`{metric(all_corr, 'spearman_transfer_rate_stealth_avg')}`。",
        f"- 提高 source ASR 过滤到 `source_asr>=0.10` 后，整体相关为：Pearson=`{metric(all_corr_010, 'pearson_transfer_rate_stealth_avg')}`，Spearman=`{metric(all_corr_010, 'spearman_transfer_rate_stealth_avg')}`。如果两个阈值下方向一致，说明结论不完全由低 source ASR 分母放大造成。",
        f"- 按数据集拆分：CIFAR-10 Spearman=`{metric(cifar_corr, 'spearman_transfer_rate_stealth_avg')}`，Tiny-ImageNet Spearman=`{metric(tiny_corr, 'spearman_transfer_rate_stealth_avg')}`，MNIST-M Spearman=`{metric(mnist_corr, 'spearman_transfer_rate_stealth_avg')}`。",
        f"- 全局 ACC 分层后：high_acc Spearman=`{metric(high_acc, 'spearman_transfer_stealth')}`，mid_acc Spearman=`{metric(mid_acc, 'spearman_transfer_stealth')}`，low_acc Spearman=`{metric(low_acc, 'spearman_transfer_stealth')}`。这部分用于判断分类难度是否改变 transfer-stealth 关系强度。",
        "",
        "更具体地说，完整旧基线支持如下结论：",
        "",
        "1. `transfer_rate` 与 `stealth_avg` 整体呈负相关，说明迁移性越强时，平均隐蔽性通常越弱。",
        "2. 这种关系在 CIFAR-10 上最强，Tiny-ImageNet 次之，MNIST-M 相对更弱。",
        "3. ACC 分层后，low/mid/high ACC 区间的相关强度不同，说明任务难度可能改变 tradeoff 的形态。",
        "4. 这个关系不是漂亮的强线性关系，而更像分组/排序型关系；普通散点图会被 transfer_rate 集中、极端值、数据集和攻击混杂影响。",
        "5. 攻击类型和检测方法差异很大，最终论文分析不能只报告 overall mean。",
        "",
        "完整旧基线最适合支撑的表述是：",
        "",
        "```text",
        "在原有多数据集、多模型、多攻击实验中，迁移性和隐蔽性整体存在负相关；",
        "但这个 tradeoff 不是固定常数，而是会随数据集、ACC 区间、攻击类型和检测器响应变化。",
        "```",
        "",
        "## 5. 为什么普通基线散点图看不出很清楚的趋势",
        "",
        "普通散点图看起来不明显，主要不是因为没有关系，而是因为 `transfer_rate` 的分布非常特殊。",
        "",
        "核心分布诊断：",
        "",
        f"- 主分析样本数：`{metric(plot_diag_all, 'n_rows', 0)}`",
        f"- transfer_rate 中位数：`{metric(plot_diag_all, 'transfer_rate_median')}`",
        f"- transfer_rate 25%/75% 分位：`{metric(plot_diag_all, 'transfer_rate_p25')}` / `{metric(plot_diag_all, 'transfer_rate_p75')}`",
        f"- transfer_rate 95% 分位和最大值：`{metric(plot_diag_all, 'transfer_rate_p95')}` / `{metric(plot_diag_all, 'transfer_rate_max')}`",
        f"- transfer_rate 落在 `[0.9, 1.1]` 的比例：`{metric(plot_diag_all, 'share_transfer_0p9_1p1')}`",
        f"- transfer_rate 大于 `1.5` 的比例：`{metric(plot_diag_all, 'share_transfer_gt_1p5')}`",
        "",
        "也就是说，将近一半样本的 transfer_rate 都挤在 1 附近，同时少数极端值会拉长横轴。因此普通散点图会形成一条很厚的竖带，视觉上不容易看到斜率。",
        "",
        "因此，汇报时不建议把普通散点图作为唯一证据。更推荐结合：",
        "",
        "```text",
        "分组相关性表",
        "ACC 分层表",
        "dataset 分面图",
        "分箱中位数趋势图",
        "rank 分箱趋势图",
        "```",
        "",
        "最适合汇报趋势的图是：",
        "",
        "```text",
        "analysis-transfer-asr2/baseline_full_analysis/figures/baseline_binned_median_trend_by_dataset.png",
        "analysis-transfer-asr2/baseline_full_analysis/figures/baseline_rank_binned_trend_by_dataset.png",
        "analysis-transfer-asr2/baseline_full_analysis/figures/baseline_transfer_vs_stealth_dataset_facets_clipped.png",
        "```",
        "",
        "这些图对应的完整说明见：",
        "",
        "```text",
        "analysis-transfer-asr2/BASELINE_FULL_FIGURE_ANALYSIS_CN.md",
        "```",
        "",
        "## 6. Dataset / Arch 汇总",
        "",
        top_table(summary, 40),
        "",
        "这张表用于回答：不同数据集和模型本身的 clean ACC、transfer_rate、stealth_avg 是否处于不同水平。",
        "",
        "汇报时重点看：",
        "",
        "- CIFAR-10 平均 ACC 高，transfer_rate 相对较低，tradeoff 最明显。",
        "- Tiny-ImageNet 平均 ACC 明显低，但 transfer_rate 较高，说明任务难度和迁移性不是简单单调关系。",
        "- MNIST-M ACC 很高，但 transfer_rate 和 stealth_avg 都偏高，说明数据集本身也强烈影响结论。",
        "",
        "这张表是完整旧基线的第一层背景证据：不同 dataset/arch 的基础水平不同，后续所有整体平均都需要谨慎解释。",
        "",
        "## 7. 最关键证据：ACC 分层下的 transfer-stealth 关系",
        "",
        top_table(acc_bins, 40),
        "",
        "这张表用于回答：高 ACC、中 ACC、低 ACC 区间里，迁移性和隐蔽性的负相关是否同样强。",
        "",
        "最重要的三行是 global ACC bin：",
        "",
        "| ACC bin | Spearman transfer-stealth | 解释 |",
        "|---|---:|---|",
        f"| low_acc | {metric(low_acc, 'spearman_transfer_stealth')} | 低 ACC 条件下仍有负相关，但强度中等 |",
        f"| mid_acc | {metric(mid_acc, 'spearman_transfer_stealth')} | 中等 ACC 条件下负相关最强 |",
        f"| high_acc | {metric(high_acc, 'spearman_transfer_stealth')} | 高 ACC 条件下负相关仍存在，但弱于 mid_acc |",
        "",
        "这部分最适合和噪声实验结合。完整旧基线说明 ACC 区间本来就会改变 tradeoff 强度；噪声实验进一步说明，在固定 CIFAR-10 + SmallCNN 下，主动降低 ACC 后 tradeoff 会明显变化。",
        "",
        "## 8. 攻击类型差异",
        "",
        top_table(attack_summary, 40),
        "",
        "这张表用于排查 attack-dependent 混杂。它说明不同攻击方法天然处于不同区域：",
        "",
        "- `badnet/basic`、`adaptive_patch` 往往更容易形成高迁移、低隐蔽的局部触发器趋势；",
        "- `SIG`、`WaNet`、`blend`、`adaptive_blend` 更容易表现出较高隐蔽性，但迁移性和数据集有关；",
        "- `UPGD` 和 `BELT` 的趋势更特殊，尤其当前 SIG/UPGD 还不是 clean-label，后续需要单独修正和分析。",
        "",
        "因此汇报时不要说“所有攻击都一致”。更准确的说法是：",
        "",
        "```text",
        "完整基线整体存在 transfer-stealth tradeoff，但该关系明显 attack-dependent。",
        "```",
        "",
        "## 9. 四个检测方法分别贡献了什么",
        "",
        top_table(defense, 60),
        "",
        "`stealth_avg` 是四个检测方法的平均值，所以必须拆开看每个 defense。否则可能误以为所有检测方法都同向变化。",
        "",
        "汇报时可以这样讲：",
        "",
        "- `STRIP` 和 `SentiNet` 在一些数据集上 stealth 很高，说明它们对部分攻击不敏感；",
        "- `ScaleUp` 和 `IBD-PSC` 对模型结构、输入分布和触发器类型更敏感；",
        "- 如果某个实验里的 `stealth_avg` 变化很大，需要回到 defense breakdown 判断到底是哪个检测器驱动的。",
        "",
        "## 10. 回归结果怎么解释",
        "",
        *regression_note,
        "",
        "完整回归表保存于 `analysis-transfer-asr2/baseline_full_analysis/baseline_full_regression.txt`。",
        "",
        "这个回归结果支持：difficulty 可能会调节 transfer-stealth 的线性斜率。但它是 pooled regression，混合了 dataset、arch、attack、poison_rate，因此不能写成“ACC 是唯一因果原因”。",
        "",
        "更稳妥的表述是：",
        "",
        "```text",
        "完整旧基线在回归和分层层面都支持 difficulty 与 transfer-stealth 关系有关；",
        "但该影响与数据集、模型架构和攻击类型混杂，需要结合噪声实验做更接近控制变量的补充验证。",
        "```",
        "",
        "## 11. 和噪声实验、新模型实验的关系",
        "",
        "- 这份报告提供的是旧完整 baseline 的总体背景：ResNet18 / MobileNetV2 / VGG19-BN 在 CIFAR-10、Tiny-ImageNet、MNIST-M 上的完整结果。",
        "- 噪声实验更像固定 `CIFAR-10 + SmallCNN` 后改变输入难度的控制变量实验。",
        "- 新模型实验更像对 baseline 的模型体系补充：`SmallCNN -> CIFAR-10`、`ResNet34 -> Tiny-ImageNet`。",
        "- 三者合起来更适合形成证据链：旧完整 baseline 说明大范围现象，新模型补充说明模型替换方向，噪声实验说明同模型下 ACC 变化的影响。",
        "",
        "建议写作顺序是：",
        "",
        "```text",
        "完整旧基线：证明大范围 tradeoff 存在，且随 dataset/ACC/attack 改变。",
        "噪声实验：固定 SmallCNN 和 CIFAR-10，主动改变 ACC，验证 difficulty effect。",
        "模型架构实验：补充 SmallCNN 和 ResNet34，观察换模型后关系是否仍成立。",
        "```",
        "",
        "## 12. 当前结果能支持的表述",
        "",
        "比较稳妥、准确的表述：",
        "",
        "```text",
        "完整旧基线显示，在多数据集、多模型、多攻击的原始实验中，迁移性和隐蔽性整体呈负相关。",
        "这种关系在 CIFAR-10 上最明显，在 Tiny-ImageNet 和 MNIST-M 上也存在，但强度不同。",
        "ACC 分层、回归和分箱趋势图都说明，任务难度会改变 tradeoff 的强弱。",
        "不过该关系不是简单线性关系，而是明显受到数据集、攻击类型和检测方法影响。",
        "```",
        "",
        "不建议直接写成：",
        "",
        "```text",
        "散点图显示明显线性负相关。",
        "ACC 是唯一导致 tradeoff 变化的原因。",
        "所有攻击方法都有一致规律。",
        "```",
        "",
        "## 13. 当前结果的限制",
        "",
        "1. `transfer_rate` 大量集中在 1 附近，并存在少数极端值，普通散点图不适合作为唯一证据。",
        "2. 完整旧基线混合了不同数据集、模型和攻击，不能单独归因 ACC。",
        "3. `SIG` 和 `UPGD` 当前仍是 dirty-label / label-flipping 版本，不是 clean-label。",
        "4. `stealth_avg` 是四个检测器平均值，必须结合 defense breakdown 解释。",
        "5. 最终论文结论应同时报告 `source_asr>=0.05` 和 `source_asr>=0.10` 的敏感性结果。",
        "",
        "## 14. 后续建议补充实验和分析",
        "",
        "1. 修正并单独重跑 clean-label SIG / UPGD。",
        "2. 对基线、噪声、新模型都补充 attack-family 分组图。",
        "3. 对 `transfer_rate` 增加 log / winsorize / source_asr>=0.10 的稳健性分析。",
        "4. 把四个 defense 的趋势作为附录，不只报告 `stealth_avg`。",
        "5. 汇报图优先使用分箱中位数趋势图和 rank 分箱趋势图，而不是普通散点图。",
        "",
        "## 15. 建议阅读顺序",
        "",
        "为了快速理解完整旧基线，建议按这个顺序看：",
        "",
        "1. `BASELINE_FULL_ACC_TRANSFER_STEALTH_REPORT_CN.md`：先读本文档，把握主结论。",
        "2. `baseline_full_plot_diagnostics.csv`：理解为什么普通散点图不明显。",
        "3. `figures/baseline_binned_median_trend_by_dataset.png`：看最适合汇报的分箱趋势图。",
        "4. `figures/baseline_rank_binned_trend_by_dataset.png`：看 Spearman 排序关系。",
        "5. `baseline_full_acc_bins.csv`：看 ACC 分层相关性。",
        "6. `baseline_full_summary_by_attack.csv`：看攻击类型差异。",
        "7. `baseline_full_defense_breakdown.csv`：看四个检测方法贡献。",
        "8. `baseline_full_regression.txt`：看完整回归统计。",
        "",
        "## 16. 输出文件作用",
        "",
        "- `baseline_full_acc_transfer_stealth_rows.csv`：逐目录明细，是所有 baseline-only 表格和回归的基础。",
        "- `baseline_full_summary_by_dataset_arch.csv`：按数据集和模型汇总。",
        "- `baseline_full_summary_by_attack.csv`：按攻击类型汇总。",
        "- `baseline_full_correlations.csv`：整体、数据集、模型、攻击类型的相关性。",
        "- `baseline_full_acc_bins.csv`：ACC 分层结果。",
        "- `baseline_full_defense_breakdown.csv`：四个检测器拆解。",
        "- `baseline_full_regression.txt`：完整线性回归输出。",
        "- `figures/baseline_transfer_vs_stealth_by_dataset_clipped.png`：按 dataset 上色的裁剪散点图。",
        "- `figures/baseline_transfer_vs_stealth_by_acc_bin_clipped.png`：按 ACC bin 上色的裁剪散点图。",
        "- `figures/baseline_transfer_vs_stealth_dataset_facets_clipped.png`：按 dataset 分面的裁剪散点图。",
        "- `figures/baseline_binned_median_trend_by_dataset.png`：按 transfer_rate 分箱后的中位数趋势图。",
        "- `figures/baseline_rank_binned_trend_by_dataset.png`：按 transfer_rate rank 分箱后的趋势图。",
        "- `baseline_full_plot_diagnostics.csv`：基线图分布诊断表，用于解释为什么普通散点图趋势不明显。",
        "- `baseline_full_binned_median_trend_by_dataset.csv`：分箱中位数趋势表。",
        "- `baseline_full_rank_binned_trend_by_dataset.csv`：rank 分箱趋势表。",
        "",
        "## 17. 一句话总结",
        "",
        "```text",
        "完整旧基线证明 transfer-stealth tradeoff 在原有实验中整体存在，但它不是简单线性斜线；",
        "该关系会随数据集、ACC 区间、攻击类型和检测器改变，最适合用分组相关性和分箱趋势图汇报。",
        "```",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_acc_moderation_result_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    attack_summary: pd.DataFrame,
    corr: pd.DataFrame,
    acc_bins: pd.DataFrame,
    acc_moderation: pd.DataFrame,
    acc_bin_slope: pd.DataFrame,
    pairwise: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    defense_breakdown: pd.DataFrame,
    diagnostics: pd.DataFrame,
    binned_trend: pd.DataFrame,
    rank_trend: pd.DataFrame,
    report_path: Path,
) -> None:
    primary = exclude_arch_attacks(df[df["primary_main_analysis"]].copy())
    baseline = primary[primary["result_group"] == "baseline_full"]
    supplement = primary[primary["result_group"] == "new_model_supplement"]

    main_mod = first_row(acc_moderation, cohort="primary_main_source_asr>=0.05")
    sens_mod = first_row(acc_moderation, cohort="primary_main_source_asr>=0.10")
    include_mod = first_row(acc_moderation, cohort="include_main_source_asr>=0.05")
    high_acc = first_row(acc_bin_slope, group_type="global_acc_bin", dataset="all", acc_bin="high_acc")
    mid_acc = first_row(acc_bin_slope, group_type="global_acc_bin", dataset="all", acc_bin="mid_acc")
    low_acc = first_row(acc_bin_slope, group_type="global_acc_bin", dataset="all", acc_bin="low_acc")
    diag_all = first_row(diagnostics, group="all")
    cifar_small = first_row(summary, dataset="cifar10", arch_base="SmallCNN")
    cifar_resnet18 = first_row(summary, dataset="cifar10", arch_base="ResNet18")
    tiny_resnet34 = first_row(summary, dataset="tiny_imagenet", arch_base="ResNet34")
    tiny_resnet18 = first_row(summary, dataset="tiny_imagenet", arch_base="ResNet18")
    small_vs_resnet18 = first_row(pairwise_summary, dataset="cifar10", new_arch="SmallCNN", base_arch="ResNet18")
    r34_vs_resnet18 = first_row(pairwise_summary, dataset="tiny_imagenet", new_arch="ResNet34", base_arch="ResNet18")

    key_mod_cols = [
        "cohort",
        "n_rows",
        "spearman_transfer_stealth",
        "spearman_clean_acc_transfer",
        "spearman_clean_acc_stealth",
        "raw_interaction_coef",
        "raw_interaction_pvalue",
        "std_interaction_coef",
        "std_interaction_pvalue",
        "std_r_squared",
    ]
    key_acc_cols = [
        "group_type",
        "dataset",
        "acc_bin",
        "n_rows",
        "clean_acc_mean",
        "difficulty_mean",
        "transfer_rate_median",
        "stealth_avg_median",
        "spearman_transfer_stealth",
    ]
    key_summary_cols = [
        "dataset",
        "arch_base",
        "result_group",
        "n_rows",
        "clean_acc_mean",
        "difficulty_mean",
        "transfer_rate_mean",
        "transfer_rate_median",
        "stealth_avg_mean",
        "stealth_avg_median",
    ]
    key_pair_cols = [
        "dataset",
        "new_arch",
        "base_arch",
        "n_rows",
        "delta_clean_acc_mean",
        "delta_transfer_rate_mean",
        "delta_stealth_avg_mean",
        "delta_source_asr_mean",
        "delta_transfer_asr_mean",
    ]
    attack_focus = attack_summary[attack_summary["dataset"].isin(PRIMARY_DATASETS)].copy()
    attack_focus["transfer_minus_stealth_gap"] = attack_focus["transfer_rate_mean"] - attack_focus["stealth_avg_mean"]
    attack_cols = [
        "dataset",
        "arch_base",
        "attack_family",
        "n_rows",
        "clean_acc_mean",
        "source_asr_mean",
        "transfer_rate_mean",
        "stealth_avg_mean",
        "transfer_minus_stealth_gap",
    ]
    defense_cols = [
        "dataset",
        "arch_base",
        "defense",
        "n_rows",
        "tpr_mean",
        "stealth_mean",
        "auc_mean",
    ]

    lines = [
        "# 模型/架构实验：ACC 是否调节迁移性与隐蔽性的关系",
        "",
        "## 0. 先看结论",
        "",
        "这份报告的重点不是单纯比较 SmallCNN、ResNet18 或 ResNet34 谁更好，而是回答一个更具体的问题：`clean_acc` 是否会影响 `transfer_rate` 与 `stealth_avg` 的关系，以及它是怎么影响的。",
        "",
        "当前最稳妥的回答是：",
        "",
        f"- **ACC 会影响两者关系，但主要不是直接边际影响，而是调节斜率。**主口径 `primary_main_analysis` 中，`transfer_rate` 与 `stealth_avg` 整体负相关，Spearman=`{metric(main_mod, 'spearman_transfer_stealth')}`，Pearson=`{metric(main_mod, 'pearson_transfer_stealth')}`。",
        f"- **ACC 对单指标的边际关系不均衡。**`clean_acc` vs `transfer_rate` 的 Spearman=`{metric(main_mod, 'spearman_clean_acc_transfer')}`，说明 ACC 与迁移性本身有关；但 `clean_acc` vs `stealth_avg` 的 Spearman=`{metric(main_mod, 'spearman_clean_acc_stealth')}`，直接解释隐蔽性很弱。因此不能简单说“ACC 高就一定更隐蔽”。",
        f"- **关键证据是交互项。**原始控制回归中 `transfer_rate:difficulty` 系数=`{metric(main_mod, 'raw_interaction_coef')}`，p=`{sci_metric(main_mod, 'raw_interaction_pvalue')}`；标准化并控制 `dataset/arch/attack/poison_rate` 后，交互系数=`{metric(main_mod, 'std_interaction_coef')}`，p=`{sci_metric(main_mod, 'std_interaction_pvalue')}`。",
        "- 因为 `difficulty = 1 - clean_acc`，负交互项表示：**ACC 下降、任务难度上升时，迁移性越强带来的隐蔽性下降更明显，transfer-stealth tradeoff 更陡。**",
        f"- `source_asr>=0.10` 敏感性检查中，标准化交互系数=`{metric(sens_mod, 'std_interaction_coef')}`，p=`{sci_metric(sens_mod, 'std_interaction_pvalue')}`，方向仍一致，说明这个结论不完全由低 source ASR 分母放大造成。",
        "",
        "结论边界也必须讲清楚：架构实验不是纯 ACC 干预，因为模型容量、归纳偏置、数据集、输入尺寸和攻击类型同时变化。因此它支持“ACC/任务难度会调节 transfer-stealth 关系”，但不能单独证明 ACC 是唯一因果原因。",
        "",
        "## 1. 建议优先看的图",
        "",
        "| 优先级 | 图 | 这张图回答什么 | 汇报时怎么说 |",
        "|---:|---|---|---|",
        "| 1 | `arch_acc_moderation_summary.png` | ACC 与 transfer 有边际关系，但对 stealth 直接解释弱；按 ACC 分层后 transfer-stealth 斜率变了 | ACC 不是简单单调变量，更像 moderator |",
        "| 2 | `arch_interaction_slope_by_acc.png` | 低/中/高 difficulty 下模型预测斜率如何变 | ACC 下降时 tradeoff 变陡，这是核心图 |",
        "| 3 | `arch_acc_bin_transfer_stealth.png` | high/mid/low ACC bin 内的散点和趋势 | 分箱说明关系强度随 ACC 区间改变 |",
        "| 4 | `arch_dataset_acc_bin_spearman.png` | CIFAR-10 和 Tiny-ImageNet 内部分箱相关性 | 防止全局分箱被 dataset 组成差异误读 |",
        "| 5 | `arch_pairwise_delta_summary.png` | 换模型后 ACC、transfer、stealth 三者共同怎么变 | 这是模型对照证据，不是主因果证据 |",
        "| 6 | `arch_attack_heatmap.png` / `arch_defense_heatmap.png` | attack 和 defense 为什么会混杂主趋势 | 解释为什么不能只讲一条直线 |",
        "",
        md_image("arch_acc_analysis/figures/arch_acc_moderation_summary.png", "ACC moderation summary"),
        "",
        md_image("arch_acc_analysis/figures/arch_interaction_slope_by_acc.png", "Interaction slope by ACC/difficulty"),
        "",
        md_image("arch_acc_analysis/figures/arch_acc_bin_transfer_stealth.png", "Transfer-stealth by ACC bin"),
        "",
        md_image("arch_acc_analysis/figures/arch_dataset_acc_bin_spearman.png", "Dataset-specific ACC bin Spearman"),
        "",
        md_image("arch_acc_analysis/figures/arch_pairwise_delta_summary.png", "Pairwise delta summary"),
        "",
        md_image("arch_acc_analysis/figures/arch_attack_heatmap.png", "Attack heatmap"),
        "",
        md_image("arch_acc_analysis/figures/arch_defense_heatmap.png", "Defense heatmap"),
        "",
        "HTML 展示页同步更新：`analysis-transfer-asr2/ARCH_ACC_TRANSFER_STEALTH_DASHBOARD_CN.html`。",
        "",
        "## 2. 核心问题：ACC 是直接变量还是调节变量？",
        "",
        "当前结果更支持第二种解释：ACC 主要是调节变量，而不是直接决定 transfer 或 stealth 的单变量。",
        "",
        "可以把问题拆成三层：",
        "",
        "1. `transfer_rate` 和 `stealth_avg` 是否相关？是，整体负相关。",
        "2. `clean_acc` 自己是否能直接解释 transfer 或 stealth？它能解释一部分 transfer 差异，但对 stealth 的直接解释弱。",
        "3. `clean_acc` 是否改变 `transfer_rate -> stealth_avg` 的斜率？有证据支持，交互项显著。",
        "",
        "这也是为什么普通 ACC 散点图看起来不够明显：真正要找的不是 `ACC -> stealth` 的单调线，而是 `ACC` 改变 `transfer-stealth` 这条线的斜率。",
        "",
        "## 3. 指标定义与分析口径",
        "",
        "固定指标定义：",
        "",
        "```text",
        "difficulty = 1 - clean_acc",
        "transfer_rate = transfer_asr^2 / source_asr",
        "stealth_avg = mean(1 - TPR)",
        "```",
        "",
        "- `clean_acc` 是源域分类准确率。",
        "- `difficulty` 是 ACC 的反向表达，数值越大表示任务越难或模型在该任务上越弱。",
        "- `transfer_rate` 是当前固定使用的迁移性指标，会受到 `source_asr` 分母影响。",
        "- `stealth_avg` 来自源域四个检测防御：`SentiNet / STRIP / ScaleUp / IBD_PSC`，数值越大表示越隐蔽。",
        "",
        "主报告口径使用 `primary_main_analysis`，即 CIFAR-10 和 Tiny-ImageNet 上当前架构分析关注的模型集合。`include_main_analysis` 会保留为补充口径，但不放进主结论，避免 MNIST-M 改变主线叙事。",
        "",
        "数据规模：",
        "",
        f"- 总解析行数：`{len(df)}`",
        f"- 主分析行数：`{len(primary)}`",
        f"- baseline_full 主分析行数：`{len(baseline)}`",
        f"- new_model_supplement 主分析行数：`{len(supplement)}`",
        f"- 完整四防御行数：`{int(df['complete_defense_results'].sum()) if not df.empty else 0}`",
        f"- 有效 transfer_rate 行数：`{int(df['valid_transfer_rate'].sum()) if not df.empty else 0}`",
        "",
        "## 4. 三类证据",
        "",
        "### 4.1 总体相关：transfer-stealth 负相关存在",
        "",
        f"主口径下，`transfer_rate` 与 `stealth_avg` 的 Spearman=`{metric(main_mod, 'spearman_transfer_stealth')}`。这说明迁移性越强时，四个源域检测方法平均越容易检出，隐蔽性越低。",
        "",
        "但这不是一条漂亮的单线性斜线。图表分布诊断显示：",
        "",
        f"- transfer_rate 中位数：`{metric(diag_all, 'transfer_rate_median')}`",
        f"- transfer_rate 25%/75% 分位：`{metric(diag_all, 'transfer_rate_p25')}` / `{metric(diag_all, 'transfer_rate_p75')}`",
        f"- transfer_rate 95% 分位和最大值：`{metric(diag_all, 'transfer_rate_p95')}` / `{metric(diag_all, 'transfer_rate_max')}`",
        f"- transfer_rate 落在 `[0.9, 1.1]` 的比例：`{metric(diag_all, 'share_transfer_0p9_1p1')}`",
        "",
        "这解释了为什么普通散点图不够显眼：大量点挤在 transfer_rate 约 1 附近，同时攻击类型和模型结构混在一起。",
        "",
        "### 4.2 ACC 边际关系不均衡：不能写成单调因果",
        "",
        "如果只看 ACC 与单个指标的边际相关，会得到一个不均衡的结果：",
        "",
        f"- `clean_acc` vs `transfer_rate`：Pearson=`{metric(main_mod, 'pearson_clean_acc_transfer')}`，Spearman=`{metric(main_mod, 'spearman_clean_acc_transfer')}`。",
        f"- `clean_acc` vs `stealth_avg`：Pearson=`{metric(main_mod, 'pearson_clean_acc_stealth')}`，Spearman=`{metric(main_mod, 'spearman_clean_acc_stealth')}`。",
        "",
        "所以，汇报时可以说 ACC 与 transfer_rate 本身有关，但不要把它写成“ACC 单独决定隐蔽性”。更准确的表达是：ACC/difficulty 还会改变迁移性和隐蔽性之间的 tradeoff 斜率。",
        "",
        "### 4.3 交互回归：ACC/difficulty 改变 transfer-stealth 斜率",
        "",
        "核心回归模型是：",
        "",
        "```text",
        "stealth_avg ~ transfer_rate * difficulty + C(dataset) + C(arch_base) + C(attack_family) + C(poison_rate)",
        "```",
        "",
        "主口径结果：",
        "",
        limited_table(acc_moderation, key_mod_cols, 10, ["cohort"]),
        "",
        f"主口径中，标准化交互项 `transfer_rate_z:difficulty_z`=`{metric(main_mod, 'std_interaction_coef')}`，p=`{sci_metric(main_mod, 'std_interaction_pvalue')}`。这个数值为负，表示 difficulty 越高，也就是 ACC 越低时，`transfer_rate` 增加对应的 `stealth_avg` 下降更明显。",
        "",
        f"敏感性口径 `source_asr>=0.10` 下，标准化交互项=`{metric(sens_mod, 'std_interaction_coef')}`，p=`{sci_metric(sens_mod, 'std_interaction_pvalue')}`。方向仍然一致，说明结论不是只靠低 source ASR 样本支撑。",
        "",
        f"补充口径 `include_main_analysis` 下，标准化交互项=`{metric(include_mod, 'std_interaction_coef')}`，p=`{sci_metric(include_mod, 'std_interaction_pvalue')}`。这个口径混入 MNIST-M，只作为稳健性背景，不作为主结论。",
        "",
        "## 5. ACC 分层：为什么表面结果会有一点“矛盾感”",
        "",
        "全局 ACC 分层结果如下：",
        "",
        f"- high_acc：Spearman=`{metric(high_acc, 'spearman_transfer_stealth')}`，clean_acc_mean=`{metric(high_acc, 'clean_acc_mean')}`。",
        f"- mid_acc：Spearman=`{metric(mid_acc, 'spearman_transfer_stealth')}`，clean_acc_mean=`{metric(mid_acc, 'clean_acc_mean')}`。",
        f"- low_acc：Spearman=`{metric(low_acc, 'spearman_transfer_stealth')}`，clean_acc_mean=`{metric(low_acc, 'clean_acc_mean')}`。",
        "",
        limited_table(acc_bin_slope, key_acc_cols, 30, ["group_type", "dataset", "acc_bin"]),
        "",
        "这里不要机械理解为“ACC 越低，Spearman 必须单调越负”。分箱 Spearman 会受到 dataset、attack、模型组成和样本密度影响。它能说明 high/mid/low ACC 区间的关系强度不同，但不能单独给出因果方向。",
        "",
        "因此报告把分层结果作为现象证据，把控制回归中的 `transfer_rate:difficulty` 作为更核心的调节证据。两者结合后的结论是：transfer-stealth 负相关存在，ACC/difficulty 会改变它的强度，但这个改变不是简单的单调分箱排序。",
        "",
        "## 6. SmallCNN / ResNet34 的作用：模型生态对照",
        "",
        "SmallCNN 和 ResNet34 不再作为主结论，而是作为模型替换后 ACC、transfer、stealth 如何共同变化的对照证据。",
        "",
        f"- CIFAR-10 SmallCNN 的平均 clean ACC=`{metric(cifar_small, 'clean_acc_mean')}`，ResNet18=`{metric(cifar_resnet18, 'clean_acc_mean')}`；SmallCNN 的 transfer_rate=`{metric(cifar_small, 'transfer_rate_mean')}`，stealth_avg=`{metric(cifar_small, 'stealth_avg_mean')}`。",
        f"- 同配置 pairwise 中，SmallCNN 相对 ResNet18：clean ACC `{signed_metric(small_vs_resnet18, 'delta_clean_acc_mean')}`，transfer_rate `{signed_metric(small_vs_resnet18, 'delta_transfer_rate_mean')}`，stealth_avg `{signed_metric(small_vs_resnet18, 'delta_stealth_avg_mean')}`。",
        f"- Tiny-ImageNet ResNet34 的平均 clean ACC=`{metric(tiny_resnet34, 'clean_acc_mean')}`，ResNet18=`{metric(tiny_resnet18, 'clean_acc_mean')}`；同配置 pairwise 中，ResNet34 相对 ResNet18：clean ACC `{signed_metric(r34_vs_resnet18, 'delta_clean_acc_mean')}`，transfer_rate `{signed_metric(r34_vs_resnet18, 'delta_transfer_rate_mean')}`，stealth_avg `{signed_metric(r34_vs_resnet18, 'delta_stealth_avg_mean')}`。",
        "",
        "这些结果说明换模型后三项指标会一起移动，但它们不是纯 ACC 干预。汇报时可以说它们支持“模型生态中 tradeoff 斜率和水平会变”，不要说它们单独证明 ACC 因果。",
        "",
        "Pairwise delta 表：",
        "",
        limited_table(pairwise_summary, key_pair_cols, 20, ["dataset", "new_arch", "base_arch"]) if not pairwise_summary.empty else "当前没有足够 pairwise 数据。",
        "",
        "Dataset/Arch 汇总表：",
        "",
        limited_table(summary[summary["dataset"].isin(PRIMARY_DATASETS)].copy(), key_summary_cols, 30, ["dataset", "result_group", "arch_base"]),
        "",
        "## 7. 攻击类型和检测器是强混杂因素",
        "",
        "攻击类型会决定样本天然落在哪个 transfer-stealth 区域。例如局部强触发攻击可能形成高迁移但低隐蔽，部分形变或混合类攻击可能更隐蔽但迁移结构不同。因此不能只报告 overall mean。",
        "",
        limited_table(attack_focus, attack_cols, 80, ["dataset", "attack_family", "arch_base"]),
        "",
        "`stealth_avg` 也是四个检测器的平均值，不代表四个防御方法同方向变化。最终汇报中如果某个模型 stealth 变化很大，需要回到 defense breakdown 判断是哪个检测器驱动。",
        "",
        limited_table(defense_breakdown, defense_cols, 80, ["dataset", "arch_base", "defense"]),
        "",
        "## 8. 分箱趋势和 rank 趋势",
        "",
        "这两张图用于解释普通散点图不明显的问题。分箱趋势看真实数值中位数，rank 趋势更接近 Spearman 排序视角。",
        "",
        "- `arch_binned_median_trend_by_dataset_arch.csv` 对应 `arch_binned_median_trend_by_dataset_arch.png`。",
        "- `arch_rank_binned_trend_by_dataset_arch.csv` 对应 `arch_rank_binned_trend_by_dataset_arch.png`。",
        "",
        "分箱趋势表预览：",
        "",
        limited_table(binned_trend, ["dataset_arch", "bin_id", "n_rows", "transfer_rate_median", "stealth_avg_median", "clean_acc_mean"], 30, ["dataset_arch", "bin_id"]),
        "",
        "Rank 趋势表预览：",
        "",
        limited_table(rank_trend, ["dataset_arch", "bin_id", "n_rows", "transfer_rank_median", "stealth_rank_median"], 30, ["dataset_arch", "bin_id"]),
        "",
        "## 9. 汇报建议",
        "",
        "建议按 5 页讲：",
        "",
        "1. **问题**：不是问哪个模型最好，而是问 ACC 是否改变 `transfer_rate -> stealth_avg` 的关系。",
        "2. **核心图**：展示 `arch_acc_moderation_summary.png`，说明 ACC 对 transfer 有边际关系、对 stealth 直接解释弱，但按 ACC 分层后 transfer-stealth 关系变了。",
        "3. **交互图**：展示 `arch_interaction_slope_by_acc.png`，说明低 ACC/高 difficulty 下 tradeoff 更陡。",
        "4. **分层证据**：展示 `arch_dataset_acc_bin_spearman.png`，说明数据集内也能看到关系强度变化，但不追求简单单调排序。",
        "5. **边界**：展示 pairwise/attack/defense 图，说明模型结构、攻击类型和检测器响应是混杂因素。",
        "",
        "可以直接使用的汇报话术：",
        "",
        "```text",
        "我们的模型分析重点不是比较某个架构的绝对性能，而是看 ACC 是否会改变迁移性和隐蔽性之间的关系。",
        "结果显示，transfer_rate 和 stealth_avg 整体呈负相关；但 clean_acc 与单独的 transfer_rate 或 stealth_avg 边际相关很弱。",
        "真正关键的是交互项：控制 dataset、arch、attack 和 poison_rate 后，transfer_rate:difficulty 显著为负。",
        "由于 difficulty = 1 - clean_acc，这说明 ACC 下降、任务难度上升时，迁移性增强带来的隐蔽性下降更明显，也就是 tradeoff 更陡。",
        "因此当前架构实验支持 ACC/任务难度是 transfer-stealth 关系的调节因素，但不能单独证明 ACC 是唯一因果原因。",
        "```",
        "",
        "论文式谨慎表述：",
        "",
        "```text",
        "Architecture-level evidence suggests that clean accuracy / task difficulty moderates the relationship between transferability and stealthiness. Lower clean accuracy, equivalently higher difficulty, is associated with a more negative transfer-stealth slope after controlling for dataset, architecture, attack family, and poisoning rate. However, because architecture changes also alter capacity, inductive bias, and detector responses, these results should be interpreted as moderation evidence rather than a pure causal intervention on ACC.",
        "```",
        "",
        "## 10. 当前结果能支持什么，不能支持什么",
        "",
        "- 可以支持：`transfer_rate` 与 `stealth_avg` 整体负相关。",
        "- 可以支持：ACC/difficulty 会调节这条负相关的强度，控制变量后交互项显著。",
        "- 可以支持：SmallCNN/ResNet34 的模型替换结果说明 tradeoff 水平和斜率会随模型生态变化。",
        "- 不能支持：ACC 是唯一原因。",
        "- 不能支持：ACC 与 transfer 或 stealth 存在简单单调关系。",
        "- 不能支持：所有攻击和所有检测器都有完全一致规律。",
        "",
        "## 11. 输出文件说明",
        "",
        "- `arch_acc_moderation_summary.csv`：主口径和敏感性口径的 ACC 调节效应摘要，是本报告最关键的表。",
        "- `arch_acc_bin_slope_summary.csv`：high/mid/low ACC bin 内的 transfer-stealth 相关性。",
        "- `arch_acc_moderation_summary.png`：ACC 直接关系与分层 transfer-stealth 的核心三联图。",
        "- `arch_interaction_slope_by_acc.png`：低/中/高 difficulty 下的预测斜率图。",
        "- `arch_acc_bin_transfer_stealth.png`：按 ACC bin 展示 transfer-stealth 散点。",
        "- `arch_dataset_acc_bin_spearman.png`：按数据集展示 ACC bin Spearman。",
        "- `arch_acc_transfer_stealth_rows.csv`：逐实验明细，所有架构分析的基础表。",
        "- `arch_acc_pairwise_delta_summary.csv`：同配置模型替换 delta。",
        "- `arch_acc_summary_by_attack.csv`：攻击类型分层。",
        "- `arch_defense_breakdown_summary.csv`：四个检测器拆解。",
        "- `ARCH_ACC_TRANSFER_STEALTH_DASHBOARD_CN.html`：静态展示页。",
        "",
        "## 12. 后续最值得补的实验",
        "",
        "1. 在同一 dataset/arch/attack 下主动改变 ACC，例如噪声、训练轮数或数据比例，用于更接近因果干预。",
        "2. 优先补齐 `SmallCNN_cifar10 vs ResNet18_cifar10` 和 `ResNet34_tiny_imagenet vs ResNet18_tiny_imagenet` 的同配置结果。",
        "3. 对核心结论同时报告 `source_asr>=0.05` 和 `source_asr>=0.10`。",
        "4. 在论文附录中按 attack family 和 defense 分开报告，不只报告 `stealth_avg`。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    attack_summary: pd.DataFrame,
    corr: pd.DataFrame,
    acc_bins: pd.DataFrame,
    pairwise: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    defense_breakdown: pd.DataFrame,
    diagnostics: pd.DataFrame,
    binned_trend: pd.DataFrame,
    rank_trend: pd.DataFrame,
    report_path: Path,
) -> None:
    primary = exclude_arch_attacks(df[df["primary_main_analysis"]].copy())
    baseline = primary[primary["result_group"] == "baseline_full"]
    supplement = primary[primary["result_group"] == "new_model_supplement"]

    overall_corr = corr[(corr["group_type"] == "all") & (corr["source_asr_filter"] == "source_asr>=0.05")]
    overall_spearman = overall_corr["spearman_transfer_rate_stealth_avg"].iloc[0] if not overall_corr.empty else float("nan")
    overall_pearson = overall_corr["pearson_transfer_rate_stealth_avg"].iloc[0] if not overall_corr.empty else float("nan")

    cifar_small = first_row(summary, dataset="cifar10", arch_base="SmallCNN")
    cifar_resnet18 = first_row(summary, dataset="cifar10", arch_base="ResNet18")
    tiny_resnet34 = first_row(summary, dataset="tiny_imagenet", arch_base="ResNet34")
    tiny_resnet18 = first_row(summary, dataset="tiny_imagenet", arch_base="ResNet18")
    small_vs_resnet18 = first_row(pairwise_summary, dataset="cifar10", new_arch="SmallCNN", base_arch="ResNet18")
    r34_vs_resnet18 = first_row(pairwise_summary, dataset="tiny_imagenet", new_arch="ResNet34", base_arch="ResNet18")
    high_acc = first_row(acc_bins, group_type="global_acc_bin", dataset="all", acc_bin="high_acc")
    mid_acc = first_row(acc_bins, group_type="global_acc_bin", dataset="all", acc_bin="mid_acc")
    low_acc = first_row(acc_bins, group_type="global_acc_bin", dataset="all", acc_bin="low_acc")
    diag_all = first_row(diagnostics, group="all")
    corr_010 = first_row(corr, group_type="all", group_name="all", source_asr_filter="source_asr>=0.1")

    key_summary_cols = [
        "dataset",
        "arch_base",
        "result_group",
        "n_rows",
        "clean_acc_mean",
        "source_asr_mean",
        "transfer_asr_mean",
        "transfer_rate_mean",
        "transfer_rate_median",
        "stealth_avg_mean",
        "stealth_avg_median",
    ]
    key_pair_cols = [
        "dataset",
        "new_arch",
        "base_arch",
        "n_rows",
        "delta_clean_acc_mean",
        "delta_transfer_rate_mean",
        "delta_stealth_avg_mean",
        "delta_source_asr_mean",
        "delta_transfer_asr_mean",
    ]
    key_acc_cols = [
        "group_type",
        "dataset",
        "acc_bin",
        "n_rows",
        "clean_acc_mean",
        "transfer_rate_median",
        "stealth_avg_median",
        "spearman_transfer_stealth",
    ]
    attack_focus = attack_summary[attack_summary["dataset"].isin(PRIMARY_DATASETS)].copy()
    attack_focus["transfer_minus_stealth_gap"] = attack_focus["transfer_rate_mean"] - attack_focus["stealth_avg_mean"]
    attack_cols = [
        "dataset",
        "arch_base",
        "attack_family",
        "n_rows",
        "source_asr_mean",
        "transfer_asr_mean",
        "transfer_rate_mean",
        "stealth_avg_mean",
        "transfer_minus_stealth_gap",
    ]
    defense_cols = [
        "dataset",
        "arch_base",
        "defense",
        "n_rows",
        "tpr_mean",
        "stealth_mean",
        "auc_mean",
    ]

    lines = [
        "# 架构/模型 ACC 对迁移性与隐蔽性关系的实验分析报告",
        "",
        "## 0. 先看结论",
        "",
        "这份报告回答的是：在已有 baseline 模型之外，加入 `CIFAR-10 SmallCNN` 和 `Tiny-ImageNet ResNet34` 后，模型能力/架构变化是否会改变 `ACC - 迁移性 - 隐蔽性` 的关系。",
        "",
        "当前最稳的结论是：",
        "",
        f"- 主分析样本中 `transfer_rate` 与 `stealth_avg` 整体负相关：Pearson=`{fmt_float(overall_pearson)}`，Spearman=`{fmt_float(overall_spearman)}`。",
        f"- 提高过滤阈值到 `source_asr>=0.10` 后，整体 Spearman=`{metric(corr_010, 'spearman_transfer_rate_stealth_avg')}`，方向仍为负，说明结论不完全由低 source ASR 分母放大造成。",
        f"- ACC 分层后，high/mid/low 三层 Spearman 分别是 `{metric(high_acc, 'spearman_transfer_stealth')}` / `{metric(mid_acc, 'spearman_transfer_stealth')}` / `{metric(low_acc, 'spearman_transfer_stealth')}`，说明 tradeoff 强度会随 ACC 区间改变。",
        f"- `SmallCNN` 相对 CIFAR-10 `ResNet18`：clean ACC `{signed_metric(small_vs_resnet18, 'delta_clean_acc_mean')}`，transfer_rate `{signed_metric(small_vs_resnet18, 'delta_transfer_rate_mean')}`，stealth_avg `{signed_metric(small_vs_resnet18, 'delta_stealth_avg_mean')}`。这更像“弱模型提高迁移率但降低隐蔽性”。",
        f"- `ResNet34` 相对 Tiny-ImageNet `ResNet18`：clean ACC `{signed_metric(r34_vs_resnet18, 'delta_clean_acc_mean')}`，transfer_rate `{signed_metric(r34_vs_resnet18, 'delta_transfer_rate_mean')}`，stealth_avg `{signed_metric(r34_vs_resnet18, 'delta_stealth_avg_mean')}`。它说明模型加深后并不是单调提升所有指标。",
        "",
        "最重要的边界也要一起讲：架构实验不是纯 ACC 干预，因为模型容量、归纳偏置、数据集、输入尺寸和攻击类型同时变化。因此它更适合做“模型生态对照”，而不是单独证明 ACC 是唯一因果变量。",
        "",
        "## 1. 建议优先看的图",
        "",
        "| 优先级 | 图 | 用途 | 汇报时怎么说 |",
        "|---:|---|---|---|",
        "| 1 | `arch_metric_overview.png` | 看各 dataset/arch 的 ACC、transfer_rate、stealth_avg 水平 | 先说明新模型把系统推到不同 ACC/能力区间 |",
        "| 2 | `arch_pairwise_delta_summary.png` | 看同配置新模型相对旧模型的均值差 | 这是最适合讲 SmallCNN/ResNet34 补充实验的图 |",
        "| 3 | `arch_binned_median_trend_by_dataset_arch.png` | 看分箱后的 transfer-stealth 趋势 | 普通散点不明显时，用分箱趋势减少点重叠 |",
        "| 4 | `arch_rank_binned_trend_by_dataset_arch.png` | 看 Spearman 排序趋势 | 解释为什么相关性比肉眼散点更稳定 |",
        "| 5 | `arch_attack_heatmap.png` | 看不同攻击在不同模型上的位置 | 说明结论 attack-dependent，不能只报 overall |",
        "| 6 | `arch_defense_heatmap.png` | 看四个检测器分别贡献 | 解释 stealth_avg 是平均值，必须拆开看 |",
        "",
        md_image("arch_acc_analysis/figures/arch_metric_overview.png", "Architecture metric overview"),
        "",
        md_image("arch_acc_analysis/figures/arch_pairwise_delta_summary.png", "Pairwise delta summary"),
        "",
        md_image("arch_acc_analysis/figures/arch_binned_median_trend_by_dataset_arch.png", "Binned median trend"),
        "",
        md_image("arch_acc_analysis/figures/arch_rank_binned_trend_by_dataset_arch.png", "Rank-binned trend"),
        "",
        md_image("arch_acc_analysis/figures/arch_attack_heatmap.png", "Attack heatmap"),
        "",
        md_image("arch_acc_analysis/figures/arch_defense_heatmap.png", "Defense heatmap"),
        "",
        "HTML 展示页也已经生成：`analysis-transfer-asr2/ARCH_ACC_TRANSFER_STEALTH_DASHBOARD_CN.html`。如果要给老师快速展示，建议打开 HTML；如果要写论文/汇报稿，建议读本文档。",
        "",
        "## 2. 分析问题与指标",
        "",
        "本报告基于当前已经完成的实验结果，分析 ACC / 分类任务难度是否会影响迁移性与隐蔽性之间的关系。",
        "",
        "固定指标定义：",
        "",
        "```text",
        "difficulty = 1 - clean_acc",
        "transfer_rate = transfer_asr^2 / source_asr",
        "stealth_avg = mean(1 - TPR)",
        "```",
        "",
        "`stealth_avg` 使用源域四个检测防御：`SentiNet / STRIP / ScaleUp / IBD_PSC`。",
        "",
        "`transfer_rate` 的分母是 `source_asr`。当 `source_asr` 很低时，该指标可能被异常放大，所以主分析使用 `source_asr>=0.05`，并保留 `source_asr>=0.10` 的敏感性检查。",
        "",
        "## 3. 数据来源与完整性",
        "",
        f"- 总解析行数：`{len(df)}`",
        f"- 主分析行数：`{len(primary)}`",
        f"- baseline_full 行数：`{len(baseline)}`",
        f"- new_model_supplement 行数：`{len(supplement)}`",
        f"- 完整四防御行数：`{int(df['complete_defense_results'].sum()) if not df.empty else 0}`",
        f"- 有效 transfer_rate 行数：`{int(df['valid_transfer_rate'].sum()) if not df.empty else 0}`",
        "",
        "主分析过滤条件：四个防御完整、`transfer_rate` 可计算、`source_asr >= 0.05`。",
        "",
        "图表分布诊断：",
        "",
        f"- 主分析 transfer_rate 中位数：`{metric(diag_all, 'transfer_rate_median')}`",
        f"- transfer_rate 25%/75% 分位：`{metric(diag_all, 'transfer_rate_p25')}` / `{metric(diag_all, 'transfer_rate_p75')}`",
        f"- transfer_rate 95% 分位和最大值：`{metric(diag_all, 'transfer_rate_p95')}` / `{metric(diag_all, 'transfer_rate_max')}`",
        f"- transfer_rate 落在 `[0.9, 1.1]` 的比例：`{metric(diag_all, 'share_transfer_0p9_1p1')}`",
        f"- transfer_rate 大于 `1.5` 的比例：`{metric(diag_all, 'share_transfer_gt_1p5')}`",
        "",
        "这解释了为什么普通散点图看起来“不明显”：很多点挤在 transfer_rate 约 1 附近，并且不同攻击/模型混在一起。更适合汇报的是 pairwise delta、分箱趋势和热力图。",
        "",
        "## 4. 关键结果摘要",
        "",
        "### 4.1 总体关系",
        "",
        f"- 主分析样本中 `transfer_rate` 与 `stealth_avg` 呈负相关：Pearson=`{fmt_float(overall_pearson)}`，Spearman=`{fmt_float(overall_spearman)}`。这说明在当前结果里，迁移性越强，四个源域检测方法平均越容易检出，隐蔽性越低。",
        f"- 全局 ACC 分层后仍保持负相关：high_acc Spearman=`{metric(high_acc, 'spearman_transfer_stealth')}`，mid_acc Spearman=`{metric(mid_acc, 'spearman_transfer_stealth')}`，low_acc Spearman=`{metric(low_acc, 'spearman_transfer_stealth')}`。因此架构实验没有推翻 transfer-stealth 反向关系，但关系强弱确实随 ACC 区间变化。",
        "",
        "### 4.2 CIFAR-10 SmallCNN 与已有模型",
        "",
        f"- SmallCNN 的平均 clean ACC 为 `{metric(cifar_small, 'clean_acc_mean')}`，低于 CIFAR-10 ResNet18 的 `{metric(cifar_resnet18, 'clean_acc_mean')}`；它的平均 `transfer_rate` 为 `{metric(cifar_small, 'transfer_rate_mean')}`，高于 ResNet18 的 `{metric(cifar_resnet18, 'transfer_rate_mean')}`；但 `stealth_avg` 为 `{metric(cifar_small, 'stealth_avg_mean')}`，低于 ResNet18 的 `{metric(cifar_resnet18, 'stealth_avg_mean')}`。",
        f"- 在同配置 pairwise 对比中，SmallCNN 相对 ResNet18 的平均差值是：clean ACC `{signed_metric(small_vs_resnet18, 'delta_clean_acc_mean')}`，transfer_rate `{signed_metric(small_vs_resnet18, 'delta_transfer_rate_mean')}`，stealth_avg `{signed_metric(small_vs_resnet18, 'delta_stealth_avg_mean')}`。这组结果更像是“更弱模型让迁移率升高，但隐蔽性下降”，而不是简单的“任务更难就更隐蔽”。",
        "",
        "### 4.3 Tiny-ImageNet ResNet34 与已有模型",
        "",
        f"- ResNet34 的平均 clean ACC 为 `{metric(tiny_resnet34, 'clean_acc_mean')}`，高于 Tiny-ImageNet ResNet18 的 `{metric(tiny_resnet18, 'clean_acc_mean')}`；整体均值下 `transfer_rate` 为 `{metric(tiny_resnet34, 'transfer_rate_mean')}`，ResNet18 为 `{metric(tiny_resnet18, 'transfer_rate_mean')}`；`stealth_avg` 为 `{metric(tiny_resnet34, 'stealth_avg_mean')}`，高于 ResNet18 的 `{metric(tiny_resnet18, 'stealth_avg_mean')}`。",
        f"- 在同配置 pairwise 对比中，ResNet34 相对 ResNet18 的平均差值是：clean ACC `{signed_metric(r34_vs_resnet18, 'delta_clean_acc_mean')}`，transfer_rate `{signed_metric(r34_vs_resnet18, 'delta_transfer_rate_mean')}`，stealth_avg `{signed_metric(r34_vs_resnet18, 'delta_stealth_avg_mean')}`。这里要注意：同配置子集和整体均值不完全相同，pairwise 更适合看“新模型替换旧模型”的方向。",
        "",
        "### 4.4 当前结论边界",
        "",
        "- 架构实验支持：模型/数据集设置会改变迁移性与隐蔽性的斜率和水平，不能只看一个统一平均值。",
        "- 架构实验暂不支持：把 ACC 作为唯一因果变量。因为模型容量、归纳偏置、数据集类别数、输入尺寸和攻击机制都同时在变。",
        "- 与噪声实验结合时，应把噪声实验作为更接近“同模型下改变任务难度”的证据，把架构实验作为“模型生态对照”的证据。",
        "",
        "## 5. Dataset/Arch 总体汇总",
        "",
        limited_table(summary, key_summary_cols, 30, ["dataset", "result_group", "arch_base"]),
        "",
        "这张表回答“不同模型本身处于什么水平”。重点不是只看 clean ACC，而是一起看三列：`clean_acc_mean`、`transfer_rate_mean`、`stealth_avg_mean`。",
        "",
        "## 6. 总体 transfer-stealth 关系",
        "",
        f"在主分析样本中，`transfer_rate` 与 `stealth_avg` 的 Pearson 相关为 `{fmt_float(overall_pearson)}`，Spearman 相关为 `{fmt_float(overall_spearman)}`。",
        "",
        "解释时需要注意：`transfer_rate` 的分母是 source ASR，低 source ASR 会放大该指标，因此报告同时保留 `source_asr>=0.10` 的敏感性分析。",
        "",
        "## 7. ACC 分层结果",
        "",
        limited_table(acc_bins, key_acc_cols, 30, ["group_type", "dataset", "acc_bin"]),
        "",
        "ACC 分层是判断 difficulty 是否改变 transfer-stealth 关系的关键证据。如果 high/mid/low ACC 层的相关性方向或强度不同，说明总体平均会掩盖难度条件下的关系变化。当前全局 high/mid/low 都是负相关，但强度不同；这说明“迁移性和隐蔽性的反向关系”存在，同时 difficulty 会改变这个反向关系的强度。",
        "",
        "## 8. 新模型与已有模型的同配置 Pairwise Delta",
        "",
        f"可匹配 pairwise 行数：`{len(pairwise)}`",
        "",
    ]

    if not pairwise_summary.empty:
        lines.extend([limited_table(pairwise_summary, key_pair_cols, 20, ["dataset", "new_arch", "base_arch"]), ""])
    else:
        lines.extend(["当前新模型与 baseline 没有匹配到足够同配置结果，后续应优先补齐 ResNet18 对照配置。", ""])

    lines.extend(
        [
            "Pairwise delta 是架构补充实验里最值得讲的表，因为它尽量控制了 attack、poison_rate、strength、cover_rate 等配置，只看替换模型后的方向变化。它比整体均值更适合支持“模型能力变化会改变 tradeoff 水平”。",
            "",
            "## 9. 攻击类型分层",
            "",
            limited_table(attack_focus, attack_cols, 80, ["dataset", "attack_family", "arch_base"]),
            "",
            "攻击类型是强混杂因素。当前结果里，`badnet`、`adaptive_patch` 等局部/强触发攻击通常更容易形成高迁移但低隐蔽；`WaNet` 一类攻击则经常表现为较高隐蔽但迁移率更低。最终论文式结论必须按攻击机制拆开讨论，不能只报告整体平均。",
            "",
            "## 10. 四个防御方法拆解",
            "",
            limited_table(defense_breakdown, defense_cols, 80, ["dataset", "arch_base", "defense"]),
            "",
            "`stealth_avg` 是平均值，不代表四个检测方法同方向变化。比如 CIFAR-10 SmallCNN 的低 `stealth_avg` 很大程度来自 `IBD_PSC` 和 `ScaleUp` 的高 TPR；Tiny-ImageNet ResNet34 的较高 `stealth_avg` 则主要受 `SentiNet`、`STRIP` 和 `ScaleUp` 影响。报告结论应同时参考 defense breakdown。",
            "",
            "## 11. 分箱趋势和 rank 趋势",
            "",
            "分箱趋势图用于解决“普通散点图不明显”的问题。它把 transfer_rate 排序/分箱后看每个箱内的 stealth 中位数，能减少点重叠和极端值影响。",
            "",
            "- `arch_binned_median_trend_by_dataset_arch.csv`：对应 `arch_binned_median_trend_by_dataset_arch.png`，看真实数值分箱趋势。",
            "- `arch_rank_binned_trend_by_dataset_arch.csv`：对应 `arch_rank_binned_trend_by_dataset_arch.png`，看 Spearman 风格的排序趋势。",
            "",
            "分箱趋势表预览：",
            "",
            limited_table(binned_trend, ["dataset_arch", "bin_id", "n_rows", "transfer_rate_median", "stealth_avg_median", "clean_acc_mean"], 30, ["dataset_arch", "bin_id"]),
            "",
            "Rank 趋势表预览：",
            "",
            limited_table(rank_trend, ["dataset_arch", "bin_id", "n_rows", "transfer_rank_median", "stealth_rank_median"], 30, ["dataset_arch", "bin_id"]),
            "",
            "## 12. 回归与统计检验",
            "",
            "完整回归表见 `analysis-transfer-asr2/arch_acc_analysis/arch_acc_regression.txt`。解读规则：",
            "",
            "- `transfer_rate:difficulty` 显著：支持 difficulty 对 transfer-stealth 关系有线性调节证据。",
            "- 不显著但 ACC 分层差异明显：说明关系可能是非线性、分段或 attack-dependent。",
            "- `C(arch)` 或 `transfer_rate:C(dataset_arch)` 显著：说明模型/架构改变隐蔽性水平或 tradeoff 斜率。",
            "",
            "当前主回归的自动摘要：",
            "",
            *regression_interaction_note(primary),
            "",
            "## 13. 与噪声实验的关系",
            "",
            "噪声实验和架构实验应作为两条互补证据链：",
            "",
            "- 噪声实验：同模型同数据集下改变输入难度，更接近控制变量。",
            "- 架构实验：利用已有全面模型和新模型补充，检查现象是否在模型/数据集体系中复现。",
            "",
            "当前架构实验显示：transfer-stealth 负相关整体存在，而且不同 ACC 层、不同模型的斜率不同。前面噪声实验则更接近固定 SmallCNN 和 CIFAR-10 后观察 ACC 下降。两者应合并回答两个层次的问题：",
            "",
            "1. 噪声实验回答：在同一个模型和数据集内，输入难度提高后，transfer-stealth 关系是否改变。",
            "2. 架构实验回答：换模型、换数据集后，这种关系是否还稳定存在，或者被模型结构放大/削弱。",
            "3. 如果两者趋势一致，可以更强地支持“ACC/任务难度会调节迁移性与隐蔽性关系”。如果两者不一致，不应直接否定假设，而应检查 attack type、source ASR 长尾、以及四个防御方法是否对模型结构特别敏感。",
            "",
            "## 14. 汇报建议",
            "",
            "建议按 5 页讲，不要从大表开始：",
            "",
            "1. **问题和指标**：说明 `transfer_rate = transfer_asr^2 / source_asr`，`stealth_avg = mean(1 - TPR)`，目标是看 ACC/模型能力是否调节 transfer-stealth 关系。",
            "2. **总体趋势**：展示 `arch_binned_median_trend_by_dataset_arch.png` 或 `arch_rank_binned_trend_by_dataset_arch.png`，说整体负相关存在，但不是一条漂亮直线。",
            "3. **新模型对比**：展示 `arch_pairwise_delta_summary.png`，重点讲 SmallCNN 与 ResNet34 的同配置 delta。",
            "4. **混杂因素**：展示 `arch_attack_heatmap.png`，说明攻击类型决定点落在哪个区域，因此不能只报 overall mean。",
            "5. **检测器拆解**：展示 `arch_defense_heatmap.png`，说明 `stealth_avg` 的变化由不同 defense 共同驱动，尤其 SmallCNN 上 IBD_PSC/ScaleUp 更强。",
            "",
            "可以直接使用的汇报话术：",
            "",
            "```text",
            "我们先用完整旧基线确认 transfer-stealth tradeoff 整体存在；",
            "然后在架构补充实验中加入 CIFAR-10 SmallCNN 和 Tiny-ImageNet ResNet34。",
            "结果显示，换模型后 clean ACC、transfer_rate 和 stealth_avg 的水平都会变；",
            "同配置 pairwise delta 说明 SmallCNN 往往提高迁移率但降低隐蔽性，ResNet34 的方向则更受 Tiny-ImageNet 攻击配置影响。",
            "因此架构实验支持：模型能力/架构会改变 transfer-stealth tradeoff 的水平和斜率。",
            "但它不能单独证明 ACC 是唯一原因，因为模型容量、归纳偏置、数据集和攻击类型同时变化。",
            "所以我们把架构实验作为生态对照，把噪声实验作为更接近控制变量的 difficulty intervention。",
            "```",
            "",
            "## 15. 当前可以支持的结论边界",
            "",
            "- 可以支持：当前结果能够分析 ACC 区间、模型/数据集架构与 transfer-stealth 关系之间的关联。",
            "- 可以支持：新模型结果即使不完整，也能作为已有 baseline 的补充观察，尤其通过同配置 pairwise delta 使用。",
            "- 不能过度声称：模型/数据集变化单独证明 ACC 是唯一因果因素，因为架构、数据集、类别数、输入尺寸也同时变化。",
            "",
            "## 16. 输出文件说明",
            "",
            "- `arch_acc_transfer_stealth_rows.csv`：逐实验明细，所有架构分析的基础表。",
            "- `arch_acc_summary_by_dataset_arch.csv`：按 dataset/arch 汇总 ACC、ASR、transfer_rate、stealth。",
            "- `arch_acc_pairwise_delta.csv`：逐配置 pairwise delta。",
            "- `arch_acc_pairwise_delta_summary.csv`：pairwise delta 均值汇总，最适合汇报新模型影响。",
            "- `arch_acc_bin_transfer_stealth.csv`：ACC 分层结果。",
            "- `arch_acc_summary_by_attack.csv`：攻击类型分层结果。",
            "- `arch_defense_breakdown_summary.csv`：四个检测器拆解。",
            "- `arch_plot_diagnostics.csv`：解释普通散点图为什么不明显。",
            "- `arch_binned_median_trend_by_dataset_arch.csv`：分箱中位数趋势。",
            "- `arch_rank_binned_trend_by_dataset_arch.csv`：rank 分箱趋势。",
            "- `ARCH_ACC_TRANSFER_STEALTH_DASHBOARD_CN.html`：静态展示页。",
            "",
            "## 17. 后续最值得补的实验",
            "",
            "1. 优先补齐 `SmallCNN_cifar10 vs ResNet18_cifar10` 的同配置结果。",
            "2. 优先补齐 `ResNet34_tiny_imagenet vs ResNet18_tiny_imagenet` 的同配置结果。",
            "3. 对噪声实验补充一个 clean/noise baseline 的重复种子，用于判断 ACC 变化是否稳定。",
            "4. 所有核心结论同时报告 `source_asr>=0.05` 和 `source_asr>=0.10`。",
            "5. 最终论文式分析中，把攻击类型和防御方法拆开，不只报告整体平均。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_arch_html_dashboard(
    summary: pd.DataFrame,
    acc_bins: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    attack_summary: pd.DataFrame,
    defense_breakdown: pd.DataFrame,
    diagnostics: pd.DataFrame,
    corr: pd.DataFrame,
    path: Path,
) -> None:
    overall_corr = first_row(corr, group_type="all", group_name="all", source_asr_filter="source_asr>=0.05")
    high_acc = first_row(acc_bins, group_type="global_acc_bin", dataset="all", acc_bin="high_acc")
    mid_acc = first_row(acc_bins, group_type="global_acc_bin", dataset="all", acc_bin="mid_acc")
    low_acc = first_row(acc_bins, group_type="global_acc_bin", dataset="all", acc_bin="low_acc")
    diag_all = first_row(diagnostics, group="all")
    attack_focus = attack_summary[attack_summary["dataset"].isin(PRIMARY_DATASETS)].copy()
    attack_focus["transfer_minus_stealth_gap"] = attack_focus["transfer_rate_mean"] - attack_focus["stealth_avg_mean"]

    key_summary_cols = [
        "dataset",
        "arch_base",
        "result_group",
        "n_rows",
        "clean_acc_mean",
        "transfer_rate_mean",
        "stealth_avg_mean",
    ]
    key_pair_cols = [
        "dataset",
        "new_arch",
        "base_arch",
        "n_rows",
        "delta_clean_acc_mean",
        "delta_transfer_rate_mean",
        "delta_stealth_avg_mean",
    ]
    key_acc_cols = [
        "group_type",
        "dataset",
        "acc_bin",
        "n_rows",
        "clean_acc_mean",
        "transfer_rate_median",
        "stealth_avg_median",
        "spearman_transfer_stealth",
    ]
    attack_cols = [
        "dataset",
        "arch_base",
        "attack_family",
        "n_rows",
        "transfer_rate_mean",
        "stealth_avg_mean",
        "transfer_minus_stealth_gap",
    ]
    defense_cols = [
        "dataset",
        "arch_base",
        "defense",
        "tpr_mean",
        "stealth_mean",
        "auc_mean",
    ]
    figure_cards = [
        image_card(
            "arch_acc_analysis/figures/arch_metric_overview.png",
            "1. 架构指标总览",
            "先看不同模型把 clean ACC、transfer_rate、stealth_avg 推到了什么水平。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_pairwise_delta_summary.png",
            "2. 同配置 Delta",
            "新模型减旧模型，最适合讲 SmallCNN 和 ResNet34 的补充价值。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_binned_median_trend_by_dataset_arch.png",
            "3. 分箱中位数趋势",
            "减少散点重叠后，看 transfer_rate 和 stealth_avg 的整体方向。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_rank_binned_trend_by_dataset_arch.png",
            "4. Rank 趋势",
            "对应 Spearman 排序相关，适合解释趋势不一定是线性斜线。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_attack_heatmap.png",
            "5. 攻击类型热力图",
            "不同攻击天然落在不同区域，说明结论必须 attack-dependent。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_defense_heatmap.png",
            "6. 检测器热力图",
            "拆开四个 defense，解释 stealth_avg 的来源。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_transfer_vs_stealth_facets.png",
            "7. 分面散点图",
            "普通散点只作为分布说明，不作为唯一证据。",
        ),
    ]

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>架构 ACC-迁移性-隐蔽性分析展示</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5f6b7a;
      --line: #d8dee8;
      --band: #f5f7fb;
      --blue: #2457a6;
      --green: #287d55;
      --red: #b23b3b;
      --gold: #a66a00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      background: #ffffff;
    }}
    header {{
      padding: 28px 36px 20px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 30px 0 12px; font-size: 20px; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    p {{ margin: 0 0 10px; }}
    main {{ padding: 0 36px 42px; max-width: 1380px; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: #fff;
      padding: 14px 16px;
      border-radius: 8px;
    }}
    .metric b {{ display: block; font-size: 24px; margin-bottom: 2px; }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    .callout {{
      border-left: 4px solid var(--blue);
      background: var(--band);
      padding: 14px 16px;
      margin: 18px 0;
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .figure-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }}
    .figure-card img {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .figure-card p {{ color: var(--muted); margin-top: 8px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 20px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eef2f7; }}
    code {{
      background: #eef2f7;
      padding: 1px 4px;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    ol, ul {{ padding-left: 22px; }}
    @media (max-width: 980px) {{
      main, header {{ padding-left: 18px; padding-right: 18px; }}
      .summary-grid, .figure-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>架构/模型 ACC 对迁移性与隐蔽性关系的分析展示</h1>
    <p>静态 dashboard，数据来自 <code>analysis-transfer-asr2/arch_acc_analysis/</code>。重点是解释 SmallCNN 和 ResNet34 架构补充如何改变 ACC、transfer_rate 和 stealth_avg。</p>
    <div class="summary-grid">
      <div class="metric"><b>{metric(overall_corr, 'spearman_transfer_rate_stealth_avg')}</b><span>overall Spearman transfer-stealth</span></div>
      <div class="metric"><b>{metric(high_acc, 'spearman_transfer_stealth')}</b><span>high ACC Spearman</span></div>
      <div class="metric"><b>{metric(mid_acc, 'spearman_transfer_stealth')}</b><span>mid ACC Spearman</span></div>
      <div class="metric"><b>{metric(low_acc, 'spearman_transfer_stealth')}</b><span>low ACC Spearman</span></div>
    </div>
  </header>
  <main>
    <section class="callout">
      <p><b>一句话结论：</b>架构实验支持 transfer-stealth 负相关整体存在，并且模型/数据集/ACC 区间会改变关系强弱；但架构变化不是纯 ACC 干预，不能单独证明 ACC 是唯一因果变量。</p>
      <p><b>为什么散点不明显：</b>transfer_rate 中位数为 <code>{metric(diag_all, 'transfer_rate_median')}</code>，落在 [0.9, 1.1] 的比例为 <code>{metric(diag_all, 'share_transfer_0p9_1p1')}</code>，所以需要看 pairwise delta、分箱趋势和热力图。</p>
    </section>

    <h2>建议展示顺序</h2>
    <ol>
      <li>先用“架构指标总览”说明新模型把 ACC/迁移性/隐蔽性推到不同区间。</li>
      <li>再用“同配置 Delta”讲 SmallCNN 和 ResNet34 相对旧模型的方向。</li>
      <li>用“分箱趋势”和“Rank 趋势”展示 transfer-stealth 关系，不把普通散点当唯一证据。</li>
      <li>用“攻击热力图”和“检测器热力图”说明 attack/defense 是强混杂因素。</li>
    </ol>

    <h2>核心图表</h2>
    <div class="figure-grid">
      {''.join(figure_cards)}
    </div>

    <h2>Dataset / Arch 汇总</h2>
    {html_table(summary, key_summary_cols, 30, ["dataset", "result_group", "arch_base"])}

    <h2>同配置 Pairwise Delta</h2>
    {html_table(pairwise_summary, key_pair_cols, 20, ["dataset", "new_arch", "base_arch"])}

    <h2>ACC 分层结果</h2>
    {html_table(acc_bins, key_acc_cols, 30, ["group_type", "dataset", "acc_bin"])}

    <h2>攻击类型分层</h2>
    {html_table(attack_focus, attack_cols, 80, ["dataset", "attack_family", "arch_base"])}

    <h2>四个检测器拆解</h2>
    {html_table(defense_breakdown, defense_cols, 80, ["dataset", "arch_base", "defense"])}
  </main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_acc_moderation_html_dashboard(
    summary: pd.DataFrame,
    acc_moderation: pd.DataFrame,
    acc_bin_slope: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    attack_summary: pd.DataFrame,
    defense_breakdown: pd.DataFrame,
    diagnostics: pd.DataFrame,
    path: Path,
) -> None:
    main_mod = first_row(acc_moderation, cohort="primary_main_source_asr>=0.05")
    sens_mod = first_row(acc_moderation, cohort="primary_main_source_asr>=0.10")
    diag_all = first_row(diagnostics, group="all")
    attack_focus = attack_summary[attack_summary["dataset"].isin(PRIMARY_DATASETS)].copy()
    attack_focus["transfer_minus_stealth_gap"] = attack_focus["transfer_rate_mean"] - attack_focus["stealth_avg_mean"]

    key_mod_cols = [
        "cohort",
        "n_rows",
        "spearman_transfer_stealth",
        "spearman_clean_acc_transfer",
        "spearman_clean_acc_stealth",
        "raw_interaction_coef",
        "raw_interaction_pvalue",
        "std_interaction_coef",
        "std_interaction_pvalue",
        "std_r_squared",
    ]
    key_acc_cols = [
        "group_type",
        "dataset",
        "acc_bin",
        "n_rows",
        "clean_acc_mean",
        "difficulty_mean",
        "transfer_rate_median",
        "stealth_avg_median",
        "spearman_transfer_stealth",
    ]
    key_summary_cols = [
        "dataset",
        "arch_base",
        "result_group",
        "n_rows",
        "clean_acc_mean",
        "difficulty_mean",
        "transfer_rate_mean",
        "stealth_avg_mean",
    ]
    key_pair_cols = [
        "dataset",
        "new_arch",
        "base_arch",
        "n_rows",
        "delta_clean_acc_mean",
        "delta_transfer_rate_mean",
        "delta_stealth_avg_mean",
    ]
    attack_cols = [
        "dataset",
        "arch_base",
        "attack_family",
        "n_rows",
        "clean_acc_mean",
        "transfer_rate_mean",
        "stealth_avg_mean",
        "transfer_minus_stealth_gap",
    ]
    defense_cols = [
        "dataset",
        "arch_base",
        "defense",
        "tpr_mean",
        "stealth_mean",
        "auc_mean",
    ]
    figure_cards = [
        image_card(
            "arch_acc_analysis/figures/arch_acc_moderation_summary.png",
            "1. ACC 调节效应总览",
            "先看 ACC 对 transfer 有边际关系、对 stealth 直接解释弱，并且按 ACC 分层后 transfer-stealth 关系发生变化。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_interaction_slope_by_acc.png",
            "2. 交互斜率",
            "低 ACC / 高 difficulty 下，transfer_rate 增加对应的 stealth_avg 下降更明显。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_acc_bin_transfer_stealth.png",
            "3. ACC 分层散点",
            "把 high/mid/low ACC bin 分开看，避免整体散点把斜率差异糊在一起。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_dataset_acc_bin_spearman.png",
            "4. 数据集内部分箱相关",
            "展示 CIFAR-10 和 Tiny-ImageNet 内部的 ACC-bin Spearman，减少全局混合误读。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_pairwise_delta_summary.png",
            "5. 同配置 Delta",
            "模型替换后 ACC、transfer、stealth 的共同变化；这是对照证据，不是主因果证据。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_attack_heatmap.png",
            "6. 攻击类型热力图",
            "不同攻击天然落在不同区域，说明结论必须 attack-dependent。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_defense_heatmap.png",
            "7. 检测器热力图",
            "拆开四个 defense，解释 stealth_avg 的来源。",
        ),
    ]

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ACC 调节迁移性-隐蔽性关系展示</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5f6b7a;
      --line: #d8dee8;
      --band: #f5f7fb;
      --blue: #2457a6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      background: #ffffff;
    }}
    header {{
      padding: 28px 36px 20px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 30px 0 12px; font-size: 20px; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    p {{ margin: 0 0 10px; }}
    main {{ padding: 0 36px 42px; max-width: 1380px; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: #fff;
      padding: 14px 16px;
      border-radius: 8px;
    }}
    .metric b {{ display: block; font-size: 24px; margin-bottom: 2px; }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    .callout {{
      border-left: 4px solid var(--blue);
      background: var(--band);
      padding: 14px 16px;
      margin: 18px 0;
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .figure-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }}
    .figure-card img {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .figure-card p {{ color: var(--muted); margin-top: 8px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 20px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eef2f7; }}
    code {{
      background: #eef2f7;
      padding: 1px 4px;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    ol, ul {{ padding-left: 22px; }}
    @media (max-width: 980px) {{
      main, header {{ padding-left: 18px; padding-right: 18px; }}
      .summary-grid, .figure-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ACC 是否调节迁移性与隐蔽性的关系</h1>
    <p>静态 dashboard，数据来自 <code>analysis-transfer-asr2/arch_acc_analysis/</code>。主线是 ACC/difficulty 是否改变 <code>transfer_rate -> stealth_avg</code> 的斜率。</p>
    <div class="summary-grid">
      <div class="metric"><b>{metric(main_mod, 'spearman_transfer_stealth')}</b><span>overall Spearman transfer-stealth</span></div>
      <div class="metric"><b>{metric(main_mod, 'std_interaction_coef')}</b><span>controlled interaction coef</span></div>
      <div class="metric"><b>{sci_metric(main_mod, 'std_interaction_pvalue')}</b><span>interaction p-value</span></div>
      <div class="metric"><b>{sci_metric(sens_mod, 'std_interaction_pvalue')}</b><span>source_asr&gt;=0.10 p-value</span></div>
    </div>
  </header>
  <main>
    <section class="callout">
      <p><b>一句话结论：</b>ACC 与 transfer_rate 本身有关，但对 stealth_avg 的直接解释弱；更关键的是 <code>difficulty = 1 - clean_acc</code> 与 <code>transfer_rate</code> 的交互项显著为负，说明 ACC 下降、任务难度上升时，迁移性增强带来的隐蔽性下降更明显。</p>
      <p><b>边界：</b>架构实验不是纯 ACC 干预，模型容量、归纳偏置、数据集和攻击类型同时变化；因此这里是调节证据，不是单一因果证明。</p>
      <p><b>散点为什么不明显：</b>transfer_rate 中位数为 <code>{metric(diag_all, 'transfer_rate_median')}</code>，落在 [0.9, 1.1] 的比例为 <code>{metric(diag_all, 'share_transfer_0p9_1p1')}</code>，所以优先看交互斜率和分层图。</p>
    </section>

    <h2>建议展示顺序</h2>
    <ol>
      <li>先讲 ACC 不是强单变量，而是 transfer-stealth 关系的 moderator。</li>
      <li>展示交互斜率图，说明低 ACC / 高 difficulty 下 tradeoff 更陡。</li>
      <li>用数据集内 ACC-bin Spearman 解释分层结果，而不是追求全局单调排序。</li>
      <li>最后用 pairwise delta、attack heatmap、defense heatmap 说明混杂因素。</li>
    </ol>

    <h2>核心图表</h2>
    <div class="figure-grid">
      {''.join(figure_cards)}
    </div>

    <h2>ACC 调节效应摘要</h2>
    {html_table(acc_moderation, key_mod_cols, 10, ["cohort"])}

    <h2>ACC 分层斜率</h2>
    {html_table(acc_bin_slope, key_acc_cols, 30, ["group_type", "dataset", "acc_bin"])}

    <h2>Dataset / Arch 汇总</h2>
    {html_table(summary[summary["dataset"].isin(PRIMARY_DATASETS)].copy(), key_summary_cols, 30, ["dataset", "result_group", "arch_base"])}

    <h2>同配置 Pairwise Delta</h2>
    {html_table(pairwise_summary, key_pair_cols, 20, ["dataset", "new_arch", "base_arch"])}

    <h2>攻击类型分层</h2>
    {html_table(attack_focus, attack_cols, 80, ["dataset", "attack_family", "arch_base"])}

    <h2>四个检测器拆解</h2>
    {html_table(defense_breakdown, defense_cols, 80, ["dataset", "arch_base", "defense"])}
  </main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_new_model_arch_result_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    attack_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    relationship_shift: pd.DataFrame,
    defense_breakdown: pd.DataFrame,
    diagnostics: pd.DataFrame,
    report_path: Path,
) -> None:
    primary = exclude_arch_attacks(df[df["primary_main_analysis"]].copy())
    baseline = primary[primary["result_group"] == "baseline_full"]
    supplement = primary[primary["result_group"] == "new_model_supplement"]
    diag_all = first_row(diagnostics, group="all")

    cifar_small = first_row(summary, dataset="cifar10", arch_base="SmallCNN")
    cifar_resnet18 = first_row(summary, dataset="cifar10", arch_base="ResNet18")
    tiny_resnet34 = first_row(summary, dataset="tiny_imagenet", arch_base="ResNet34")
    tiny_resnet18 = first_row(summary, dataset="tiny_imagenet", arch_base="ResNet18")

    small_vs_resnet18 = first_row(pairwise_summary, dataset="cifar10", new_arch="SmallCNN", base_arch="ResNet18")
    small_vs_mobile = first_row(pairwise_summary, dataset="cifar10", new_arch="SmallCNN", base_arch="mobilenetv2")
    small_vs_vgg = first_row(pairwise_summary, dataset="cifar10", new_arch="SmallCNN", base_arch="vgg19_bn")
    r34_vs_resnet18 = first_row(pairwise_summary, dataset="tiny_imagenet", new_arch="ResNet34", base_arch="ResNet18")
    r34_vs_mobile = first_row(pairwise_summary, dataset="tiny_imagenet", new_arch="ResNet34", base_arch="mobilenetv2")
    r34_vs_vgg = first_row(pairwise_summary, dataset="tiny_imagenet", new_arch="ResNet34", base_arch="vgg19_bn")
    small_rel_resnet18 = first_row(relationship_shift, dataset="cifar10", new_arch="SmallCNN", base_arch="ResNet18")
    r34_rel_resnet18 = first_row(relationship_shift, dataset="tiny_imagenet", new_arch="ResNet34", base_arch="ResNet18")

    key_summary_cols = [
        "dataset",
        "arch_base",
        "result_group",
        "n_rows",
        "clean_acc_mean",
        "difficulty_mean",
        "source_asr_mean",
        "transfer_asr_mean",
        "transfer_rate_mean",
        "transfer_rate_median",
        "stealth_avg_mean",
        "stealth_avg_median",
    ]
    key_pair_cols = [
        "dataset",
        "new_arch",
        "base_arch",
        "n_rows",
        "delta_clean_acc_mean",
        "delta_transfer_rate_mean",
        "delta_stealth_avg_mean",
        "delta_source_asr_mean",
        "delta_transfer_asr_mean",
    ]
    key_relationship_cols = [
        "dataset",
        "new_arch",
        "base_arch",
        "n_rows",
        "base_spearman_transfer_stealth",
        "new_spearman_transfer_stealth",
        "delta_spearman_transfer_stealth",
        "base_pearson_transfer_stealth",
        "new_pearson_transfer_stealth",
        "delta_pearson_transfer_stealth",
        "base_spearman_acc_transfer",
        "new_spearman_acc_transfer",
        "base_spearman_acc_stealth",
        "new_spearman_acc_stealth",
    ]
    attack_focus = attack_summary[attack_summary["dataset"].isin(PRIMARY_DATASETS)].copy()
    attack_focus["transfer_minus_stealth_gap"] = attack_focus["transfer_rate_mean"] - attack_focus["stealth_avg_mean"]
    attack_cols = [
        "dataset",
        "arch_base",
        "attack_family",
        "n_rows",
        "clean_acc_mean",
        "transfer_rate_mean",
        "stealth_avg_mean",
        "transfer_minus_stealth_gap",
    ]
    defense_cols = [
        "dataset",
        "arch_base",
        "defense",
        "n_rows",
        "tpr_mean",
        "stealth_mean",
        "auc_mean",
    ]

    lines = [
        "# 新增架构模型对 ACC、迁移性与隐蔽性关系的影响",
        "",
        "## 0. 先看结论",
        "",
        "这份报告只聚焦新增的两个模型：`CIFAR-10 SmallCNN` 和 `Tiny-ImageNet ResNet34`。核心问题是：新增模型造成 ACC 变化后，相对已有 baseline，迁移性和隐蔽性的关系发生了什么变化。",
        "",
        "本轮按你的问题重新生成，主表和主图都排除 `SIG` 与 `upgd`。因此这里看的不是完整攻击集合，而是去掉这两个强驱动方法后的架构对照。",
        "",
        "最重要的结论是：",
        "",
        f"- **SmallCNN 让 CIFAR-10 的 ACC 下降，但迁移性上升、隐蔽性下降。**相对 CIFAR-10 ResNet18，同配置 pairwise delta 为：clean ACC `{signed_metric(small_vs_resnet18, 'delta_clean_acc_mean')}`，transfer_rate `{signed_metric(small_vs_resnet18, 'delta_transfer_rate_mean')}`，stealth_avg `{signed_metric(small_vs_resnet18, 'delta_stealth_avg_mean')}`。",
        f"- **SmallCNN 还让 transfer-stealth 负相关明显变强。**在 SmallCNN - ResNet18 的同配置子集里，Spearman 从 `{metric(small_rel_resnet18, 'base_spearman_transfer_stealth')}` 变为 `{metric(small_rel_resnet18, 'new_spearman_transfer_stealth')}`，Pearson 从 `{metric(small_rel_resnet18, 'base_pearson_transfer_stealth')}` 变为 `{metric(small_rel_resnet18, 'new_pearson_transfer_stealth')}`。这说明不是只有均值变了，迁移性和隐蔽性的 tradeoff 也更陡了。",
        f"- SmallCNN 相对 MobileNetV2 / VGG19-BN 也呈现同样方向：transfer_rate 分别 `{signed_metric(small_vs_mobile, 'delta_transfer_rate_mean')}` / `{signed_metric(small_vs_vgg, 'delta_transfer_rate_mean')}`，stealth_avg 分别 `{signed_metric(small_vs_mobile, 'delta_stealth_avg_mean')}` / `{signed_metric(small_vs_vgg, 'delta_stealth_avg_mean')}`。这说明 SmallCNN 补充实验的主要现象是：**更弱模型更容易迁移，但更不隐蔽**。",
        f"- **ResNet34 让 Tiny-ImageNet 的 ACC 上升，但去掉 SIG/UPGD 后，迁移性提升不再稳健，隐蔽性仍下降。**相对 Tiny-ImageNet ResNet18，同配置 pairwise delta 为：clean ACC `{signed_metric(r34_vs_resnet18, 'delta_clean_acc_mean')}`，transfer_rate `{signed_metric(r34_vs_resnet18, 'delta_transfer_rate_mean')}`，stealth_avg `{signed_metric(r34_vs_resnet18, 'delta_stealth_avg_mean')}`。",
        f"- ResNet34 - ResNet18 的 transfer-stealth 相关性变化较小：Spearman 从 `{metric(r34_rel_resnet18, 'base_spearman_transfer_stealth')}` 变为 `{metric(r34_rel_resnet18, 'new_spearman_transfer_stealth')}`，Pearson 从 `{metric(r34_rel_resnet18, 'base_pearson_transfer_stealth')}` 变为 `{metric(r34_rel_resnet18, 'new_pearson_transfer_stealth')}`。所以 ResNet34 更适合讲“迁移性提升不稳、隐蔽性不改善”，不适合讲明显加强 tradeoff。",
        f"- ResNet34 相对 MobileNetV2 / VGG19-BN 的 transfer_rate 只剩接近 0 的小幅变化 `{signed_metric(r34_vs_mobile, 'delta_transfer_rate_mean')}` / `{signed_metric(r34_vs_vgg, 'delta_transfer_rate_mean')}`，stealth_avg 分别 `{signed_metric(r34_vs_mobile, 'delta_stealth_avg_mean')}` / `{signed_metric(r34_vs_vgg, 'delta_stealth_avg_mean')}`。这说明之前 ResNet34 的“迁移性上升”很大一部分来自 SIG/UPGD，不能把它写成所有攻击上的普遍架构规律。",
        "",
        "一句话总结：去掉 SIG/UPGD 后，SmallCNN 仍然稳定表现为“ACC 降低、迁移性上升、隐蔽性下降”；ResNet34 不再支持“迁移性明显上升”，更像是“ACC 提高，但迁移性接近不变，隐蔽性不改善甚至下降”。架构影响需要按攻击类型拆开看。",
        "",
        "## 1. 最应该看的图",
        "",
        "| 优先级 | 图 | 重点看什么 | 汇报时怎么说 |",
        "|---:|---|---|---|",
        "| 1 | `arch_pairwise_delta_summary.png` | 排除 SIG/UPGD 后，新模型减 baseline 的 ACC、transfer_rate、stealth_avg 方向 | 这是架构补充实验最核心的图 |",
        "| 2 | `arch_pairwise_relationship_shift.png` | transfer-stealth 相关性在 baseline 和新模型之间怎么变 | 回答“关系本身有没有变强/变弱” |",
        "| 3 | `arch_acc_correlation_shift.png` | ACC 与 transfer/stealth 的相关性怎么变 | 回答“ACC 怎么影响二者关系” |",
        "| 4 | `arch_metric_overview.png` | 每个 dataset/arch 的三项均值水平 | 用它说明新增模型落在哪个位置 |",
        "| 5 | `arch_transfer_vs_stealth_facets.png` | 新模型在 transfer-stealth 平面上的分布 | 看新增模型是否移动到高迁移/低隐蔽区域 |",
        "| 6 | `arch_attack_heatmap.png` | 哪些攻击驱动变化 | 说明现象 attack-dependent |",
        "| 7 | `arch_defense_heatmap.png` | 哪些检测器驱动 stealth_avg 变化 | 说明隐蔽性下降来自检测器响应 |",
        "",
        md_image("arch_acc_analysis/figures/arch_pairwise_delta_summary.png", "Pairwise delta summary"),
        "",
        md_image("arch_acc_analysis/figures/arch_pairwise_relationship_shift.png", "Pairwise relationship shift"),
        "",
        md_image("arch_acc_analysis/figures/arch_acc_correlation_shift.png", "ACC correlation shift"),
        "",
        md_image("arch_acc_analysis/figures/arch_metric_overview.png", "Architecture metric overview"),
        "",
        md_image("arch_acc_analysis/figures/arch_transfer_vs_stealth_facets.png", "Transfer-stealth facets"),
        "",
        md_image("arch_acc_analysis/figures/arch_attack_heatmap.png", "Attack heatmap"),
        "",
        md_image("arch_acc_analysis/figures/arch_defense_heatmap.png", "Defense heatmap"),
        "",
        "HTML 展示页同步更新：`analysis-transfer-asr2/ARCH_ACC_TRANSFER_STEALTH_DASHBOARD_CN.html`。",
        "",
        "## 2. 分析口径",
        "",
        "新增模型和 baseline 的对照关系：",
        "",
        "- `CIFAR-10 SmallCNN` 对照已有 `CIFAR-10 ResNet18 / MobileNetV2 / VGG19-BN`。",
        "- `Tiny-ImageNet ResNet34` 对照已有 `Tiny-ImageNet ResNet18 / MobileNetV2 / VGG19-BN`。",
        "- 本轮主结果排除 `SIG` 和 `upgd`；如果要讲完整攻击集合，需要单独说明 SIG/UPGD 会显著抬高 ResNet34 的 transfer_rate。",
        "",
        "指标定义：",
        "",
        "```text",
        "difficulty = 1 - clean_acc",
        "transfer_rate = transfer_asr^2 / source_asr",
        "stealth_avg = mean(1 - TPR)",
        "```",
        "",
        "这里最关键的是同配置 pairwise delta。它尽量固定 attack、poison_rate、strength、cover_rate，只看把模型从 baseline 换成新增模型后，三项指标怎么变。整体均值只作为背景，pairwise delta 才是主证据。",
        "",
        "数据规模：",
        "",
        f"- 主分析行数：`{len(primary)}`",
        f"- baseline_full 主分析行数：`{len(baseline)}`",
        f"- new_model_supplement 主分析行数：`{len(supplement)}`",
        f"- pairwise 可匹配行数：`{len(pairwise)}`",
        f"- transfer_rate 中位数：`{metric(diag_all, 'transfer_rate_median')}`",
        "",
        "## 3. CIFAR-10 SmallCNN：ACC 降低后，迁移性升高、隐蔽性降低",
        "",
        "整体均值对比：",
        "",
        f"- SmallCNN：clean ACC=`{metric(cifar_small, 'clean_acc_mean')}`，transfer_rate=`{metric(cifar_small, 'transfer_rate_mean')}`，stealth_avg=`{metric(cifar_small, 'stealth_avg_mean')}`。",
        f"- ResNet18：clean ACC=`{metric(cifar_resnet18, 'clean_acc_mean')}`，transfer_rate=`{metric(cifar_resnet18, 'transfer_rate_mean')}`，stealth_avg=`{metric(cifar_resnet18, 'stealth_avg_mean')}`。",
        "",
        "同配置 pairwise 结果：",
        "",
        "| 对照 | Δ clean ACC | Δ transfer_rate | Δ stealth_avg | 解读 |",
        "|---|---:|---:|---:|---|",
        f"| SmallCNN - ResNet18 | {signed_metric(small_vs_resnet18, 'delta_clean_acc_mean')} | {signed_metric(small_vs_resnet18, 'delta_transfer_rate_mean')} | {signed_metric(small_vs_resnet18, 'delta_stealth_avg_mean')} | ACC 降低，迁移性上升，隐蔽性下降 |",
        f"| SmallCNN - MobileNetV2 | {signed_metric(small_vs_mobile, 'delta_clean_acc_mean')} | {signed_metric(small_vs_mobile, 'delta_transfer_rate_mean')} | {signed_metric(small_vs_mobile, 'delta_stealth_avg_mean')} | 同方向，说明不是只相对 ResNet18 成立 |",
        f"| SmallCNN - VGG19-BN | {signed_metric(small_vs_vgg, 'delta_clean_acc_mean')} | {signed_metric(small_vs_vgg, 'delta_transfer_rate_mean')} | {signed_metric(small_vs_vgg, 'delta_stealth_avg_mean')} | 同方向，支持 SmallCNN 高迁移低隐蔽趋势 |",
        "",
        "这个结果可以这样讲：SmallCNN 降低了源模型的分类能力，但后门在迁移域上更容易保持效果；与此同时，四个源域检测方法平均更容易检出，所以 stealth_avg 下降。它体现的是“更容易迁移，但更不隐蔽”的 tradeoff。",
        "",
        "## 4. Tiny-ImageNet ResNet34：ACC 提高后，迁移性提升基本消失，隐蔽性不改善",
        "",
        "整体均值对比：",
        "",
        f"- ResNet34：clean ACC=`{metric(tiny_resnet34, 'clean_acc_mean')}`，transfer_rate=`{metric(tiny_resnet34, 'transfer_rate_mean')}`，stealth_avg=`{metric(tiny_resnet34, 'stealth_avg_mean')}`。",
        f"- ResNet18：clean ACC=`{metric(tiny_resnet18, 'clean_acc_mean')}`，transfer_rate=`{metric(tiny_resnet18, 'transfer_rate_mean')}`，stealth_avg=`{metric(tiny_resnet18, 'stealth_avg_mean')}`。",
        "",
        "同配置 pairwise 结果：",
        "",
        "| 对照 | Δ clean ACC | Δ transfer_rate | Δ stealth_avg | 解读 |",
        "|---|---:|---:|---:|---|",
        f"| ResNet34 - ResNet18 | {signed_metric(r34_vs_resnet18, 'delta_clean_acc_mean')} | {signed_metric(r34_vs_resnet18, 'delta_transfer_rate_mean')} | {signed_metric(r34_vs_resnet18, 'delta_stealth_avg_mean')} | ACC 提高，迁移性没有上升，隐蔽性下降 |",
        f"| ResNet34 - MobileNetV2 | {signed_metric(r34_vs_mobile, 'delta_clean_acc_mean')} | {signed_metric(r34_vs_mobile, 'delta_transfer_rate_mean')} | {signed_metric(r34_vs_mobile, 'delta_stealth_avg_mean')} | ACC 提高，迁移性接近不变，隐蔽性小幅下降 |",
        f"| ResNet34 - VGG19-BN | {signed_metric(r34_vs_vgg, 'delta_clean_acc_mean')} | {signed_metric(r34_vs_vgg, 'delta_transfer_rate_mean')} | {signed_metric(r34_vs_vgg, 'delta_stealth_avg_mean')} | ACC 提高，迁移性仅小幅变化，隐蔽性下降 |",
        "",
        "这个结果很重要：ResNet34 的 ACC 提高了，但 transfer_rate 没有明显上升，stealth_avg 也没有同步提高。这说明去掉 SIG/UPGD 后，ResNet34 不再是“高 ACC 也高迁移”的强证据，而更适合用来说明：更深模型会改变检测器响应，但 ACC 不能单独决定迁移性或隐蔽性。",
        "",
        "## 5. 总表：Dataset / Arch 均值",
        "",
        limited_table(summary[summary["dataset"].isin(PRIMARY_DATASETS)].copy(), key_summary_cols, 30, ["dataset", "result_group", "arch_base"]),
        "",
        "这张表用于看每个模型的绝对水平。汇报时不要只读均值，重点还是下一张 pairwise delta 表，因为 pairwise 更接近公平对照。",
        "",
        "## 6. 主证据：同配置 Pairwise Delta",
        "",
        limited_table(pairwise_summary, key_pair_cols, 20, ["dataset", "new_arch", "base_arch"]) if not pairwise_summary.empty else "当前没有足够 pairwise 数据。",
        "",
        "读这张表时只看三个 delta：",
        "",
        "- `delta_clean_acc_mean`：新增模型是否改变 ACC。",
        "- `delta_transfer_rate_mean`：迁移性是否上升。",
        "- `delta_stealth_avg_mean`：隐蔽性是否上升。负值表示更容易被检测。",
        "",
        "当前最值得汇报的现象是：SmallCNN 的高迁移、低隐蔽非常稳定；ResNet34 在排除 SIG/UPGD 后，transfer_rate 提升基本消失，但 stealth_avg 仍不改善。这正好解释了为什么需要单独拿掉 SIG/UPGD 做敏感性检查。",
        "",
        "## 7. 相关性变化：ACC 怎么影响迁移性与隐蔽性的关系",
        "",
        "这里回答你关心的第二层问题：不是只看均值升降，而是看 `transfer_rate` 和 `stealth_avg` 的关系本身有没有因为换模型而改变。",
        "",
        limited_table(relationship_shift, key_relationship_cols, 20, ["dataset", "new_arch", "base_arch"]) if not relationship_shift.empty else "当前没有足够 relationship shift 数据。",
        "",
        "重点读法：",
        "",
        f"- **SmallCNN - ResNet18：transfer-stealth 负相关明显增强。**同配置子集里，Spearman 从 `{metric(small_rel_resnet18, 'base_spearman_transfer_stealth')}` 变为 `{metric(small_rel_resnet18, 'new_spearman_transfer_stealth')}`，变化 `{signed_metric(small_rel_resnet18, 'delta_spearman_transfer_stealth')}`；Pearson 从 `{metric(small_rel_resnet18, 'base_pearson_transfer_stealth')}` 变为 `{metric(small_rel_resnet18, 'new_pearson_transfer_stealth')}`，变化 `{signed_metric(small_rel_resnet18, 'delta_pearson_transfer_stealth')}`。这说明 SmallCNN 不只是把平均 transfer_rate 拉高、stealth_avg 拉低，还让“迁移越强、隐蔽越差”的关系更陡。",
        f"- **SmallCNN 中 ACC 对两个单独指标的直接相关不强。**同配置子集里，Spearman(ACC, transfer_rate) 从 ResNet18 `{metric(small_rel_resnet18, 'base_spearman_acc_transfer')}` 到 SmallCNN `{metric(small_rel_resnet18, 'new_spearman_acc_transfer')}`；Spearman(ACC, stealth_avg) 从 ResNet18 `{metric(small_rel_resnet18, 'base_spearman_acc_stealth')}` 到 SmallCNN `{metric(small_rel_resnet18, 'new_spearman_acc_stealth')}`。所以更准确的解释不是“ACC 单独决定 transfer 或 stealth”，而是 **ACC 下降伴随模型结构变弱，使 transfer-stealth tradeoff 更明显**。",
        f"- **ResNet34 - ResNet18：相关性没有明显变强。**Spearman 从 `{metric(r34_rel_resnet18, 'base_spearman_transfer_stealth')}` 变为 `{metric(r34_rel_resnet18, 'new_spearman_transfer_stealth')}`，变化 `{signed_metric(r34_rel_resnet18, 'delta_spearman_transfer_stealth')}`；Pearson 从 `{metric(r34_rel_resnet18, 'base_pearson_transfer_stealth')}` 变为 `{metric(r34_rel_resnet18, 'new_pearson_transfer_stealth')}`，变化 `{signed_metric(r34_rel_resnet18, 'delta_pearson_transfer_stealth')}`。这支持前面的判断：ResNet34 的 ACC 提高没有让迁移性更强，也没有让 tradeoff 更陡。",
        "",
        "这一节的结论可以浓缩为一句：**ACC 不是稳定的直接解释变量；在 SmallCNN 这种模型能力明显下降的场景里，ACC 变化伴随架构变化，会让 transfer-stealth 的负相关变强。ResNet34 则说明 ACC 提高不一定强化或缓解这个关系。**",
        "",
        "## 8. 攻击类型和检测器拆解",
        "",
        "如果老师追问“为什么 stealth 下降”，看这两张表和对应 heatmap。",
        "",
        "攻击类型分层：",
        "",
        limited_table(attack_focus, attack_cols, 80, ["dataset", "attack_family", "arch_base"]),
        "",
        "检测器拆解：",
        "",
        limited_table(defense_breakdown, defense_cols, 80, ["dataset", "arch_base", "defense"]),
        "",
        "解释规则很简单：`stealth_avg` 是四个检测器平均值。如果新增模型的 stealth_avg 下降，需要检查是 SentiNet、STRIP、ScaleUp 还是 IBD-PSC 贡献最大。",
        "",
        "## 9. 汇报建议",
        "",
        "建议按 5 页讲，不要从复杂回归开始：",
        "",
        "1. **新增了什么模型**：CIFAR-10 加 SmallCNN，Tiny-ImageNet 加 ResNet34。",
        "2. **ACC 怎么变**：SmallCNN 降低 ACC；ResNet34 提高 Tiny-ImageNet ACC。",
        "3. **迁移和隐蔽怎么变**：SmallCNN 仍然 transfer_rate 上升、stealth_avg 下降；ResNet34 去掉 SIG/UPGD 后 transfer_rate 接近不变，但 stealth_avg 不改善。",
        "4. **相关性怎么变**：SmallCNN - ResNet18 的 transfer-stealth Spearman 从约 -0.46 变到约 -0.89，说明 tradeoff 明显变陡；ResNet34 - ResNet18 没有这种增强。",
        "5. **怎么解释**：新增架构改变了模型能力和检测器响应，因此改变了迁移性与隐蔽性的 tradeoff；SIG/UPGD 是 ResNet34 原先迁移性上升的重要驱动，ACC 是重要表征但不是唯一原因。",
        "",
        "可以直接使用的汇报话术：",
        "",
        "```text",
        "这部分架构实验主要看新增模型 SmallCNN 和 ResNet34。",
        "本轮结果排除了 SIG 和 UPGD，主要看去掉这两个强驱动方法后结论是否还稳。",
        "在 CIFAR-10 上，SmallCNN 相对 baseline ACC 更低，但同配置下 transfer_rate 上升、stealth_avg 下降，说明更弱模型让后门更容易迁移，但更容易被检测。",
        "更重要的是，SmallCNN - ResNet18 的同配置相关性显示，transfer_rate 和 stealth_avg 的负相关明显变强，说明 ACC/模型能力变化主要影响二者 tradeoff 的强度。",
        "在 Tiny-ImageNet 上，ResNet34 相对 ResNet18 ACC 更高，但去掉 SIG/UPGD 后 transfer_rate 不再明显上升，stealth_avg 仍下降，说明之前 ResNet34 的迁移性上升主要受攻击类型驱动。",
        "因此新增架构实验的核心发现是：SmallCNN 的架构效应很稳；ResNet34 的迁移性结论对攻击类型敏感。",
        "这个结果支持架构会影响 transfer-stealth tradeoff，但不能把 ACC 写成唯一因果变量。",
        "```",
        "",
        "## 10. 当前结论边界",
        "",
        "- 可以支持：新增 SmallCNN / ResNet34 相对 baseline 改变了 ACC、transfer_rate 和 stealth_avg 的共同位置。",
        "- 可以支持：SmallCNN 更明显地体现低 ACC、高迁移、低隐蔽。",
        "- 可以支持：去掉 SIG/UPGD 后，ResNet34 说明更高 ACC 不一定带来更高迁移性或更高隐蔽性。",
        "- 不建议声称：ACC 单独决定迁移性或隐蔽性。",
        "- 不建议声称：所有模型都满足同一个单调规律。",
        "",
        "## 11. 输出文件说明",
        "",
        "- `arch_acc_summary_by_dataset_arch.csv`：每个 dataset/arch 的均值水平。",
        "- `arch_acc_pairwise_delta_summary.csv`：新增模型相对 baseline 的同配置 delta，是本报告最关键的表。",
        "- `arch_pairwise_relationship_shift.csv`：同配置下 baseline 和新增模型的相关性变化，回答 transfer-stealth 关系是否变强。",
        "- `arch_relationship_summary_by_dataset_arch.csv`：每个 dataset/arch 内部的相关性系数。",
        "- `arch_pairwise_delta_summary.png`：最适合汇报的核心图。",
        "- `arch_pairwise_relationship_shift.png`：展示 transfer-stealth 相关性怎么变化。",
        "- `arch_acc_correlation_shift.png`：展示 ACC 与 transfer/stealth 直接相关性怎么变化。",
        "- `arch_metric_overview.png`：各模型 ACC、transfer_rate、stealth_avg 的总览图。",
        "- `arch_transfer_vs_stealth_facets.png`：看新增模型在 transfer-stealth 平面上的位置。",
        "- `arch_attack_heatmap.png`：攻击类型拆解。",
        "- `arch_defense_heatmap.png`：检测器拆解。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_new_model_arch_html_dashboard(
    summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    relationship_shift: pd.DataFrame,
    attack_summary: pd.DataFrame,
    defense_breakdown: pd.DataFrame,
    diagnostics: pd.DataFrame,
    path: Path,
) -> None:
    small_vs_resnet18 = first_row(pairwise_summary, dataset="cifar10", new_arch="SmallCNN", base_arch="ResNet18")
    r34_vs_resnet18 = first_row(pairwise_summary, dataset="tiny_imagenet", new_arch="ResNet34", base_arch="ResNet18")
    small_rel_resnet18 = first_row(relationship_shift, dataset="cifar10", new_arch="SmallCNN", base_arch="ResNet18")
    diag_all = first_row(diagnostics, group="all")
    attack_focus = attack_summary[attack_summary["dataset"].isin(PRIMARY_DATASETS)].copy()
    attack_focus["transfer_minus_stealth_gap"] = attack_focus["transfer_rate_mean"] - attack_focus["stealth_avg_mean"]

    key_summary_cols = [
        "dataset",
        "arch_base",
        "result_group",
        "n_rows",
        "clean_acc_mean",
        "transfer_rate_mean",
        "stealth_avg_mean",
    ]
    key_pair_cols = [
        "dataset",
        "new_arch",
        "base_arch",
        "n_rows",
        "delta_clean_acc_mean",
        "delta_transfer_rate_mean",
        "delta_stealth_avg_mean",
    ]
    key_relationship_cols = [
        "dataset",
        "new_arch",
        "base_arch",
        "n_rows",
        "base_spearman_transfer_stealth",
        "new_spearman_transfer_stealth",
        "delta_spearman_transfer_stealth",
        "base_pearson_transfer_stealth",
        "new_pearson_transfer_stealth",
        "delta_pearson_transfer_stealth",
        "base_spearman_acc_transfer",
        "new_spearman_acc_transfer",
        "base_spearman_acc_stealth",
        "new_spearman_acc_stealth",
    ]
    attack_cols = [
        "dataset",
        "arch_base",
        "attack_family",
        "n_rows",
        "transfer_rate_mean",
        "stealth_avg_mean",
        "transfer_minus_stealth_gap",
    ]
    defense_cols = [
        "dataset",
        "arch_base",
        "defense",
        "tpr_mean",
        "stealth_mean",
        "auc_mean",
    ]
    figure_cards = [
        image_card(
            "arch_acc_analysis/figures/arch_pairwise_delta_summary.png",
            "1. 同配置 Delta",
            "排除 SIG/UPGD 后，新增模型减 baseline，看 ACC、transfer_rate、stealth_avg 三项指标如何一起变。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_pairwise_relationship_shift.png",
            "2. 相关性变化",
            "看 baseline 和新增模型的 transfer-stealth 相关性是否变强或变弱。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_acc_correlation_shift.png",
            "3. ACC 关系变化",
            "看 ACC 与 transfer/stealth 的直接相关是否稳定。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_metric_overview.png",
            "4. 指标总览",
            "看 SmallCNN 和 ResNet34 在各自数据集里落在哪个水平。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_transfer_vs_stealth_facets.png",
            "5. 迁移性-隐蔽性分面",
            "看新增模型是否移动到高迁移、低隐蔽区域。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_attack_heatmap.png",
            "6. 攻击类型热力图",
            "当前热力图已排除 SIG/UPGD，用剩余攻击解释不同位置。",
        ),
        image_card(
            "arch_acc_analysis/figures/arch_defense_heatmap.png",
            "7. 检测器热力图",
            "拆开四个 defense，解释 stealth_avg 下降来自哪里。",
        ),
    ]

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>新增架构模型影响展示</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5f6b7a;
      --line: #d8dee8;
      --band: #f5f7fb;
      --blue: #2457a6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      background: #ffffff;
    }}
    header {{
      padding: 28px 36px 20px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 30px 0 12px; font-size: 20px; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    p {{ margin: 0 0 10px; }}
    main {{ padding: 0 36px 42px; max-width: 1380px; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: #fff;
      padding: 14px 16px;
      border-radius: 8px;
    }}
    .metric b {{ display: block; font-size: 24px; margin-bottom: 2px; }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    .callout {{
      border-left: 4px solid var(--blue);
      background: var(--band);
      padding: 14px 16px;
      margin: 18px 0;
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .figure-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }}
    .figure-card img {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .figure-card p {{ color: var(--muted); margin-top: 8px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 20px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eef2f7; }}
    code {{
      background: #eef2f7;
      padding: 1px 4px;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    ol, ul {{ padding-left: 22px; }}
    @media (max-width: 980px) {{
      main, header {{ padding-left: 18px; padding-right: 18px; }}
      .summary-grid, .figure-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>新增 SmallCNN / ResNet34 对迁移性与隐蔽性的影响</h1>
    <p>静态 dashboard，主口径已排除 SIG / upgd，重点看新增模型相对 baseline 的同配置 delta。</p>
    <div class="summary-grid">
      <div class="metric"><b>{signed_metric(small_vs_resnet18, 'delta_clean_acc_mean')}</b><span>SmallCNN - ResNet18 ACC</span></div>
      <div class="metric"><b>{signed_metric(small_vs_resnet18, 'delta_transfer_rate_mean')}</b><span>SmallCNN transfer_rate</span></div>
      <div class="metric"><b>{metric(small_rel_resnet18, 'new_spearman_transfer_stealth')}</b><span>SmallCNN transfer-stealth Spearman</span></div>
      <div class="metric"><b>{signed_metric(r34_vs_resnet18, 'delta_transfer_rate_mean')}</b><span>ResNet34 transfer_rate</span></div>
    </div>
  </header>
  <main>
    <section class="callout">
      <p><b>一句话结论：</b>排除 SIG/UPGD 后，SmallCNN 在 CIFAR-10 上仍然降低 ACC、提高 transfer_rate、降低 stealth_avg，并且 transfer-stealth 负相关明显变强；ResNet34 在 Tiny-ImageNet 上提高 ACC，但 transfer_rate 提升基本消失，stealth_avg 仍不改善。</p>
      <p><b>主证据：</b>优先看 <code>arch_pairwise_delta_summary.png</code>，因为它按同配置比较新增模型和 baseline。</p>
      <p><b>相关性证据：</b>SmallCNN - ResNet18 的 Spearman 从 <code>{metric(small_rel_resnet18, 'base_spearman_transfer_stealth')}</code> 变为 <code>{metric(small_rel_resnet18, 'new_spearman_transfer_stealth')}</code>，说明 ACC/模型能力变化后，迁移性与隐蔽性的 tradeoff 更陡。</p>
      <p><b>图表注意：</b>transfer_rate 中位数为 <code>{metric(diag_all, 'transfer_rate_median')}</code>，普通散点容易重叠，所以 delta 图比散点图更适合汇报。</p>
    </section>

    <h2>建议展示顺序</h2>
    <ol>
      <li>先讲新增了 SmallCNN 和 ResNet34。</li>
      <li>用同配置 delta 说明 ACC、transfer_rate、stealth_avg 的方向。</li>
      <li>再用相关性变化图说明 transfer-stealth 关系有没有变陡。</li>
      <li>用指标总览和分面散点说明新增模型的位置。</li>
      <li>最后用 attack/defense heatmap 解释删掉 SIG/UPGD 后差异还剩在哪里。</li>
    </ol>

    <h2>核心图表</h2>
    <div class="figure-grid">
      {''.join(figure_cards)}
    </div>

    <h2>同配置 Pairwise Delta</h2>
    {html_table(pairwise_summary, key_pair_cols, 20, ["dataset", "new_arch", "base_arch"])}

    <h2>同配置相关性变化</h2>
    {html_table(relationship_shift, key_relationship_cols, 20, ["dataset", "new_arch", "base_arch"])}

    <h2>Dataset / Arch 汇总</h2>
    {html_table(summary[summary["dataset"].isin(PRIMARY_DATASETS)].copy(), key_summary_cols, 30, ["dataset", "result_group", "arch_base"])}

    <h2>攻击类型分层</h2>
    {html_table(attack_focus, attack_cols, 80, ["dataset", "attack_family", "arch_base"])}

    <h2>四个检测器拆解</h2>
    {html_table(defense_breakdown, defense_cols, 80, ["dataset", "arch_base", "defense"])}
  </main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    roots = [
        (Path(args.baseline_root), "baseline_full"),
        (Path(args.supplement_root), "new_model_supplement"),
    ]
    df = discover_rows(roots)
    if df.empty:
        raise SystemExit("No result rows discovered.")
    df = add_analysis_flags(df, args.source_asr_threshold)

    if args.dry_run:
        print("Dry run: discovered rows")
        print(
            df.groupby(["result_root", "dataset", "arch_base"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["result_root", "dataset", "arch_base"])
            .to_string(index=False)
        )
        print()
        print("complete_defense_results:", int(df["complete_defense_results"].sum()))
        print("valid_transfer_rate:", int(df["valid_transfer_rate"].sum()))
        print("primary_main_analysis:", int(df["primary_main_analysis"].sum()))
        return

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows_path = output_dir / "arch_acc_transfer_stealth_rows.csv"
    df.to_csv(rows_path, index=False)

    summary = build_summary_by_dataset_arch(df)
    attack_summary = build_summary_by_attack(df)
    corr = build_correlations(df, [None, 0.01, 0.05, 0.10])
    acc_bins = build_acc_bin_table(df)
    acc_moderation = build_acc_moderation_summary(df)
    acc_bin_slope = build_acc_bin_slope_summary(df)
    pairwise = build_pairwise_delta(df)
    pairwise_summary = build_pairwise_delta_summary(pairwise)
    relationship_summary = build_arch_relationship_summary(df)
    relationship_shift = build_pairwise_relationship_shift(pairwise)
    defense_breakdown = build_defense_breakdown(df)
    diagnostics = arch_plot_diagnostics(df)
    binned_trend = arch_binned_trend(df, by="dataset_arch", bins=6)
    rank_trend = arch_rank_trend(df, by="dataset_arch", bins=8)

    summary.to_csv(output_dir / "arch_acc_summary_by_dataset_arch.csv", index=False)
    attack_summary.to_csv(output_dir / "arch_acc_summary_by_attack.csv", index=False)
    corr.to_csv(output_dir / "arch_acc_correlations.csv", index=False)
    acc_bins.to_csv(output_dir / "arch_acc_bin_transfer_stealth.csv", index=False)
    acc_moderation.to_csv(output_dir / "arch_acc_moderation_summary.csv", index=False)
    acc_bin_slope.to_csv(output_dir / "arch_acc_bin_slope_summary.csv", index=False)
    pairwise.to_csv(output_dir / "arch_acc_pairwise_delta.csv", index=False)
    pairwise_summary.to_csv(output_dir / "arch_acc_pairwise_delta_summary.csv", index=False)
    relationship_summary.to_csv(output_dir / "arch_relationship_summary_by_dataset_arch.csv", index=False)
    relationship_shift.to_csv(output_dir / "arch_pairwise_relationship_shift.csv", index=False)
    defense_breakdown.to_csv(output_dir / "arch_defense_breakdown_summary.csv", index=False)
    diagnostics.to_csv(output_dir / "arch_plot_diagnostics.csv", index=False)
    binned_trend.to_csv(output_dir / "arch_binned_median_trend_by_dataset_arch.csv", index=False)
    rank_trend.to_csv(output_dir / "arch_rank_binned_trend_by_dataset_arch.csv", index=False)

    write_missing_report(df, output_dir / "arch_missing_report.txt")
    write_regression_report(df, output_dir / "arch_acc_regression.txt")

    plot_scatter(df, "clean_acc", "transfer_rate", "dataset_arch", figures_dir / "arch_acc_vs_transfer_rate.png", "ACC vs transfer_rate")
    plot_scatter(df, "clean_acc", "stealth_avg", "dataset_arch", figures_dir / "arch_acc_vs_stealth.png", "ACC vs stealth_avg")
    plot_scatter(
        df,
        "transfer_rate",
        "stealth_avg",
        "dataset_arch",
        figures_dir / "arch_transfer_vs_stealth_by_dataset_arch.png",
        "transfer_rate vs stealth_avg",
    )
    plot_transfer_stealth_by_acc_bin(df, figures_dir / "arch_transfer_vs_stealth_by_acc_bin.png")
    plot_pairwise_delta(pairwise, figures_dir / "arch_pairwise_delta.png")
    plot_defense_breakdown(defense_breakdown, figures_dir / "arch_defense_stealth_breakdown.png")
    plot_arch_metric_overview(summary, figures_dir / "arch_metric_overview.png")
    plot_arch_transfer_stealth_facets(df, figures_dir / "arch_transfer_vs_stealth_facets.png")
    plot_arch_binned_trend(binned_trend, figures_dir / "arch_binned_median_trend_by_dataset_arch.png")
    plot_arch_rank_trend(rank_trend, figures_dir / "arch_rank_binned_trend_by_dataset_arch.png")
    plot_arch_acc_moderation_summary(df, acc_bin_slope, figures_dir / "arch_acc_moderation_summary.png")
    plot_arch_interaction_slope_by_acc(df, figures_dir / "arch_interaction_slope_by_acc.png")
    plot_transfer_stealth_by_acc_bin(df, figures_dir / "arch_acc_bin_transfer_stealth.png")
    plot_arch_dataset_acc_bin_spearman(acc_bin_slope, figures_dir / "arch_dataset_acc_bin_spearman.png")
    plot_pairwise_delta_summary(pairwise_summary, figures_dir / "arch_pairwise_delta_summary.png")
    plot_pairwise_relationship_shift(relationship_shift, figures_dir / "arch_pairwise_relationship_shift.png")
    plot_acc_correlation_shift(relationship_shift, figures_dir / "arch_acc_correlation_shift.png")
    plot_attack_heatmaps(attack_summary, figures_dir / "arch_attack_heatmap.png")
    plot_defense_heatmap(defense_breakdown, figures_dir / "arch_defense_heatmap.png")

    report_path = Path("analysis-transfer-asr2/ARCH_ACC_TRANSFER_STEALTH_RESULT_REPORT_CN.md")
    write_new_model_arch_result_report(
        df,
        summary,
        attack_summary,
        pairwise,
        pairwise_summary,
        relationship_shift,
        defense_breakdown,
        diagnostics,
        report_path,
    )
    write_new_model_arch_html_dashboard(
        summary,
        pairwise_summary,
        relationship_shift,
        attack_summary,
        defense_breakdown,
        diagnostics,
        Path("analysis-transfer-asr2/ARCH_ACC_TRANSFER_STEALTH_DASHBOARD_CN.html"),
    )

    noise_csv = Path(args.noise_rows_csv)
    combined = build_combined_with_noise(df, noise_csv)
    combined.to_csv(output_dir / "combined_acc_effect_rows.csv", index=False)
    write_combined_report(
        combined,
        Path("analysis-transfer-asr2/ACC_DIFFICULTY_NOISE_ARCH_COMBINED_REPORT_CN.md"),
        output_dir / "combined_acc_effect_summary.csv",
        output_dir / "combined_acc_effect_regression.txt",
    )

    baseline_report_path = Path("analysis-transfer-asr2/BASELINE_FULL_ACC_TRANSFER_STEALTH_REPORT_CN.md")
    write_baseline_full_report(
        df,
        Path("analysis-transfer-asr2/baseline_full_analysis"),
        baseline_report_path,
    )

    print(f"Wrote rows: {rows_path}")
    print(f"Wrote report: {report_path}")
    print("Wrote HTML dashboard: analysis-transfer-asr2/ARCH_ACC_TRANSFER_STEALTH_DASHBOARD_CN.html")
    print(f"Wrote combined report: analysis-transfer-asr2/ACC_DIFFICULTY_NOISE_ARCH_COMBINED_REPORT_CN.md")
    print(f"Wrote baseline full report: {baseline_report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default="poisoned_train_set1")
    parser.add_argument("--supplement-root", default="poisoned_train_set")
    parser.add_argument("--output-dir", default="analysis-transfer-asr2/arch_acc_analysis")
    parser.add_argument("--noise-rows-csv", default=str(NOISE_DEFAULT))
    parser.add_argument("--source-asr-threshold", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
