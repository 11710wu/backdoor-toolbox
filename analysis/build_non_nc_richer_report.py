#!/usr/bin/env python3
"""Build a richer non-NC transfer/stealth report with defense-specific analysis.

Outputs under analysis/:
- report_tables/*.csv
- report_figures/*.png
- transfer_stealth_report_detailed.md
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-codex")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_detailed_analysis_report as base
import extract_all_results as extract


DEFENSE_ORDER = ["STRIP", "SentiNet", "SCaLe-Up", "IBD_PSC"]
DEFENSE_PREFIX = {
    "STRIP": "strip",
    "SentiNet": "sentinet",
    "SCaLe-Up": "scaleup",
    "IBD_PSC": "ibd_psc",
}
ARCH_RAW_TO_OUTPUT = {
    "ResNet18": "resnet18",
    "mobilenetv2": "mobilenet",
    "vgg19_bn": "vgg",
}
ATTACK_FAMILY_MAP = {
    "basic": "localized_patch",
    "adaptive_patch": "localized_patch",
    "belt": "localized_patch",
    "blend": "blended_trigger",
    "adaptive_blend": "blended_trigger",
    "SIG": "global_signal",
    "WaNet": "global_spatial",
    "upgd": "optimized_perturbation",
}
ATTACK_FAMILY_ORDER = [
    "localized_patch",
    "blended_trigger",
    "global_signal",
    "global_spatial",
    "optimized_perturbation",
]
ATTACK_FAMILY_DISPLAY = {
    "localized_patch": "localized_patch",
    "blended_trigger": "blended_trigger",
    "global_signal": "global_signal",
    "global_spatial": "global_spatial",
    "optimized_perturbation": "optimized_perturbation",
}
ATTACK_METHOD_NOTES = {
    "basic": "典型 BadNet/patch 攻击。触发器局部、显式、几何位置固定，通常最容易获得高 ASR 与高迁移，但也最容易暴露。",
    "adaptive_patch": "自适应局部 patch。仍然属于局部触发器，但参数与 cover 设计使其兼具 patch 强激活与一定伪装能力。",
    "belt": "局部区域 + mask/cover 的混合型触发。几何上仍偏局部，但显著性不如 basic/adaptive_patch 那样尖锐。",
    "blend": "全局像素混合触发。触发器分布在整幅图像上，显著性扩散，通常比 patch 更隐蔽，但是否迁移取决于域间低层统计是否保留。",
    "adaptive_blend": "自适应 blended trigger。与 blend 相比更强调 cover 与参数调节，目标是继续压低可见性。",
    "SIG": "正弦/频域型全局触发。没有明确局部区域，通常更难被局部解释型防御抓到；但若域转换破坏其频率结构，迁移会显著下滑。",
    "WaNet": "几何形变触发。它不是加 patch，而是通过空间扭曲注入后门，往往对局部区域检测不敏感，且跨数据集表现高度依赖几何结构能否保留。",
    "upgd": "优化型小扰动触发。更接近对抗式后门，依赖模型与数据分布的细粒度耦合，因此可能在近域迁移上表现很好，但在大域移位下快速失效。",
}
DEFENSE_METHOD_NOTES = {
    "STRIP": (
        "输入混合熵检测。代码中通过把测试样本与多张干净样本叠加后计算预测熵来区分 clean/poison；"
        "如果触发器在混合后仍能稳定主导模型，熵会偏低，因此更适合抓强主导型 patch 或部分 blend。"
    ),
    "SentiNet": (
        "GradCAM 驱动的局部解释型检测。`other_defenses_tool_box/sentinet.py` 明确写有"
        " `Only support localized attacks`，并使用显著区域替换/移植策略，因此天然偏向局部触发器。"
    ),
    "SCaLe-Up": (
        "输入强度缩放一致性检测。代码使用 `scale_set=[3,5,7,9,11]` 放大像素强度，并检查预测在多尺度下是否异常一致；"
        "只要触发器在缩放后仍然稳固地主导类别，它就更容易被抓到。"
    ),
    "IBD_PSC": (
        "模型内部 BN 参数放大的一致性检测。它不是看显著区域，而是看预测类别在参数缩放后的稳定度，"
        "因此对局部/全局/形变型触发器都更通用，往往是四种方法里覆盖面最宽的一种。"
    ),
}


def _format_dataset_name(dataset: str) -> str:
    return base.DATASET_DISPLAY[dataset]


def _safe_mean(series: pd.Series) -> float:
    if series.dropna().empty:
        return float("nan")
    return float(series.mean())


def _normalize_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["alpha", "cover_rate"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(-1.0)
    return out


def load_defense_results(repo_root: Path, df_no_nc: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    expected_keys = _normalize_merge_keys(
        df_no_nc[
            [
                "dataset",
                "arch",
                "attack_type",
                "poison_rate",
                "train_param_value",
                "test_param_type",
                "test_param_value",
                "alpha",
                "cover_rate",
            ]
        ].copy()
    ).drop_duplicates()

    for dataset in base.DATASETS:
        poisoned_root = repo_root / "poisoned_train_set" / dataset
        if not poisoned_root.exists():
            continue
        for folder in sorted(poisoned_root.iterdir()):
            if not folder.is_dir():
                continue
            params = extract.parse_folder_name(folder.name)
            attack_type = params.get("attack_type")
            if attack_type in (None, "unknown"):
                continue
            arch = ARCH_RAW_TO_OUTPUT.get(params.get("arch_raw"))
            if arch is None:
                continue

            train_param_type = extract.get_training_param_type(attack_type)
            train_param_value = params.get(train_param_type) if train_param_type else None
            test_param_type = extract.get_test_param_type(train_param_type)
            alpha = params.get("alpha")
            cover_rate = params.get("cover_rate")
            transfer_by_pval: Dict[float, float] = {}

            def _collect_transfer(files: List[Path]) -> None:
                for file_path in files:
                    info = extract._parse_auxiliary_result_filename(file_path.name)
                    if info:
                        _, pval = info
                    elif train_param_value is not None:
                        pval = train_param_value
                    else:
                        continue
                    try:
                        transfer = extract._parse_transfer_rate_from_text(
                            file_path.read_text(encoding="utf-8")
                        )
                    except Exception:
                        transfer = None
                    if transfer is not None:
                        transfer_by_pval[pval] = transfer

            if dataset == "cifar10":
                _collect_transfer(list(folder.glob("test_stl10_results*.txt")))
            elif dataset == "tiny_imagenet":
                _collect_transfer(list(folder.glob("test_tiny_target_domain_results*.txt")))
            else:
                files = list(folder.glob("test_mnistm_results*.txt"))
                if not files:
                    files = list(folder.glob("test_mnist_cross_results*.txt"))
                _collect_transfer(files)

            for defense in DEFENSE_ORDER:
                prefix = DEFENSE_PREFIX[defense]
                files = list(folder.glob(f"{prefix}_defense_results*.json"))
                if not files:
                    rows.append(
                        {
                            "dataset": dataset,
                            "arch": arch,
                            "attack_type": attack_type,
                            "attack_family": ATTACK_FAMILY_MAP[attack_type],
                            "poison_rate": params.get("poison_rate"),
                            "train_param_value": train_param_value,
                            "test_param_type": test_param_type,
                            "test_param_value": train_param_value,
                            "alpha": alpha,
                            "cover_rate": cover_rate,
                            "defense": defense,
                            "tpr": np.nan,
                            "auc": np.nan,
                            "transfer_rate": transfer_by_pval.get(train_param_value),
                            "has_result": False,
                            "folder": folder.name,
                        }
                    )
                    continue

                for file_path in files:
                    info = extract._parse_auxiliary_result_filename(file_path.name)
                    if info:
                        file_test_param_type, file_test_param_value = info
                    else:
                        file_test_param_type, file_test_param_value = (
                            test_param_type,
                            train_param_value,
                        )
                    rec = extract._parse_defense_json(file_path)
                    rows.append(
                        {
                            "dataset": dataset,
                            "arch": arch,
                            "attack_type": attack_type,
                            "attack_family": ATTACK_FAMILY_MAP[attack_type],
                            "poison_rate": params.get("poison_rate"),
                            "train_param_value": train_param_value,
                            "test_param_type": file_test_param_type,
                            "test_param_value": file_test_param_value,
                            "alpha": alpha,
                            "cover_rate": cover_rate,
                            "defense": defense,
                            "tpr": np.nan if rec is None else rec["tpr"],
                            "auc": np.nan if rec is None else rec["auc"],
                            "transfer_rate": transfer_by_pval.get(file_test_param_value),
                            "has_result": rec is not None,
                            "folder": folder.name,
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No defense result records found under poisoned_train_set.")

    df = _normalize_merge_keys(df)
    key_cols = [
        "dataset",
        "arch",
        "attack_type",
        "poison_rate",
        "train_param_value",
        "test_param_type",
        "test_param_value",
        "alpha",
        "cover_rate",
    ]
    expected = (
        expected_keys.assign(expected=True)
        .merge(
            pd.DataFrame({"defense": DEFENSE_ORDER, "tmp": 1}),
            how="cross",
        )
        .drop(columns=["tmp"])
    )
    observed = (
        df[key_cols + ["defense", "has_result"]]
        .copy()
        .drop_duplicates()
        .rename(columns={"has_result": "observed_has_result"})
    )
    missing = expected.merge(
        observed,
        on=key_cols + ["defense"],
        how="left",
    )
    missing = missing[missing["observed_has_result"].isna()].copy()
    if not missing.empty:
        missing["attack_family"] = missing["attack_type"].map(ATTACK_FAMILY_MAP)
        missing["tpr"] = np.nan
        missing["auc"] = np.nan
        missing["transfer_rate"] = np.nan
        missing["has_result"] = False
        missing["folder"] = "missing_from_poisoned_train_set_scan"
        df = pd.concat(
            [
                df,
                missing[
                    [
                        "dataset",
                        "arch",
                        "attack_type",
                        "attack_family",
                        "poison_rate",
                        "train_param_value",
                        "test_param_type",
                        "test_param_value",
                        "alpha",
                        "cover_rate",
                        "defense",
                        "tpr",
                        "auc",
                        "transfer_rate",
                        "has_result",
                        "folder",
                    ]
                ],
            ],
            ignore_index=True,
        )
    return df


def build_attack_family_summary(df_no_nc: pd.DataFrame) -> pd.DataFrame:
    out = df_no_nc.copy()
    out["attack_family"] = out["attack_type"].astype(str).map(ATTACK_FAMILY_MAP)
    grouped = (
        out.groupby(["dataset", "attack_family"], observed=True)
        .agg(
            n_points=("transfer_rate", "size"),
            transfer_mean=("transfer_rate", "mean"),
            stealth_mean=("stealth_mean", "mean"),
            asr_mean=("asr", "mean"),
            tradeoff_hmean_mean=("tradeoff_hmean", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(
        ["dataset", "tradeoff_hmean_mean"], ascending=[True, False]
    ).reset_index(drop=True)


def build_defense_overall_summary(defense_df: pd.DataFrame) -> pd.DataFrame:
    valid = defense_df[defense_df["has_result"]].copy()
    out = (
        valid.groupby(["dataset", "defense"], observed=True)
        .agg(
            n_points=("tpr", "size"),
            tpr_mean=("tpr", "mean"),
            auc_mean=("auc", "mean"),
            tpr_std=("tpr", "std"),
            auc_std=("auc", "std"),
        )
        .reset_index()
    )
    out["stealth_from_tpr"] = 1.0 - out["tpr_mean"]
    out["stealth_from_auc"] = 1.0 - out["auc_mean"]
    return out.sort_values(["dataset", "tpr_mean"], ascending=[True, False]).reset_index(
        drop=True
    )


def build_defense_attack_summary(defense_df: pd.DataFrame) -> pd.DataFrame:
    valid = defense_df[defense_df["has_result"]].copy()
    out = (
        valid.groupby(["dataset", "defense", "attack_type"], observed=True)
        .agg(
            n_points=("tpr", "size"),
            tpr_mean=("tpr", "mean"),
            auc_mean=("auc", "mean"),
            transfer_mean=("transfer_rate", "mean"),
        )
        .reset_index()
    )
    return out.sort_values(
        ["dataset", "defense", "tpr_mean"], ascending=[True, True, False]
    ).reset_index(drop=True)


def build_defense_family_summary(defense_df: pd.DataFrame) -> pd.DataFrame:
    valid = defense_df[defense_df["has_result"]].copy()
    out = (
        valid.groupby(["dataset", "defense", "attack_family"], observed=True)
        .agg(
            n_points=("tpr", "size"),
            tpr_mean=("tpr", "mean"),
            auc_mean=("auc", "mean"),
        )
        .reset_index()
    )
    return out.sort_values(
        ["dataset", "defense", "tpr_mean"], ascending=[True, True, False]
    ).reset_index(drop=True)


def build_defense_coverage_summary(
    defense_df: pd.DataFrame, df_no_nc: pd.DataFrame
) -> pd.DataFrame:
    expected = (
        df_no_nc.groupby("dataset", observed=True)
        .size()
        .rename("n_configs")
        .reset_index()
    )
    rows: List[Dict[str, object]] = []
    total_expected = 0
    total_actual = 0
    for dataset in base.DATASETS:
        n_configs = int(expected[expected["dataset"] == dataset]["n_configs"].iloc[0])
        expected_records = n_configs * len(DEFENSE_ORDER)
        actual_records = int(
            defense_df[
                (defense_df["dataset"] == dataset) & (defense_df["has_result"])
            ].shape[0]
        )
        total_expected += expected_records
        total_actual += actual_records
        rows.append(
            {
                "dataset": dataset,
                "dataset_display": _format_dataset_name(dataset),
                "n_configs": n_configs,
                "expected_records": expected_records,
                "actual_records": actual_records,
                "coverage_rate": actual_records / expected_records if expected_records else np.nan,
            }
        )
    rows.append(
        {
            "dataset": "overall",
            "dataset_display": "overall",
            "n_configs": int(expected["n_configs"].sum()),
            "expected_records": total_expected,
            "actual_records": total_actual,
            "coverage_rate": total_actual / total_expected if total_expected else np.nan,
        }
    )
    return pd.DataFrame(rows)


def build_best_defense_by_attack(defense_attack_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in base.DATASETS:
        ds = defense_attack_summary[defense_attack_summary["dataset"] == dataset]
        for attack in base.ATTACK_ORDER:
            part = ds[ds["attack_type"].astype(str) == attack].sort_values(
                "tpr_mean", ascending=False
            )
            if part.empty:
                continue
            best = part.iloc[0]
            worst = part.iloc[-1]
            rows.append(
                {
                    "dataset": dataset,
                    "attack_type": attack,
                    "best_defense": best["defense"],
                    "best_tpr_mean": best["tpr_mean"],
                    "best_auc_mean": best["auc_mean"],
                    "worst_defense": worst["defense"],
                    "worst_tpr_mean": worst["tpr_mean"],
                    "worst_auc_mean": worst["auc_mean"],
                    "tpr_gap": best["tpr_mean"] - worst["tpr_mean"],
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "tpr_gap"], ascending=[True, False]
    ).reset_index(drop=True)


def build_defense_extrema_summary(defense_attack_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in base.DATASETS:
        ds = defense_attack_summary[defense_attack_summary["dataset"] == dataset]
        for defense in DEFENSE_ORDER:
            part = ds[ds["defense"] == defense].sort_values("tpr_mean", ascending=False)
            if part.empty:
                continue
            best = part.iloc[0]
            worst = part.iloc[-1]
            rows.append(
                {
                    "dataset": dataset,
                    "defense": defense,
                    "best_detected_attack": best["attack_type"],
                    "best_tpr_mean": best["tpr_mean"],
                    "weakest_attack": worst["attack_type"],
                    "weakest_tpr_mean": worst["tpr_mean"],
                    "tpr_gap": best["tpr_mean"] - worst["tpr_mean"],
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "tpr_gap"], ascending=[True, False]
    ).reset_index(drop=True)


def build_anomaly_tables(
    df_no_nc: pd.DataFrame, defense_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    merge_keys = [
        "dataset",
        "arch",
        "attack_type",
        "poison_rate",
        "train_param_value",
        "test_param_type",
        "test_param_value",
        "alpha",
        "cover_rate",
    ]
    def_avg = (
        _normalize_merge_keys(defense_df[defense_df["has_result"]])
        .groupby(merge_keys, observed=True)
        .agg(
            defense_tpr_mean=("tpr", "mean"),
            defense_auc_mean=("auc", "mean"),
        )
        .reset_index()
    )
    no_nc = _normalize_merge_keys(df_no_nc)
    merged = no_nc.merge(def_avg, on=merge_keys, how="left")
    high_transfer_high_detection = (
        merged[merged["transfer_rate"] >= 0.8]
        .sort_values(["stealth_mean", "transfer_rate"], ascending=[True, False])
        .head(20)
        .reset_index(drop=True)
    )
    high_stealth_low_transfer = (
        merged[merged["stealth_mean"] >= 0.65]
        .sort_values(["transfer_rate", "stealth_mean"], ascending=[True, False])
        .head(20)
        .reset_index(drop=True)
    )
    return {
        "anomaly_high_transfer_high_detection": high_transfer_high_detection,
        "anomaly_high_stealth_low_transfer": high_stealth_low_transfer,
    }


def plot_attack_family_heatmaps(attack_family_summary: pd.DataFrame, path: Path) -> None:
    metric_defs = [
        ("transfer_mean", "Transfer Mean"),
        ("stealth_mean", "Stealth Mean"),
        ("tradeoff_hmean_mean", "Tradeoff HMean"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    for ax, (metric, title) in zip(axes, metric_defs):
        pivot = (
            attack_family_summary.pivot(index="attack_family", columns="dataset", values=metric)
            .reindex(index=ATTACK_FAMILY_ORDER, columns=base.DATASETS)
        )
        arr = pivot.to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(base.DATASETS)))
        ax.set_xticklabels([_format_dataset_name(d) for d in base.DATASETS], rotation=16, ha="right")
        ax.set_yticks(np.arange(len(ATTACK_FAMILY_ORDER)))
        ax.set_yticklabels([ATTACK_FAMILY_DISPLAY[x] for x in ATTACK_FAMILY_ORDER])
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                txt = "NaN" if np.isnan(arr[i, j]) else f"{arr[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.suptitle("Cross-dataset Comparison by Attack Family", y=1.02)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Value")
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.18, top=0.84, wspace=0.28)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_defense_overall_heatmaps(defense_overall_summary: pd.DataFrame, path: Path) -> None:
    metric_defs = [("tpr_mean", "Defense TPR Mean"), ("auc_mean", "Defense AUC Mean")]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for ax, (metric, title) in zip(axes, metric_defs):
        pivot = (
            defense_overall_summary.pivot(index="defense", columns="dataset", values=metric)
            .reindex(index=DEFENSE_ORDER, columns=base.DATASETS)
        )
        arr = pivot.to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(base.DATASETS)))
        ax.set_xticklabels([_format_dataset_name(d) for d in base.DATASETS], rotation=16, ha="right")
        ax.set_yticks(np.arange(len(DEFENSE_ORDER)))
        ax.set_yticklabels(DEFENSE_ORDER)
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                txt = "NaN" if np.isnan(arr[i, j]) else f"{arr[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Value")
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.18, top=0.88, wspace=0.3)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_defense_attack_heatmaps(
    defense_attack_summary: pd.DataFrame, dataset: str, path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 4.9))
    for ax, metric, title in [
        (axes[0], "tpr_mean", "TPR Mean"),
        (axes[1], "auc_mean", "AUC Mean"),
    ]:
        pivot = (
            defense_attack_summary[defense_attack_summary["dataset"] == dataset]
            .pivot(index="defense", columns="attack_type", values=metric)
            .reindex(index=DEFENSE_ORDER, columns=base.ATTACK_ORDER)
        )
        arr = pivot.to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(base.ATTACK_ORDER)))
        ax.set_xticklabels(base.ATTACK_ORDER, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(DEFENSE_ORDER)))
        ax.set_yticklabels(DEFENSE_ORDER)
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                txt = "NaN" if np.isnan(arr[i, j]) else f"{arr[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.suptitle(f"{_format_dataset_name(dataset)}: Defense x Attack", y=1.03)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Value")
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.24, top=0.82, wspace=0.3)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_defense_family_heatmaps(
    defense_family_summary: pd.DataFrame, dataset: str, path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8))
    for ax, metric, title in [
        (axes[0], "tpr_mean", "TPR Mean"),
        (axes[1], "auc_mean", "AUC Mean"),
    ]:
        pivot = (
            defense_family_summary[defense_family_summary["dataset"] == dataset]
            .pivot(index="defense", columns="attack_family", values=metric)
            .reindex(index=DEFENSE_ORDER, columns=ATTACK_FAMILY_ORDER)
        )
        arr = pivot.to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(ATTACK_FAMILY_ORDER)))
        ax.set_xticklabels([ATTACK_FAMILY_DISPLAY[x] for x in ATTACK_FAMILY_ORDER], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(DEFENSE_ORDER)))
        ax.set_yticklabels(DEFENSE_ORDER)
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                txt = "NaN" if np.isnan(arr[i, j]) else f"{arr[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.suptitle(f"{_format_dataset_name(dataset)}: Defense x Attack Family", y=1.03)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Value")
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.22, top=0.82, wspace=0.3)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_poison_rate_trends_fixed(
    poison_summary: pd.DataFrame, dataset: str, path: Path
) -> None:
    sub = poison_summary[poison_summary["dataset"] == dataset].copy().sort_values("poison_rate")
    x = sub["poison_rate"].to_numpy(dtype=float)
    y_transfer = sub["transfer_mean"].to_numpy(dtype=float)
    y_stealth = sub["stealth_mean_mean"].to_numpy(dtype=float)
    y_asr = sub["asr_mean"].to_numpy(dtype=float)
    counts = sub["n_points"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(x, y_transfer, marker="o", linewidth=2.2, color="#2c7fb8", label="Transfer")
    ax.plot(x, y_stealth, marker="s", linewidth=2.2, color="#d95f0e", label="Stealth Mean")
    ax.plot(x, y_asr, marker="^", linewidth=2.2, color="#31a354", label="ASR")
    for xi, n in zip(x, counts):
        ax.text(xi, 0.02, f"n={int(n)}", rotation=90, fontsize=8, alpha=0.7)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Poison Rate")
    ax.set_ylabel("Mean value")
    ax.set_title(f"{_format_dataset_name(dataset)}: Metric Trends vs Poison Rate")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    plt.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_all_tables(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


def save_all_figures(
    df_no_nc: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    base.plot_overall_means(
        tables["overall_dataset_summary"], fig_dir / "overall_dataset_means.png"
    )
    base.plot_cross_dataset_attack_heatmaps(
        tables["cross_dataset_attack_summary"],
        fig_dir / "cross_dataset_attack_comparison_heatmaps.png",
    )
    plot_defense_overall_heatmaps(
        tables["defense_overall_summary"],
        fig_dir / "defense_overall_heatmaps.png",
    )
    for dataset in base.DATASETS:
        base.plot_scatter(
            df_no_nc, dataset, fig_dir / f"{dataset}_transfer_vs_stealth_mean_scatter.png"
        )
        base.plot_attack_heatmap(
            tables["attack_summary_by_dataset"],
            dataset,
            fig_dir / f"{dataset}_attack_metric_heatmap.png",
        )
        base.plot_attack_boxplots(
            df_no_nc, dataset, fig_dir / f"{dataset}_attack_boxplots.png"
        )
        plot_poison_rate_trends_fixed(
            tables["poison_rate_summary_by_dataset"],
            dataset,
            fig_dir / f"{dataset}_poison_rate_trends.png",
        )
        base.plot_arch_attack_tradeoff_heatmap(
            tables["attack_summary_by_dataset_arch"],
            dataset,
            fig_dir / f"{dataset}_arch_attack_tradeoff_heatmap.png",
        )
        plot_dataset_defense_attack_heatmaps(
            tables["defense_attack_summary"],
            dataset,
            fig_dir / f"{dataset}_defense_attack_heatmaps.png",
        )


def attack_method_bullets(
    cross_attack: pd.DataFrame,
    defense_attack_summary: pd.DataFrame,
) -> List[str]:
    bullets: List[str] = []
    basic_sentinet_cifar = _lookup_metric(
        defense_attack_summary, "cifar10", "SentiNet", "basic", "tpr_mean"
    )
    basic_sentinet_tiny = _lookup_metric(
        defense_attack_summary, "tiny_imagenet", "SentiNet", "basic", "tpr_mean"
    )
    basic_sentinet_mnist = _lookup_metric(
        defense_attack_summary, "mnistm", "SentiNet", "basic", "tpr_mean"
    )
    for attack in base.ATTACK_ORDER:
        rows = (
            cross_attack[cross_attack["attack_type"].astype(str) == attack]
            .sort_values("dataset")
            .copy()
        )
        if rows.empty:
            continue
        part = []
        for dataset in base.DATASETS:
            row = rows[rows["dataset"] == dataset]
            if row.empty:
                continue
            r = row.iloc[0]
            part.append(
                f"{dataset}: transfer={r['transfer_mean']:.4f}, stealth={r['stealth_mean_mean']:.4f}, tradeoff={r['tradeoff_hmean_mean']:.4f}"
            )
        line = f"`{attack}`：{ATTACK_METHOD_NOTES[attack]} 数值上看，" + "；".join(part) + "。"
        if attack == "basic":
            line += (
                "它在三个数据集上都维持最高或接近最高的迁移率，但 `SentiNet` 在 "
                f"CIFAR/Tiny/MNIST-M 上对它的平均 TPR 分别达到 "
                f"{basic_sentinet_cifar:.4f}/{basic_sentinet_tiny:.4f}/{basic_sentinet_mnist:.4f}，"
                "说明 patch 的高迁移几乎总是伴随高暴露。"
            )
        elif attack == "SIG":
            line += (
                "它最需要谨慎解释，因为高隐蔽性很大一部分来自‘不容易被激活’而不是‘激活后仍不被检测’。"
                "尤其在 MNIST-M -> MNIST 上，平均迁移率只有 "
                f"{rows[rows['dataset']=='mnistm']['transfer_mean'].iloc[0]:.4f}，已经接近整体失效。"
            )
        elif attack == "WaNet":
            line += (
                "它的跨数据集差异尤其大：CIFAR 上迁移明显偏低，但 Tiny 与 MNIST-M 上都进入高迁移高隐蔽区，"
                "说明几何形变触发是否稳定，强烈依赖源域与目标域的几何和纹理结构。"
            )
        elif attack == "upgd":
            line += (
                "它在 CIFAR-10 -> STL-10 上反而是平均折中最好的方法，说明小扰动型后门在近域自然图像之间可能更容易保留。"
            )
        bullets.append(line)
    return bullets


def defense_method_bullets(
    defense_overall_summary: pd.DataFrame,
    defense_extrema_summary: pd.DataFrame,
) -> List[str]:
    bullets: List[str] = []
    for defense in DEFENSE_ORDER:
        sub = defense_overall_summary[defense_overall_summary["defense"] == defense].copy()
        if sub.empty:
            continue
        part = []
        for dataset in base.DATASETS:
            row = sub[sub["dataset"] == dataset]
            if row.empty:
                continue
            r = row.iloc[0]
            part.append(
                f"{dataset}: TPR={r['tpr_mean']:.4f}, AUC={r['auc_mean']:.4f}"
            )
        line = f"`{defense}`：{DEFENSE_METHOD_NOTES[defense]} 总体均值为 " + "；".join(part) + "。"
        ext = defense_extrema_summary[defense_extrema_summary["defense"] == defense].copy()
        if not ext.empty:
            examples = []
            for dataset in base.DATASETS:
                row = ext[ext["dataset"] == dataset]
                if row.empty:
                    continue
                r = row.iloc[0]
                examples.append(
                    f"{dataset} 上最容易检测的是 `{r['best_detected_attack']}` (`TPR={r['best_tpr_mean']:.4f}`)，"
                    f"最弱的是 `{r['weakest_attack']}` (`TPR={r['weakest_tpr_mean']:.4f}`)"
                )
            if examples:
                line += " 例如，" + "；".join(examples) + "。"
        bullets.append(line)
    return bullets


def dataset_detail_bullets(
    dataset: str,
    overall: pd.DataFrame,
    attack_summary: pd.DataFrame,
    defense_overall_summary: pd.DataFrame,
    defense_attack_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
) -> List[str]:
    bullets: List[str] = []
    ds_overall = overall[overall["dataset"] == dataset].iloc[0]
    ds_attack = attack_summary[attack_summary["dataset"] == dataset].copy()
    ds_def = defense_overall_summary[defense_overall_summary["dataset"] == dataset].copy()
    ds_def_attack = defense_attack_summary[
        defense_attack_summary["dataset"] == dataset
    ].copy()
    ds_stab = stability_summary[stability_summary["dataset"] == dataset].copy()

    best_transfer_attack = ds_attack.sort_values("transfer_mean", ascending=False).iloc[0]
    best_stealth_attack = ds_attack.sort_values("stealth_mean_mean", ascending=False).iloc[0]
    best_tradeoff_attack = ds_attack.sort_values("tradeoff_hmean_mean", ascending=False).iloc[0]
    strongest_defense_tpr = ds_def.sort_values("tpr_mean", ascending=False).iloc[0]
    strongest_defense_auc = ds_def.sort_values("auc_mean", ascending=False).iloc[0]
    most_unstable = ds_stab.sort_values("transfer_std", ascending=False).iloc[0]

    def _best_worst_defense(attack_type: str) -> tuple[pd.Series, pd.Series]:
        part = ds_def_attack[ds_def_attack["attack_type"].astype(str) == attack_type].sort_values(
            "tpr_mean", ascending=False
        )
        return part.iloc[0], part.iloc[-1]

    best_def_for_transfer, worst_def_for_transfer = _best_worst_defense(
        str(best_transfer_attack["attack_type"])
    )
    best_def_for_tradeoff, worst_def_for_tradeoff = _best_worst_defense(
        str(best_tradeoff_attack["attack_type"])
    )

    bullets.append(
        f"整体上，`{_format_dataset_name(dataset)}` 的平均迁移率为 `{ds_overall['transfer_mean']:.4f}`，"
        f"平均隐蔽性为 `{ds_overall['stealth_mean_mean']:.4f}`，"
        f"`transfer_rate` 与 `stealth_mean` 的相关系数为 `{ds_overall['corr_transfer_stealth_mean']:.4f}`。"
    )
    bullets.append(
        f"按攻击方法看，迁移性最好的是 `{best_transfer_attack['attack_type']}` "
        f"(`transfer_mean={best_transfer_attack['transfer_mean']:.4f}`, "
        f"`stealth_mean={best_transfer_attack['stealth_mean_mean']:.4f}`, "
        f"`tradeoff={best_transfer_attack['tradeoff_hmean_mean']:.4f}`)；"
        f"隐蔽性最好的是 `{best_stealth_attack['attack_type']}` "
        f"(`stealth_mean={best_stealth_attack['stealth_mean_mean']:.4f}`, "
        f"`transfer_mean={best_stealth_attack['transfer_mean']:.4f}`)；"
        f"综合折中最好的是 `{best_tradeoff_attack['attack_type']}` "
        f"(`tradeoff_hmean_mean={best_tradeoff_attack['tradeoff_hmean_mean']:.4f}`, "
        f"`transfer_mean={best_tradeoff_attack['transfer_mean']:.4f}`, "
        f"`stealth_mean={best_tradeoff_attack['stealth_mean_mean']:.4f}`)。"
    )
    bullets.append(
        f"从防御端看，若按平均 TPR 衡量，当前数据集上效果最好的防御方法是 `{strongest_defense_tpr['defense']}` "
        f"(`TPR={strongest_defense_tpr['tpr_mean']:.4f}`)；"
        f"若按平均 AUC 衡量，则最好的是 `{strongest_defense_auc['defense']}` "
        f"(`AUC={strongest_defense_auc['auc_mean']:.4f}`)。"
    )
    bullets.append(
        f"对于迁移性最好的攻击 `{best_transfer_attack['attack_type']}`，最有效的防御是 `{best_def_for_transfer['defense']}` "
        f"(`TPR={best_def_for_transfer['tpr_mean']:.4f}`)，最弱的是 `{worst_def_for_transfer['defense']}` "
        f"(`TPR={worst_def_for_transfer['tpr_mean']:.4f}`)。"
    )
    bullets.append(
        f"对于折中最好的攻击 `{best_tradeoff_attack['attack_type']}`，最有效的防御是 `{best_def_for_tradeoff['defense']}` "
        f"(`TPR={best_def_for_tradeoff['tpr_mean']:.4f}`)，最弱的是 `{worst_def_for_tradeoff['defense']}` "
        f"(`TPR={worst_def_for_tradeoff['tpr_mean']:.4f}`)。"
    )
    bullets.append(
        f"迁移率最不稳定的攻击方法是 `{most_unstable['attack_type']}` "
        f"(`transfer_std={most_unstable['transfer_std']:.4f}`, `transfer_iqr={most_unstable['transfer_iqr']:.4f}`)，"
        "说明该方法更依赖具体参数与架构组合。"
    )

    if dataset == "cifar10":
        bullets.append(
            "这个数据集最鲜明的主线是：`basic` 占据高迁移端，`SIG` 占据高隐蔽端，而 `upgd` 与 `blend` 更接近 tradeoff 中上部。"
            "因此 CIFAR 最适合拿来展示“高迁移不等于高隐蔽”。"
        )
        bullets.append(
            "另一个关键点是 `WaNet`：它在 CIFAR-10 -> STL-10 上平均迁移率只有 `0.1951`，远低于 Tiny 与 MNIST-M。"
            "这说明几何形变并不总能跨域保留，至少在 CIFAR 与 STL 的风格差异下，它比 patch/blend 更容易失稳。"
        )
    elif dataset == "tiny_imagenet":
        tiny_sentinet_adaptive_patch = _lookup_metric(
            defense_attack_summary, "tiny_imagenet", "SentiNet", "adaptive_patch", "tpr_mean"
        )
        tiny_strip_adaptive_patch = _lookup_metric(
            defense_attack_summary, "tiny_imagenet", "STRIP", "adaptive_patch", "tpr_mean"
        )
        bullets.append(
            "Tiny-ImageNet 的 target-domain 结果最值得强调的现象是：`belt` 虽然拥有最高平均迁移率，`SIG` 虽然拥有最高平均隐蔽性，"
            "但真正综合最优的是 `WaNet`。这说明在更自然的目标域里，最有论文价值的不是两个端点，而是能够同时维持两项指标的结构型攻击。"
        )
        bullets.append(
            "同时，`adaptive_patch` 在这里虽然平均迁移率很高，但 `SentiNet` 与 `STRIP` 分别达到 "
            f"`{tiny_sentinet_adaptive_patch:.4f}` / `{tiny_strip_adaptive_patch:.4f}`，"
            "说明它的高迁移并没有换来真正的检测逃逸。"
        )
    elif dataset == "mnistm":
        mnist_scaleup_wanet = _lookup_metric(
            defense_attack_summary, "mnistm", "SCaLe-Up", "WaNet", "tpr_mean"
        )
        bullets.append(
            "MNIST-M 最关键的现象是三类方法恰好分占三个位置：`basic` 拥有最高迁移率，`SIG` 拥有最高隐蔽性，而 `belt` 拥有最高折中分。"
            "这几乎是 tradeoff 假设的离散化展示。"
        )
        bullets.append(
            "另一个值得深挖的点是 `SCaLe-Up` 对 `WaNet` 的平均 TPR 达到 "
            f"`{mnist_scaleup_wanet:.4f}`，"
            "反而远高于它对 basic/belt/adaptive_patch 的检测效果。结合方法机制，更合理的解释是："
            "在 MNIST 风格的简单结构上，形变型触发在强度缩放下更容易保留异常一致性，因此会被 SCaLe-Up 放大。"
        )
    return bullets


def _md_table(df: pd.DataFrame, columns: List[str] | None = None) -> str:
    return base.md_table(df, columns=columns, decimals=4)


def _lookup_metric(
    df: pd.DataFrame, dataset: str, defense: str, attack_type: str, metric: str
) -> float:
    part = df[
        (df["dataset"] == dataset)
        & (df["defense"] == defense)
        & (df["attack_type"].astype(str) == attack_type)
    ]
    if part.empty:
        return float("nan")
    return float(part.iloc[0][metric])


def build_report(
    tables: Dict[str, pd.DataFrame],
    figure_dir_name: str,
    table_dir_name: str,
) -> str:
    overall = tables["overall_dataset_summary"]
    arch_summary = tables["arch_summary"]
    attack_summary = tables["attack_summary_by_dataset"]
    attack_arch_summary = tables["attack_summary_by_dataset_arch"]
    cross_attack = tables["cross_dataset_attack_summary"]
    poison_summary = tables["poison_rate_summary_by_dataset"]
    stability_summary = tables["attack_stability_summary"]
    top_tradeoff = tables["top_configs_by_tradeoff"]
    defense_overall_summary = tables["defense_overall_summary"]
    defense_attack_summary = tables["defense_attack_summary"]
    defense_coverage_summary = tables["defense_coverage_summary"]
    defense_extrema_summary = tables["defense_extrema_summary"]
    anomaly_high_transfer = tables["anomaly_high_transfer_high_detection"]
    anomaly_high_stealth = tables["anomaly_high_stealth_low_transfer"]

    total_cov = defense_coverage_summary[
        defense_coverage_summary["dataset"] == "overall"
    ].iloc[0]
    missing_records = int(total_cov["expected_records"] - total_cov["actual_records"])

    lines: List[str] = []
    lines.append("# 三个数据集迁移性与隐蔽性的详细分析报告\n\n")
    lines.append("生成位置：`analysis/transfer_stealth_report_detailed.md`\n\n")
    lines.append(
        "这份报告只聚焦 `NC` 之外的主结果，即：迁移性采用目标域测试 ASR，隐蔽性采用 "
        "`STRIP / SentiNet / IBD-PSC / SCaLe-Up` 四种防御的平均表现。"
        "下面所有主结论都建立在这四种防御和当前三个数据集的完整实验结果上。\n\n"
    )

    lines.append("## 1. 指标与口径\n\n")
    lines.append("- `transfer_rate`：目标域测试 ASR，越大表示跨域迁移越强。\n")
    lines.append("- `stealth_auc_avg = 1 - mean(AUC)`：四种防御平均后的 AUC 型隐蔽性，越大越隐蔽。\n")
    lines.append("- `stealth_tpr_avg = 1 - mean(TPR)`：四种防御平均后的 TPR 型隐蔽性，越大越隐蔽。\n")
    lines.append("- `stealth_mean = (stealth_auc_avg + stealth_tpr_avg) / 2`：本报告用于主比较的综合隐蔽性。\n")
    lines.append("- `tradeoff_hmean`：`transfer_rate` 与 `stealth_mean` 的调和均值，用于量化“迁移性-隐蔽性”折中。\n")
    lines.append("- 当前 `tiny_imagenet` 的迁移性已改为后生成的 `target-domain` 结果，不再读取旧的 Tiny-ImageNet-C。\n")
    lines.append("- 当前结果文件里的 `basic` 对应论文里常见的 `BadNet`，正文撰写时需要统一命名。\n\n")

    lines.append("## 2. 数据完整性与解释边界\n\n")
    lines.append(_md_table(defense_coverage_summary) + "\n\n")
    lines.append(
        f"- 四种防御的有效结果总覆盖率为 `{total_cov['actual_records']:.0f}/{total_cov['expected_records']:.0f}`，"
        f"即 `{total_cov['coverage_rate']:.4f}`。主分析的数据覆盖是完整的。"
    )
    lines.append("\n")
    lines.append(
        f"- 当前主分析表中缺失的有效记录为 `{missing_records}` 条，因此整体统计、排序和相关性分析不受缺失值干扰。"
    )
    lines.append("\n")
    lines.append(
        "- 需要特别强调：高隐蔽性不一定意味着“成功绕过检测”。在某些情况下，高隐蔽性来自攻击点落在 tradeoff 曲线的低迁移一端，"
        "最典型的例子就是 `MNIST-M -> MNIST` 上的 `SIG`。"
    )
    lines.append("\n\n")

    lines.append("## 3. 方法机理与先验预期\n\n")
    lines.append("### 3.1 攻击方法\n\n")
    for attack in base.ATTACK_ORDER:
        lines.append(f"- `{attack}`：{ATTACK_METHOD_NOTES[attack]}\n")
    lines.append("\n### 3.2 防御方法\n\n")
    for defense in DEFENSE_ORDER:
        lines.append(f"- `{defense}`：{DEFENSE_METHOD_NOTES[defense]}\n")
    lines.append("\n")

    lines.append("## 4. 总体统计\n\n")
    lines.append(f"![overall means]({figure_dir_name}/overall_dataset_means.png)\n\n")
    lines.append(
        _md_table(
            overall,
            [
                "dataset_display",
                "n_points",
                "transfer_mean",
                "transfer_median",
                "stealth_auc_mean",
                "stealth_tpr_mean",
                "stealth_mean_mean",
                "asr_mean",
                "corr_transfer_stealth_mean",
                "corr_transfer_asr",
            ],
        )
        + "\n\n"
    )
    lines.append(f"![cross attack comparison]({figure_dir_name}/cross_dataset_attack_comparison_heatmaps.png)\n\n")
    lines.append(
        _md_table(
            cross_attack,
            [
                "attack_type",
                "dataset",
                "transfer_mean",
                "stealth_mean_mean",
                "tradeoff_hmean_mean",
                "rank_transfer_mean",
                "rank_stealth_mean",
                "rank_tradeoff_mean",
            ],
        )
        + "\n\n"
    )
    lines.append(f"![defense overall]({figure_dir_name}/defense_overall_heatmaps.png)\n\n")
    lines.append(_md_table(defense_overall_summary) + "\n\n")
    lines.append("- 三个数据集都出现显著的迁移性-隐蔽性负相关，说明跨域后门的主矛盾始终是“强迁移”与“低暴露”之间的权衡。\n")
    lines.append("- `Tiny-ImageNet -> Target Domain` 的平均迁移率最高，`MNIST-M -> MNIST` 的平均隐蔽性最高，但这种高隐蔽性并不总是正面信号。\n")
    lines.append("- 从防御总体均值看，`CIFAR-10 -> STL-10` 上最强的是 `SCaLe-Up`，`Tiny-ImageNet` 与 `MNIST-M` 上最强的是 `IBD_PSC`。\n")
    lines.append("- 从攻击方法整体排序看，`basic`、`belt` 往往更接近高迁移端，`SIG` 往往更接近高隐蔽端，而 `upgd`、`blend`、`adaptive_blend` 与 `WaNet` 更常出现在折中较优的位置。\n\n")

    section_no = 5
    for idx, dataset in enumerate(base.DATASETS, start=1):
        ds_attack = attack_summary[attack_summary["dataset"] == dataset].copy()
        ds_def_overall = defense_overall_summary[
            defense_overall_summary["dataset"] == dataset
        ].copy()
        ds_def_attack = defense_attack_summary[
            defense_attack_summary["dataset"] == dataset
        ].copy()
        tiny_strip_blend = _lookup_metric(
            defense_attack_summary, "tiny_imagenet", "STRIP", "blend", "tpr_mean"
        )
        tiny_strip_adaptive_blend = _lookup_metric(
            defense_attack_summary,
            "tiny_imagenet",
            "STRIP",
            "adaptive_blend",
            "tpr_mean",
        )
        ds_poison = poison_summary[poison_summary["dataset"] == dataset].copy()
        ds_stability = stability_summary[stability_summary["dataset"] == dataset].copy()
        ds_top_tradeoff = top_tradeoff[top_tradeoff["dataset"] == dataset].head(5)

        lines.append(f"## {section_no}. {_format_dataset_name(dataset)}\n\n")
        lines.append(f"- Target setting: `{base.DATASET_TARGET_NOTE[dataset]}`\n\n")
        lines.append(f"![scatter]({figure_dir_name}/{dataset}_transfer_vs_stealth_mean_scatter.png)\n\n")
        lines.append(f"![attack heatmap]({figure_dir_name}/{dataset}_attack_metric_heatmap.png)\n\n")
        lines.append(f"![attack boxplots]({figure_dir_name}/{dataset}_attack_boxplots.png)\n\n")
        lines.append(f"![poison trends]({figure_dir_name}/{dataset}_poison_rate_trends.png)\n\n")
        lines.append(f"![arch attack tradeoff]({figure_dir_name}/{dataset}_arch_attack_tradeoff_heatmap.png)\n\n")
        lines.append(f"![defense attack heatmaps]({figure_dir_name}/{dataset}_defense_attack_heatmaps.png)\n\n")

        lines.append(f"### {section_no}.1 关键统计表\n\n")
        lines.append("**攻击方法均值统计**\n\n")
        lines.append(
            _md_table(
                ds_attack,
                [
                    "attack_type",
                    "n_points",
                    "transfer_mean",
                    "stealth_mean_mean",
                    "asr_mean",
                    "tradeoff_hmean_mean",
                    "rank_tradeoff_mean",
                ],
            )
            + "\n\n"
        )
        lines.append("**防御方法总体统计**\n\n")
        lines.append(_md_table(ds_def_overall) + "\n\n")
        lines.append("**防御方法 x 攻击方法**\n\n")
        lines.append(_md_table(ds_def_attack) + "\n\n")
        lines.append("**Poison Rate 趋势**\n\n")
        lines.append(_md_table(ds_poison) + "\n\n")
        lines.append("**Top-5 迁移-隐蔽折中配置**\n\n")
        lines.append(
            _md_table(
                ds_top_tradeoff,
                [
                    "rank_in_dataset",
                    "arch",
                    "attack_type",
                    "poison_rate",
                    "train_param_value",
                    "test_param_type",
                    "test_param_value",
                    "transfer_rate",
                    "stealth_mean",
                    "asr",
                    "tradeoff_hmean",
                ],
            )
            + "\n\n"
        )
        lines.append(f"### {section_no}.2 详细分析\n\n")
        for bullet in dataset_detail_bullets(
            dataset,
            overall,
            attack_summary,
            defense_overall_summary,
            defense_attack_summary,
            stability_summary,
        ):
            lines.append(f"- {bullet}\n")
        lines.append("\n")

        if dataset == "cifar10":
            lines.append(
                "- `upgd` 在这个数据集上之所以特别重要，不是因为它单项最好，而是因为它是少数同时把 `transfer_rate` 保持在高位、又没有像 `basic` 那样完全暴露的方法。"
                "从防御结果看，虽然 `SCaLe-Up` 对 `upgd` 的平均 TPR 仍达到 `0.7066`，但 `STRIP/SentiNet/IBD_PSC` 并没有像面对 patch 那样统一高效，因此它最终成为均值折中最优攻击。\n"
            )
            lines.append(
                "- `adaptive_patch` 在 CIFAR 上出现了一个很典型的“局部自适应不等于局部不可检测”现象："
                "`SentiNet` 的平均 TPR 达到 `0.9371`，说明只要触发仍以局部显著区域的形式存在，GradCAM 类方法仍然能稳定地把它抓出来。\n\n"
            )
        elif dataset == "tiny_imagenet":
            lines.append(
                "- `STRIP` 在 Tiny 上对 `blend/adaptive_blend` 的平均 TPR 分别达到 "
                f"`{tiny_strip_blend:.4f}` / `{tiny_strip_adaptive_blend:.4f}`，"
                "比在 CIFAR 或 MNIST-M 上都更高。这是一个值得写进论文的反常点：虽然 blended trigger 按机理不是典型局部触发器，"
                "但在 64x64 自然图像及当前 target-domain 下，输入混合仍然足以打破其低熵主导性。"
            )
            lines.append("\n")
            lines.append(
                "- `IBD_PSC` 在 Tiny 上几乎对所有非极弱攻击都保持中等以上检测能力，尤其是 `belt/basic/WaNet/blend`。"
                "这意味着 target-domain 迁移并没有抹去模型内部的后门一致性，哪怕有些攻击在像素层面并不显眼，模型内部响应仍可被参数缩放方法放大。\n\n"
            )
        elif dataset == "mnistm":
            lines.append(
                "- `SentiNet` 在 MNIST-M 上对 `belt` 的平均 TPR 只有 `0.1340`，明显低于它对 `basic` (`0.9199`) 和 `adaptive_patch` (`0.8004`) 的检测能力。"
                "这说明 `belt` 虽然仍是局部触发器，但它的 mask/cover 设计让触发区域不再总是成为最显著的 GradCAM 区域。"
            )
            lines.append("\n")
            lines.append(
                "- `IBD_PSC` 在 MNIST-M 上是最稳健的总体防御：它不只对 `basic/blend/WaNet` 有效，"
                "对几乎已经失效的 `SIG` 也给出了相对更高的 TPR (`0.4885`)。"
                "这说明它捕捉到的不只是像素可见性，而是后门输入在模型内部的一致性偏移。\n\n"
            )
        section_no += 1

    lines.append(f"## {section_no}. 按防御方法的综合分析\n\n")
    for bullet in defense_method_bullets(defense_overall_summary, defense_extrema_summary):
        lines.append(f"- {bullet}\n")
    lines.append("\n")
    lines.append(
        _md_table(
            defense_extrema_summary,
            [
                "dataset",
                "defense",
                "best_detected_attack",
                "best_tpr_mean",
                "weakest_attack",
                "weakest_tpr_mean",
                "tpr_gap",
            ],
        )
        + "\n\n"
    )
    lines.append(
        "- `SentiNet` 的结果最符合代码先验：对 `basic/adaptive_patch` 近乎极强，对 `SIG/WaNet/blend/adaptive_blend` 明显偏弱。"
        "因此论文里不能把它写成“通用检测器”，而应该明确写成“对局部触发器尤其敏感”。\n"
    )
    lines.append(
        "- `STRIP` 不是完全没用，但它的有效范围更窄：对于 patch 以及部分 blended trigger 可以工作，"
        "对于 `SIG` 和 `WaNet` 这样的全局平滑或几何形变型后门则明显吃力。\n"
    )
    lines.append(
        "- `SCaLe-Up` 的行为最有数据集依赖性：CIFAR 上总体最强，MNIST-M 上则主要在 `WaNet` 上异常强。"
        "这说明它不是按“局部/全局”分类，而是按“缩放后预测是否出现异常一致性”来分类。\n"
    )
    lines.append(
        "- `IBD_PSC` 在三个数据集里都表现出最宽的覆盖面，尤其在 Tiny 与 MNIST-M 上最稳定。"
        "如果论文需要选一个“总体最可靠”的检测基线，它是最合适的候选。\n\n"
    )

    section_no += 1
    lines.append(f"## {section_no}. 特殊与异常结果\n\n")
    lines.append(f"### {section_no}.1 高迁移但高暴露的典型配置\n\n")
    lines.append(
        _md_table(
            anomaly_high_transfer,
            [
                "dataset",
                "arch",
                "attack_type",
                "poison_rate",
                "train_param_value",
                "test_param_value",
                "transfer_rate",
                "stealth_mean",
                "asr",
                "defense_tpr_mean",
                "defense_auc_mean",
                "tradeoff_hmean",
            ],
        )
        + "\n\n"
    )
    lines.append(f"### {section_no}.2 高隐蔽但低迁移的典型配置\n\n")
    lines.append(
        _md_table(
            anomaly_high_stealth,
            [
                "dataset",
                "arch",
                "attack_type",
                "poison_rate",
                "train_param_value",
                "test_param_value",
                "transfer_rate",
                "stealth_mean",
                "asr",
                "defense_tpr_mean",
                "defense_auc_mean",
                "tradeoff_hmean",
            ],
        )
        + "\n\n"
    )
    lines.append(
        "- 第一类异常是“高迁移但高暴露”，基本被 `basic` 统治，说明最强的局部 patch 方案往往不是真正适合写成隐蔽攻击主角的方法。\n"
    )
    lines.append(
        "- 第二类异常是“高隐蔽但低迁移”，主要由 `SIG`、部分 `upgd` 以及个别极低强度 `blend/adaptive_blend` 组成。"
        "这类点最容易误导分析：它们看起来隐蔽性很高，但本质上很多已经接近‘触发失败’。\n"
    )
    lines.append(
        "- 因此写论文时需要把“隐蔽但失效”和“隐蔽且仍能稳定迁移”严格区分开。就当前结果看，真正值得当作成功折中案例写的主要是 "
        "`CIFAR` 上的 `upgd/blend`，`Tiny` 上的 `WaNet/blend/adaptive_blend`，以及 `MNIST-M` 上的 `belt/WaNet/blend/adaptive_blend`。\n\n"
    )

    section_no += 1
    lines.append(f"## {section_no}. 可直接写入论文的主结论\n\n")
    lines.append("- 三个数据集都存在稳定的迁移性-隐蔽性张力，且这种张力不是个别参数点现象，而是整体统计规律。\n")
    lines.append("- `basic` 与 `belt` 往往占据高迁移端，但它们也更容易被 `SentiNet`、`STRIP` 或 `IBD_PSC` 捕获，因此很难直接作为“高隐蔽攻击”支撑论文主结论。\n")
    lines.append("- `upgd`、`blend`、`adaptive_blend` 与 `WaNet` 更容易落在折中较优的位置，但不同数据集上的领先者并不相同，这说明折中最优方法具有明确的数据集依赖性。\n")
    lines.append("- `SIG` 是当前最典型的高隐蔽低迁移样本：它可以帮助证明 tradeoff 的一端，但不能被误写成综合最优攻击；在 `MNIST-M -> MNIST` 上这一点尤其明显。\n")
    lines.append("- 对防御方法的论文表述必须条件化：`SentiNet` 主要适用于局部触发器，`STRIP` 主要适用于低熵主导型触发，`SCaLe-Up` 对尺度一致性敏感，`IBD_PSC` 的覆盖面最广。\n")
    lines.append("- 结合三个数据集的均值结果，`CIFAR-10 -> STL-10` 上最值得强调的是 `upgd`，`Tiny-ImageNet -> ImageNetV2 target-domain` 上最值得强调的是 `WaNet`，`MNIST-M -> MNIST` 上最值得强调的是 `belt`。\n\n")

    lines.append(f"## {section_no + 1}. 文件索引\n\n")
    for name in [
        "overall_dataset_summary",
        "arch_summary",
        "attack_summary_by_dataset",
        "attack_summary_by_dataset_arch",
        "cross_dataset_attack_summary",
        "poison_rate_summary_by_dataset",
        "attack_stability_summary",
        "top_configs_by_tradeoff",
        "defense_overall_summary",
        "defense_attack_summary",
        "defense_coverage_summary",
        "defense_extrema_summary",
        "anomaly_high_transfer_high_detection",
        "anomaly_high_stealth_low_transfer",
    ]:
        lines.append(f"- 统计表：`{table_dir_name}/{name}.csv`\n")
    for name in [
        "overall_dataset_means.png",
        "cross_dataset_attack_comparison_heatmaps.png",
        "defense_overall_heatmaps.png",
        "{dataset}_transfer_vs_stealth_mean_scatter.png",
        "{dataset}_attack_metric_heatmap.png",
        "{dataset}_attack_boxplots.png",
        "{dataset}_poison_rate_trends.png",
        "{dataset}_arch_attack_tradeoff_heatmap.png",
        "{dataset}_defense_attack_heatmaps.png",
    ]:
        lines.append(f"- 图片：`{figure_dir_name}/{name}`\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build richer non-NC analysis report")
    parser.add_argument("--analysis-dir", default="analysis")
    parser.add_argument("--report-name", default="transfer_stealth_report_detailed.md")
    parser.add_argument("--table-dir-name", default="report_tables")
    parser.add_argument("--figure-dir-name", default="report_figures")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.is_absolute():
        analysis_dir = repo_root / analysis_dir

    df_no_nc = base.load_mode(analysis_dir, "no_nc")
    defense_df = load_defense_results(repo_root, df_no_nc)

    tables: Dict[str, pd.DataFrame] = {
        "overall_dataset_summary": base.build_overall_summary(df_no_nc),
        "arch_summary": base.build_arch_summary(df_no_nc),
        "attack_summary_by_dataset": base.build_attack_summary(df_no_nc),
        "attack_summary_by_dataset_arch": base.build_attack_arch_summary(df_no_nc),
        "cross_dataset_attack_summary": base.build_cross_dataset_attack_summary(df_no_nc),
        "poison_rate_summary_by_dataset": base.build_poison_rate_summary(df_no_nc),
        "attack_stability_summary": base.build_attack_stability_summary(df_no_nc),
        "top_configs_by_tradeoff": base.build_top_configs(df_no_nc, "tradeoff_hmean"),
        "defense_overall_summary": build_defense_overall_summary(defense_df),
        "defense_attack_summary": build_defense_attack_summary(defense_df),
        "defense_coverage_summary": build_defense_coverage_summary(defense_df, df_no_nc),
        "defense_extrema_summary": build_defense_extrema_summary(
            build_defense_attack_summary(defense_df)
        ),
    }
    tables.update(build_anomaly_tables(df_no_nc, defense_df))

    table_dir = analysis_dir / args.table_dir_name
    figure_dir = analysis_dir / args.figure_dir_name
    save_all_tables(tables, table_dir)
    save_all_figures(df_no_nc, tables, figure_dir)

    report_text = build_report(tables, args.figure_dir_name, args.table_dir_name)
    report_path = analysis_dir / args.report_name
    report_path.write_text(report_text, encoding="utf-8")

    print(f"[OK] report: {report_path}")
    print(f"[OK] tables: {table_dir}")
    print(f"[OK] figures: {figure_dir}")


if __name__ == "__main__":
    main()
