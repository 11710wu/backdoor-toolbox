#!/usr/bin/env python3
"""Clean-target manifold anomaly feature-stealth pilot.

This reuses the existing feature pilot selection and feature extraction code,
but replaces the supervised clean-vs-poison linear probe with a kNN anomaly
score against clean target-class features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_feature_stealth_pilot as base  # noqa: E402


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def mean_knn_distance(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    k: int,
    metric: str,
) -> Tuple[np.ndarray, int]:
    if len(reference_features) < 1:
        raise ValueError("No reference features available")
    k_eff = min(int(k), len(reference_features))
    ref = reference_features.astype(np.float64)
    query = query_features.astype(np.float64)
    if metric == "cosine":
        ref = l2_normalize(ref)
        query = l2_normalize(query)
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nbrs.fit(ref)
    distances, _ = nbrs.kneighbors(query, return_distance=True)
    return distances.mean(axis=1), k_eff


def percentile_ranks(reference_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
    ref = np.sort(reference_scores.astype(np.float64))
    return np.searchsorted(ref, query_scores, side="right") / max(1, len(ref))


def stable_offset(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def split_clean_features(
    clean_features: np.ndarray,
    clean_ids: np.ndarray,
    reference_ratio: float,
    seed: int,
    experiment_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(clean_features)
    if n < 2:
        raise ValueError("Need at least two clean target samples for reference/calibration split")
    rng = np.random.default_rng(seed + stable_offset(experiment_id))
    order = np.arange(n)
    rng.shuffle(order)
    split = int(round(n * reference_ratio))
    split = min(max(1, split), n - 1)
    ref_pos = order[:split]
    calib_pos = order[split:]
    return clean_features[ref_pos], clean_features[calib_pos], clean_ids[ref_pos], clean_ids[calib_pos]


def defense_lookup_for_result(result: Dict[str, Any], lookup: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    result_dir = str(Path(str(result.get("checkpoint_path", ""))).parent)
    return lookup.get(result_dir, {})


def run_checkpoint(candidate: base.Candidate, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    started = time.time()
    out: Dict[str, Any] = {
        "experiment_id": candidate.experiment_id,
        "attack": candidate.attack,
        "dataset": candidate.dataset,
        "architecture": candidate.architecture,
        "attack_strength": candidate.attack_strength,
        "poison_rate": candidate.poison_rate,
        "cover_rate": candidate.cover_rate,
        "mask_rate": candidate.mask_rate,
        "label_mode": candidate.label_mode,
        "victim_seed": candidate.victim_seed,
        "target_class": candidate.target_class,
        "checkpoint_path": str(candidate.checkpoint_path),
        "poisoned_dataset_path": "",
        "source_asr": candidate.source_asr,
        "target_transfer_asr": candidate.target_transfer_asr,
        "selection_asr_type": candidate.selection_asr_type,
        "selection_asr_value": candidate.selection_asr_value,
        "selection_reason": candidate.selection_reason,
        "feature_layer": "penultimate_before_final_classifier",
        "feature_dimension": None,
        "knn_k_requested": args.knn_k,
        "knn_k_effective": None,
        "knn_metric": args.knn_metric,
        "target_fpr": args.target_fpr,
        "status": "failed",
        "error": "",
    }
    try:
        subset = base.build_clean_poison_subset(candidate, args.sample_seed, args.max_samples_per_class)
        out.update({k: v for k, v in subset.items() if k.startswith("n_")})
        n = min(subset["n_clean_used"], subset["n_poison_used"])
        out["low_sample_warning"] = bool(n < 100)
        out["exploratory_only"] = bool(n < 50)
        out["source_attack_failure_warning"] = bool(
            base.is_finite(candidate.source_asr) and candidate.source_asr < base.SOURCE_FAILURE_THRESHOLD
        )
        if n < 2:
            raise ValueError("No balanced clean/poison samples available")

        actual_img_set, dataset_path = base.load_actual_img_set(candidate.result_dir, candidate.dataset)
        out["poisoned_dataset_path"] = dataset_path
        reconstruct_poison = actual_img_set is None
        indices = subset["clean_indices"] + subset["poison_indices"]
        probe_labels = [0] * len(subset["clean_indices"]) + [1] * len(subset["poison_indices"])
        features, labels, sample_ids, feature_layer = base.extract_features_for_indices(
            candidate,
            indices,
            probe_labels,
            actual_img_set=actual_img_set,
            reconstruct_poison=reconstruct_poison,
            poison_index_set=set(subset["poison_indices"]),
            batch_size=args.batch_size,
            device=device,
        )
        out["feature_layer"] = feature_layer
        out["feature_dimension"] = int(features.shape[1])

        clean_features = features[labels == 0]
        poison_features = features[labels == 1]
        clean_ids = sample_ids[labels == 0]
        poison_ids = sample_ids[labels == 1]
        if len(clean_features) < args.min_clean_samples:
            raise ValueError(f"Too few clean target samples for manifold split: {len(clean_features)}")
        if len(poison_features) < args.min_poison_samples:
            raise ValueError(f"Too few poison samples for manifold scoring: {len(poison_features)}")

        ref_features, calib_features, ref_ids, calib_ids = split_clean_features(
            clean_features,
            clean_ids,
            reference_ratio=args.reference_ratio,
            seed=args.split_seed,
            experiment_id=candidate.experiment_id,
        )
        calib_scores, k_eff = mean_knn_distance(ref_features, calib_features, args.knn_k, args.knn_metric)
        poison_scores, _ = mean_knn_distance(ref_features, poison_features, args.knn_k, args.knn_metric)
        threshold = float(np.quantile(calib_scores, 1.0 - args.target_fpr))
        clean_fpr_empirical = float(np.mean(calib_scores > threshold))
        feature_tpr = float(np.mean(poison_scores > threshold))
        poison_percentiles = percentile_ranks(calib_scores, poison_scores)

        out.update(
            {
                "status": "ok",
                "n_clean_reference": int(len(ref_features)),
                "n_clean_calibration": int(len(calib_features)),
                "n_poison_scored": int(len(poison_features)),
                "clean_reference_ids": [int(x) for x in ref_ids.tolist()],
                "clean_calibration_ids": [int(x) for x in calib_ids.tolist()],
                "poison_scored_ids": [int(x) for x in poison_ids.tolist()],
                "knn_k_effective": int(k_eff),
                "threshold_score": threshold,
                "clean_fpr_empirical": clean_fpr_empirical,
                "feature_tpr_at_fpr": feature_tpr,
                "manifold_feature_stealth": float(1.0 - feature_tpr),
                "clean_score_mean": float(np.mean(calib_scores)),
                "clean_score_median": float(np.median(calib_scores)),
                "clean_score_std": float(np.std(calib_scores, ddof=1)) if len(calib_scores) > 1 else float("nan"),
                "poison_score_mean": float(np.mean(poison_scores)),
                "poison_score_median": float(np.median(poison_scores)),
                "poison_score_std": float(np.std(poison_scores, ddof=1)) if len(poison_scores) > 1 else float("nan"),
                "score_mean_gap": float(np.mean(poison_scores) - np.mean(calib_scores)),
                "poison_score_percentile_median": float(np.median(poison_percentiles)),
                "poison_score_percentile_mean": float(np.mean(poison_percentiles)),
                "runtime_seconds": time.time() - started,
            }
        )
        return out
    except Exception as exc:
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["runtime_seconds"] = time.time() - started
        return out


def correlation_summary(rows: Sequence[Dict[str, Any]], x_key: str, y_key: str) -> Dict[str, Any]:
    valid = [r for r in rows if base.is_finite(r.get(x_key)) and base.is_finite(r.get(y_key))]
    x = np.array([float(r[x_key]) for r in valid], dtype=float)
    y = np.array([float(r[y_key]) for r in valid], dtype=float)
    if len(valid) >= 3 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
        rho, pval = stats.spearmanr(x, y)
    else:
        rho, pval = float("nan"), float("nan")
    return {"spearman_rho": float(rho), "spearman_p_value": float(pval), "n": len(valid)}


def summarize(data: Dict[str, Any]) -> None:
    defense_lookup = base.load_defense_stealth_lookup()
    rows = [r for r in data.get("checkpoint_results", []) if r.get("status") == "ok"]
    for r in rows:
        d = defense_lookup_for_result(r, defense_lookup)
        if d:
            r["defense_stealth"] = d.get("stealthiness")
    valid = [r for r in rows if not r.get("exploratory_only")]
    data["statistical_analysis"] = {
        "valid_n": len(valid),
        "target_asr_vs_manifold_feature_stealth": correlation_summary(
            valid, "target_transfer_asr", "manifold_feature_stealth"
        ),
        "defense_stealth_vs_manifold_feature_stealth": correlation_summary(
            [r for r in valid if base.is_finite(r.get("defense_stealth"))],
            "defense_stealth",
            "manifold_feature_stealth",
        ),
        "target_asr_vs_feature_tpr": correlation_summary(valid, "target_transfer_asr", "feature_tpr_at_fpr"),
    }
    data["dataset_level_summary"] = {}
    for dataset in sorted({r.get("dataset") for r in rows if r.get("dataset")}):
        sub = [r for r in rows if r.get("dataset") == dataset]
        valid_sub = [r for r in sub if not r.get("exploratory_only")]
        data["dataset_level_summary"][dataset] = {
            "selected_count": len(sub),
            "success_count": len([r for r in sub if r.get("status") == "ok"]),
            "failed_count": len([r for r in data.get("checkpoint_results", []) if r.get("dataset") == dataset and r.get("status") != "ok"]),
            "valid_n": len(valid_sub),
            "target_transfer_asr_mean": float(np.mean([r["target_transfer_asr"] for r in valid_sub])) if valid_sub else float("nan"),
            "defense_stealth_mean": float(np.mean([r["defense_stealth"] for r in valid_sub if base.is_finite(r.get("defense_stealth"))]))
            if any(base.is_finite(r.get("defense_stealth")) for r in valid_sub)
            else float("nan"),
            "manifold_feature_stealth_mean": float(np.mean([r["manifold_feature_stealth"] for r in valid_sub])) if valid_sub else float("nan"),
            "feature_tpr_at_fpr_mean": float(np.mean([r["feature_tpr_at_fpr"] for r in valid_sub])) if valid_sub else float("nan"),
            "clean_fpr_empirical_mean": float(np.mean([r["clean_fpr_empirical"] for r in valid_sub])) if valid_sub else float("nan"),
            "target_asr_vs_manifold_feature_stealth": correlation_summary(
                valid_sub, "target_transfer_asr", "manifold_feature_stealth"
            ),
        }
    data["attack_level_summary"] = {}
    for attack in base.ATTACKS:
        sub = [r for r in rows if r.get("attack") == attack]
        valid_sub = [r for r in sub if not r.get("exploratory_only")]
        data["attack_level_summary"][attack] = {
            "success_count": len(sub),
            "failed_count": len([r for r in data.get("checkpoint_results", []) if r.get("attack") == attack and r.get("status") != "ok"]),
            "valid_n": len(valid_sub),
            "target_transfer_asr_mean": float(np.mean([r["target_transfer_asr"] for r in valid_sub])) if valid_sub else float("nan"),
            "defense_stealth_mean": float(np.mean([r["defense_stealth"] for r in valid_sub if base.is_finite(r.get("defense_stealth"))]))
            if any(base.is_finite(r.get("defense_stealth")) for r in valid_sub)
            else float("nan"),
            "manifold_feature_stealth_mean": float(np.mean([r["manifold_feature_stealth"] for r in valid_sub])) if valid_sub else float("nan"),
            "feature_tpr_at_fpr_mean": float(np.mean([r["feature_tpr_at_fpr"] for r in valid_sub])) if valid_sub else float("nan"),
            "target_asr_vs_manifold_feature_stealth": correlation_summary(
                valid_sub, "target_transfer_asr", "manifold_feature_stealth"
            ),
        }


def plot_results(data: Dict[str, Any], output_dir: Path) -> None:
    rows = [
        r
        for r in data.get("checkpoint_results", [])
        if r.get("status") == "ok"
        and base.is_finite(r.get("target_transfer_asr"))
        and base.is_finite(r.get("manifold_feature_stealth"))
    ]
    colors = dict(zip(base.ATTACKS, plt.cm.tab10(np.linspace(0, 1, len(base.ATTACKS)))))

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for attack in base.ATTACKS:
        sub = [r for r in rows if r.get("attack") == attack]
        if not sub:
            continue
        ax.scatter(
            [r["target_transfer_asr"] for r in sub],
            [r["manifold_feature_stealth"] for r in sub],
            s=38,
            alpha=0.82,
            color=colors[attack],
            edgecolor="black",
            linewidth=0.3,
            label=attack,
        )
    stat = data.get("statistical_analysis", {}).get("target_asr_vs_manifold_feature_stealth", {})
    ax.text(
        0.02,
        0.03,
        f"Spearman rho={base.fmt(stat.get('spearman_rho'))}, "
        f"p={base.fmt(stat.get('spearman_p_value'), 4)}, n={stat.get('n', 0)}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    ax.set_xlabel("Target-side transfer ASR")
    ax.set_ylabel("Manifold feature stealth @5% FPR")
    ax.set_title("Transfer ASR vs Clean-Target Manifold Feature Stealth")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "target_asr_vs_manifold_feature_stealth.png", dpi=220)
    plt.close(fig)

    rows2 = [r for r in rows if base.is_finite(r.get("defense_stealth"))]
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    for attack in base.ATTACKS:
        sub = [r for r in rows2 if r.get("attack") == attack]
        if not sub:
            continue
        ax.scatter(
            [r["defense_stealth"] for r in sub],
            [r["manifold_feature_stealth"] for r in sub],
            s=38,
            alpha=0.82,
            color=colors[attack],
            edgecolor="black",
            linewidth=0.3,
            label=attack,
        )
    ax.plot([0, 1], [0, 1], color="#555555", linewidth=1.0, linestyle="--")
    stat = data.get("statistical_analysis", {}).get("defense_stealth_vs_manifold_feature_stealth", {})
    ax.text(
        0.02,
        0.03,
        f"Spearman rho={base.fmt(stat.get('spearman_rho'))}, "
        f"p={base.fmt(stat.get('spearman_p_value'), 4)}, n={stat.get('n', 0)}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    ax.set_xlabel("Defense stealth")
    ax.set_ylabel("Manifold feature stealth @5% FPR")
    ax.set_title("Defense Stealth vs Manifold Feature Stealth")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "defense_stealth_vs_manifold_feature_stealth.png", dpi=220)
    plt.close(fig)


def format_attack_counts(counts: Dict[str, Any]) -> str:
    return ", ".join(f"{k}:{v}" for k, v in counts.items() if v) or "NA"


def write_report(data: Dict[str, Any], output_dir: Path) -> None:
    cfg = data.get("pilot_config", {})
    ds_sel = data.get("dataset_selection_summary", {})
    sel = data.get("selection_summary", {})
    ds_sum = data.get("dataset_level_summary", {})
    atk_sum = data.get("attack_level_summary", {})
    stat = data.get("statistical_analysis", {})
    results = data.get("checkpoint_results", [])
    ok = [r for r in results if r.get("status") == "ok"]
    lines: List[str] = []
    lines.append("# Clean-Target Manifold Feature Stealth Pilot Report\n")
    lines.append("## 1. 实验目的\n")
    lines.append(
        "本实验重新定义 feature-level stealth：不再把 oracle clean/poison 线性 probe 作为主指标，"
        "而是先用 clean target-class feature 建立正常特征流形，再看 payload poison 有多少落在该流形之外。"
        "它回答的问题是：poison feature 是否仍然嵌在 clean target-class manifold 内，还是表现为特征异常点/异常子群。\n"
    )
    lines.append("主指标定义如下：\n")
    lines.append(
        "`feature_tpr@5%fpr = P(score_knn(poison) > tau_95(clean_calibration))`；"
        "`manifold_feature_stealth@5%fpr = 1 - feature_tpr@5%fpr`。"
        "分数越高表示 poison 越少被 clean-target manifold anomaly detector 检出，特征层越隐蔽。\n"
    )
    lines.append("## 2. 配置选择\n")
    if cfg.get("selection_mode") == "coverage_balanced":
        lines.append(
            f"- 数据根目录：`{cfg.get('data_root')}`。\n"
            f"- 数据集：`{', '.join(cfg.get('datasets', []))}`。\n"
            f"- 选择规则和线性 probe 扩展实验一致：总计 {cfg.get('total_checkpoints')} 个 checkpoint/config，"
            "先按 dataset 尽量均分名额，再在每个 dataset 内贪心覆盖 attack、poison rate、attack strength、architecture 和 target transfer ASR 范围。\n"
            "- 选择前过滤没有 balanced clean/poison 对照样本的配置；source ASR 只用于确认源域攻击是否成功。\n"
        )
    else:
        lines.append(
            f"- 数据根目录：`{cfg.get('data_root')}`。\n"
            f"- 数据集：`{', '.join(cfg.get('datasets', []))}`。\n"
            f"- 选择规则和前一版 feature stealth pilot 一致：每个数据集选 {cfg.get('checkpoints_per_dataset')} 个 checkpoint，"
            "按 target-side transfer ASR 覆盖低、中、高迁移强度；选择前过滤没有 balanced clean/poison 对照样本的配置。\n"
            "- 选择指标仍只使用 target-side transfer ASR；source ASR 只用于确认源域攻击是否成功。\n"
        )
    if ds_sel:
        lines.append("| Dataset | quota | raw candidates | feasible candidates | filtered infeasible | selected | ASR min | ASR max | selected attacks | poison rates | strengths | filtered attacks |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
        for dataset, s in ds_sel.items():
            lines.append(
                f"| {dataset} | {s.get('quota', cfg.get('checkpoints_per_dataset', 'NA'))} | "
                f"{s.get('raw_candidate_count', 0)} | {s.get('candidate_count', 0)} | "
                f"{s.get('filtered_infeasible_count', 0)} | {s.get('selected_count', 0)} | "
                f"{base.fmt(s.get('asr_min'))} | {base.fmt(s.get('asr_max'))} | "
                f"{base.format_count_map(s.get('selected_attack_counts', {}))} | "
                f"{base.format_count_map(s.get('selected_poison_rate_counts', {}))} | "
                f"{base.format_count_map(s.get('selected_strength_counts', {}))} | "
                f"{base.format_count_map(s.get('filtered_infeasible_attack_counts', {}))} |"
            )
        lines.append("")
    lines.append("Attack 覆盖：\n")
    lines.append("| Attack | candidates | selected | success | failed |")
    lines.append("|---|---:|---:|---:|---:|")
    for attack in base.ATTACKS:
        s = sel.get(attack, {})
        a = atk_sum.get(attack, {})
        lines.append(
            f"| {attack} | {s.get('candidate_count', 0)} | {s.get('selected_count', 0)} | "
            f"{a.get('success_count', 0)} | {a.get('failed_count', 0)} |"
        )
    lines.append("\n## 3. 计算流程\n")
    lines.append(
        "对每个 checkpoint 固定 victim model，提取 penultimate feature。clean target 样本来自 poisoned train set 中 `final_label == target_class` "
        "且不属于 poison/cover 的样本；payload poison 使用 `poison_indices`，BELT 使用 `pmarks==1`，cover 样本排除。"
        "clean target feature 按固定 seed 分成 reference/calibration 两半，reference 建 kNN 邻居集合，calibration 定阈值。"
        f"本次默认 `k={cfg.get('knn_k')}`、`metric={cfg.get('knn_metric')}`、`target_fpr={cfg.get('target_fpr')}`。"
        "cosine 距离下先对 feature 做 L2 normalize。UPGD/BELT 继续沿用项目中的 raw `[0,1]` / `no_normalize` 兼容路径。\n"
    )
    lines.append("## 4. 数据质量检查\n")
    lines.append("| Dataset | selected | success | mean clean empirical FPR | mean poison TPR@5%FPR | mean manifold stealth |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dataset, s in ds_sum.items():
        lines.append(
            f"| {dataset} | {s.get('selected_count', 0)} | {s.get('success_count', 0)} | "
            f"{base.fmt(s.get('clean_fpr_empirical_mean'))} | {base.fmt(s.get('feature_tpr_at_fpr_mean'))} | "
            f"{base.fmt(s.get('manifold_feature_stealth_mean'))} |"
        )
    lines.append(
        "\n`clean empirical FPR` 应接近但不必精确等于 0.05，因为阈值来自有限 calibration 样本的 95 分位数，且使用 `>` 判断异常。\n"
    )
    lines.append("## 5. 总体结果\n")
    tstat = stat.get("target_asr_vs_manifold_feature_stealth", {})
    dstat = stat.get("defense_stealth_vs_manifold_feature_stealth", {})
    lines.append(
        f"有效 checkpoint 数量 n={stat.get('valid_n', 0)}。target transfer ASR 与 manifold feature stealth 的 Spearman "
        f"rho={base.fmt(tstat.get('spearman_rho'))}, p={base.fmt(tstat.get('spearman_p_value'), 4)}。"
        f"Defense stealth 与 manifold feature stealth 的 Spearman rho={base.fmt(dstat.get('spearman_rho'))}, "
        f"p={base.fmt(dstat.get('spearman_p_value'), 4)}。\n"
    )
    lines.append("图 `target_asr_vs_manifold_feature_stealth.png` 展示迁移 ASR 与新特征隐蔽性的关系；图 `defense_stealth_vs_manifold_feature_stealth.png` 展示防御隐蔽性与特征流形隐蔽性的差异。\n")
    lines.append("### Checkpoint-level 明细\n")
    lines.append("| Dataset | Attack | Poison rate | Strength | Source ASR | Target transfer ASR | Defense stealth | Feature TPR@5%FPR | Manifold feature stealth | Clean FPR | Score gap | n ref/calib/poison | Experiment |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in sorted(ok, key=lambda x: (str(x.get("dataset")), float(x.get("target_transfer_asr", 0)), str(x.get("attack")))):
        lines.append(
            f"| {r.get('dataset')} | {r.get('attack')} | {base.fmt(r.get('poison_rate'))} | "
            f"{r.get('attack_strength') or 'NA'} | {base.fmt(r.get('source_asr'))} | "
            f"{base.fmt(r.get('target_transfer_asr'))} | {base.fmt(r.get('defense_stealth'))} | "
            f"{base.fmt(r.get('feature_tpr_at_fpr'))} | {base.fmt(r.get('manifold_feature_stealth'))} | "
            f"{base.fmt(r.get('clean_fpr_empirical'))} | {base.fmt(r.get('score_mean_gap'))} | "
            f"{r.get('n_clean_reference', 'NA')}/{r.get('n_clean_calibration', 'NA')}/{r.get('n_poison_scored', 'NA')} | "
            f"`{base.short_experiment_id(r.get('experiment_id'))}` |"
        )
    lines.append("\n## 6. 攻击类型层面结果\n")
    lines.append("| Attack | valid n | mean target ASR | mean defense stealth | mean manifold stealth | mean feature TPR@5%FPR | ASR-stealth rho | interpretation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for attack in base.ATTACKS:
        a = atk_sum.get(attack, {})
        corr = a.get("target_asr_vs_manifold_feature_stealth", {})
        n = int(a.get("valid_n", 0) or 0)
        if n < 3:
            interp = "样本太少，只作记录"
        elif base.is_finite(corr.get("spearman_rho")) and float(corr.get("spearman_rho")) < -0.1:
            interp = "迁移越强，越容易偏离 clean target manifold"
        elif base.is_finite(corr.get("spearman_rho")) and float(corr.get("spearman_rho")) > 0.1:
            interp = "出现反向/异质性信号"
        else:
            interp = "趋势不明显"
        lines.append(
            f"| {attack} | {n} | {base.fmt(a.get('target_transfer_asr_mean'))} | "
            f"{base.fmt(a.get('defense_stealth_mean'))} | {base.fmt(a.get('manifold_feature_stealth_mean'))} | "
            f"{base.fmt(a.get('feature_tpr_at_fpr_mean'))} | {base.fmt(corr.get('spearman_rho'))} | {interp} |"
        )
    lines.append("\n## 7. 结果解释\n")
    high_def_low_feat = [
        r
        for r in ok
        if base.is_finite(r.get("defense_stealth"))
        and r["defense_stealth"] >= 0.7
        and r.get("manifold_feature_stealth", 1.0) <= 0.3
    ]
    high_feat = sorted(ok, key=lambda r: r.get("manifold_feature_stealth", -1), reverse=True)[:5]
    lines.append(
        f"高防御隐蔽但低特征流形隐蔽的 checkpoint 数量为 {len(high_def_low_feat)}。"
        "这类点说明现有 SentiNet/ScaleUp/STRIP/IBD-PSC 平均不一定能捕获 clean-target feature manifold 外的异常。"
        "相反，如果 manifold feature stealth 高，则说明 poison feature 大多落在 clean target calibration 的正常范围内。\n"
    )
    lines.append("Manifold feature stealth 最高的代表点：\n")
    for r in high_feat:
        lines.append(
            f"- `{base.short_experiment_id(r.get('experiment_id'))}`: target ASR={base.fmt(r.get('target_transfer_asr'))}, "
            f"defense stealth={base.fmt(r.get('defense_stealth'))}, manifold stealth={base.fmt(r.get('manifold_feature_stealth'))}, "
            f"feature TPR={base.fmt(r.get('feature_tpr_at_fpr'))}\n"
        )
    failures = [r for r in results if r.get("status") != "ok"]
    if failures:
        lines.append("失败 checkpoint 保留在 JSON 中，示例：\n")
        for r in failures[:10]:
            lines.append(f"- `{r.get('experiment_id')}`: {r.get('error')}\n")
    lines.append("## 8. 限制\n")
    lines.append(
        "这个指标仍然需要 poison indices 来评估 TPR，但检测阈值只由 clean target calibration 决定。"
        "dirty-label 攻击的 poison 可能因为原始语义不是 target class 而偏离 target manifold，因此后续最好加入 semantic control。"
        "此外，kNN score 受 feature 层、距离度量、k 值和 clean reference 覆盖度影响；当前结果应作为 pilot，而不是最终全量结论。\n"
    )
    (output_dir / "feature_manifold_stealth_pilot_report.md").write_text("\n".join(lines), encoding="utf-8")


def initial_results(
    args: argparse.Namespace,
    selection_summary: Dict[str, Any],
    dataset_selection_summary: Dict[str, Any],
    discovery: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "pilot_config": {
            "data_root": str(Path(args.data_root).resolve()),
            "datasets": args.datasets_list,
            "selection_mode": args.selection_mode,
            "checkpoints_per_dataset": args.checkpoints_per_dataset,
            "total_checkpoints": args.total_checkpoints,
            "selection_metric": args.selection_metric,
            "asr_bins_internal_selection": args.asr_bins,
            "samples_per_bin_internal_selection": args.samples_per_bin,
            "sample_feasibility_precheck": args.sample_feasibility_precheck,
            "max_samples_per_class": args.max_samples_per_class,
            "reference_ratio": args.reference_ratio,
            "target_fpr": args.target_fpr,
            "knn_k": args.knn_k,
            "knn_metric": args.knn_metric,
            "sample_seed": args.sample_seed,
            "split_seed": args.split_seed,
            "discovery": discovery,
        },
        "selection_summary": selection_summary,
        "dataset_selection_summary": dataset_selection_summary,
        "checkpoint_results": [],
        "dataset_level_summary": {},
        "attack_level_summary": {},
        "statistical_analysis": {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean-target manifold feature-stealth pilot.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default="cifar10,tiny_imagenet")
    parser.add_argument("--checkpoints-per-dataset", type=int, default=20)
    parser.add_argument("--selection-mode", default="asr_stratified", choices=["asr_stratified", "coverage_balanced"])
    parser.add_argument("--total-checkpoints", type=int, default=40)
    parser.add_argument("--selection-metric", default="target_transfer_asr", choices=["target_transfer_asr"])
    parser.add_argument("--asr-bins", type=int, default=4)
    parser.add_argument("--samples-per-bin", type=int, default=5)
    parser.add_argument("--sample-feasibility-precheck", action="store_true")
    parser.add_argument("--max-samples-per-class", type=int, default=3000)
    parser.add_argument("--reference-ratio", type=float, default=0.5)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--knn-metric", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--min-clean-samples", type=int, default=20)
    parser.add_argument("--min-poison-samples", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    args.datasets_list = [x.strip().replace("-", "_") for x in args.datasets.split(",") if x.strip()]
    unknown = [x for x in args.datasets_list if x not in base.NUM_CLASSES]
    if unknown:
        raise ValueError(f"Unknown datasets in --datasets: {unknown}")
    if args.asr_bins != 4:
        raise ValueError("This pilot expects --asr-bins 4 for compatibility with the previous selection logic")
    if not (0.0 < args.reference_ratio < 1.0):
        raise ValueError("--reference-ratio must be in (0, 1)")
    if not (0.0 < args.target_fpr < 1.0):
        raise ValueError("--target-fpr must be in (0, 1)")
    return args


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["POISONED_TRAIN_SET_ROOT"] = str(data_root)
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates, discovery = base.discover_experiments(data_root, output_dir)
    candidates = [c for c in candidates if c.dataset in args.datasets_list]
    if args.selection_mode == "coverage_balanced":
        selected, selection_summary, dataset_selection_summary = base.select_coverage_balanced_checkpoints(
            candidates,
            datasets_wanted=args.datasets_list,
            total_checkpoints=args.total_checkpoints,
            feasibility_precheck=args.sample_feasibility_precheck,
        )
    else:
        selected, selection_summary, dataset_selection_summary = base.select_dataset_stratified_checkpoints(
            candidates,
            datasets_wanted=args.datasets_list,
            checkpoints_per_dataset=args.checkpoints_per_dataset,
            samples_per_bin=args.samples_per_bin,
            feasibility_precheck=args.sample_feasibility_precheck,
        )

    result_path = output_dir / "feature_manifold_stealth_pilot_results.json"
    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        existing = {r.get("experiment_id") for r in data.get("checkpoint_results", []) if r.get("status") == "ok"}
        data["selection_summary"] = selection_summary
        data["dataset_selection_summary"] = dataset_selection_summary
        data["pilot_config"]["discovery"] = discovery
        data["pilot_config"]["selection_mode"] = args.selection_mode
        data["pilot_config"]["total_checkpoints"] = args.total_checkpoints
    else:
        data = initial_results(args, selection_summary, dataset_selection_summary, discovery)
        existing = set()
    base.safe_write_json(result_path, data)

    for i, candidate in enumerate(selected, 1):
        if candidate.experiment_id in existing:
            print(f"[{i}/{len(selected)}] skip existing {candidate.experiment_id}")
            continue
        print(
            f"[{i}/{len(selected)}] {candidate.attack} ASR={candidate.selection_asr_value:.4f} {candidate.experiment_id}",
            flush=True,
        )
        result = run_checkpoint(candidate, args, device)
        data["checkpoint_results"] = [
            r for r in data.get("checkpoint_results", []) if r.get("experiment_id") != candidate.experiment_id
        ]
        data["checkpoint_results"].append(result)
        summarize(data)
        base.safe_write_json(result_path, data)

    summarize(data)
    plot_results(data, output_dir)
    write_report(data, output_dir)
    base.safe_write_json(result_path, data)
    print(f"[done] {result_path}")
    print(f"[done] {output_dir / 'target_asr_vs_manifold_feature_stealth.png'}")
    print(f"[done] {output_dir / 'defense_stealth_vs_manifold_feature_stealth.png'}")
    print(f"[done] {output_dir / 'feature_manifold_stealth_pilot_report.md'}")


if __name__ == "__main__":
    main()
