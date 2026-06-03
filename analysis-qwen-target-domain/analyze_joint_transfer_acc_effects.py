"""Joint-transfer and ACC-shape analysis for Tiny target-domain results.

This script extends the previous ACC analysis with:
- joint_transfer = source_ASR * transfer_ASR / (source_ASR + transfer_ASR)
- joint_transfer_norm = 2 * joint_transfer
- transfer_retention = transfer_ASR / source_ASR
- ACC effect-shape analysis (continuous, binned, scan-based, attack interaction)
- WaNet pilot analysis and basic contrast

Outputs live in analysis-qwen-target-domain:
- acc_joint_transfer_rows.csv
- acc_joint_transfer_paired_rows.csv
- acc_joint_transfer_summary.json
- acc_effect_shape_summary.csv
- acc_caution_region_scan.csv
- final_acc_joint_transfer_tradeoff_report.md
- figures/*.png
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis-qwen-target-domain"
BASE = ROOT / "poisoned_train_set" / "tiny_imagenet"
FIG_DIR = ANALYSIS_DIR / "figures"

ACC_RE = re.compile(r"准确率:\s*([0-9.]+)")
ASR_RE = re.compile(r"攻击成功率\s*\(ASR\):\s*([0-9.]+)")
ATTACK_RE = re.compile(r"攻击类型:\s*(\S+)")

DEFENSE_FILES = {
    "strip": "strip_defense_results.json",
    "sentinet": "sentinet_defense_results.json",
    "scaleup": "scaleup_defense_results.json",
    "ibd_psc": "ibd_psc_defense_results.json",
}


def norm_arch_for_csv(arch: str) -> str:
    if arch == "mobilenetv2":
        return "mobilenet"
    if arch == "vgg19_bn":
        return "vgg"
    return arch


def load_source_asr_fallback() -> dict[tuple[str, str, float, float], float]:
    """Load source ASR from the no-NC extraction CSVs.

    Some SIG/WaNet folders do not contain test_results_seed=2333.json, but the
    extracted analysis CSVs already contain their source-domain ASR (`asr`).
    Key: (attack, csv_arch, poison_rate, test_param_value).
    """
    out: dict[tuple[str, str, float, float], float] = {}
    for path in (ROOT / "analysis-testASR").glob("data_tiny_imagenet_*_no_nc.csv"):
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = (
                        row["attack_type"],
                        row["arch"],
                        round(float(row["poison_rate"]), 6),
                        round(float(row["test_param_value"]), 6),
                    )
                    out[key] = float(row["asr"])
                except (KeyError, TypeError, ValueError):
                    continue
    return out


SOURCE_ASR_FALLBACK = load_source_asr_fallback()


def extract_test_param(config: str, attack: str) -> float | None:
    patterns = {
        "SIG": r"_delta=([0-9.]+)",
        "WaNet": r"_s=([0-9.]+)",
        "upgd": r"_eps=([0-9.]+)",
        "basic": r"_alpha=([0-9.]+)",
        "blend": r"_alpha=([0-9.]+)",
        "adaptive_blend": r"_alpha=([0-9.]+)",
        "adaptive_patch": r"_alpha=([0-9.]+)",
        "belt": r"_alpha=([0-9.]+)",
    }
    pattern = patterns.get(attack)
    if not pattern:
        return None
    matches = re.findall(pattern, config)
    if not matches:
        return None
    return float(matches[-1])


def extract_poison_rate(config: str, attack: str) -> float | None:
    match = re.search(rf"^{re.escape(attack)}_([0-9.]+)", config)
    return float(match.group(1)) if match else None

ATTACK_COLORS = {
    "SIG": "#4C78A8",
    "upgd": "#F58518",
    "WaNet": "#54A24B",
    "basic": "#E45756",
    "blend": "#72B7B2",
    "adaptive_blend": "#B279A2",
    "adaptive_patch": "#FF9DA6",
    "belt": "#9D755D",
}


def parse_result(path: Path) -> tuple[float, float, str]:
    text = path.read_text(encoding="utf-8")
    return (
        float(ACC_RE.search(text).group(1)),
        float(ASR_RE.search(text).group(1)),
        ATTACK_RE.search(text).group(1),
    )


def result_suffix(name: str, prefix: str) -> str:
    return name[len(prefix):-4]


def arch_from_dir(name: str) -> str:
    if "arch=ResNet18" in name:
        return "resnet18"
    if "arch=mobilenetv2" in name:
        return "mobilenetv2"
    if "arch=vgg19_bn" in name:
        return "vgg19_bn"
    return "unknown"


def read_source_metrics(model_dir: Path) -> dict[str, float | None]:
    path = model_dir / "test_results_seed=2333.json"
    if not path.exists():
        return {"source_clean_acc": None, "source_asr": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source_clean_acc": data.get("clean_acc"),
        "source_asr": data.get("asr"),
    }


def read_defense_metrics(model_dir: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    aucs: list[float] = []
    tprs: list[float] = []
    for key, filename in DEFENSE_FILES.items():
        path = model_dir / filename
        if not path.exists():
            out[f"{key}_auc"] = None
            out[f"{key}_tpr"] = None
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        auc = float(data["auc"]) if data.get("auc") is not None else None
        tpr = float(data["tpr"]) if data.get("tpr") is not None else None
        if tpr is not None and tpr > 1:
            tpr /= 100.0
        out[f"{key}_auc"] = auc
        out[f"{key}_tpr"] = tpr
        if auc is not None:
            aucs.append(auc)
        if tpr is not None:
            tprs.append(tpr)
    out["detection_auc"] = mean(aucs) if aucs else None
    out["detection_tpr"] = mean(tprs) if tprs else None
    out["stealth_auc"] = 1 - out["detection_auc"] if out["detection_auc"] is not None else None
    out["stealth_tpr"] = 1 - out["detection_tpr"] if out["detection_tpr"] is not None else None
    return out


def joint_metrics(source_asr: float | None, transfer_asr: float) -> dict[str, float | None]:
    if source_asr is None or source_asr <= 0 or transfer_asr < 0:
        return {
            "joint_transfer": None,
            "joint_transfer_norm": None,
            "transfer_retention": None,
        }
    denom = source_asr + transfer_asr
    joint = (source_asr * transfer_asr / denom) if denom > 0 else 0.0
    return {
        "joint_transfer": joint,
        "joint_transfer_norm": 2 * joint,
        "transfer_retention": transfer_asr / source_asr,
    }


def collect_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for model_dir in sorted(p for p in BASE.iterdir() if p.is_dir()):
        iv2_files = {
            result_suffix(p.name, "test_tiny_target_domain_results"): p
            for p in model_dir.iterdir()
            if p.name.startswith("test_tiny_target_domain_results") and p.name.endswith(".txt")
        }
        qwen_files = {
            result_suffix(p.name, "test_tiny_target_domain_qwen_results"): p
            for p in model_dir.iterdir()
            if p.name.startswith("test_tiny_target_domain_qwen_results") and p.name.endswith(".txt")
        }
        if not iv2_files or not qwen_files:
            continue

        source = read_source_metrics(model_dir)
        defenses = read_defense_metrics(model_dir)
        arch = arch_from_dir(model_dir.name)

        for variant in sorted(set(iv2_files) & set(qwen_files)):
            iv2_acc, iv2_asr, attack = parse_result(iv2_files[variant])
            qwen_acc, qwen_asr, _ = parse_result(qwen_files[variant])
            config = model_dir.name + variant
            source_for_config = dict(source)
            if source_for_config.get("source_asr") is None:
                poison_rate = extract_poison_rate(config, attack)
                test_param = extract_test_param(config, attack)
                if poison_rate is not None and test_param is not None:
                    fallback_key = (
                        attack,
                        norm_arch_for_csv(arch),
                        round(poison_rate, 6),
                        round(test_param, 6),
                    )
                    fallback_asr = SOURCE_ASR_FALLBACK.get(fallback_key)
                    if fallback_asr is not None:
                        source_for_config["source_asr"] = fallback_asr

            common = {
                "config": config,
                "model_dir": str(model_dir.relative_to(ROOT)),
                "attack": attack,
                "arch": arch,
                **source_for_config,
                **defenses,
            }

            iv2_joint = joint_metrics(source_for_config["source_asr"], iv2_asr)
            qwen_joint = joint_metrics(source_for_config["source_asr"], qwen_asr)
            long_rows.append({
                **common,
                "domain": "ImageNetV2",
                "target_acc": iv2_acc,
                "transfer_asr": iv2_asr,
                **iv2_joint,
            })
            long_rows.append({
                **common,
                "domain": "Qwen",
                "target_acc": qwen_acc,
                "transfer_asr": qwen_asr,
                **qwen_joint,
            })
            paired_rows.append({
                **common,
                "iv2_acc": iv2_acc,
                "iv2_transfer_asr": iv2_asr,
                "iv2_joint_transfer": iv2_joint["joint_transfer"],
                "iv2_joint_transfer_norm": iv2_joint["joint_transfer_norm"],
                "qwen_acc": qwen_acc,
                "qwen_transfer_asr": qwen_asr,
                "qwen_joint_transfer": qwen_joint["joint_transfer"],
                "qwen_joint_transfer_norm": qwen_joint["joint_transfer_norm"],
                "delta_acc": qwen_acc - iv2_acc,
                "delta_transfer_asr": qwen_asr - iv2_asr,
                "delta_joint_transfer": (
                    qwen_joint["joint_transfer"] - iv2_joint["joint_transfer"]
                    if qwen_joint["joint_transfer"] is not None and iv2_joint["joint_transfer"] is not None
                    else None
                ),
            })

    return long_rows, paired_rows


def clean_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ok = True
        for key in keys:
            value = row.get(key)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                ok = False
                break
        if ok:
            out.append(row)
    return out


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) is not None]


def avg(rows: list[dict[str, Any]], key: str) -> float:
    vals = values(rows, key)
    return mean(vals) if vals else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ranks(vals: list[float]) -> list[float]:
    order = sorted((v, i) for i, v in enumerate(vals))
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k][1]] = rank
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def slope_per_10pp(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float:
    sub = clean_rows(rows, [x_key, y_key])
    if len(sub) < 3:
        return float("nan")
    x = np.asarray(values(sub, x_key), dtype=float)
    y = np.asarray(values(sub, y_key), dtype=float)
    if np.std(x) == 0:
        return float("nan")
    beta = np.polyfit(x, y, 1)[0]
    return float(beta * 0.10)


def partial_corr(rows: list[dict[str, Any]], x_key: str, y_key: str, z_key: str) -> float:
    sub = clean_rows(rows, [x_key, y_key, z_key])
    if len(sub) < 5:
        return float("nan")
    x = np.asarray(values(sub, x_key), dtype=float)
    y = np.asarray(values(sub, y_key), dtype=float)
    z = np.asarray(values(sub, z_key), dtype=float)
    X = np.column_stack([np.ones(len(z)), z])
    bx = np.linalg.pinv(X) @ x
    by = np.linalg.pinv(X) @ y
    return pearson(list(x - X @ bx), list(y - X @ by))


def corr_pair(rows: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, float]:
    sub = clean_rows(rows, [x_key, y_key])
    xs = values(sub, x_key)
    ys = values(sub, y_key)
    return {
        "n": len(sub),
        "pearson": pearson(xs, ys),
        "spearman": spearman(xs, ys),
    }


def ols_r2(rows: list[dict[str, Any]], y_key: str, feature_keys: list[str]) -> float:
    sub = clean_rows(rows, [y_key] + [k for k in feature_keys if not k.startswith(("attack=", "arch=", "domain="))])
    if len(sub) < 10:
        return float("nan")
    y = np.asarray(values(sub, y_key), dtype=float)
    cols = [np.ones(len(sub))]
    for key in feature_keys:
        if key.startswith("attack="):
            val = key.split("=", 1)[1]
            cols.append(np.asarray([1.0 if r["attack"] == val else 0.0 for r in sub]))
        elif key.startswith("arch="):
            val = key.split("=", 1)[1]
            cols.append(np.asarray([1.0 if r["arch"] == val else 0.0 for r in sub]))
        elif key.startswith("domain="):
            val = key.split("=", 1)[1]
            cols.append(np.asarray([1.0 if r["domain"] == val else 0.0 for r in sub]))
        else:
            cols.append(np.asarray(values(sub, key), dtype=float))
    X = np.column_stack(cols)
    beta = np.linalg.pinv(X) @ y
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def summarize_scope(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    sub = clean_rows(rows, ["target_acc", "transfer_asr", "joint_transfer", "detection_auc", "stealth_auc"])
    return {
        "scope": scope,
        "n": len(sub),
        "mean_target_acc": avg(sub, "target_acc"),
        "mean_transfer_asr": avg(sub, "transfer_asr"),
        "mean_joint_transfer": avg(sub, "joint_transfer"),
        "mean_joint_transfer_norm": avg(sub, "joint_transfer_norm"),
        "mean_detection_auc": avg(sub, "detection_auc"),
        "mean_stealth_auc": avg(sub, "stealth_auc"),
        "acc_transfer_corr": corr_pair(sub, "target_acc", "transfer_asr"),
        "acc_joint_corr": corr_pair(sub, "target_acc", "joint_transfer"),
        "acc_stealth_corr": corr_pair(sub, "target_acc", "stealth_auc"),
        "transfer_stealth_corr": corr_pair(sub, "transfer_asr", "stealth_auc"),
        "joint_stealth_corr": corr_pair(sub, "joint_transfer", "stealth_auc"),
        "transfer_detection_corr": corr_pair(sub, "transfer_asr", "detection_auc"),
        "joint_detection_corr": corr_pair(sub, "joint_transfer", "detection_auc"),
        "partial_transfer_detection_control_acc": partial_corr(sub, "transfer_asr", "detection_auc", "target_acc"),
        "partial_joint_detection_control_acc": partial_corr(sub, "joint_transfer", "detection_auc", "target_acc"),
        "acc_to_transfer_slope_per_10pp": slope_per_10pp(sub, "target_acc", "transfer_asr"),
        "acc_to_joint_slope_per_10pp": slope_per_10pp(sub, "target_acc", "joint_transfer"),
        "acc_to_stealth_slope_per_10pp": slope_per_10pp(sub, "target_acc", "stealth_auc"),
    }


def binned_summary(rows: list[dict[str, Any]], scope: str, bins: int = 4) -> list[dict[str, Any]]:
    sub = sorted(clean_rows(rows, ["target_acc", "transfer_asr", "joint_transfer", "detection_auc", "stealth_auc"]), key=lambda r: r["target_acc"])
    if not sub:
        return []
    chunks = np.array_split(sub, bins)
    out = []
    for idx, chunk in enumerate(chunks, start=1):
        part = list(chunk)
        out.append({
            "scope": scope,
            "bin": idx,
            "n": len(part),
            "acc_min": min(values(part, "target_acc")),
            "acc_max": max(values(part, "target_acc")),
            "mean_acc": avg(part, "target_acc"),
            "mean_transfer_asr": avg(part, "transfer_asr"),
            "mean_joint_transfer": avg(part, "joint_transfer"),
            "mean_joint_transfer_norm": avg(part, "joint_transfer_norm"),
            "mean_detection_auc": avg(part, "detection_auc"),
            "mean_stealth_auc": avg(part, "stealth_auc"),
        })
    return out


def scan_acc_regions(rows: list[dict[str, Any]], scope: str, thresholds: list[float]) -> list[dict[str, Any]]:
    out = []
    base = clean_rows(rows, ["target_acc", "transfer_asr", "joint_transfer", "stealth_auc", "detection_auc"])
    for t in thresholds:
        low = [r for r in base if float(r["target_acc"]) <= t]
        high = [r for r in base if float(r["target_acc"]) > t]
        if len(low) < 20 or len(high) < 20:
            continue
        low_acc_transfer = corr_pair(low, "target_acc", "transfer_asr")["pearson"]
        low_acc_stealth = corr_pair(low, "target_acc", "stealth_auc")["pearson"]
        out.append({
            "scope": scope,
            "threshold": t,
            "low_n": len(low),
            "high_n": len(high),
            "low_mean_acc": avg(low, "target_acc"),
            "high_mean_acc": avg(high, "target_acc"),
            "low_mean_transfer_asr": avg(low, "transfer_asr"),
            "high_mean_transfer_asr": avg(high, "transfer_asr"),
            "low_mean_joint_transfer": avg(low, "joint_transfer"),
            "high_mean_joint_transfer": avg(high, "joint_transfer"),
            "low_mean_detection_auc": avg(low, "detection_auc"),
            "high_mean_detection_auc": avg(high, "detection_auc"),
            "low_mean_stealth_auc": avg(low, "stealth_auc"),
            "high_mean_stealth_auc": avg(high, "stealth_auc"),
            "low_corr_acc_transfer": low_acc_transfer,
            "low_corr_acc_joint": corr_pair(low, "target_acc", "joint_transfer")["pearson"],
            "low_corr_acc_stealth": low_acc_stealth,
            "low_corr_transfer_stealth": corr_pair(low, "transfer_asr", "stealth_auc")["pearson"],
            "low_corr_joint_stealth": corr_pair(low, "joint_transfer", "stealth_auc")["pearson"],
            "caution_score": (
                max(0.0, -low_acc_transfer)
                + max(0.0, 0.20 - abs(low_acc_stealth))
                + abs(avg(low, "mean_placeholder") if False else 0.0)
            ),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(x: float | None) -> str:
    if x is None or math.isnan(float(x)):
        return "NA"
    return f"{float(x) * 100:.2f}%"


def fmt_pp(x: float | None) -> str:
    if x is None or math.isnan(float(x)):
        return "NA"
    return f"{float(x) * 100:+.2f}pp"


def fmt_num(x: float | None, digits: int = 3) -> str:
    if x is None or math.isnan(float(x)):
        return "NA"
    return f"{float(x):+.{digits}f}"


def plot_scatter_by_domain(rows: list[dict[str, Any]], y_key: str, ylabel: str, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, domain in zip(axes, ["ImageNetV2", "Qwen"]):
        sub_domain = [r for r in rows if r["domain"] == domain]
        for attack in sorted(ATTACK_COLORS):
            sub = clean_rows([r for r in sub_domain if r["attack"] == attack], ["target_acc", y_key])
            if not sub:
                continue
            ax.scatter(
                [float(r["target_acc"]) * 100 for r in sub],
                [float(r[y_key]) * 100 for r in sub],
                s=18,
                alpha=0.65,
                color=ATTACK_COLORS[attack],
                label=attack,
            )
        clean = clean_rows(sub_domain, ["target_acc", y_key])
        if len(clean) > 3:
            xs = np.asarray([float(r["target_acc"]) * 100 for r in clean])
            ys = np.asarray([float(r[y_key]) * 100 for r in clean])
            beta = np.polyfit(xs, ys, 1)
            xline = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(xline, beta[0] * xline + beta[1], color="black", linewidth=1.5)
        ax.set_title(f"{domain}: ACC vs {ylabel}")
        ax.set_xlabel("Target clean ACC (%)")
        ax.set_ylabel(f"{ylabel} (%)")
        ax.grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=180)
    plt.close(fig)


def plot_tradeoff(rows: list[dict[str, Any]], x_key: str, xlabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for attack in sorted(ATTACK_COLORS):
        sub = clean_rows([r for r in rows if r["attack"] == attack], [x_key, "stealth_auc"])
        if not sub:
            continue
        ax.scatter(
            [float(r[x_key]) * 100 for r in sub],
            [float(r["stealth_auc"]) * 100 for r in sub],
            s=18,
            alpha=0.65,
            color=ATTACK_COLORS[attack],
            label=attack,
        )
    ax.set_xlabel(f"{xlabel} (%)")
    ax.set_ylabel("Stealth AUC = 1 - detection AUC (%)")
    ax.set_title(f"{xlabel} vs stealth AUC")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=180)
    plt.close(fig)


def plot_binned(bins: list[dict[str, Any]], scope: str, filename: str) -> None:
    sub = [b for b in bins if b["scope"] == scope]
    if not sub:
        return
    labels = [f"{b['acc_min']*100:.0f}-{b['acc_max']*100:.0f}%" for b in sub]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    ax.bar(x - width * 1.5, [b["mean_transfer_asr"] * 100 for b in sub], width, label="transfer_ASR")
    ax.bar(x - width * 0.5, [b["mean_joint_transfer_norm"] * 100 if "mean_joint_transfer_norm" in b else b["mean_joint_transfer"] * 200 for b in sub], width, label="joint_transfer_norm")
    ax.bar(x + width * 0.5, [b["mean_detection_auc"] * 100 for b in sub], width, label="detection_AUC")
    ax.bar(x + width * 1.5, [b["mean_stealth_auc"] * 100 for b in sub], width, label="stealth_AUC")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Mean value (%)")
    ax.set_title(f"ACC effect shape by bins: {scope}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=180)
    plt.close(fig)


def plot_scan(scan_rows: list[dict[str, Any]], scope: str, filename: str) -> None:
    sub = [r for r in scan_rows if r["scope"] == scope]
    if not sub:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = [r["threshold"] * 100 for r in sub]
    ax.plot(xs, [r["low_corr_acc_transfer"] for r in sub], marker="o", label="corr(ACC, transfer_ASR) in low_ACC")
    ax.plot(xs, [r["low_corr_acc_joint"] for r in sub], marker="o", label="corr(ACC, joint_transfer) in low_ACC")
    ax.plot(xs, [r["low_corr_acc_stealth"] for r in sub], marker="o", label="corr(ACC, stealth_AUC) in low_ACC")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Candidate ACC threshold (%)")
    ax.set_ylabel("Pearson correlation inside low_ACC group")
    ax.set_title(f"ACC caution-region scan: {scope}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=180)
    plt.close(fig)


def plot_attack_delta(paired_rows: list[dict[str, Any]]) -> None:
    by_attack = []
    for attack in sorted({r["attack"] for r in paired_rows}):
        sub = [r for r in paired_rows if r["attack"] == attack]
        by_attack.append({
            "attack": attack,
            "delta_acc": avg(sub, "delta_acc"),
            "delta_transfer_asr": avg(sub, "delta_transfer_asr"),
            "delta_joint_transfer": avg(sub, "delta_joint_transfer"),
        })
    by_attack.sort(key=lambda r: r["delta_transfer_asr"])
    labels = [r["attack"] for r in by_attack]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.25
    ax.bar(x - width, [r["delta_acc"] * 100 for r in by_attack], width, label="delta_ACC")
    ax.bar(x, [r["delta_transfer_asr"] * 100 for r in by_attack], width, label="delta_transfer_ASR")
    ax.bar(x + width, [r["delta_joint_transfer"] * 100 for r in by_attack], width, label="delta_joint_transfer")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Qwen - ImageNetV2 difference (percentage points)")
    ax.set_title("Attack-level ACC, transfer_ASR, and joint_transfer differences")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attack_level_acc_transfer_joint_delta.png", dpi=180)
    plt.close(fig)


def build_summaries(long_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary: dict[str, Any] = {
        "n_long": len(long_rows),
        "n_paired": len(paired_rows),
        "overall": summarize_scope(long_rows, "overall"),
        "by_domain": {},
        "by_attack": {},
        "by_arch": {},
        "regression_r2": {},
    }
    for domain in sorted({r["domain"] for r in long_rows}):
        summary["by_domain"][domain] = summarize_scope([r for r in long_rows if r["domain"] == domain], domain)
    for attack in sorted({r["attack"] for r in long_rows}):
        summary["by_attack"][attack] = summarize_scope([r for r in long_rows if r["attack"] == attack], attack)
    for arch in sorted({r["arch"] for r in long_rows}):
        summary["by_arch"][arch] = summarize_scope([r for r in long_rows if r["arch"] == arch], arch)

    attacks = sorted({r["attack"] for r in long_rows})[1:]
    archs = sorted({r["arch"] for r in long_rows})[1:]
    domains = sorted({r["domain"] for r in long_rows})[1:]
    base_features = ["transfer_asr"]
    acc_features = ["transfer_asr", "target_acc"]
    fe_features = ["transfer_asr", "target_acc"] + [f"attack={a}" for a in attacks] + [f"arch={a}" for a in archs] + [f"domain={d}" for d in domains]
    summary["regression_r2"]["stealth_transfer_only"] = ols_r2(long_rows, "stealth_auc", base_features)
    summary["regression_r2"]["stealth_transfer_plus_acc"] = ols_r2(long_rows, "stealth_auc", acc_features)
    summary["regression_r2"]["stealth_transfer_acc_fixed_effects"] = ols_r2(long_rows, "stealth_auc", fe_features)
    summary["regression_r2"]["stealth_joint_only"] = ols_r2(long_rows, "stealth_auc", ["joint_transfer"])
    summary["regression_r2"]["stealth_joint_plus_acc"] = ols_r2(long_rows, "stealth_auc", ["joint_transfer", "target_acc"])

    binned_rows: list[dict[str, Any]] = []
    binned_rows.extend(binned_summary(long_rows, "overall"))
    for domain in sorted({r["domain"] for r in long_rows}):
        binned_rows.extend(binned_summary([r for r in long_rows if r["domain"] == domain], domain))
    for attack in ["WaNet", "basic"]:
        binned_rows.extend(binned_summary([r for r in long_rows if r["attack"] == attack], attack))

    thresholds = [i / 100 for i in range(20, 66, 5)]
    scan_rows: list[dict[str, Any]] = []
    scopes = {
        "overall": long_rows,
        "ImageNetV2": [r for r in long_rows if r["domain"] == "ImageNetV2"],
        "Qwen": [r for r in long_rows if r["domain"] == "Qwen"],
        "WaNet": [r for r in long_rows if r["attack"] == "WaNet"],
        "basic": [r for r in long_rows if r["attack"] == "basic"],
    }
    for scope, rows in scopes.items():
        scan_rows.extend(scan_acc_regions(rows, scope, thresholds))

    summary["paired_delta"] = {
        "mean_delta_acc": avg(paired_rows, "delta_acc"),
        "mean_delta_transfer_asr": avg(paired_rows, "delta_transfer_asr"),
        "mean_delta_joint_transfer": avg(paired_rows, "delta_joint_transfer"),
        "delta_acc_vs_delta_transfer": corr_pair(paired_rows, "delta_acc", "delta_transfer_asr"),
        "delta_acc_vs_delta_joint": corr_pair(paired_rows, "delta_acc", "delta_joint_transfer"),
    }
    return summary, binned_rows, scan_rows


def write_report(summary: dict[str, Any], binned_rows: list[dict[str, Any]], scan_rows: list[dict[str, Any]]) -> None:
    overall = summary["overall"]
    iv2 = summary["by_domain"]["ImageNetV2"]
    qwen = summary["by_domain"]["Qwen"]
    delta = summary["paired_delta"]
    wanet = summary["by_attack"]["WaNet"]
    basic = summary["by_attack"]["basic"]

    attack_lines = []
    for attack, item in summary["by_attack"].items():
        attack_lines.append(
            f"| {attack} | {item['n']} | {fmt_pct(item['mean_target_acc'])} | "
            f"{fmt_pct(item['mean_transfer_asr'])} | {fmt_pct(item['mean_joint_transfer'])} | "
            f"{fmt_pct(item['mean_detection_auc'])} | {fmt_pct(item['mean_stealth_auc'])} | "
            f"{fmt_num(item['acc_transfer_corr']['pearson'])} | {fmt_num(item['acc_stealth_corr']['pearson'])} |"
        )

    binned_lines = []
    for row in binned_rows:
        if row["scope"] in {"overall", "WaNet", "basic"}:
            binned_lines.append(
                f"| {row['scope']} | {row['bin']} | {row['n']} | "
                f"{fmt_pct(row['acc_min'])}-{fmt_pct(row['acc_max'])} | "
                f"{fmt_pct(row['mean_transfer_asr'])} | {fmt_pct(row['mean_joint_transfer'])} | "
                f"{fmt_pct(row['mean_detection_auc'])} | {fmt_pct(row['mean_stealth_auc'])} |"
            )

    def best_scan(scope: str) -> dict[str, Any] | None:
        candidates = [r for r in scan_rows if r["scope"] == scope]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r["caution_score"])

    global_scan = best_scan("overall")
    wanet_scan = best_scan("WaNet")
    basic_scan = best_scan("basic")

    md = f"""# ACC、联合迁移性与隐蔽性总分析报告

本报告基于 `poisoned_train_set/tiny_imagenet` 下同一批 465 个配置，在 ImageNetV2 与 Qwen 两个目标域上形成 930 条目标域记录。报告保留原始 `target-domain ASR` 迁移性口径，同时新增 `joint_transfer = source_ASR * transfer_ASR / (source_ASR + transfer_ASR)`，用于判断 ACC 是否影响原有迁移性和隐蔽性的平衡。

## 1. 指标含义

- `source_ASR`：源域后门攻击成功率。它回答“攻击是否在源域建立成功”。
- `transfer_ASR`：目标域 ASR，也就是原始 target-domain ASR。它回答“加触发器后目标域样本有多少被打到目标类”。
- `target_ACC`：目标域 clean accuracy。它回答“模型在目标域本身是否可靠”。这是本报告的核心解释变量。
- `joint_transfer`：`source_ASR * transfer_ASR / (source_ASR + transfer_ASR)`。它同时惩罚低 source_ASR 和低 transfer_ASR，避免把源域失败或目标域偶然偏置误解成稳定迁移。
- `joint_transfer_norm`：`2 * joint_transfer`，是归一化版本，最大值为 1，仅用于可视化辅助。
- `detection_AUC`：STRIP、SentiNet、SCaLe-Up、IBD-PSC 四个防御 AUC 均值，越高越容易检测。
- `stealth_AUC = 1 - detection_AUC`：越高越隐蔽。

统计量解释：

- `Pearson` 衡量线性相关；越接近 +1 表示线性同升，越接近 -1 表示线性反向。
- `Spearman` 衡量秩相关；适合非线性但单调的趋势。
- `partial correlation` 表示控制某个变量后的相关性。本报告用它看“控制 ACC 后，迁移性与检测强度是否仍相关”。
- 分组/分段/扫描不是为了强造阈值，而是为了判断 ACC 影响是连续的、分段的，还是由攻击类型交互造成的。

## 2. 总体结果

| 范围 | n | mean ACC | mean transfer_ASR | mean joint_transfer | mean detection_AUC | mean stealth_AUC |
|---|---:|---:|---:|---:|---:|---:|
| ImageNetV2 | {iv2['n']} | {fmt_pct(iv2['mean_target_acc'])} | {fmt_pct(iv2['mean_transfer_asr'])} | {fmt_pct(iv2['mean_joint_transfer'])} | {fmt_pct(iv2['mean_detection_auc'])} | {fmt_pct(iv2['mean_stealth_auc'])} |
| Qwen | {qwen['n']} | {fmt_pct(qwen['mean_target_acc'])} | {fmt_pct(qwen['mean_transfer_asr'])} | {fmt_pct(qwen['mean_joint_transfer'])} | {fmt_pct(qwen['mean_detection_auc'])} | {fmt_pct(qwen['mean_stealth_auc'])} |

同一批模型从 ImageNetV2 切到 Qwen 后：

- `delta_ACC` = {fmt_pp(delta['mean_delta_acc'])}
- `delta_transfer_ASR` = {fmt_pp(delta['mean_delta_transfer_asr'])}
- `delta_joint_transfer` = {fmt_pp(delta['mean_delta_joint_transfer'])}
- `delta_ACC` vs `delta_transfer_ASR` Pearson = {fmt_num(delta['delta_acc_vs_delta_transfer']['pearson'])}
- `delta_ACC` vs `delta_joint_transfer` Pearson = {fmt_num(delta['delta_acc_vs_delta_joint']['pearson'])}

解释：Qwen 的 ACC 明显更高，但原始 transfer_ASR 和 joint_transfer 都下降，说明低 ACC 目标域确实会放大迁移性的表观值。joint_transfer 下降也说明这种现象不只是 source_ASR 低导致的，而是目标域 ASR 本身受 ACC/域统计影响。

## 3. ACC 如何影响迁移性与隐蔽性

| 关系 | Pearson | Spearman | 解释 |
|---|---:|---:|---|
| ACC vs transfer_ASR | {fmt_num(overall['acc_transfer_corr']['pearson'])} | {fmt_num(overall['acc_transfer_corr']['spearman'])} | 负相关表示 ACC 越低，原始目标域 ASR 越容易偏高 |
| ACC vs joint_transfer | {fmt_num(overall['acc_joint_corr']['pearson'])} | {fmt_num(overall['acc_joint_corr']['spearman'])} | 若仍为负，说明联合迁移性也受到 ACC 影响 |
| ACC vs stealth_AUC | {fmt_num(overall['acc_stealth_corr']['pearson'])} | {fmt_num(overall['acc_stealth_corr']['spearman'])} | 接近 0 表示 ACC 不直接决定隐蔽性 |
| transfer_ASR vs stealth_AUC | {fmt_num(overall['transfer_stealth_corr']['pearson'])} | {fmt_num(overall['transfer_stealth_corr']['spearman'])} | 负相关支持原始 trade-off |
| joint_transfer vs stealth_AUC | {fmt_num(overall['joint_stealth_corr']['pearson'])} | {fmt_num(overall['joint_stealth_corr']['spearman'])} | 新迁移性口径下的 trade-off |

连续斜率解释：

- ACC 每上升 10pp，`transfer_ASR` 平均变化 {fmt_pp(overall['acc_to_transfer_slope_per_10pp'])}。
- ACC 每上升 10pp，`joint_transfer` 平均变化 {fmt_pp(overall['acc_to_joint_slope_per_10pp'])}。
- ACC 每上升 10pp，`stealth_AUC` 平均变化 {fmt_pp(overall['acc_to_stealth_slope_per_10pp'])}。

这说明 ACC 主要改变迁移性解释，而对隐蔽性的直接影响远弱于对 ASR 的影响。

## 4. 控制 ACC 后，trade-off 是否仍存在

| 迁移性口径 | 与 detection_AUC 的 Pearson | 控制 ACC 后 |
|---|---:|---:|
| transfer_ASR | {fmt_num(overall['transfer_detection_corr']['pearson'])} | {fmt_num(overall['partial_transfer_detection_control_acc'])} |
| joint_transfer | {fmt_num(overall['joint_detection_corr']['pearson'])} | {fmt_num(overall['partial_joint_detection_control_acc'])} |

如果控制 ACC 后相关仍然明显为正，说明“迁移性越强越容易检测”的 trade-off 并不是 ACC 低造成的伪相关。若 joint_transfer 的相关弱于 transfer_ASR，则说明原始 ASR 口径中确实有部分由 target_ACC 或 source_ASR 混杂放大。

## 5. 攻击类型交互

| 攻击 | n | mean ACC | transfer_ASR | joint_transfer | detection_AUC | stealth_AUC | ACC-transfer r | ACC-stealth r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(attack_lines)}

结论：

- `WaNet` 是低 ACC、高 transfer_ASR 的代表，最适合说明 ACC 会干扰迁移性解释；同时它也显示 ACC 与攻击机制会发生交互，低 ACC 往往伴随更强形变触发、更高 detection、更低 stealth。
- `basic` 是原 trade-off 正例，通常表现为高 transfer_ASR、高 detection_AUC、低 stealth_AUC。
- `SIG` 和 `upgd` 更适合说明目标域统计与 ACC 会显著影响迁移性，因为它们在 Qwen 上 ASR 下降更明显。
- 因此 ACC 影响不是全局单一规律，而是强烈依赖攻击机制。

## 6. WaNet pilot 与 basic 对照

| 攻击 | n | mean ACC | transfer_ASR | joint_transfer | detection_AUC | stealth_AUC | ACC-transfer r | ACC-stealth r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WaNet | {wanet['n']} | {fmt_pct(wanet['mean_target_acc'])} | {fmt_pct(wanet['mean_transfer_asr'])} | {fmt_pct(wanet['mean_joint_transfer'])} | {fmt_pct(wanet['mean_detection_auc'])} | {fmt_pct(wanet['mean_stealth_auc'])} | {fmt_num(wanet['acc_transfer_corr']['pearson'])} | {fmt_num(wanet['acc_stealth_corr']['pearson'])} |
| basic | {basic['n']} | {fmt_pct(basic['mean_target_acc'])} | {fmt_pct(basic['mean_transfer_asr'])} | {fmt_pct(basic['mean_joint_transfer'])} | {fmt_pct(basic['mean_detection_auc'])} | {fmt_pct(basic['mean_stealth_auc'])} | {fmt_num(basic['acc_transfer_corr']['pearson'])} | {fmt_num(basic['acc_stealth_corr']['pearson'])} |

WaNet 的解释重点：它不是简单“迁移强所以隐蔽性差”，而是“空间形变同时降低 clean ACC 并保持触发能力”。在 WaNet 内部，ACC 与 stealth 的相关也很强，这更像是攻击强度/形变机制的交互效应，而不是目标域 ACC 对隐蔽性的普遍直接因果。因此，WaNet 中的高 target-domain ASR 和 stealth 变化都必须结合 ACC 与攻击机制解释。

basic 的解释重点：它是原 trade-off 的正例。其高迁移通常伴随高 detection_AUC 和低 stealth_AUC，说明原 trade-off 仍有机制基础。

## 7. 分组与影响形态

| scope | bin | n | ACC range | transfer_ASR | joint_transfer | detection_AUC | stealth_AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(binned_lines)}

如果低 ACC 组中 `transfer_ASR` 明显高于高 ACC 组，而总体 `stealth_AUC` 变化很小，就支持“ACC 主要影响迁移性解释”。如果某个攻击（例如 WaNet）中 `stealth_AUC` 也随 ACC 明显变化，则应写成“ACC 与攻击机制交互影响 stealth”，而不是全局 ACC 直接决定隐蔽性。

## 8. 谨慎区间探索

阈值扫描不是为了强行找硬阈值，而是辅助判断是否存在 ACC 低区间。当前最值得关注的候选：

- 全局候选：{fmt_pct(global_scan['threshold']) if global_scan else 'NA'}
- WaNet 候选：{fmt_pct(wanet_scan['threshold']) if wanet_scan else 'NA'}
- basic 候选：{fmt_pct(basic_scan['threshold']) if basic_scan else 'NA'}

如果这些候选在图中形成稳定平台，可写成“ACC 低于某一区间时需谨慎解释”；如果不同攻击给出不同候选，结论应写成“攻击机制调节 ACC 影响，不存在统一阈值”。

## 9. 回归解释力

| 模型 | R2 |
|---|---:|
| `stealth_AUC ~ transfer_ASR` | {summary['regression_r2']['stealth_transfer_only']:.3f} |
| `stealth_AUC ~ transfer_ASR + target_ACC` | {summary['regression_r2']['stealth_transfer_plus_acc']:.3f} |
| `stealth_AUC ~ transfer_ASR + target_ACC + attack + arch + domain` | {summary['regression_r2']['stealth_transfer_acc_fixed_effects']:.3f} |
| `stealth_AUC ~ joint_transfer` | {summary['regression_r2']['stealth_joint_only']:.3f} |
| `stealth_AUC ~ joint_transfer + target_ACC` | {summary['regression_r2']['stealth_joint_plus_acc']:.3f} |

若加入 `target_ACC` 后 R2 提升很小，而加入攻击/架构固定效应后 R2 提升明显，说明隐蔽性主要由攻击机制决定，而不是由 ACC 直接决定。

## 10. 最终结论

1. `target-domain ASR` 应继续保留，因为它与已有结论和实验流程一致，但在低 ACC 目标域上不能单独解释为真实迁移能力。
2. `joint_transfer` 是必要补充，因为它同时要求源域攻击成功和目标域迁移成功，可以减弱源域失败或目标域偶然偏置造成的误读。
3. ACC 对迁移性有实质影响：ACC 降低通常会抬高原始 target-domain ASR 的表观值，且这种影响可能表现为连续趋势、低 ACC 谨慎区间或攻击机制交互。
4. ACC 对隐蔽性的全局直接影响很弱；隐蔽性主要由攻击类型、触发器形态和防御机制决定。但在 WaNet 等攻击内部，ACC 可能通过攻击强度/形变机制与 stealth 发生交互。
5. 原始迁移性-隐蔽性 trade-off 不应被推翻，但需要修正为：高迁移通常更容易检测，但在低 ACC 或 WaNet 等特殊攻击下，target-domain ASR 必须结合 ACC 与 joint_transfer 一起解释。

## 11. 论文可用表述

在 Tiny-ImageNet 的目标域迁移实验中，target-domain ASR 受到目标域 clean accuracy 的影响。较低的 target ACC 会放大 ASR 的表观值，使部分配置看起来具有更强迁移性。为避免将分类不可靠性误解释为真实迁移能力，我们在保留 target-domain ASR 的同时，引入 `joint_transfer = source_ASR * transfer_ASR / (source_ASR + transfer_ASR)` 作为补充指标。进一步分析显示，ACC 对迁移性有明显影响，但对四防御平均 stealth AUC 的全局直接影响较弱；隐蔽性主要由攻击机制决定。不过在 WaNet 等攻击内部，ACC 会与攻击机制发生交互，从而同时影响 ASR 和 stealth。因此，迁移性与隐蔽性之间的 trade-off 仍然存在，但在低 ACC 或特定攻击机制下需要结合 ACC、joint_transfer 和攻击类型进行解释。
"""
    (ANALYSIS_DIR / "final_acc_joint_transfer_tradeoff_report.md").write_text(md, encoding="utf-8")


def write_outputs(long_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], summary: dict[str, Any], binned_rows: list[dict[str, Any]], scan_rows: list[dict[str, Any]]) -> None:
    ANALYSIS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    write_csv(ANALYSIS_DIR / "acc_joint_transfer_rows.csv", long_rows)
    write_csv(ANALYSIS_DIR / "acc_joint_transfer_paired_rows.csv", paired_rows)
    write_csv(ANALYSIS_DIR / "acc_effect_shape_summary.csv", binned_rows)
    write_csv(ANALYSIS_DIR / "acc_caution_region_scan.csv", scan_rows)
    (ANALYSIS_DIR / "acc_joint_transfer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_all(long_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], binned_rows: list[dict[str, Any]], scan_rows: list[dict[str, Any]]) -> None:
    plot_scatter_by_domain(long_rows, "transfer_asr", "transfer_ASR", "acc_vs_transfer_asr.png")
    plot_scatter_by_domain(long_rows, "joint_transfer_norm", "joint_transfer_norm", "acc_vs_joint_transfer.png")
    plot_tradeoff(long_rows, "transfer_asr", "transfer_ASR", "transfer_asr_vs_stealth_auc.png")
    plot_tradeoff(long_rows, "joint_transfer_norm", "joint_transfer_norm", "joint_transfer_vs_stealth_auc.png")
    plot_binned(binned_rows, "overall", "acc_effect_shape_global.png")
    plot_binned(binned_rows, "WaNet", "acc_effect_shape_wanet.png")
    plot_binned(binned_rows, "basic", "acc_effect_shape_basic.png")
    plot_scan(scan_rows, "overall", "acc_caution_region_scan_global.png")
    plot_attack_delta(paired_rows)


def main() -> None:
    long_rows, paired_rows = collect_rows()
    if len(paired_rows) != 465:
        print(f"[WARN] expected 465 paired rows, got {len(paired_rows)}")
    summary, binned_rows, scan_rows = build_summaries(long_rows, paired_rows)
    write_outputs(long_rows, paired_rows, summary, binned_rows, scan_rows)
    plot_all(long_rows, paired_rows, binned_rows, scan_rows)
    write_report(summary, binned_rows, scan_rows)
    print(f"paired rows: {len(paired_rows)}")
    print(f"long rows: {len(long_rows)}")
    print(f"report: {ANALYSIS_DIR / 'final_acc_joint_transfer_tradeoff_report.md'}")
    print(f"figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
