#!/usr/bin/env python3
"""Metric definitions used by every paper-analysis script."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


def normalize_rate(value: object) -> float:
    """Normalize a scalar rate to [0, 1], accepting percent-style values."""
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(out):
        return float("nan")
    if out > 1.0:
        out /= 100.0
    return out


def compute_transfer_rate(transfer_asr: object, source_asr: object) -> float:
    """Legacy transfer score: transfer_asr^2 / source_asr."""
    transfer = normalize_rate(transfer_asr)
    source = normalize_rate(source_asr)
    if math.isnan(transfer) or math.isnan(source) or source <= 0:
        return float("nan")
    return (transfer * transfer) / source


def compute_transferability(transfer_asr: object) -> float:
    """Main transferability definition: target-domain ASR."""
    return normalize_rate(transfer_asr)


def chance_rate_for_dataset(dataset: object, transfer_dataset: object = "") -> float:
    """Return the random target-class rate for chance-adjusted ASR."""
    dataset_s = str(dataset)
    transfer_s = str(transfer_dataset)
    if dataset_s == "tiny_imagenet" or "tiny" in transfer_s or "qwen" in transfer_s or "imagenetv2" in transfer_s:
        return 1.0 / 200.0
    if dataset_s in {"cifar10", "mnistm"} or transfer_s in {"stl10", "mnist_cross"}:
        return 1.0 / 10.0
    return 0.0


def compute_chance_adjusted_rate(value: object, chance_rate: float) -> float:
    val = normalize_rate(value)
    if math.isnan(val):
        return float("nan")
    if chance_rate >= 1.0:
        return float("nan")
    return max(0.0, (val - chance_rate) / (1.0 - chance_rate))


def compute_transfer_retention_rate(transfer_asr: object, source_asr: object) -> float:
    transfer = normalize_rate(transfer_asr)
    source = normalize_rate(source_asr)
    if math.isnan(transfer) or math.isnan(source) or source <= 0:
        return float("nan")
    return transfer / source


def compute_transfer_gap(transfer_asr: object, source_asr: object) -> float:
    transfer = normalize_rate(transfer_asr)
    source = normalize_rate(source_asr)
    if math.isnan(transfer) or math.isnan(source):
        return float("nan")
    return transfer - source


def compute_joint_transfer(source_asr: object, transfer_asr: object, chance_rate: float) -> float:
    source_adj = compute_chance_adjusted_rate(source_asr, chance_rate)
    transfer_adj = compute_chance_adjusted_rate(transfer_asr, chance_rate)
    if math.isnan(source_adj) or math.isnan(transfer_adj):
        return float("nan")
    return math.sqrt(source_adj * transfer_adj)


def compute_difficulty(clean_acc: object) -> float:
    clean = normalize_rate(clean_acc)
    if math.isnan(clean):
        return float("nan")
    return 1.0 - clean


def compute_stealth_from_tprs(tprs: Iterable[object]) -> float:
    vals = [normalize_rate(v) for v in tprs]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan")
    return 1.0 - float(np.mean(vals))


def stealth_component(tpr: object) -> float:
    val = normalize_rate(tpr)
    if math.isnan(val):
        return float("nan")
    return 1.0 - val
