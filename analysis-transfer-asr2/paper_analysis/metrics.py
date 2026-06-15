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
    """transfer_rate = transfer_asr^2 / source_asr."""
    transfer = normalize_rate(transfer_asr)
    source = normalize_rate(source_asr)
    if math.isnan(transfer) or math.isnan(source) or source <= 0:
        return float("nan")
    return (transfer * transfer) / source


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
