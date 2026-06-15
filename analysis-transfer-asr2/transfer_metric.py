"""迁移性指标：ASRt^2 / ASRs（目标域 ASR 平方 / 源域 ASR）。"""

from __future__ import annotations

import math
from typing import Optional

METRIC_NAME = "transfer_asr2_over_source_asr"
METRIC_FORMULA = "transfer_rate = transfer_asr^2 / asr"
METRIC_LABEL = "Transfer (ASRt²/ASRs)"

SIGMOID_LINEAR_K = 11.0
LOGSIGMOID_K = 11.5
TRANSFER_EPS = 1e-6


def compute_transfer_rate(transfer_asr: Optional[float], source_asr: Optional[float]) -> Optional[float]:
    """ASRt^2 / ASRs；源域 ASR 无效时返回 None。"""
    if transfer_asr is None or source_asr is None:
        return None
    try:
        t = float(transfer_asr)
        s = float(source_asr)
    except (TypeError, ValueError):
        return None
    if s <= 0.0:
        return None
    return (t * t) / s


def sigmoid(value: float) -> float:
    """Numerically stable sigmoid."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def compute_transfer_sigmoid_score(transfer_rate: Optional[float], k: float = SIGMOID_LINEAR_K) -> Optional[float]:
    """Centered sigmoid score: sigmoid(k * (transfer_rate - 1)).

    transfer_rate=1 maps to 0.5; values below/above 1 map below/above 0.5.
    """
    if transfer_rate is None:
        return None
    try:
        r = float(transfer_rate)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(r):
        return None
    return sigmoid(k * (r - 1.0))


def compute_transfer_logsigmoid_score(
    transfer_rate: Optional[float],
    k: float = LOGSIGMOID_K,
    eps: float = TRANSFER_EPS,
) -> Optional[float]:
    """Log-ratio sigmoid score: sigmoid(k * log(transfer_rate)).

    This is ratio-symmetric around 1: r and 1/r are equally far from the center.
    """
    if transfer_rate is None:
        return None
    try:
        r = float(transfer_rate)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(r):
        return None
    r = max(r, eps)
    return sigmoid(k * math.log(r))


def score_to_contrast(score: Optional[float]) -> Optional[float]:
    """Map a [0, 1] sigmoid score to [-1, 1], centered at 0."""
    if score is None:
        return None
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(s):
        return None
    return 2.0 * s - 1.0
