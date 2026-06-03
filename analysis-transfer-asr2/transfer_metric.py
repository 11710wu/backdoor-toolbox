"""迁移性指标：ASRt^2 / ASRs（目标域 ASR 平方 / 源域 ASR）。"""

from __future__ import annotations

from typing import Optional

METRIC_NAME = "transfer_asr2_over_source_asr"
METRIC_FORMULA = "transfer_rate = transfer_asr^2 / asr"
METRIC_LABEL = "Transfer (ASRt²/ASRs)"


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
