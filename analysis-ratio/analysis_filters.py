"""分析阶段共用筛选与迁移性指标（提取阶段不过滤，保留全量 CSV）。"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# 分析时仅保留源域 ASR（同域攻击成功率）高于该阈值的配置
MIN_ASR_FOR_ANALYSIS = 0.3


def normalize_asr_value(v: Optional[float]) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    x = float(v)
    if x > 1.5:
        x /= 100.0
    return x


def compute_transfer_metric(test_asr: Optional[float], source_asr: Optional[float]) -> Optional[float]:
    """迁移性 = test_ASR / (test_ASR + ASR)，取值 (0, 1)。"""
    test = normalize_asr_value(test_asr)
    source = normalize_asr_value(source_asr)
    if test is None or source is None:
        return None
    denom = test + source
    if denom <= 0:
        return None
    return test / denom


def filter_for_analysis(
    df: pd.DataFrame,
    min_asr: float = MIN_ASR_FOR_ANALYSIS,
    asr_col: str = "asr",
) -> pd.DataFrame:
    """分析用：仅保留源域 ASR > min_asr 的配置。"""
    if df.empty or asr_col not in df.columns:
        return df.copy()
    out = df.copy()
    asr = pd.to_numeric(out[asr_col], errors="coerce")
    if asr.notna().any() and asr.max() > 1.5:
        asr = asr / 100.0
    return out.loc[asr > min_asr].copy()
