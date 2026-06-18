#!/usr/bin/env python3
"""Statistical helpers for the analysis pipeline."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from config import BOOTSTRAP_N, MIN_GROUP_N, RANDOM_SEED


STRICT_CONFIG_BASE_KEYS = [
    "dataset",
    "transfer_dataset",
    "transfer_variant",
    "attack_type",
    "poison_rate",
    "strength_name",
    "strength_value",
    "cover_rate",
    "label_mode",
]


def strict_config_key_columns(df: pd.DataFrame) -> List[str]:
    keys = [col for col in STRICT_CONFIG_BASE_KEYS if col in df.columns]
    if "mask_rate" in df.columns:
        keys.append("mask_rate")
    return keys


def _stat_value(result) -> float:
    if hasattr(result, "statistic"):
        return result.statistic
    if hasattr(result, "correlation"):
        return result.correlation
    return result[0]


def _p_value(result) -> float:
    if hasattr(result, "pvalue"):
        return result.pvalue
    return result[1]


def _matched_resnet18_row_ids(df: pd.DataFrame, row_mask: pd.Series, *, dataset: str | None = None, transfer_dataset: str | None = None) -> set:
    if "result_group" not in df.columns or "arch_base" not in df.columns:
        return set()
    baseline_mask = (df["result_group"] == "baseline_strength") & (df["arch_base"] == "ResNet18")
    if dataset is not None and "dataset" in df.columns:
        baseline_mask &= df["dataset"] == dataset
    if transfer_dataset is not None and "transfer_dataset" in df.columns:
        baseline_mask &= df["transfer_dataset"] == transfer_dataset

    baseline = df[baseline_mask].copy()
    rows = df[row_mask].copy()
    keys = strict_config_key_columns(df)
    missing = [c for c in keys if c not in baseline.columns or c not in rows.columns]
    if baseline.empty or rows.empty or missing:
        return set()

    rows["_strict_row_id"] = rows.index
    matched = rows.merge(baseline[keys].drop_duplicates(), on=keys, how="inner")
    return set(matched["_strict_row_id"])


def _filter_unmatched_noise_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "result_group" not in df.columns:
        return df
    input_noise = df["input_noise_type"] if "input_noise_type" in df.columns else pd.Series("", index=df.index)
    noise_mask = (df["result_group"] == "noise_acc") & (input_noise.astype(str) != "")
    if not noise_mask.any():
        return df

    matched_ids = _matched_resnet18_row_ids(df, noise_mask, dataset="cifar10", transfer_dataset="stl10")
    keep_mask = ~noise_mask | df.index.isin(matched_ids)
    return df[keep_mask].copy()


def _filter_unmatched_arch_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "result_group" not in df.columns:
        return df
    arch_mask = df["result_group"] == "arch_acc"
    if not arch_mask.any():
        return df
    matched_ids = _matched_resnet18_row_ids(df, arch_mask)
    keep_mask = ~arch_mask | df.index.isin(matched_ids)
    return df[keep_mask].copy()


def analysis_df(
    df: pd.DataFrame,
    *,
    main_transfer_only: bool = True,
    include_unmatched_noise: bool = False,
    include_unmatched_arch: bool = False,
) -> pd.DataFrame:
    flag_col = "include_main_analysis" if main_transfer_only else "complete_analysis_row"
    if flag_col not in df.columns:
        out = df.copy()
    else:
        out = df[df[flag_col].astype(bool)].copy()
    if not include_unmatched_noise:
        out = _filter_unmatched_noise_rows(out)
    if not include_unmatched_arch:
        out = _filter_unmatched_arch_rows(out)
    return out


def main_df(df: pd.DataFrame, *, include_unmatched_noise: bool = False, include_unmatched_arch: bool = False) -> pd.DataFrame:
    return analysis_df(
        df,
        main_transfer_only=True,
        include_unmatched_noise=include_unmatched_noise,
        include_unmatched_arch=include_unmatched_arch,
    )


def numeric_df(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def correlation_record(df: pd.DataFrame, group_name: str, x: str = "transfer_rate", y: str = "stealthiness") -> Dict[str, object]:
    sub = numeric_df(df, [x, y, "source_asr", "transfer_asr", "clean_acc"]).dropna(subset=[x, y])
    rec: Dict[str, object] = {
        "group": group_name,
        "n": int(len(sub)),
        "pearson": np.nan,
        "pearson_pvalue": np.nan,
        "spearman": np.nan,
        "spearman_pvalue": np.nan,
        "kendall": np.nan,
        "kendall_pvalue": np.nan,
        "transfer_rate_median": sub[x].median() if len(sub) else np.nan,
        "stealthiness_median": sub[y].median() if len(sub) else np.nan,
        "source_asr_median": sub["source_asr"].median() if "source_asr" in sub and len(sub) else np.nan,
    }
    if len(sub) >= MIN_GROUP_N and sub[x].nunique(dropna=True) > 1 and sub[y].nunique(dropna=True) > 1:
        pearson = stats.pearsonr(sub[x], sub[y])
        spearman = stats.spearmanr(sub[x], sub[y])
        kendall = stats.kendalltau(sub[x], sub[y])
        rec.update(
            {
                "pearson": _stat_value(pearson),
                "pearson_pvalue": _p_value(pearson),
                "spearman": _stat_value(spearman),
                "spearman_pvalue": _p_value(spearman),
                "kendall": _stat_value(kendall),
                "kendall_pvalue": _p_value(kendall),
            }
        )
    return rec


def grouped_correlations(df: pd.DataFrame, group_cols: Sequence[str], x: str = "transfer_rate", y: str = "stealthiness") -> pd.DataFrame:
    if not group_cols:
        return pd.DataFrame([correlation_record(df, "overall", x=x, y=y)])
    records: List[Dict[str, object]] = []
    for keys, sub in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = correlation_record(sub, "|".join(str(k) for k in keys), x=x, y=y)
        for col, val in zip(group_cols, keys):
            rec[col] = val
        records.append(rec)
    return pd.DataFrame(records)


def ols_records(df: pd.DataFrame, formula_name: str, y_col: str, x_cols: Sequence[str], categorical_cols: Sequence[str] = ()) -> pd.DataFrame:
    cols = [y_col] + list(x_cols) + list(categorical_cols)
    sub = df[cols].copy()
    for col in [y_col] + list(x_cols):
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=[y_col] + list(x_cols))
    if len(sub) < len(x_cols) + 3:
        return pd.DataFrame(
            [{"formula": formula_name, "term": "", "coef": np.nan, "std_err": np.nan, "p_value": np.nan, "r2": np.nan, "n": len(sub)}]
        )

    x = sub[list(x_cols)].copy()
    if categorical_cols:
        dummies = pd.get_dummies(sub[list(categorical_cols)].astype(str), drop_first=True, dtype=float)
        x = pd.concat([x, dummies], axis=1)
    x.insert(0, "intercept", 1.0)
    y = sub[y_col].astype(float).to_numpy()
    x_mat = x.astype(float).to_numpy()

    try:
        beta, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
        pred = x_mat @ beta
        resid = y - pred
        dof = max(len(y) - x_mat.shape[1], 1)
        sigma2 = float((resid @ resid) / dof)
        cov = sigma2 * np.linalg.pinv(x_mat.T @ x_mat)
        stderr = np.sqrt(np.diag(cov))
        tvals = beta / stderr
        pvals = 2 * stats.t.sf(np.abs(tvals), df=dof)
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    except Exception:
        return pd.DataFrame(
            [{"formula": formula_name, "term": "", "coef": np.nan, "std_err": np.nan, "p_value": np.nan, "r2": np.nan, "n": len(sub)}]
        )

    records = []
    for term, coef, se, pval in zip(x.columns, beta, stderr, pvals):
        records.append(
            {
                "formula": formula_name,
                "term": term,
                "coef": coef,
                "std_err": se,
                "p_value": pval,
                "r2": r2,
                "n": len(sub),
            }
        )
    return pd.DataFrame(records)


def add_acc_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["clean_acc"] = pd.to_numeric(out["clean_acc"], errors="coerce")
    valid = out["clean_acc"].dropna()
    if len(valid) < 3 or valid.nunique() < 3:
        out["acc_bin"] = "unknown"
        return out
    try:
        out["acc_bin"] = pd.qcut(out["clean_acc"], q=3, labels=["low_acc", "mid_acc", "high_acc"], duplicates="drop")
        out["acc_bin"] = out["acc_bin"].astype(str).replace("nan", "unknown")
    except Exception:
        out["acc_bin"] = "unknown"
    return out


def pairwise_adjacent_delta(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    order_col: str,
    label_col: str,
    metrics: Sequence[str] = ("clean_acc", "source_asr", "transfer_asr", "transfer_rate", "stealthiness"),
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    sub = numeric_df(df, [order_col] + list(metrics)).dropna(subset=[order_col])
    for keys, group in sub.groupby(list(group_cols), dropna=False):
        group = group.sort_values(order_col)
        if len(group) < 2:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        for (_, prev), (_, curr) in zip(group.iloc[:-1].iterrows(), group.iloc[1:].iterrows()):
            rec = {col: val for col, val in zip(group_cols, keys)}
            rec["comparison_group"] = label_col
            rec["from_value"] = prev[order_col]
            rec["to_value"] = curr[order_col]
            for metric in metrics:
                rec[f"delta_{metric}"] = curr[metric] - prev[metric]
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize_delta(delta_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    delta_cols = [c for c in delta_df.columns if c.startswith("delta_")]
    records: List[Dict[str, object]] = []
    for keys, sub in delta_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: val for col, val in zip(group_cols, keys)}
        rec["n_pairs"] = len(sub)
        for col in delta_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            rec[f"{col}_mean"] = vals.mean() if len(vals) else np.nan
            rec[f"{col}_median"] = vals.median() if len(vals) else np.nan
            if len(vals) >= 3:
                try:
                    rec[f"{col}_wilcoxon_p"] = stats.wilcoxon(vals).pvalue
                except Exception:
                    rec[f"{col}_wilcoxon_p"] = np.nan
            else:
                rec[f"{col}_wilcoxon_p"] = np.nan
        records.append(rec)
    return pd.DataFrame(records)


def bootstrap_ci(values: Sequence[float], statistic: str = "mean", n_boot: int = BOOTSTRAP_N) -> Dict[str, float]:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(vals) < 2:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": len(vals), "n_boot": n_boot}
    rng = np.random.default_rng(RANDOM_SEED)
    samples = []
    for _ in range(n_boot):
        draw = rng.choice(vals, size=len(vals), replace=True)
        samples.append(np.median(draw) if statistic == "median" else np.mean(draw))
    estimate = np.median(vals) if statistic == "median" else np.mean(vals)
    return {
        "estimate": float(estimate),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "n": int(len(vals)),
        "n_boot": int(n_boot),
    }


def save_csv_and_md(df: pd.DataFrame, csv_path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    md_path = csv_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("No rows available.\n")
        else:
            f.write(df.head(80).to_markdown(index=False))
            if len(df) > 80:
                f.write(f"\n\nShowing first 80 of {len(df)} rows.\n")
