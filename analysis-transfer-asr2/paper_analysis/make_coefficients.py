#!/usr/bin/env python3
"""Additional coefficient products shared across analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import COEFFICIENT_DIR
from stats_utils import bootstrap_ci, main_df, save_csv_and_md


def _bootstrap_spearman(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    d = main_df(df).dropna(subset=["transfer_rate", "stealthiness"]).copy()
    groups = [(("overall",), d)] if not group_cols else list(d.groupby(group_cols, dropna=False))
    rng = np.random.default_rng(2333)
    for keys, sub in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: val for col, val in zip(group_cols or ["group"], keys)}
        rec["metric"] = "spearman_transfer_stealth"
        rec["n"] = len(sub)
        if len(sub) < 5 or sub["transfer_rate"].nunique() < 2 or sub["stealthiness"].nunique() < 2:
            rec.update({"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 500})
        else:
            vals = []
            for _ in range(500):
                sample = sub.sample(n=len(sub), replace=True, random_state=int(rng.integers(0, 1_000_000_000)))
                vals.append(sample["transfer_rate"].corr(sample["stealthiness"], method="spearman"))
            rec.update(
                {
                    "estimate": sub["transfer_rate"].corr(sub["stealthiness"], method="spearman"),
                    "ci_low": float(np.nanpercentile(vals, 2.5)),
                    "ci_high": float(np.nanpercentile(vals, 97.5)),
                    "n_boot": 500,
                }
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def _bootstrap_delta_tables(outputs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for family, key in [
        ("strength", "delta"),
        ("cover_rate", "delta"),
        ("label_mode", "delta"),
        ("architecture", "delta"),
        ("noise", "delta"),
    ]:
        table = outputs.get(family if family != "cover_rate" else "cover", {}).get(key, pd.DataFrame())
        if table is None or table.empty:
            continue
        for col in [c for c in table.columns if c.startswith("delta_")]:
            ci = bootstrap_ci(pd.to_numeric(table[col], errors="coerce"), statistic="mean")
            ci["family"] = family
            ci["metric"] = col
            rows.append(ci)
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, outputs: dict[str, dict]) -> dict[str, pd.DataFrame]:
    corr_overall = _bootstrap_spearman(df, [])
    corr_dataset = _bootstrap_spearman(df, ["dataset"])
    corr_transfer_dataset = _bootstrap_spearman(df, ["dataset", "transfer_dataset"])
    corr_attack = _bootstrap_spearman(df, ["attack_type"])
    corr = pd.concat([corr_overall, corr_dataset, corr_transfer_dataset, corr_attack], ignore_index=True)
    delta = _bootstrap_delta_tables(outputs)
    save_csv_and_md(corr, COEFFICIENT_DIR / "bootstrap_ci_correlations.csv", "Bootstrap CI Correlations")
    save_csv_and_md(delta, COEFFICIENT_DIR / "bootstrap_ci_pairwise_delta.csv", "Bootstrap CI Pairwise Delta")
    return {"bootstrap_correlations": corr, "bootstrap_delta": delta}


if __name__ == "__main__":
    from config import OUTPUT_DIR

    run(pd.read_csv(OUTPUT_DIR / "master_results.csv"), {})
