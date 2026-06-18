#!/usr/bin/env python3
"""Build the unified master result table."""

from __future__ import annotations

import pandas as pd

from config import MASTER_COLUMNS, OUTPUT_DIR, RESULT_SOURCES, ensure_output_dirs
from parsing_utils import iter_result_folders, parse_result_folder
from stats_utils import add_acc_bins


def build_master_table() -> pd.DataFrame:
    ensure_output_dirs()
    rows = []
    for source in RESULT_SOURCES:
        root = source["root"]
        result_group = source["result_group"]
        dataset_hint = source.get("dataset")
        for folder, folder_dataset_hint in iter_result_folders(root, dataset_hint=dataset_hint):
            rows.extend(parse_result_folder(folder, result_group=result_group, dataset_hint=folder_dataset_hint))
    df = pd.DataFrame(rows)
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[MASTER_COLUMNS + [c for c in df.columns if c not in MASTER_COLUMNS]]
    df = add_acc_bins(df)
    out = OUTPUT_DIR / "master_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    build_master_table()
