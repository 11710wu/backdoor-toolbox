#!/usr/bin/env python3
"""Validate the official CTR results and their Table II aggregation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PAPER_ROOT = Path("/workspace/tifs_backdoor_paper_restructure_20260815")
MANIFEST = PAPER_ROOT / "data/primary_baseline_belt_completed.csv"
OFFICIAL = ROOT / "clean_target_rates_all_configurations.csv"
ARCHIVED = ROOT / "archive/clean_target_rates_all_configurations_batch256_legacy_threshold.csv"
TABLE_VALUES = PAPER_ROOT / "tables/table2_reference_results_values.csv"
TABLE_TEX = PAPER_ROOT / "tables/table2_reference_results.tex"
OUTPUT = ROOT / "ctr_validation_summary.json"
TOLERANCE_PP = 0.2

ATTACK_NAMES = {
    "basic": "BadNets",
    "blend": "Blend",
    "adaptive_blend": "Adaptive-Blend",
    "adaptive_patch": "Adaptive-Patch",
    "WaNet": "WaNet",
    "SIG": "SIG",
    "upgd": "UPGD",
    "belt": "BELT",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    missing = frame["strict_config_key"].isna() | frame["strict_config_key"].eq("")
    legacy_row = frame["strict_config_key"].astype(str).str.match(r"^manifest_row_\d{4}$")
    replace = missing | legacy_row
    if replace.any():
        rows = frame.loc[replace]
        assert rows["dataset"].eq("tiny_imagenet").all()
        assert rows["attack_type"].eq("belt").all()
        assert rows["folder_name"].notna().all()
        assert not rows["folder_name"].duplicated().any()
        frame.loc[replace, "strict_config_key"] = (
            "legacy_belt|" + rows["folder_name"].astype(str)
        )
    assert frame["strict_config_key"].notna().all()
    assert not frame["strict_config_key"].duplicated().any()
    return frame


def reference_population(
    manifest: pd.DataFrame, results: pd.DataFrame, require_ok: bool = True
) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest["Attack"] = manifest["attack_type"].map(ATTACK_NAMES)
    selected = []
    for dataset in ("cifar10", "tiny_imagenet"):
        for attack in ATTACK_NAMES.values():
            rate = 0.005 if dataset == "tiny_imagenet" and attack in {"SIG", "UPGD"} else 0.01
            selected.append(
                manifest[
                    manifest["dataset"].eq(dataset)
                    & manifest["Attack"].eq(attack)
                    & manifest["arch_base"].eq("ResNet18")
                    & np.isclose(manifest["poison_rate"], rate)
                    & manifest["source_asr"].ge(0.05)
                ]
            )
    population = pd.concat(selected, ignore_index=True).merge(
        results[["strict_config_key", "source_ctr", "target_ctr", "status"]],
        on="strict_config_key",
        how="left",
        validate="one_to_one",
    )
    assert len(population) == population["strict_config_key"].nunique() == 107
    assert population[["source_ctr", "target_ctr", "status"]].notna().all().all()
    if require_ok:
        assert population["status"].eq("ok").all()
    return population


def table_mismatches(population: pd.DataFrame, table: pd.DataFrame) -> list[dict[str, str]]:
    mismatches = []
    for dataset, prefix in (("cifar10", "C"), ("tiny_imagenet", "T")):
        for attack in ATTACK_NAMES.values():
            group = population[
                population["dataset"].eq(dataset) & population["Attack"].eq(attack)
            ]
            row = table[table["Attack"].eq(attack)].iloc[0]
            for metric, column in (("source_ctr", f"{prefix} CTRs"), ("target_ctr", f"{prefix} CTRt")):
                expected = f"{100.0 * group[metric].median():.1f}"
                actual = str(row[column])
                if actual != expected:
                    mismatches.append(
                        {"dataset": dataset, "attack": attack, "column": column, "actual": actual, "expected": expected}
                    )
    return mismatches


def main() -> None:
    manifest = stable_keys(pd.read_csv(MANIFEST, low_memory=False))
    official = stable_keys(pd.read_csv(OFFICIAL, low_memory=False))
    archived = stable_keys(pd.read_csv(ARCHIVED, low_memory=False))

    assert len(manifest) == manifest["strict_config_key"].nunique() == 889
    assert len(official) == official["strict_config_key"].nunique() == 889
    assert set(manifest["strict_config_key"]) == set(official["strict_config_key"])
    assert official["status"].eq("ok").all()
    assert official["error"].isna().all()
    assert official["target_class"].eq(0).all()
    assert official["resolved_model_path"].nunique() == 889
    assert all(Path(path).is_file() for path in official["resolved_model_path"])
    assert not official["strict_config_key"].str.startswith("manifest_row_").any()
    assert official["strict_config_key"].str.startswith("legacy_belt|").sum() == 52

    required = [
        "source_clean_correct", "source_clean_total", "source_ctr_numerator",
        "source_ctr_denominator", "source_ctr", "target_clean_correct",
        "target_clean_total", "target_ctr_numerator", "target_ctr_denominator",
        "target_ctr", "recomputed_source_clean_acc", "source_clean_acc_diff_pp",
        "recomputed_target_clean_acc", "target_clean_acc_diff_pp", "evaluated_at_utc",
    ]
    assert not official[required].isna().any().any()

    arithmetic_max_error = {}
    for side, recorded in (("source", "recorded_source_clean_acc"), ("target", "recorded_target_clean_acc")):
        numerator = official[f"{side}_ctr_numerator"].astype(int)
        denominator = official[f"{side}_ctr_denominator"].astype(int)
        ctr = official[f"{side}_ctr"].astype(float)
        correct = official[f"{side}_clean_correct"].astype(int)
        total = official[f"{side}_clean_total"].astype(int)
        accuracy = official[f"recomputed_{side}_clean_acc"].astype(float)
        difference = official[f"{side}_clean_acc_diff_pp"].astype(float)
        assert ((numerator >= 0) & (numerator <= denominator)).all()
        assert ctr.between(0, 1, inclusive="both").all()
        assert difference.lt(TOLERANCE_PP).all()
        assert official[f"{side}_accuracy_valid"].eq(True).all()  # noqa: E712
        arithmetic_max_error[f"{side}_ctr"] = float((ctr - numerator / denominator).abs().max())
        arithmetic_max_error[f"{side}_accuracy"] = float((accuracy - correct / total).abs().max())
        arithmetic_max_error[f"{side}_difference_pp"] = float(
            (difference - (accuracy - official[recorded].astype(float)).abs() * 100).abs().max()
        )
        assert arithmetic_max_error[f"{side}_ctr"] < 1e-12
        assert arithmetic_max_error[f"{side}_accuracy"] < 1e-12
        assert arithmetic_max_error[f"{side}_difference_pp"] < 1e-12

    expected_denominators = {
        "source": {"cifar10": [7193], "tiny_imagenet": [7962]},
        "target": {"cifar10": [6400], "tiny_imagenet": [1990]},
    }
    observed_denominators = {
        side: {
            dataset: sorted(
                official.loc[official["dataset"].eq(dataset), f"{side}_ctr_denominator"].astype(int).unique().tolist()
            )
            for dataset in ("cifar10", "tiny_imagenet")
        }
        for side in ("source", "target")
    }
    assert observed_denominators == expected_denominators

    comparison = official.merge(
        archived,
        on="strict_config_key",
        suffixes=("_batch128", "_batch256"),
        validate="one_to_one",
    )
    batch_comparison = {}
    any_ctr_difference = pd.Series(False, index=comparison.index)
    for side in ("source", "target"):
        ctr_difference = (
            comparison[f"{side}_ctr_batch128"] - comparison[f"{side}_ctr_batch256"]
        ).abs() * 100
        accuracy_difference = (
            comparison[f"recomputed_{side}_clean_acc_batch128"]
            - comparison[f"recomputed_{side}_clean_acc_batch256"]
        ).abs() * 100
        any_ctr_difference |= ctr_difference.gt(0)
        batch_comparison[side] = {
            "ctr_rows_different": int(ctr_difference.gt(0).sum()),
            "max_ctr_difference_pp": float(ctr_difference.max()),
            "accuracy_rows_different": int(accuracy_difference.gt(0).sum()),
            "max_accuracy_difference_pp": float(accuracy_difference.max()),
        }
    batch_comparison["rows_with_any_ctr_difference"] = int(any_ctr_difference.sum())

    population = reference_population(manifest, official)
    archived_population = reference_population(manifest, archived, require_ok=False)
    table = pd.read_csv(TABLE_VALUES, dtype=str)
    assert table.shape == (8, 17)
    mismatches = table_mismatches(population, table)
    assert not mismatches
    rounded_cells_different = 0
    for dataset in ("cifar10", "tiny_imagenet"):
        for attack in ATTACK_NAMES.values():
            current = population[population["dataset"].eq(dataset) & population["Attack"].eq(attack)]
            old = archived_population[
                archived_population["dataset"].eq(dataset) & archived_population["Attack"].eq(attack)
            ]
            for metric in ("source_ctr", "target_ctr"):
                rounded_cells_different += (
                    f"{100.0 * current[metric].median():.1f}"
                    != f"{100.0 * old[metric].median():.1f}"
                )
    assert rounded_cells_different == 0

    tex_lines = TABLE_TEX.read_text(encoding="utf-8").splitlines()
    attacks = "|".join(re.escape(name) for name in ATTACK_NAMES.values())
    data_lines = [line for line in tex_lines if re.match(rf"^({attacks})\s*&", line.strip())]
    assert len(data_lines) == 8
    assert all(line.count("&") + 1 == 17 for line in data_lines)

    group_sizes = {
        f"{dataset}|{attack}": int(size)
        for (dataset, attack), size in population.groupby(["dataset", "Attack"], sort=False).size().items()
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_status": "VERIFIED",
        "official_batch_size": 128,
        "clean_accuracy_acceptance_rule": "absolute difference < 0.2 percentage points",
        "formal_result": {
            "path": str(OFFICIAL),
            "sha256": sha256(OFFICIAL),
            "rows": len(official),
            "unique_configuration_keys": int(official["strict_config_key"].nunique()),
            "status_counts": {str(k): int(v) for k, v in official["status"].value_counts().items()},
            "stable_legacy_belt_keys": int(official["strict_config_key"].str.startswith("legacy_belt|").sum()),
            "row_number_keys": int(official["strict_config_key"].str.startswith("manifest_row_").sum()),
            "unique_existing_model_paths": int(official["resolved_model_path"].nunique()),
            "max_source_clean_accuracy_difference_pp": float(official["source_clean_acc_diff_pp"].max()),
            "max_target_clean_accuracy_difference_pp": float(official["target_clean_acc_diff_pp"].max()),
            "arithmetic_max_absolute_error": arithmetic_max_error,
            "ctr_denominators": observed_denominators,
        },
        "archived_batch256": {
            "path": str(ARCHIVED),
            "sha256": sha256(ARCHIVED),
            "status_fields_authoritative": False,
        },
        "batch_size_comparison": batch_comparison,
        "table2": {
            "reference_configurations": len(population),
            "all_status_ok": True,
            "group_sizes": group_sizes,
            "csv_rows": int(table.shape[0]),
            "csv_columns": int(table.shape[1]),
            "tex_data_rows": len(data_lines),
            "tex_cells_per_row": [line.count("&") + 1 for line in data_lines],
            "median_mismatches": mismatches,
            "rounded_cells_changed_between_batch_sizes": int(rounded_cells_different),
        },
    }
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
