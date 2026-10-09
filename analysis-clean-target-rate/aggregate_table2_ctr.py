#!/usr/bin/env python3
"""Add source/target clean target-label rates to the paper's Table II.

The script selects the same 107 ResNet-18 reference configurations used by
Table II, computes configuration-level CTR medians, and updates only the Table
II CSV/TeX assets.  It never interpolates missing configurations.
"""

from pathlib import Path

import numpy as np
import pandas as pd


TOOLBOX_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = Path("/workspace/tifs_backdoor_paper_restructure_20260815")
MANIFEST = PAPER_ROOT / "data/primary_baseline_belt_completed.csv"
CTR_RESULTS = TOOLBOX_ROOT / "analysis-clean-target-rate/clean_target_rates_all_configurations.csv"
TABLE_VALUES = PAPER_ROOT / "tables/table2_reference_results_values.csv"
TABLE_TEX = PAPER_ROOT / "tables/table2_reference_results.tex"

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
ATTACK_ORDER = list(ATTACK_NAMES.values())


def assign_stable_configuration_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    missing = frame["strict_config_key"].isna() | frame["strict_config_key"].eq("")
    if missing.any():
        legacy = frame.loc[missing]
        if not legacy["dataset"].eq("tiny_imagenet").all() or not legacy[
            "attack_type"
        ].eq("belt").all():
            raise ValueError("missing strict_config_key outside legacy Tiny-ImageNet BELT rows")
        if legacy["folder_name"].isna().any() or legacy["folder_name"].duplicated().any():
            raise ValueError("legacy BELT rows require unique, non-empty folder_name values")
        frame.loc[missing, "strict_config_key"] = (
            "legacy_belt|" + legacy["folder_name"].astype(str)
        )
    if frame["strict_config_key"].isna().any() or frame["strict_config_key"].duplicated().any():
        raise ValueError("configuration keys must be present and unique")
    return frame


def reference_rate(dataset: str, attack: str) -> float:
    return 0.005 if dataset == "tiny_imagenet" and attack in {"SIG", "UPGD"} else 0.01


def load_reference_population() -> pd.DataFrame:
    manifest = assign_stable_configuration_keys(pd.read_csv(MANIFEST, low_memory=False))
    manifest["Attack"] = manifest["attack_type"].map(ATTACK_NAMES)
    results = pd.read_csv(CTR_RESULTS, low_memory=False)
    selected = []
    for dataset in ("cifar10", "tiny_imagenet"):
        for attack in ATTACK_ORDER:
            group = manifest[
                manifest["dataset"].eq(dataset)
                & manifest["Attack"].eq(attack)
                & manifest["arch_base"].eq("ResNet18")
                & np.isclose(manifest["poison_rate"], reference_rate(dataset, attack))
                & manifest["source_asr"].ge(0.05)
            ]
            selected.append(group)
    population = pd.concat(selected, ignore_index=True)
    if len(population) != 107 or population["strict_config_key"].nunique() != 107:
        raise ValueError("Table II reference population is not the expected 107 configurations")
    population = population.merge(
        results[
            [
                "strict_config_key",
                "source_ctr",
                "target_ctr",
                "status",
            ]
        ],
        on="strict_config_key",
        how="left",
        validate="one_to_one",
    )
    if population[["source_ctr", "target_ctr", "status"]].isna().any().any():
        raise ValueError("Table II configuration is missing a CTR result")
    if not population["status"].eq("ok").all():
        raise ValueError("Table II contains a configuration that failed CTR validation")
    return population


def format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}"


def update_values(population: pd.DataFrame) -> pd.DataFrame:
    values = pd.read_csv(TABLE_VALUES, dtype=object)
    for column in ("C CTRs", "C CTRt", "T CTRs", "T CTRt"):
        if column in values.columns:
            values = values.drop(columns=column)
    values.insert(values.columns.get_loc("C ACCs") + 1, "C CTRs", "")
    values.insert(values.columns.get_loc("C ACCt") + 1, "C CTRt", "")
    values.insert(values.columns.get_loc("T ACCs") + 1, "T CTRs", "")
    values.insert(values.columns.get_loc("T ACCt") + 1, "T CTRt", "")
    for index, row in values.iterrows():
        attack = row["Attack"]
        for dataset, prefix in (("cifar10", "C"), ("tiny_imagenet", "T")):
            group = population[
                population["dataset"].eq(dataset) & population["Attack"].eq(attack)
            ]
            if group.empty:
                raise ValueError(f"Missing Table II group: {dataset}/{attack}")
            values.loc[index, f"{prefix} CTRs"] = format_percent(group["source_ctr"].median())
            values.loc[index, f"{prefix} CTRt"] = format_percent(group["target_ctr"].median())
    expected = [
        "Attack", "C ACCs", "C CTRs", "C As", "C Ss", "C ACCt", "C CTRt", "C At", "C JSTS",
        "T ACCs", "T CTRs", "T As", "T Ss", "T ACCt", "T CTRt", "T At", "T JSTS",
    ]
    return values[expected]


def update_tex(values: pd.DataFrame) -> None:
    rows = []
    for _, row in values.iterrows():
        cells = [row["Attack"]]
        for prefix in ("C", "T"):
            cells.extend(
                [
                    row[f"{prefix} ACCs"],
                    row[f"{prefix} CTRs"],
                    row[f"{prefix} As"],
                    row[f"{prefix} Ss"],
                    row[f"{prefix} ACCt"],
                    row[f"{prefix} CTRt"],
                    row[f"{prefix} At"],
                    rf"\textbf{{{row[f'{prefix} JSTS']}}}",
                ]
            )
        rows.append("\n" + " & ".join(map(str, cells)) + r" \\" + "\n")
    text = r"""\begin{table*}[!t]
\centering
\caption{Joint evaluation of backdoor attacks across distribution shifts. $\mathrm{CTR}_s$ and $\mathrm{CTR}_t$ denote the clean target-label rate on source and target domains, respectively; their denominators exclude samples whose true label is the attack target.}
\label{tab:reference_results}

\begin{minipage}{0.99\textwidth}
\centering
\tiny
\setlength{\tabcolsep}{1.0pt}
\renewcommand{\arraystretch}{1.05}

\begin{tabular*}{\linewidth}{l@{\extracolsep{\fill}}cccccccc@{\extracolsep{\fill}}cccccccc}
\toprule
& \multicolumn{8}{c}{CIFAR-10$\rightarrow$STL-10}
& \multicolumn{8}{c}{Tiny-ImageNet$\rightarrow$ImageNetV2} \\
\cmidrule(lr){2-9}
\cmidrule(lr){10-17}

Attack
& $\mathrm{ACC}_s$ & $\mathrm{CTR}_s$ & $A_s$ & $S_s$ & $\mathrm{ACC}_t$ & $\mathrm{CTR}_t$ & $A_t$ & JSTS [IQR]
& $\mathrm{ACC}_s$ & $\mathrm{CTR}_s$ & $A_s$ & $S_s$ & $\mathrm{ACC}_t$ & $\mathrm{CTR}_t$ & $A_t$ & JSTS [IQR] \\
\midrule
""" + "".join(rows) + r"""
\bottomrule
\end{tabular*}

\end{minipage}
\end{table*}
"""
    TABLE_TEX.write_text(text, encoding="utf-8")


def main() -> None:
    population = load_reference_population()
    values = update_values(population)
    values.to_csv(TABLE_VALUES, index=False)
    update_tex(values)
    print(values[["Attack", "C CTRs", "C CTRt", "T CTRs", "T CTRt"]].to_string(index=False))
    print(f"Updated {len(population)} configuration-level results across {len(values)} attacks.")


if __name__ == "__main__":
    main()
