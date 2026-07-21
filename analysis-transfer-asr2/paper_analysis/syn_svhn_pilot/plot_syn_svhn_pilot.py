#!/usr/bin/env python3
"""Plot three-point SYN -> SVHN pilot scatter diagnostics (never corridors)."""

import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from run.syn_svhn_pilot_config import ATTACKS, PILOT_ROOT


def main():
    root = Path(PILOT_ROOT)
    source = root / "pilot_summary.csv"
    if not source.exists():
        raise FileNotFoundError(
            "Run analysis-transfer-asr2/paper_analysis/syn_svhn_pilot/"
            f"collect_syn_svhn_pilot.py first: {source}"
        )
    frame = pd.read_csv(source)
    plot_dir = root / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    exclusions = []
    diagnostics = []
    for attack in ATTACKS:
        subset = frame[frame["attack"] == attack].copy()
        if subset.empty:
            continue
        usable = subset.dropna(subset=["target_transfer_asr", "stealthiness"])
        valid_usable = usable[usable["source_asr_ge_5pct"] == True]
        for _, row in subset.loc[~subset.index.isin(usable.index)].iterrows():
            missing = []
            if pd.isna(row.get("target_transfer_asr")):
                missing.append("target_transfer_asr")
            if pd.isna(row.get("stealthiness")):
                missing.append("stealthiness_or_detector")
            exclusions.append({
                "attack": attack, "strength_name": row.get("strength_name"),
                "strength_value": row.get("strength_value"), "reason": ";".join(missing),
                "result_dir": row.get("result_dir"),
            })
        fig, ax = plt.subplots(figsize=(5.4, 4.5))
        for _, row in usable.iterrows():
            valid = bool(row.get("source_asr_ge_5pct"))
            ax.scatter(
                row["target_transfer_asr"], row["stealthiness"],
                marker="o" if valid else "x", s=72,
                color="#2878B5" if valid else "#888888", linewidths=1.8, zorder=3,
            )
            ax.annotate(
                f"{row['strength_name']}={row['strength_value']:g}",
                (row["target_transfer_asr"], row["stealthiness"]),
                xytext=(5, 5), textcoords="offset points", fontsize=8,
            )
        rho = pvalue = float("nan")
        if len(valid_usable) == 3:
            rho, pvalue = spearmanr(
                valid_usable["target_transfer_asr"], valid_usable["stealthiness"]
            )
        diagnostics.append({
            "attack": attack, "n": len(valid_usable), "spearman_rho": rho,
            "spearman_pvalue": pvalue, "interpretation": "diagnostic only",
        })
        ax.set_xlabel("Target-side transfer ASR")
        ax.set_ylabel("Stealthiness")
        ax.set_title(f"SYN→SVHN {attack}: n={len(usable)}, diagnostic only")
        ax.grid(alpha=0.22)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(plot_dir / f"{attack}_transfer_stealth_scatter.{suffix}", dpi=300)
        plt.close(fig)
    pd.DataFrame(exclusions, columns=[
        "attack", "strength_name", "strength_value", "reason", "result_dir",
    ]).to_csv(root / "pilot_plot_exclusions.csv", index=False)
    pd.DataFrame(diagnostics, columns=[
        "attack", "n", "spearman_rho", "spearman_pvalue", "interpretation",
    ]).to_csv(root / "pilot_spearman_diagnostic.csv", index=False)
    print(f"[pilot plots] wrote scatter figures and diagnostics under {root}")


if __name__ == "__main__":
    main()
