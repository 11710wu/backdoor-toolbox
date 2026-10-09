"""Stratified paired bootstrap for the frozen WaNet diagnostic CSVs.

This writes only to diagnostics_wanet/ and never alters the formal CTR files.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "diagnostics_wanet"
SEED = 2333
N_BOOT = 2000


def metric(y, clean, pred, target):
    non_target = y != target
    ctr = np.mean(clean[non_target] == target) if non_target.any() else np.nan
    asr = np.mean(pred[non_target] == target) if non_target.any() else np.nan
    eligible = non_target & (clean != target)
    conv = np.mean(pred[eligible] == target) if eligible.any() else np.nan
    return ctr, asr, conv


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for path in sorted(OUT.glob("*_paired.csv")):
        raw = pd.read_csv(path)
        for (domain, variant), z in raw.groupby(["domain", "variant"], sort=True):
            # Bootstrap within true class so class composition cannot drive the CI.
            target = 0
            eligible = z[z.true_label != target]
            groups = [g for _, g in eligible.groupby("true_label")]
            if not groups:
                continue
            # Vectorize the resampling per class; this keeps the requested
            # 2000 replicates fast even for the 8k-image source split.
            vals = np.empty((N_BOOT, 3), dtype=float)
            clean_parts, pred_parts = [], []
            for g in groups:
                n = len(g)
                ix = rng.integers(0, n, size=(N_BOOT, n))
                clean_parts.append(g.clean_pred.to_numpy()[ix])
                pred_parts.append(g.pred.to_numpy()[ix])
            clean = np.concatenate(clean_parts, axis=1)
            pred = np.concatenate(pred_parts, axis=1)
            vals[:, 0] = np.mean(clean == target, axis=1)
            vals[:, 1] = np.mean(pred == target, axis=1)
            eligible = clean != target
            vals[:, 2] = np.divide(np.sum((pred == target) & eligible, axis=1),
                                   np.sum(eligible, axis=1),
                                   out=np.full(N_BOOT, np.nan), where=np.sum(eligible, axis=1) > 0)
            for name, col in zip(("ctr", "asr", "conditional_conversion"), range(3)):
                q = np.nanpercentile(vals[:, col], [2.5, 50, 97.5])
                rows.append({
                    "file": path.name, "domain": domain, "variant": variant,
                    "metric": name, "n": len(z), "bootstrap": N_BOOT,
                    "seed": SEED, "mean": float(np.nanmean(vals[:, col])),
                    "p2_5": float(q[0]), "median": float(q[1]), "p97_5": float(q[2]),
                })
    pd.DataFrame(rows).to_csv(OUT / "bootstrap_summary.csv", index=False)


if __name__ == "__main__":
    main()
