#!/usr/bin/env python3
"""Quality checks for generated analysis outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from config import COEFFICIENT_DIR, FIGURE_DIR, FIGURE_DOC_DIR, OUTPUT_DIR, RECOMMENDED_FIGURES, REPORT_DIR, REQUIRED_REPORTS, TABLE_DIR


def check_png(path: Path) -> dict[str, object]:
    rec = {"path": str(path), "exists": path.exists(), "ok": False, "width": 0, "height": 0, "size_bytes": 0, "reason": ""}
    if not path.exists():
        rec["reason"] = "missing"
        return rec
    rec["size_bytes"] = path.stat().st_size
    try:
        with Image.open(path) as img:
            rec["width"], rec["height"] = img.size
    except Exception as exc:
        rec["reason"] = f"open_failed:{exc}"
        return rec
    if rec["size_bytes"] < 10_000:
        rec["reason"] = "too_small"
    elif rec["width"] < 400 or rec["height"] < 300:
        rec["reason"] = "dimensions_too_small"
    else:
        rec["ok"] = True
    return rec


def run() -> pd.DataFrame:
    records = []
    for fig in sorted(FIGURE_DIR.glob("*.png")):
        rec = check_png(fig)
        doc = FIGURE_DOC_DIR / fig.with_suffix(".md").name
        rec["doc_exists"] = doc.exists()
        rec["doc_ok"] = doc.exists() and all(section in doc.read_text(encoding="utf-8") for section in ["Purpose", "Data Source", "How To Read", "Focus"])
        records.append(rec)
    for fig in RECOMMENDED_FIGURES:
        if not (FIGURE_DIR / fig).exists():
            records.append({"path": str(FIGURE_DIR / fig), "exists": False, "ok": False, "width": 0, "height": 0, "size_bytes": 0, "reason": "recommended_missing", "doc_exists": False, "doc_ok": False})
    report_rows = []
    for report in REQUIRED_REPORTS:
        path = REPORT_DIR / report
        report_rows.append({"path": str(path), "exists": path.exists(), "ok": path.exists(), "type": "report"})
    for required in [
        OUTPUT_DIR / "master_results.csv",
        OUTPUT_DIR / "completeness_report.csv",
        COEFFICIENT_DIR / "rq1_overall_correlations.csv",
        COEFFICIENT_DIR / "bootstrap_ci_correlations.csv",
        COEFFICIENT_DIR / "bootstrap_ci_pairwise_delta.csv",
        TABLE_DIR / "table_1_experiment_coverage.csv",
    ]:
        report_rows.append({"path": str(required), "exists": required.exists(), "ok": required.exists(), "type": "required_file"})
    df = pd.DataFrame(records + report_rows)
    df.to_csv(OUTPUT_DIR / "output_quality_report.csv", index=False)
    return df


if __name__ == "__main__":
    out = run()
    failed = out[out["ok"] != True]
    print(f"Checked {len(out)} outputs; failures={len(failed)}")
    if len(failed):
        print(failed.head(40).to_string(index=False))
