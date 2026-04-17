#!/usr/bin/env python3
"""Review utilities for tiny target-domain preview set.

Functions:
- validate image completeness and shape
- build a compact review report
- export failed_cases template for manual labeling
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image


def load_manifest(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def inspect_images(rows: List[Dict]) -> Dict:
    missing = []
    bad_size = []
    failed_gen = []
    ok = 0
    for row in rows:
        if row.get("status") != "ok":
            failed_gen.append(
                {
                    "wnid": row.get("wnid"),
                    "class_name": row.get("class_name"),
                    "reason": row.get("error", "generation_failed"),
                }
            )
            continue
        p = Path(row["output_path"])
        if not p.exists():
            missing.append({"wnid": row.get("wnid"), "class_name": row.get("class_name"), "path": str(p)})
            continue
        try:
            with Image.open(p) as img:
                if img.size != (64, 64):
                    bad_size.append(
                        {
                            "wnid": row.get("wnid"),
                            "class_name": row.get("class_name"),
                            "path": str(p),
                            "size": f"{img.size[0]}x{img.size[1]}",
                        }
                    )
                else:
                    ok += 1
        except Exception as e:  # noqa: BLE001
            missing.append({"wnid": row.get("wnid"), "class_name": row.get("class_name"), "path": str(p), "error": str(e)})
    return {
        "total_rows": len(rows),
        "ok_64x64": ok,
        "missing_or_unreadable": missing,
        "bad_size": bad_size,
        "failed_generation_rows": failed_gen,
    }


def build_manual_failed_template(rows: List[Dict]) -> List[Dict]:
    template = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        template.append(
            {
                "wnid": row.get("wnid"),
                "class_name": row.get("class_name"),
                "image_path": row.get("output_path"),
                "accept": None,
                "fail_reason": "",
                "notes": "",
            }
        )
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description="Review tiny target-domain preview outputs.")
    parser.add_argument(
        "--preview-dir",
        type=str,
        default="./data/tiny-target-domain-preview-1shot",
    )
    args = parser.parse_args()

    preview_dir = Path(args.preview_dir).resolve()
    manifest_path = preview_dir / "manifest_preview_1shot.jsonl"
    report_path = preview_dir / "review_report.json"
    manual_template_path = preview_dir / "failed_cases.manual_template.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

    rows = load_manifest(manifest_path)
    report = inspect_images(rows)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    manual_template = build_manual_failed_template(rows)
    manual_template_path.write_text(json.dumps(manual_template, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] review_report: {report_path}")
    print(f"[OK] manual_template: {manual_template_path}")
    print(f"[INFO] ok_64x64={report['ok_64x64']} / total={report['total_rows']}")


if __name__ == "__main__":
    main()

