#!/usr/bin/env python3
"""Build quality report for generated Tiny target-domain dataset."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageStat


def is_gray_image(path: Path, std_threshold: float = 8.0) -> bool:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            stat = ImageStat.Stat(img)
            # Average RGB std as a simple low-texture / gray proxy.
            avg_std = sum(stat.stddev) / 3.0
            return avg_std < std_threshold
    except Exception:
        return False


def load_manual_review(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wnid = (row.get("wnid") or "").strip()
            if wnid:
                rows[wnid] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create generation quality report JSON.")
    parser.add_argument("--images-root", required=True, type=str, help="Root directory with class subfolders.")
    parser.add_argument("--manual-review-csv", type=str, default=None, help="Optional manual review CSV (Top50).")
    parser.add_argument("--output-json", required=True, type=str)
    parser.add_argument("--gray-std-threshold", type=float, default=8.0)
    args = parser.parse_args()

    images_root = Path(args.images_root)
    class_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])

    per_class_counts: Dict[str, int] = {}
    total_images = 0
    gray_images = 0

    for class_dir in class_dirs:
        files = sorted(class_dir.glob("*.png"))
        per_class_counts[class_dir.name] = len(files)
        total_images += len(files)
        for f in files:
            if is_gray_image(f, std_threshold=args.gray_std_threshold):
                gray_images += 1

    counts = list(per_class_counts.values())
    manual_rows = load_manual_review(Path(args.manual_review_csv)) if args.manual_review_csv else {}
    reviewed = [r for r in manual_rows.values() if (r.get("pass_fail") or "").strip()]
    reviewed_pass = [
        r for r in reviewed
        if (r.get("pass_fail") or "").strip().lower() in {"pass", "true", "1", "y", "yes"}
    ]
    failed_examples: List[Dict] = []
    for r in reviewed:
        pf = (r.get("pass_fail") or "").strip().lower()
        if pf in {"fail", "false", "0", "n", "no"}:
            failed_examples.append(
                {
                    "wnid": r.get("wnid", ""),
                    "name": r.get("name", ""),
                    "failure_type": r.get("failure_type", ""),
                    "reason": r.get("reason", ""),
                }
            )

    report = {
        "images_root": str(images_root),
        "num_classes": len(class_dirs),
        "total_images": total_images,
        "per_class_min": min(counts) if counts else 0,
        "per_class_max": max(counts) if counts else 0,
        "per_class_counts": per_class_counts,
        "gray_image_threshold_std": args.gray_std_threshold,
        "gray_image_count": gray_images,
        "gray_image_rate": (gray_images / total_images) if total_images else 0.0,
        "manual_review_rows": len(manual_rows),
        "manual_reviewed_rows": len(reviewed),
        "ambiguous_class_pass_rate": (len(reviewed_pass) / len(reviewed)) if reviewed else None,
        "manual_fail_examples": failed_examples[:50],
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] quality report: {out}")
    print(f"[OK] classes={report['num_classes']} total_images={report['total_images']} gray_rate={report['gray_image_rate']:.4f}")


if __name__ == "__main__":
    main()

