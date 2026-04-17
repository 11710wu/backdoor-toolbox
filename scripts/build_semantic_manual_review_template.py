#!/usr/bin/env python3
"""Build manual review CSV for high-risk Tiny-ImageNet classes."""

import argparse
import csv
import json
import re
from pathlib import Path


def sanitize_class_dir_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown_class"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create manual review CSV for semantic ambiguity classes.")
    parser.add_argument("--risk-topk", required=True, type=str)
    parser.add_argument("--semantic-spec", required=True, type=str)
    parser.add_argument("--images-root", required=True, type=str)
    parser.add_argument("--output-csv", required=True, type=str)
    parser.add_argument("--sample-file", type=str, default="0001.png")
    args = parser.parse_args()

    topk = json.loads(Path(args.risk_topk).read_text(encoding="utf-8"))
    spec = json.loads(Path(args.semantic_spec).read_text(encoding="utf-8"))
    spec_by_wnid = {c["wnid"]: c for c in spec["classes"]}
    images_root = Path(args.images_root)

    rows = []
    for item in topk:
        wnid = item["wnid"]
        name = item["name"]
        sem = spec_by_wnid.get(wnid, {})
        class_dir = sanitize_class_dir_name(name)
        image_path = images_root / class_dir / args.sample_file
        rows.append(
            {
                "wnid": wnid,
                "name": name,
                "risk_score": item.get("risk_score", ""),
                "risk_reasons": "|".join(item.get("risk_reasons", [])),
                "image_path": str(image_path),
                "sense": sem.get("sense", ""),
                "include_keywords": "|".join(sem.get("include_keywords", [])),
                "exclude_keywords": "|".join(sem.get("exclude_keywords", [])),
                "pass_fail": "",
                "failure_type": "",
                "reason": "",
            }
        )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "wnid", "name", "risk_score", "risk_reasons", "image_path",
        "sense", "include_keywords", "exclude_keywords",
        "pass_fail", "failure_type", "reason",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] manual review template: {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

