#!/usr/bin/env python3
"""Organize generated target-domain images into a stable ImageFolder dataset.

The output directory uses **wnid** as subdirectory names so that
``torchvision.datasets.ImageFolder`` assigns label indices in exactly the same
sorted order as the original Tiny-ImageNet train/val sets.

Usage:
    python scripts/organize_tiny_target_domain.py \
        --src-images /workspace/data/tiny-target-domain-50shot-v1/images \
        --class-map  /workspace/data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json \
        --output-dir /workspace/data/tiny-target-domain

Outputs under ``--output-dir``:
    images/<wnid>/*.png          ImageFolder structure (wnid dirs, alphabetically sorted = label idx)
    class_to_idx.json            {wnid: int}  – matches ImageFolder assignment
    idx_to_class.json            {int_str: {wnid, name}}
    class_to_wnid.json           {display_name: wnid}
    wnid_to_class.json           {wnid: display_name}
    dataset_manifest.jsonl        per-image record
    build_validation_report.json  automated QC report
    build_validation_summary.txt  human-readable summary
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def load_class_map(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = sorted(data["classes"], key=lambda c: int(c["index"]))
    assert len(classes) == 200, f"Expected 200 classes, got {len(classes)}"
    return classes


def build_name_to_entry(classes: List[Dict]) -> Dict[str, Dict]:
    """Build lookup: sanitized display name -> class entry (for matching src dirs)."""
    lookup: Dict[str, Dict] = {}
    for c in classes:
        key = sanitize_name(c["name"])
        lookup[key] = c
        for syn in c.get("synonyms", []):
            skey = sanitize_name(syn)
            if skey not in lookup:
                lookup[skey] = c
    return lookup


def discover_src_images(src_root: Path) -> Dict[str, List[Path]]:
    """Return {dir_name: [image_paths]} for every subdirectory in src_root."""
    result: Dict[str, List[Path]] = {}
    if not src_root.is_dir():
        return result
    for d in sorted(src_root.iterdir()):
        if not d.is_dir():
            continue
        imgs = sorted(
            p for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        )
        result[d.name] = imgs
    return result


def organize(
    src_root: Path,
    class_map_path: Path,
    output_dir: Path,
    copy_mode: str = "copy",
) -> Dict:
    classes = load_class_map(class_map_path)
    name_lookup = build_name_to_entry(classes)

    # ImageFolder label index = sorted wnid order
    wnid_sorted = sorted(classes, key=lambda c: c["wnid"])
    class_to_idx = {c["wnid"]: idx for idx, c in enumerate(wnid_sorted)}
    idx_to_class = {idx: {"wnid": c["wnid"], "name": c["name"]} for idx, c in enumerate(wnid_sorted)}
    wnid_to_name = {c["wnid"]: c["name"] for c in classes}
    name_to_wnid = {c["name"]: c["wnid"] for c in classes}

    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    src_dirs = discover_src_images(src_root)

    manifest_rows: List[Dict] = []
    matched_wnids = set()
    unmatched_dirs: List[str] = []
    per_class_counts: Dict[str, int] = {}
    bad_images: List[Dict] = []

    for dir_name, src_paths in src_dirs.items():
        entry = name_lookup.get(dir_name)
        if entry is None:
            entry = name_lookup.get(sanitize_name(dir_name))
        if entry is None:
            # Try wnid directly
            if dir_name in wnid_to_name:
                entry = next(c for c in classes if c["wnid"] == dir_name)
        if entry is None:
            unmatched_dirs.append(dir_name)
            continue

        wnid = entry["wnid"]
        matched_wnids.add(wnid)
        dst_dir = images_out / wnid
        dst_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for src_path in src_paths:
            dst_path = dst_dir / src_path.name
            # Validate image
            try:
                with Image.open(src_path) as img:
                    w, h = img.size
                    if (w, h) != (64, 64):
                        bad_images.append({
                            "src": str(src_path),
                            "reason": f"size={w}x{h}, expected 64x64",
                            "wnid": wnid,
                        })
                        continue
            except Exception as e:
                bad_images.append({
                    "src": str(src_path),
                    "reason": f"unreadable: {e}",
                    "wnid": wnid,
                })
                continue

            if copy_mode == "symlink":
                if dst_path.exists() or dst_path.is_symlink():
                    dst_path.unlink()
                dst_path.symlink_to(src_path.resolve())
            else:
                shutil.copy2(src_path, dst_path)

            count += 1
            manifest_rows.append({
                "wnid": wnid,
                "class_name": entry["name"],
                "label_idx": class_to_idx[wnid],
                "src_path": str(src_path),
                "dst_path": str(dst_path),
                "filename": src_path.name,
            })

        per_class_counts[wnid] = count

    # Find missing classes
    all_wnids = {c["wnid"] for c in classes}
    missing_wnids = all_wnids - matched_wnids

    # Write mappings
    _write_json(output_dir / "class_to_idx.json", class_to_idx)
    _write_json(output_dir / "idx_to_class.json", {str(k): v for k, v in idx_to_class.items()})
    _write_json(output_dir / "wnid_to_class.json", wnid_to_name)
    _write_json(output_dir / "class_to_wnid.json", name_to_wnid)

    # Write manifest
    manifest_path = output_dir / "dataset_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Build validation report
    counts = list(per_class_counts.values())
    report = {
        "num_classes_expected": 200,
        "num_classes_matched": len(matched_wnids),
        "num_classes_missing": len(missing_wnids),
        "missing_wnids": sorted(missing_wnids),
        "unmatched_src_dirs": unmatched_dirs,
        "total_images_copied": len(manifest_rows),
        "per_class_min": min(counts) if counts else 0,
        "per_class_max": max(counts) if counts else 0,
        "per_class_mean": round(sum(counts) / len(counts), 2) if counts else 0,
        "classes_with_zero_images": sorted(
            wnid for wnid in all_wnids if per_class_counts.get(wnid, 0) == 0
        ),
        "bad_images_count": len(bad_images),
        "bad_images_sample": bad_images[:20],
        "class_to_idx_matches_imagefolder_sort": (
            list(class_to_idx.keys()) == sorted(class_to_idx.keys())
        ),
    }
    _write_json(output_dir / "build_validation_report.json", report)

    # Human-readable summary
    summary_lines = [
        "=== Tiny Target Domain Build Validation ===",
        f"Classes expected:  200",
        f"Classes matched:   {report['num_classes_matched']}",
        f"Classes missing:   {report['num_classes_missing']}",
        f"Total images:      {report['total_images_copied']}",
        f"Per-class min/max/mean: {report['per_class_min']}/{report['per_class_max']}/{report['per_class_mean']}",
        f"Bad images:        {report['bad_images_count']}",
        f"Unmatched src dirs:{len(unmatched_dirs)}",
        f"class_to_idx sorted correctly: {report['class_to_idx_matches_imagefolder_sort']}",
    ]
    if report["num_classes_missing"] > 0:
        summary_lines.append(f"Missing wnids: {report['missing_wnids'][:10]}...")
    if unmatched_dirs:
        summary_lines.append(f"Unmatched dirs: {unmatched_dirs[:10]}...")
    if report["classes_with_zero_images"]:
        summary_lines.append(f"Zero-image classes: {report['classes_with_zero_images'][:10]}...")

    ok = (
        report["num_classes_matched"] == 200
        and report["num_classes_missing"] == 0
        and report["per_class_min"] > 0
        and report["bad_images_count"] == 0
        and len(unmatched_dirs) == 0
    )
    summary_lines.append(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    summary_text = "\n".join(summary_lines)
    (output_dir / "build_validation_summary.txt").write_text(summary_text, encoding="utf-8")

    print(summary_text)
    return report


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize generated images into a stable Tiny target-domain dataset."
    )
    parser.add_argument(
        "--src-images", type=str, required=True,
        help="Root of generated images, e.g. data/tiny-target-domain-50shot-v1/images",
    )
    parser.add_argument(
        "--class-map", type=str,
        default="./data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json",
        help="Path to class_map_tiny_imagenet_200.json",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/tiny-target-domain",
        help="Output dataset root",
    )
    parser.add_argument(
        "--copy-mode", type=str, choices=["copy", "symlink"], default="copy",
        help="How to place images: copy files or create symlinks",
    )
    args = parser.parse_args()

    organize(
        src_root=Path(args.src_images).resolve(),
        class_map_path=Path(args.class_map).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        copy_mode=args.copy_mode,
    )


if __name__ == "__main__":
    main()
