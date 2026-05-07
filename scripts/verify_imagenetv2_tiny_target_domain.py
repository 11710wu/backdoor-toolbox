#!/usr/bin/env python3
"""Verify ImageNetV2 -> Tiny-ImageNet target-domain extraction.

Checks performed:
1. Source ImageNetV2 root has numeric class dirs 0..999.
2. Tiny-ImageNet train class order is used exactly.
3. Output ``class_to_idx.json`` matches Tiny-ImageNet sorted ``wnid`` order.
4. Frozen ``Tiny wnid -> ImageNet1k index`` mapping matches the official
   ImageNet class-index JSON when requested.
5. Every manifest row is self-consistent.
6. Every output image is 64x64 (or the requested expected size).
7. For every class, output filenames match the expected source filenames from
   the mapped ImageNetV2 class directory.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen

from PIL import Image

from extract_imagenetv2_tiny_target_domain import (
    TINY_WNID_TO_IMAGENET1K_IDX,
    load_tiny_metadata,
)


OFFICIAL_IMAGENET_CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_manifest_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def list_image_names(directory: Path) -> List[str]:
    if not directory.is_dir():
        return []
    return sorted(
        p.name for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def fetch_official_class_index() -> Dict[str, List[str]]:
    with urlopen(OFFICIAL_IMAGENET_CLASS_INDEX_URL, timeout=20) as r:  # nosec B310
        return json.load(r)


def verify(
    src_root: Path,
    tiny_root: Path,
    output_dir: Path,
    class_map_path: Path | None,
    expected_size: int,
    check_official: bool,
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    errors: List[str] = []

    sorted_train_wnids, wnid_to_synonyms = load_tiny_metadata(tiny_root, class_map_path)
    expected_class_to_idx = {wnid: i for i, wnid in enumerate(sorted_train_wnids)}

    # ------------------------------------------------------------------
    # 1. Source root check: numeric dirs 0..999 and per-class image counts
    # ------------------------------------------------------------------
    src_dirs = sorted(p for p in src_root.iterdir() if p.is_dir())
    src_dir_names = [p.name for p in src_dirs]
    try:
        src_indices = sorted(int(name) for name in src_dir_names)
        source_is_numeric_0_999 = src_indices == list(range(1000))
    except ValueError:
        src_indices = []
        source_is_numeric_0_999 = False
    src_counts = {p.name: len(list_image_names(p)) for p in src_dirs}
    results["source"] = {
        "num_class_dirs": len(src_dirs),
        "is_numeric_0_999": source_is_numeric_0_999,
        "min_images_per_class": min(src_counts.values()) if src_counts else 0,
        "max_images_per_class": max(src_counts.values()) if src_counts else 0,
        "all_classes_have_10_images": all(v == 10 for v in src_counts.values()) if src_counts else False,
    }
    if not source_is_numeric_0_999:
        errors.append("Source ImageNetV2 dirs are not exactly numeric 0..999.")

    # ------------------------------------------------------------------
    # 2. Output metadata and class order
    # ------------------------------------------------------------------
    class_to_idx_path = output_dir / "class_to_idx.json"
    tiny_to_imagenet_path = output_dir / "tiny_to_imagenetv2_index.json"
    manifest_path = output_dir / "dataset_manifest.jsonl"
    test_root = output_dir / "test"

    class_to_idx = json.loads(class_to_idx_path.read_text(encoding="utf-8"))
    tiny_to_imagenet = json.loads(tiny_to_imagenet_path.read_text(encoding="utf-8"))
    manifest_rows = load_manifest_rows(manifest_path)
    output_class_dirs = sorted(p.name for p in test_root.iterdir() if p.is_dir())

    results["output"] = {
        "num_output_classes": len(output_class_dirs),
        "class_to_idx_matches_tiny_train": class_to_idx == expected_class_to_idx,
        "output_class_dirs_match_tiny_train": output_class_dirs == sorted_train_wnids,
        "tiny_to_imagenet_mapping_matches_frozen": tiny_to_imagenet == TINY_WNID_TO_IMAGENET1K_IDX,
        "manifest_rows": len(manifest_rows),
        "images_alias_exists": (output_dir / "images").exists(),
        "images_alias_is_symlink": (output_dir / "images").is_symlink(),
    }
    if class_to_idx != expected_class_to_idx:
        errors.append("Output class_to_idx.json does not match Tiny-ImageNet train order.")
    if output_class_dirs != sorted_train_wnids:
        errors.append("Output class dirs do not match Tiny-ImageNet sorted wnids.")
    if tiny_to_imagenet != TINY_WNID_TO_IMAGENET1K_IDX:
        errors.append("Output tiny_to_imagenetv2_index.json does not match frozen mapping.")

    # ------------------------------------------------------------------
    # 3. Official ImageNet class-index cross-check
    # ------------------------------------------------------------------
    official_mismatches: List[Dict[str, object]] = []
    if check_official:
        official = fetch_official_class_index()
        for wnid, idx in TINY_WNID_TO_IMAGENET1K_IDX.items():
            official_wnid, official_name = official[str(idx)]
            if official_wnid != wnid:
                official_mismatches.append(
                    {
                        "tiny_wnid": wnid,
                        "mapped_index": idx,
                        "official_wnid": official_wnid,
                        "official_name": official_name,
                    }
                )
        results["official_cross_check"] = {
            "checked": True,
            "mismatch_count": len(official_mismatches),
            "mismatches_sample": official_mismatches[:20],
        }
        if official_mismatches:
            errors.append(f"Official ImageNet class-index mismatches: {len(official_mismatches)}")
    else:
        results["official_cross_check"] = {"checked": False}

    # ------------------------------------------------------------------
    # 4. Manifest + per-class filename/order checks
    # ------------------------------------------------------------------
    per_class_manifest_count = Counter()
    src_seen = set()
    dst_seen = set()
    bad_manifest_rows: List[Dict[str, object]] = []
    bad_sizes: List[Dict[str, object]] = []

    for row in manifest_rows:
        wnid = row["wnid"]
        src_path = Path(row["src_path"])
        dst_path = Path(row["dst_path"])
        mapped_idx = TINY_WNID_TO_IMAGENET1K_IDX[wnid]

        row_errors: List[str] = []
        if wnid not in class_to_idx:
            row_errors.append("wnid_missing_in_class_to_idx")
        else:
            if row["tiny_label_idx"] != class_to_idx[wnid]:
                row_errors.append("tiny_label_idx_mismatch")
        if row["imagenetv2_index"] != mapped_idx:
            row_errors.append("imagenetv2_index_mismatch")
        if src_path.parent.name != str(mapped_idx):
            row_errors.append("src_parent_dir_mismatch")
        if not src_path.exists():
            row_errors.append("src_missing")
        if not dst_path.exists():
            row_errors.append("dst_missing")
        if src_path in src_seen:
            row_errors.append("duplicate_src_path")
        if dst_path in dst_seen:
            row_errors.append("duplicate_dst_path")

        if dst_path.exists():
            with Image.open(dst_path) as im:
                if im.size != (expected_size, expected_size):
                    bad_sizes.append({"dst_path": str(dst_path), "size": list(im.size)})

        src_seen.add(src_path)
        dst_seen.add(dst_path)
        per_class_manifest_count[wnid] += 1

        if row_errors:
            bad_manifest_rows.append({"row": row, "errors": row_errors})

    per_class_filename_checks: Dict[str, Dict[str, object]] = {}
    per_class_filename_errors: List[Dict[str, object]] = []

    for wnid in sorted_train_wnids:
        mapped_idx = TINY_WNID_TO_IMAGENET1K_IDX[wnid]
        src_dir = src_root / str(mapped_idx)
        dst_dir = test_root / wnid / "images"

        src_names = list_image_names(src_dir)
        dst_names = list_image_names(dst_dir)
        expected_dst_names = [Path(name).stem + ".jpeg" for name in src_names]

        ok = dst_names == expected_dst_names
        per_class_filename_checks[wnid] = {
            "imagenetv2_index": mapped_idx,
            "src_count": len(src_names),
            "dst_count": len(dst_names),
            "match": ok,
        }
        if not ok:
            per_class_filename_errors.append(
                {
                    "wnid": wnid,
                    "imagenetv2_index": mapped_idx,
                    "src_names_sample": src_names[:5],
                    "dst_names_sample": dst_names[:5],
                    "expected_dst_names_sample": expected_dst_names[:5],
                }
            )

    results["integrity"] = {
        "manifest_error_count": len(bad_manifest_rows),
        "bad_manifest_rows_sample": bad_manifest_rows[:20],
        "bad_size_count": len(bad_sizes),
        "bad_size_sample": bad_sizes[:20],
        "unique_src_paths": len(src_seen),
        "unique_dst_paths": len(dst_seen),
        "per_class_manifest_count_min": min(per_class_manifest_count.values()) if per_class_manifest_count else 0,
        "per_class_manifest_count_max": max(per_class_manifest_count.values()) if per_class_manifest_count else 0,
        "per_class_filename_match_all": len(per_class_filename_errors) == 0,
        "per_class_filename_errors_sample": per_class_filename_errors[:20],
    }

    if bad_manifest_rows:
        errors.append(f"Manifest consistency errors: {len(bad_manifest_rows)}")
    if bad_sizes:
        errors.append(f"Output image size mismatches: {len(bad_sizes)}")
    if per_class_filename_errors:
        errors.append(f"Per-class filename/order mismatches: {len(per_class_filename_errors)}")

    # ------------------------------------------------------------------
    # 5. Human-readable samples
    # ------------------------------------------------------------------
    sample_rows = []
    for wnid in [sorted_train_wnids[0], sorted_train_wnids[32], sorted_train_wnids[109], sorted_train_wnids[-1]]:
        sample_rows.append(
            {
                "wnid": wnid,
                "class_name": wnid_to_synonyms.get(wnid, [wnid])[0],
                "tiny_label_idx": class_to_idx[wnid],
                "imagenetv2_index": TINY_WNID_TO_IMAGENET1K_IDX[wnid],
                "images_written": per_class_manifest_count[wnid],
            }
        )
    results["samples"] = sample_rows

    results["ok"] = not errors
    results["errors"] = errors
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify ImageNetV2 -> Tiny-ImageNet target-domain extraction."
    )
    parser.add_argument(
        "--src-root",
        type=str,
        default="/workspace/imagenetv2-matched-frequency-format-val",
        help="ImageNetV2 root with numeric 0..999 class dirs",
    )
    parser.add_argument(
        "--tiny-root",
        type=str,
        default="/workspace/backdoor-toolbox-new1/data/Tiny-imagenet/tiny-imagenet-200",
        help="Tiny-ImageNet root",
    )
    parser.add_argument(
        "--class-map",
        type=str,
        default="/workspace/data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json",
        help="Optional Tiny-ImageNet class map JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/data/imagenetv2-matched-frequency-tiny-organized",
        help="Extracted Tiny-compatible target-domain dataset root",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=64,
        help="Expected square output image size",
    )
    parser.add_argument(
        "--check-official-class-index",
        action="store_true",
        help="Download the official ImageNet class-index JSON and cross-check all 200 mappings",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional JSON report output path",
    )
    args = parser.parse_args()

    report = verify(
        src_root=Path(args.src_root).resolve(),
        tiny_root=Path(args.tiny_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        class_map_path=Path(args.class_map).resolve() if args.class_map else None,
        expected_size=args.expected_size,
        check_official=args.check_official_class_index,
    )

    report_path = (
        Path(args.report_path).resolve()
        if args.report_path
        else Path(args.output_dir).resolve() / "verification_report.json"
    )
    write_json(report_path, report)

    print(f"[VERIFY] report_path={report_path}")
    print(f"[VERIFY] ok={report['ok']}")
    if report["errors"]:
        for err in report["errors"]:
            print(f"[VERIFY][ERROR] {err}")
    else:
        print("[VERIFY] All checks passed.")
    print(f"[VERIFY] source={report['source']}")
    print(f"[VERIFY] output={report['output']}")
    print(f"[VERIFY] integrity={report['integrity']}")


if __name__ == "__main__":
    main()
