#!/usr/bin/env python3
"""Extract the Tiny-ImageNet 200-class subset from ImageNetV2.

This script turns ImageNetV2's ``0..999`` class folders into a Tiny-ImageNet
compatible target-domain dataset that can be consumed by
``test_tiny_target_domain.py``.

Output layout:
    output_dir/
      test/<wnid>/images/*.jpeg
      images -> test
      class_to_idx.json
      idx_to_class.json
      class_to_wnid.json
      wnid_to_class.json
      tiny_to_imagenetv2_index.json
      dataset_manifest.jsonl
      build_validation_report.json
      build_validation_summary.txt

Important design choice:
    ``test_tiny_target_domain.py`` assumes Tiny-ImageNet-style 64x64 inputs for
    the ``tiny_imagenet`` source domain. ImageNetV2 images are larger and have
    varying aspect ratios, so this script center-crops and resizes them to
    64x64 by default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageOps


# Frozen from the official ImageNet 1k class-index JSON:
# https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
TINY_WNID_TO_IMAGENET1K_IDX: Dict[str, int] = {
    "n01443537": 1,
    "n01629819": 25,
    "n01641577": 30,
    "n01644900": 32,
    "n01698640": 50,
    "n01742172": 61,
    "n01768244": 69,
    "n01770393": 71,
    "n01774384": 75,
    "n01774750": 76,
    "n01784675": 79,
    "n01855672": 99,
    "n01882714": 105,
    "n01910747": 107,
    "n01917289": 109,
    "n01944390": 113,
    "n01945685": 114,
    "n01950731": 115,
    "n01983481": 122,
    "n01984695": 123,
    "n02002724": 128,
    "n02056570": 145,
    "n02058221": 146,
    "n02074367": 149,
    "n02085620": 151,
    "n02094433": 187,
    "n02099601": 207,
    "n02099712": 208,
    "n02106662": 235,
    "n02113799": 267,
    "n02123045": 281,
    "n02123394": 283,
    "n02124075": 285,
    "n02125311": 286,
    "n02129165": 291,
    "n02132136": 294,
    "n02165456": 301,
    "n02190166": 308,
    "n02206856": 309,
    "n02226429": 311,
    "n02231487": 313,
    "n02233338": 314,
    "n02236044": 315,
    "n02268443": 319,
    "n02279972": 323,
    "n02281406": 325,
    "n02321529": 329,
    "n02364673": 338,
    "n02395406": 341,
    "n02403003": 345,
    "n02410509": 347,
    "n02415577": 349,
    "n02423022": 353,
    "n02437312": 354,
    "n02480495": 365,
    "n02481823": 367,
    "n02486410": 372,
    "n02504458": 386,
    "n02509815": 387,
    "n02666196": 398,
    "n02669723": 400,
    "n02699494": 406,
    "n02730930": 411,
    "n02769748": 414,
    "n02788148": 421,
    "n02791270": 424,
    "n02793495": 425,
    "n02795169": 427,
    "n02802426": 430,
    "n02808440": 435,
    "n02814533": 436,
    "n02814860": 437,
    "n02815834": 438,
    "n02823428": 440,
    "n02837789": 445,
    "n02841315": 447,
    "n02843684": 448,
    "n02883205": 457,
    "n02892201": 458,
    "n02906734": 462,
    "n02909870": 463,
    "n02917067": 466,
    "n02927161": 467,
    "n02948072": 470,
    "n02950826": 471,
    "n02963159": 474,
    "n02977058": 480,
    "n02988304": 485,
    "n02999410": 488,
    "n03014705": 492,
    "n03026506": 496,
    "n03042490": 500,
    "n03085013": 508,
    "n03089624": 509,
    "n03100240": 511,
    "n03126707": 517,
    "n03160309": 525,
    "n03179701": 526,
    "n03201208": 532,
    "n03250847": 542,
    "n03255030": 543,
    "n03355925": 557,
    "n03388043": 562,
    "n03393912": 565,
    "n03400231": 567,
    "n03404251": 568,
    "n03424325": 570,
    "n03444034": 573,
    "n03447447": 576,
    "n03544143": 604,
    "n03584254": 605,
    "n03599486": 612,
    "n03617480": 614,
    "n03637318": 619,
    "n03649909": 621,
    "n03662601": 625,
    "n03670208": 627,
    "n03706229": 635,
    "n03733131": 645,
    "n03763968": 652,
    "n03770439": 655,
    "n03796401": 675,
    "n03804744": 677,
    "n03814639": 678,
    "n03837869": 682,
    "n03838899": 683,
    "n03854065": 687,
    "n03891332": 704,
    "n03902125": 707,
    "n03930313": 716,
    "n03937543": 720,
    "n03970156": 731,
    "n03976657": 733,
    "n03977966": 734,
    "n03980874": 735,
    "n03983396": 737,
    "n03992509": 739,
    "n04008634": 744,
    "n04023962": 747,
    "n04067472": 758,
    "n04070727": 760,
    "n04074963": 761,
    "n04099969": 765,
    "n04118538": 768,
    "n04133789": 774,
    "n04146614": 779,
    "n04149813": 781,
    "n04179913": 786,
    "n04251144": 801,
    "n04254777": 806,
    "n04259630": 808,
    "n04265275": 811,
    "n04275548": 815,
    "n04285008": 817,
    "n04311004": 821,
    "n04328186": 826,
    "n04356056": 837,
    "n04366367": 839,
    "n04371430": 842,
    "n04376876": 845,
    "n04398044": 849,
    "n04399382": 850,
    "n04417672": 853,
    "n04456115": 862,
    "n04465501": 866,
    "n04486054": 873,
    "n04487081": 874,
    "n04501370": 877,
    "n04507155": 879,
    "n04532106": 887,
    "n04532670": 888,
    "n04540053": 890,
    "n04560804": 899,
    "n04562935": 900,
    "n04596742": 909,
    "n04597913": 910,
    "n06596364": 917,
    "n07579787": 923,
    "n07583066": 924,
    "n07614500": 928,
    "n07615774": 929,
    "n07695742": 932,
    "n07711569": 935,
    "n07715103": 938,
    "n07720875": 945,
    "n07734744": 947,
    "n07747607": 950,
    "n07749582": 951,
    "n07753592": 954,
    "n07768694": 957,
    "n07871810": 962,
    "n07873807": 963,
    "n07875152": 964,
    "n07920052": 967,
    "n09193705": 970,
    "n09246464": 972,
    "n09256479": 973,
    "n09332890": 975,
    "n09428293": 978,
    "n12267677": 988,
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_words(words_path: Path) -> Dict[str, List[str]]:
    wnid_to_synonyms: Dict[str, List[str]] = {}
    with words_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or "\t" not in line:
                continue
            wnid, names = line.split("\t", 1)
            synonyms = [x.strip() for x in names.split(",") if x.strip()]
            wnid_to_synonyms[wnid] = synonyms
    return wnid_to_synonyms


def load_tiny_metadata(tiny_root: Path, class_map_path: Path | None) -> Tuple[List[str], Dict[str, List[str]]]:
    train_dir = tiny_root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Tiny-ImageNet train dir not found: {train_dir}")

    sorted_train_wnids = sorted(
        d.name for d in train_dir.iterdir()
        if d.is_dir()
    )
    if len(sorted_train_wnids) != 200:
        raise ValueError(f"Expected 200 Tiny-ImageNet train classes, got {len(sorted_train_wnids)}")

    if class_map_path is not None and class_map_path.exists():
        payload = json.loads(class_map_path.read_text(encoding="utf-8"))
        wnid_to_synonyms = {
            row["wnid"]: row.get("synonyms", [row["name"]])
            for row in payload["classes"]
        }
    else:
        words_path = tiny_root / "words.txt"
        if not words_path.exists():
            raise FileNotFoundError(
                f"Missing class map and Tiny words.txt: {words_path}"
            )
        wnid_to_synonyms = parse_words(words_path)

    return sorted_train_wnids, wnid_to_synonyms


def iter_images(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def resize_image(img: Image.Image, size: int, mode: str) -> Image.Image:
    if size <= 0:
        return img.convert("RGB")

    resampling_ns = getattr(Image, "Resampling", Image)
    resample = resampling_ns.BICUBIC
    rgb = img.convert("RGB")
    if mode == "fit":
        return ImageOps.fit(rgb, (size, size), method=resample, centering=(0.5, 0.5))
    if mode == "stretch":
        return rgb.resize((size, size), resample=resample)
    raise ValueError(f"Unsupported resize mode: {mode}")


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dir already exists: {path}. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_dataset(
    src_root: Path,
    tiny_root: Path,
    output_dir: Path,
    class_map_path: Path | None,
    image_size: int,
    resize_mode: str,
    max_images_per_class: int | None,
    jpeg_quality: int,
    overwrite: bool,
) -> Dict[str, object]:
    sorted_train_wnids, wnid_to_synonyms = load_tiny_metadata(tiny_root, class_map_path)

    missing_mapping = sorted(set(sorted_train_wnids) - set(TINY_WNID_TO_IMAGENET1K_IDX))
    if missing_mapping:
        raise ValueError(f"Frozen ImageNet1k mapping is missing Tiny wnids: {missing_mapping[:10]}")

    ensure_clean_dir(output_dir, overwrite=overwrite)
    test_root = output_dir / "test"
    test_root.mkdir(parents=True, exist_ok=True)

    class_to_idx = {wnid: idx for idx, wnid in enumerate(sorted_train_wnids)}
    idx_to_class = {
        str(idx): {
            "wnid": wnid,
            "name": wnid_to_synonyms.get(wnid, [wnid])[0],
            "imagenetv2_index": TINY_WNID_TO_IMAGENET1K_IDX[wnid],
        }
        for wnid, idx in class_to_idx.items()
    }
    wnid_to_class = {wnid: wnid_to_synonyms.get(wnid, [wnid])[0] for wnid in sorted_train_wnids}
    class_to_wnid = {name: wnid for wnid, name in wnid_to_class.items()}

    manifest_rows: List[Dict[str, object]] = []
    per_class_counts: Dict[str, int] = {}
    missing_source_dirs: List[Dict[str, object]] = []
    bad_images: List[Dict[str, object]] = []

    for wnid in sorted_train_wnids:
        imagenet_idx = TINY_WNID_TO_IMAGENET1K_IDX[wnid]
        src_dir = src_root / str(imagenet_idx)
        if not src_dir.is_dir():
            missing_source_dirs.append({"wnid": wnid, "imagenetv2_index": imagenet_idx, "src_dir": str(src_dir)})
            per_class_counts[wnid] = 0
            continue

        dst_dir = test_root / wnid / "images"
        dst_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for i, src_path in enumerate(iter_images(src_dir)):
            if max_images_per_class is not None and i >= max_images_per_class:
                break
            try:
                with Image.open(src_path) as img:
                    orig_w, orig_h = img.size
                    out_img = resize_image(img, size=image_size, mode=resize_mode)
            except Exception as exc:
                bad_images.append({
                    "wnid": wnid,
                    "imagenetv2_index": imagenet_idx,
                    "src_path": str(src_path),
                    "reason": str(exc),
                })
                continue

            dst_name = f"{src_path.stem}.jpeg"
            dst_path = dst_dir / dst_name
            out_img.save(dst_path, format="JPEG", quality=jpeg_quality)
            count += 1

            manifest_rows.append({
                "wnid": wnid,
                "tiny_label_idx": class_to_idx[wnid],
                "class_name": wnid_to_class[wnid],
                "imagenetv2_index": imagenet_idx,
                "src_path": str(src_path),
                "dst_path": str(dst_path),
                "orig_size": [orig_w, orig_h],
                "output_size": list(out_img.size),
            })

        per_class_counts[wnid] = count

    images_alias = output_dir / "images"
    if images_alias.exists() or images_alias.is_symlink():
        if images_alias.is_symlink() or images_alias.is_file():
            images_alias.unlink()
        else:
            shutil.rmtree(images_alias)
    try:
        images_alias.symlink_to(test_root, target_is_directory=True)
    except OSError:
        shutil.copytree(test_root, images_alias)

    write_json(output_dir / "class_to_idx.json", class_to_idx)
    write_json(output_dir / "idx_to_class.json", idx_to_class)
    write_json(output_dir / "wnid_to_class.json", wnid_to_class)
    write_json(output_dir / "class_to_wnid.json", class_to_wnid)
    write_json(output_dir / "tiny_to_imagenetv2_index.json", TINY_WNID_TO_IMAGENET1K_IDX)

    manifest_path = output_dir / "dataset_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = list(per_class_counts.values())
    report = {
        "num_classes_expected": 200,
        "num_classes_written": len(per_class_counts),
        "total_images_written": len(manifest_rows),
        "per_class_min": min(counts) if counts else 0,
        "per_class_max": max(counts) if counts else 0,
        "per_class_mean": round(mean(counts), 2) if counts else 0,
        "classes_with_zero_images": sorted([wnid for wnid, c in per_class_counts.items() if c == 0]),
        "missing_source_dirs": missing_source_dirs,
        "bad_images_count": len(bad_images),
        "bad_images_sample": bad_images[:20],
        "image_size": image_size,
        "resize_mode": resize_mode,
        "max_images_per_class": max_images_per_class,
        "jpeg_quality": jpeg_quality,
        "class_to_idx_matches_sorted_tiny_train": (
            class_to_idx == {wnid: i for i, wnid in enumerate(sorted_train_wnids)}
        ),
        "recommended_target_domain_dir": str(output_dir),
        "recommended_imagefolder_root": str(output_dir / "images"),
        "compatible_test_script": "test_tiny_target_domain.py",
    }
    write_json(output_dir / "build_validation_report.json", report)

    summary_lines = [
        "ImageNetV2 -> Tiny-ImageNet target-domain extraction summary",
        f"output_dir: {output_dir}",
        f"classes: {len(per_class_counts)}",
        f"images_written: {len(manifest_rows)}",
        f"per_class_min/max/mean: {report['per_class_min']}/{report['per_class_max']}/{report['per_class_mean']}",
        f"zero_image_classes: {len(report['classes_with_zero_images'])}",
        f"image_size: {image_size}",
        f"resize_mode: {resize_mode}",
        f"recommended_test_cmd_target_domain_dir: {output_dir}",
    ]
    (output_dir / "build_validation_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the Tiny-ImageNet 200-class subset from ImageNetV2."
    )
    parser.add_argument(
        "--src-root",
        type=str,
        default="/workspace/imagenetv2-matched-frequency-format-val",
        help="ImageNetV2 root with 0..999 class folders",
    )
    parser.add_argument(
        "--tiny-root",
        type=str,
        default="/workspace/backdoor-toolbox-new1/data/Tiny-imagenet/tiny-imagenet-200",
        help="Tiny-ImageNet root used as the source of class order and metadata",
    )
    parser.add_argument(
        "--class-map",
        type=str,
        default="/workspace/data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json",
        help="Optional Tiny-ImageNet class_map_tiny_imagenet_200.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/data/imagenetv2-matched-frequency-tiny-organized",
        help="Output dataset root compatible with test_tiny_target_domain.py",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Final square image size. Use 64 for direct Tiny-ImageNet testing.",
    )
    parser.add_argument(
        "--resize-mode",
        type=str,
        default="fit",
        choices=["fit", "stretch"],
        help="fit=center-crop to square then resize, stretch=direct resize",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Optional cap on how many ImageNetV2 images to keep per class",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for resized output files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild output-dir if it already exists",
    )
    args = parser.parse_args()

    report = build_dataset(
        src_root=Path(args.src_root).resolve(),
        tiny_root=Path(args.tiny_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        class_map_path=Path(args.class_map).resolve() if args.class_map else None,
        image_size=args.image_size,
        resize_mode=args.resize_mode,
        max_images_per_class=args.max_images_per_class,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )

    print("[OK] ImageNetV2 Tiny target-domain dataset is ready.")
    print(f"[OK] output_dir={Path(args.output_dir).resolve()}")
    print(f"[OK] total_images_written={report['total_images_written']}")
    print(
        f"[OK] per_class_min/max/mean="
        f"{report['per_class_min']}/{report['per_class_max']}/{report['per_class_mean']}"
    )
    if report["classes_with_zero_images"]:
        print(f"[WARN] classes_with_zero_images={len(report['classes_with_zero_images'])}")
    print(
        "[OK] Compatible with test_tiny_target_domain.py via "
        f"-target_domain_dir {Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
