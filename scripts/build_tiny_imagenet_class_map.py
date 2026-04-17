#!/usr/bin/env python3
"""Build Tiny-ImageNet 200-class taxonomy mapping.

Output JSON schema:
{
  "source_dir": ".../tiny-imagenet-200",
  "num_classes": 200,
  "classes": [
    {"index": 0, "wnid": "n01443537", "name": "goldfish", "synonyms": ["goldfish", "..."]},
    ...
  ]
}
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from urllib.request import urlopen


def parse_words(words_path: Path) -> Dict[str, List[str]]:
    wnid_to_synonyms: Dict[str, List[str]] = {}
    with words_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "\t" not in line:
                continue
            wnid, names = line.split("\t", 1)
            synonyms = [x.strip() for x in names.split(",") if x.strip()]
            wnid_to_synonyms[wnid] = synonyms
    return wnid_to_synonyms


def parse_words_from_text(text: str) -> Dict[str, List[str]]:
    wnid_to_synonyms: Dict[str, List[str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "\t" not in line:
            continue
        wnid, names = line.split("\t", 1)
        synonyms = [x.strip() for x in names.split(",") if x.strip()]
        wnid_to_synonyms[wnid] = synonyms
    return wnid_to_synonyms


def download_official_metadata() -> Dict[str, str]:
    urls = {
        "wnids": "https://raw.githubusercontent.com/rmccorm4/Tiny-Imagenet-200/master/sets/wnids.txt",
        "words": "https://raw.githubusercontent.com/rmccorm4/Tiny-Imagenet-200/master/sets/words.txt",
    }
    out = {}
    for key, url in urls.items():
        with urlopen(url, timeout=20) as r:  # nosec B310 - trusted static metadata files
            out[key] = r.read().decode("utf-8")
    return out


def build_class_map(dataset_dir: Path, allow_download: bool = False) -> Dict:
    wnids_path = dataset_dir / "wnids.txt"
    words_path = dataset_dir / "words.txt"
    classes = []
    if wnids_path.exists() and words_path.exists():
        wnid_to_synonyms = parse_words(words_path)
        wnids = [x.strip() for x in wnids_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    elif allow_download:
        remote = download_official_metadata()
        wnid_to_synonyms = parse_words_from_text(remote["words"])
        wnids = [x.strip() for x in remote["wnids"].splitlines() if x.strip()]
    else:
        raise FileNotFoundError(
            f"Missing {wnids_path} / {words_path}. "
            "Use --allow-download-fallback to fetch official metadata."
        )

    for idx, wnid in enumerate(wnids):
        synonyms = wnid_to_synonyms.get(wnid, [wnid])
        classes.append(
            {
                "index": idx,
                "wnid": wnid,
                "name": synonyms[0],
                "synonyms": synonyms,
            }
        )
    return {
        "source_dir": str(dataset_dir),
        "num_classes": len(classes),
        "classes": classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Tiny-ImageNet class map JSON.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/Tiny-imagenet/tiny-imagenet-200",
        help="Tiny-ImageNet root containing wnids.txt and words.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--allow-download-fallback",
        action="store_true",
        help="Download official wnids/words from public GitHub when local files are missing",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_class_map(dataset_dir, allow_download=args.allow_download_fallback)
    if payload["num_classes"] != 200:
        print(f"[WARN] Expected 200 classes, got {payload['num_classes']}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved class map: {output_path}")
    print(f"[OK] num_classes={payload['num_classes']}")


if __name__ == "__main__":
    main()

