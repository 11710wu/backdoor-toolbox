#!/usr/bin/env python3
"""Analyze domain shift between Tiny-ImageNet train and a generated target domain.

This script is designed for the Tiny target-domain workflow in this repo.
It compares:

- source domain: Tiny-ImageNet train set
- target domain: generated 64x64 target-domain images

It computes three families of measurements:

1. Domain-classifier test
2. Feature-space visualization
3. Same-class cross-domain distance vs different-class distance

Important behavior:
- Only target files named ``image_*`` are used by default.
- Files named ``raw_image_*`` are ignored.
- No torch dependency is required. The default feature backend is:
  64x64 RGB pixels -> flatten -> standardize -> PCA

Example:
  python scripts/analyze_tiny_target_domain_shift.py \
    --source-train /workspace/backdoor-toolbox-new1/data/Tiny-imagenet/tiny-imagenet-200/train \
    --target-images /workspace/data/tiny-target-domain-qwen-200class-inline-raw641/images \
    --class-map /workspace/data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json \
    --output-dir /workspace/backdoor-toolbox-new1/scripts/domain_shift_qwen200_vs_tiny_train
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class ClassEntry:
    index: int
    wnid: str
    name: str
    synonyms: Tuple[str, ...]


@dataclass(frozen=True)
class ImageRecord:
    domain: str
    wnid: str
    class_name: str
    path: str


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def load_class_map(path: Path) -> List[ClassEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = [
        ClassEntry(
            index=int(c["index"]),
            wnid=str(c["wnid"]),
            name=str(c["name"]),
            synonyms=tuple(str(x) for x in c.get("synonyms", [])),
        )
        for c in payload["classes"]
    ]
    classes.sort(key=lambda c: c.index)
    if len(classes) != 200:
        raise ValueError(f"Expected 200 classes in class_map, got {len(classes)}")
    return classes


def build_name_to_entry(classes: Sequence[ClassEntry]) -> Dict[str, ClassEntry]:
    lookup: Dict[str, ClassEntry] = {}
    for entry in classes:
        lookup[sanitize_name(entry.name)] = entry
        for syn in entry.synonyms:
            key = sanitize_name(syn)
            lookup.setdefault(key, entry)
    return lookup


def collect_target_records(
    target_images: Path,
    classes: Sequence[ClassEntry],
) -> List[ImageRecord]:
    name_lookup = build_name_to_entry(classes)
    entries: List[ImageRecord] = []
    for class_dir in sorted(p for p in target_images.iterdir() if p.is_dir()):
        entry = name_lookup.get(class_dir.name) or name_lookup.get(sanitize_name(class_dir.name))
        if entry is None:
            raise ValueError(f"Unrecognized target class directory: {class_dir.name}")
        candidates = sorted(
            p for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            and p.name.startswith("image_")
        )
        if not candidates:
            raise ValueError(
                f"No target 64x64 image_* file found under {class_dir}. "
                "This script intentionally ignores raw_image_* files."
            )
        if len(candidates) > 1:
            # For this analysis we keep the first 64x64 target image per class.
            candidates = candidates[:1]
        for img_path in candidates:
            entries.append(
                ImageRecord(
                    domain="target",
                    wnid=entry.wnid,
                    class_name=entry.name,
                    path=str(img_path),
                )
            )
    entries.sort(key=lambda r: next(c.index for c in classes if c.wnid == r.wnid))
    if len(entries) != 200:
        raise ValueError(f"Expected 200 target records, got {len(entries)}")
    return entries


def collect_source_records(
    source_train: Path,
    classes: Sequence[ClassEntry],
    samples_per_class: int,
    rng: random.Random,
) -> List[ImageRecord]:
    entries: List[ImageRecord] = []
    for entry in classes:
        class_dir = source_train / entry.wnid / "images"
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing Tiny-ImageNet train image dir: {class_dir}")
        candidates = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png", ".bmp", ".webp"}
        )
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"Class {entry.wnid} has only {len(candidates)} source images; "
                f"cannot sample {samples_per_class}"
            )
        chosen = rng.sample(candidates, samples_per_class)
        chosen.sort()
        for img_path in chosen:
            entries.append(
                ImageRecord(
                    domain="source",
                    wnid=entry.wnid,
                    class_name=entry.name,
                    path=str(img_path),
                )
            )
    return entries


def load_image_feature(path: str, image_size: int = 64) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def extract_pixel_features(records: Sequence[ImageRecord], image_size: int = 64) -> np.ndarray:
    feats = [load_image_feature(r.path, image_size=image_size) for r in records]
    return np.stack(feats, axis=0)


if TORCH_AVAILABLE:
    class ImagePathDataset(Dataset):
        def __init__(self, records: Sequence[ImageRecord], transform):
            self.records = list(records)
            self.transform = transform

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int):
            rec = self.records[idx]
            with Image.open(rec.path) as img:
                img = img.convert("RGB")
                x = self.transform(img)
            return x


def extract_deep_features_resnet18_tiny(
    records: Sequence[ImageRecord],
    checkpoint_path: str,
    batch_size: int = 128,
) -> np.ndarray:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch/torchvision not available; cannot use deep feature backend")

    import sys

    repo_root = "/workspace/backdoor-toolbox-new1"
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from utils import resnet

    model = resnet.ResNet18_tiny_imagenet(num_classes=200)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tfm = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    )
    ds = ImagePathDataset(records, tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    feats: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, hidden = model(batch, return_hidden=True)
            feats.append(hidden.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


def prepare_feature_space(
    raw_features: np.ndarray,
    pca_components: int,
) -> Tuple[np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(raw_features)
    n_components = min(pca_components, scaled.shape[0] - 1, scaled.shape[1])
    if n_components < 2:
        raise ValueError("Not enough samples to fit PCA")
    pca = PCA(n_components=n_components, random_state=0)
    embedded = pca.fit_transform(scaled)
    return scaled.astype(np.float32), embedded.astype(np.float32), scaler, pca


def build_record_table(records: Sequence[ImageRecord], features: np.ndarray, embedded: np.ndarray) -> List[Dict]:
    table: List[Dict] = []
    for rec, feat, emb in zip(records, features, embedded):
        table.append(
            {
                "domain": rec.domain,
                "wnid": rec.wnid,
                "class_name": rec.class_name,
                "path": rec.path,
                "feature": feat,
                "pca_feature": emb,
            }
        )
    return table


def run_domain_classifier(
    rows: Sequence[Dict],
    repeats: int,
    seed: int,
) -> Dict:
    rng = random.Random(seed)
    target_rows = [r for r in rows if r["domain"] == "target"]
    source_by_wnid: Dict[str, List[Dict]] = {}
    for r in rows:
        if r["domain"] != "source":
            continue
        source_by_wnid.setdefault(r["wnid"], []).append(r)

    accs: List[float] = []
    aucs: List[float] = []
    for rep in range(repeats):
        chosen_source: List[Dict] = []
        for target_row in target_rows:
            pool = source_by_wnid[target_row["wnid"]]
            chosen_source.append(rng.choice(pool))

        batch = chosen_source + target_rows
        X = np.stack([r["feature"] for r in batch], axis=0)
        y = np.array([0] * len(chosen_source) + [1] * len(target_rows), dtype=np.int64)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=y,
            random_state=seed + rep,
        )
        clf = LogisticRegression(max_iter=2000, random_state=seed + rep)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)[:, 1]
        accs.append(float(accuracy_score(y_test, pred)))
        aucs.append(float(roc_auc_score(y_test, prob)))

    return {
        "repeats": repeats,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "accuracy_values": [round(x, 6) for x in accs],
        "auc_values": [round(x, 6) for x in aucs],
    }


def compute_distance_stats(rows: Sequence[Dict], class_order: Sequence[ClassEntry]) -> Tuple[List[Dict], Dict]:
    source_by_wnid: Dict[str, np.ndarray] = {}
    target_by_wnid: Dict[str, np.ndarray] = {}
    for entry in class_order:
        source_feats = [r["feature"] for r in rows if r["domain"] == "source" and r["wnid"] == entry.wnid]
        target_feats = [r["feature"] for r in rows if r["domain"] == "target" and r["wnid"] == entry.wnid]
        if not source_feats:
            raise ValueError(f"No source features for {entry.wnid}")
        if len(target_feats) != 1:
            raise ValueError(f"Expected exactly one target feature for {entry.wnid}, got {len(target_feats)}")
        source_by_wnid[entry.wnid] = np.stack(source_feats, axis=0)
        target_by_wnid[entry.wnid] = np.stack(target_feats, axis=0)

    all_target = np.stack([target_by_wnid[e.wnid][0] for e in class_order], axis=0)
    all_target_wnids = [e.wnid for e in class_order]

    per_class: List[Dict] = []
    same_vals: List[float] = []
    diff_vals: List[float] = []
    for entry in class_order:
        src = source_by_wnid[entry.wnid]
        tgt_same = target_by_wnid[entry.wnid]
        same_cross = float(cdist(src, tgt_same, metric="cosine").mean())

        other_idx = [i for i, wnid in enumerate(all_target_wnids) if wnid != entry.wnid]
        tgt_other = all_target[other_idx]
        diff_cross = float(cdist(src, tgt_other, metric="cosine").mean())
        ratio = same_cross / diff_cross if diff_cross > 1e-12 else math.inf
        margin = diff_cross - same_cross

        row = {
            "index": entry.index,
            "wnid": entry.wnid,
            "class_name": entry.name,
            "same_cross_cosine": same_cross,
            "diff_cross_cosine": diff_cross,
            "ratio_same_over_diff": ratio,
            "margin_diff_minus_same": margin,
        }
        per_class.append(row)
        same_vals.append(same_cross)
        diff_vals.append(diff_cross)

    ratio_vals = [r["ratio_same_over_diff"] for r in per_class if np.isfinite(r["ratio_same_over_diff"])]
    summary = {
        "same_cross_mean": float(np.mean(same_vals)),
        "same_cross_std": float(np.std(same_vals)),
        "diff_cross_mean": float(np.mean(diff_vals)),
        "diff_cross_std": float(np.std(diff_vals)),
        "ratio_mean": float(np.mean(ratio_vals)),
        "ratio_std": float(np.std(ratio_vals)),
        "fraction_ratio_below_1": float(np.mean([x < 1.0 for x in ratio_vals])),
        "fraction_same_less_than_diff": float(
            np.mean([r["same_cross_cosine"] < r["diff_cross_cosine"] for r in per_class])
        ),
    }
    return per_class, summary


def make_pca_domain_plot(rows: Sequence[Dict], output_path: Path) -> None:
    X = np.stack([r["pca_feature"][:2] for r in rows], axis=0)
    domains = [r["domain"] for r in rows]
    plt.figure(figsize=(8, 6))
    for domain, color in [("source", "#1f77b4"), ("target", "#d62728")]:
        mask = np.array([d == domain for d in domains])
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            s=16 if domain == "source" else 28,
            alpha=0.55 if domain == "source" else 0.9,
            c=color,
            label=domain,
        )
    plt.title("PCA Feature Space: Source vs Target")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_pca_class_shift_plot(rows: Sequence[Dict], class_order: Sequence[ClassEntry], output_path: Path) -> None:
    source_centroid: Dict[str, np.ndarray] = {}
    target_point: Dict[str, np.ndarray] = {}
    for entry in class_order:
        src = np.stack([r["pca_feature"][:2] for r in rows if r["domain"] == "source" and r["wnid"] == entry.wnid], axis=0)
        tgt = np.stack([r["pca_feature"][:2] for r in rows if r["domain"] == "target" and r["wnid"] == entry.wnid], axis=0)
        source_centroid[entry.wnid] = src.mean(axis=0)
        target_point[entry.wnid] = tgt[0]

    plt.figure(figsize=(9, 8))
    for entry in class_order:
        s = source_centroid[entry.wnid]
        t = target_point[entry.wnid]
        plt.plot([s[0], t[0]], [s[1], t[1]], color="#b0b0b0", linewidth=0.6, alpha=0.7)
    sxy = np.stack([source_centroid[e.wnid] for e in class_order], axis=0)
    txy = np.stack([target_point[e.wnid] for e in class_order], axis=0)
    plt.scatter(sxy[:, 0], sxy[:, 1], c="#1f77b4", s=18, alpha=0.65, label="source centroid")
    plt.scatter(txy[:, 0], txy[:, 1], c="#d62728", s=22, alpha=0.85, label="target point")
    plt.title("Per-Class Shift in PCA Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_tsne_domain_plot(rows: Sequence[Dict], output_path: Path, seed: int) -> None:
    X = np.stack([r["feature"] for r in rows], axis=0)
    n_samples = X.shape[0]
    perplexity = max(5, min(30, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    Z = tsne.fit_transform(X)
    domains = [r["domain"] for r in rows]
    plt.figure(figsize=(8, 6))
    for domain, color in [("source", "#1f77b4"), ("target", "#d62728")]:
        mask = np.array([d == domain for d in domains])
        plt.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=16 if domain == "source" else 28,
            alpha=0.55 if domain == "source" else 0.9,
            c=color,
            label=domain,
        )
    plt.title("t-SNE Feature Space: Source vs Target")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def make_ratio_plot(per_class_rows: Sequence[Dict], output_path: Path) -> None:
    rows = sorted(per_class_rows, key=lambda r: r["ratio_same_over_diff"])
    labels = [r["class_name"] for r in rows]
    values = [r["ratio_same_over_diff"] for r in rows]
    plt.figure(figsize=(14, 5))
    plt.bar(np.arange(len(rows)), values, color="#4c72b0")
    plt.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label="ratio = 1")
    plt.xticks(np.arange(len(rows)), labels, rotation=90, fontsize=6)
    plt.ylabel("same_cross / diff_cross")
    plt.title("Per-Class Cross-Domain Distance Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_summary(
    path: Path,
    config: Dict,
    domain_result: Dict,
    distance_summary: Dict,
    worst_rows: Sequence[Dict],
) -> None:
    lines = [
        "# Tiny Target-Domain Shift Analysis",
        "",
        "## Config",
        "",
        f"- source_train: `{config['source_train']}`",
        f"- target_images: `{config['target_images']}`",
        f"- class_map: `{config['class_map']}`",
        f"- source_samples_per_class: `{config['source_samples_per_class']}`",
        f"- feature_backend: `{config['feature_backend']}`",
        f"- pca_components: `{config['pca_components']}`",
        "",
        "## Domain Classifier",
        "",
        f"- accuracy_mean: `{domain_result['accuracy_mean']:.6f}`",
        f"- accuracy_std: `{domain_result['accuracy_std']:.6f}`",
        f"- auc_mean: `{domain_result['auc_mean']:.6f}`",
        f"- auc_std: `{domain_result['auc_std']:.6f}`",
        "",
        "## Cross-Domain Distance",
        "",
        f"- same_cross_mean: `{distance_summary['same_cross_mean']:.6f}`",
        f"- diff_cross_mean: `{distance_summary['diff_cross_mean']:.6f}`",
        f"- ratio_mean: `{distance_summary['ratio_mean']:.6f}`",
        f"- fraction_ratio_below_1: `{distance_summary['fraction_ratio_below_1']:.6f}`",
        f"- fraction_same_less_than_diff: `{distance_summary['fraction_same_less_than_diff']:.6f}`",
        "",
        "## Hardest Classes",
        "",
        "| class_name | wnid | same_cross | diff_cross | ratio |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in worst_rows:
        lines.append(
            f"| {row['class_name']} | {row['wnid']} | "
            f"{row['same_cross_cosine']:.6f} | {row['diff_cross_cosine']:.6f} | "
            f"{row['ratio_same_over_diff']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze domain shift for Tiny target-domain data.")
    parser.add_argument(
        "--source-train",
        type=str,
        default="/workspace/backdoor-toolbox-new1/data/Tiny-imagenet/tiny-imagenet-200/train",
    )
    parser.add_argument(
        "--target-images",
        type=str,
        default="/workspace/data/tiny-target-domain-qwen-200class-inline-raw641/images",
    )
    parser.add_argument(
        "--class-map",
        type=str,
        default="/workspace/data/tiny-target-domain-preview-1shot/class_map_tiny_imagenet_200.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/backdoor-toolbox-new1/scripts/domain_shift_qwen200_vs_tiny_train",
    )
    parser.add_argument(
        "--feature-backend",
        type=str,
        choices=["pixel_flatten_then_pca", "tiny_resnet18_hidden"],
        default="pixel_flatten_then_pca",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/workspace/backdoor-toolbox-new1/poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=ResNet18_tiny_imagenet/ResNet18_tiny_imagenet.pt",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--source-samples-per-class", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--domain-classifier-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260420)
    args = parser.parse_args()

    source_train = Path(args.source_train).resolve()
    target_images = Path(args.target_images).resolve()
    class_map = Path(args.class_map).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    classes = load_class_map(class_map)
    target_records = collect_target_records(target_images, classes)
    source_records = collect_source_records(source_train, classes, args.source_samples_per_class, rng)
    all_records = source_records + target_records

    print(f"[INFO] source records: {len(source_records)}")
    print(f"[INFO] target records: {len(target_records)}")
    if args.feature_backend == "pixel_flatten_then_pca":
        print(f"[INFO] extracting 64x64 pixel features from {len(all_records)} images")
        raw_features = extract_pixel_features(all_records, image_size=args.image_size)
    else:
        print(f"[INFO] extracting deep ResNet18 hidden features from {len(all_records)} images")
        raw_features = extract_deep_features_resnet18_tiny(
            all_records,
            checkpoint_path=args.checkpoint_path,
            batch_size=args.batch_size,
        )
    features, embedded, scaler, pca = prepare_feature_space(raw_features, pca_components=args.pca_components)
    rows = build_record_table(all_records, features, embedded)

    print("[INFO] running domain classifier")
    domain_result = run_domain_classifier(
        rows,
        repeats=args.domain_classifier_repeats,
        seed=args.seed,
    )

    print("[INFO] computing cross-domain distance statistics")
    per_class_rows, distance_summary = compute_distance_stats(rows, classes)

    print("[INFO] generating plots")
    make_pca_domain_plot(rows, output_dir / "pca_domain_scatter.png")
    make_pca_class_shift_plot(rows, classes, output_dir / "pca_class_shift.png")
    make_tsne_domain_plot(rows, output_dir / "tsne_domain_scatter.png", seed=args.seed)
    make_ratio_plot(per_class_rows, output_dir / "distance_ratio_per_class.png")

    config_payload = {
        "source_train": str(source_train),
        "target_images": str(target_images),
        "class_map": str(class_map),
        "source_samples_per_class": args.source_samples_per_class,
        "image_size": args.image_size,
        "feature_backend": args.feature_backend,
        "checkpoint_path": args.checkpoint_path if args.feature_backend != "pixel_flatten_then_pca" else None,
        "batch_size": args.batch_size if args.feature_backend != "pixel_flatten_then_pca" else None,
        "pca_components": min(args.pca_components, raw_features.shape[0] - 1, raw_features.shape[1]),
        "domain_classifier_repeats": args.domain_classifier_repeats,
        "seed": args.seed,
        "target_image_filter": "image_* only",
    }

    worst_rows = sorted(per_class_rows, key=lambda r: r["ratio_same_over_diff"], reverse=True)[:20]

    (output_dir / "analysis_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "domain_classifier_result.json").write_text(
        json.dumps(domain_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "distance_summary.json").write_text(
        json.dumps(distance_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "hardest_classes.json").write_text(
        json.dumps(worst_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_csv(
        output_dir / "per_class_distance_report.csv",
        per_class_rows,
        fieldnames=[
            "index",
            "wnid",
            "class_name",
            "same_cross_cosine",
            "diff_cross_cosine",
            "ratio_same_over_diff",
            "margin_diff_minus_same",
        ],
    )
    write_markdown_summary(
        output_dir / "summary.md",
        config_payload,
        domain_result,
        distance_summary,
        worst_rows,
    )

    print("[OK] output_dir:", output_dir)
    print(
        "[OK] domain classifier: "
        f"acc={domain_result['accuracy_mean']:.4f}, auc={domain_result['auc_mean']:.4f}"
    )
    print(
        "[OK] distance summary: "
        f"same_cross={distance_summary['same_cross_mean']:.4f}, "
        f"diff_cross={distance_summary['diff_cross_mean']:.4f}, "
        f"ratio={distance_summary['ratio_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
