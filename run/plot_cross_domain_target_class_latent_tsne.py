#!/usr/bin/env python3
"""Plot cross-domain target-class latent separability t-SNE panels.

This figure is an AC-style latent separability diagnostic adapted to dataset
transfer. For each attack setting, it jointly embeds four groups from the
source-trained poisoned model:

  A clean target:   source-domain clean target-class samples.
  A poison target:  source-domain payload poison samples.
  B clean target:   target-domain clean target-class samples.
  B poison target:  target-domain triggered non-target samples that transfer to
                    the target class by default.

The figure is qualitative mechanism evidence. It should not replace target-side
transfer ASR or detector TPR as the primary metrics.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PAPER_ANALYSIS = REPO / "analysis-transfer-asr2" / "paper_analysis"
if str(PAPER_ANALYSIS) not in sys.path:
    sys.path.append(str(PAPER_ANALYSIS))

import config  # noqa: E402
from utils import supervisor  # noqa: E402
from utils.tools import IMG_Dataset  # noqa: E402
from make_target_class_latent_tsne import (  # noqa: E402
    SPECS,
    find_model_path,
    get_poison_transform,
    load_model,
    make_args,
    split_target_class_indices,
)
from plot_cross_domain_trigger_response_and_retention import (  # noqa: E402
    DISPLAY_NAMES,
    FilteredSTL10,
    collect_group,
    sample_indices_by_label,
)


def normalize_2d(z: np.ndarray) -> np.ndarray:
    z = z.copy()
    z -= z.mean(axis=0, keepdims=True)
    scale = np.percentile(np.abs(z), 98)
    if scale > 0:
        z /= scale
    return z


def panel_tsne(features: np.ndarray, seed: int, perplexity: int) -> np.ndarray:
    x = StandardScaler().fit_transform(features.astype(np.float64))
    pca_dim = min(50, x.shape[1], x.shape[0] - 1)
    if pca_dim >= 2:
        x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    px = min(perplexity, max(5, (x.shape[0] - 1) // 5))
    z = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=px,
        max_iter=1000,
        random_state=seed,
        metric="euclidean",
    ).fit_transform(x)
    return normalize_2d(z)


def sample_list(values: list[int], max_n: int, seed: int) -> list[int]:
    values = list(values)
    if max_n and len(values) > max_n:
        rng = random.Random(seed)
        values = rng.sample(values, max_n)
    return sorted(values)


def selected_specs(attack_arg: str):
    if attack_arg.lower() in {"all", "*"}:
        keys = None
    else:
        keys = {x.strip() for x in attack_arg.split(",") if x.strip()}
    specs = [s for s in SPECS if s.key != "none"]
    if keys is not None:
        specs = [s for s in specs if s.key in keys or DISPLAY_NAMES.get(s.key, "").lower() in keys]
    return specs


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default="all", help="Comma-separated attack keys or all.")
    ap.add_argument("--dataset_source", default="cifar10")
    ap.add_argument("--dataset_target", default="stl10")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--poison_root", default="poisoned_train_set")
    ap.add_argument("--output_dir", default=str(REPO / "analysis-transfer-asr2/paper_analysis_outputs/dataset_transfer_mechanism"))
    ap.add_argument("--plot_mode", choices=["cross_domain", "target_only"], default="cross_domain")
    ap.add_argument("--max_source_clean", type=int, default=2200)
    ap.add_argument("--max_source_poison", type=int, default=250)
    ap.add_argument("--max_target_clean", type=int, default=450)
    ap.add_argument("--max_target_poison", type=int, default=50)
    ap.add_argument("--target_clean_scope", choices=["target_class", "all"], default="target_class")
    ap.add_argument("--target_poison_candidates", type=int, default=1800)
    ap.add_argument("--target_poison_mode", choices=["success", "all"], default="success")
    ap.add_argument("--feature_layer", choices=["penultimate", "layer4"], default="penultimate")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--perplexity", type=int, default=35)
    ap.add_argument("--seed", type=int, default=2333)
    args = ap.parse_args()

    if args.dataset_source != "cifar10" or args.dataset_target != "stl10" or args.arch != "resnet18":
        raise NotImplementedError("This script currently supports CIFAR10 -> STL10 with ResNet18.")

    os.environ["POISONED_TRAIN_SET_ROOT"] = args.poison_root
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_class = int(config.target_class[args.dataset_source])
    poisoned_root = REPO / args.poison_root / args.dataset_source
    clean_dir = poisoned_root / "none_0.000_poison_seed=2333_arch=ResNet18_cifar10"
    if not clean_dir.exists():
        raise FileNotFoundError(clean_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panels = []
    count_rows = []
    point_rows = []

    # Source images: prefer saved clean_dir/data; fall back to torchvision CIFAR-10 train
    # (same 50k order used when creating poisoned_train_set).
    import torchvision

    clean_data_dir = clean_dir / "data"
    clean_label_path = clean_dir / "labels"
    use_torchvision_source = not clean_data_dir.exists()
    if use_torchvision_source:
        print(f"[info] {clean_data_dir} missing; using torchvision CIFAR-10 train as source images")

    for panel_idx, spec in enumerate(selected_specs(args.attack)):
        attack_dir = poisoned_root / spec.dirname
        if not attack_dir.exists():
            print(f"[warning] skip missing attack directory: {attack_dir}")
            continue
        spec_args = make_args(spec)
        data_transform = supervisor.get_transforms(spec_args)[1]
        skip_normalize = spec.poison_type in {"upgd", "belt"}
        target_dataset = FilteredSTL10(REPO / "data", skip_normalize=skip_normalize)
        is_normalized_input = spec.poison_type not in {"upgd", "belt"} and not spec_args.no_normalize
        poison_transform = get_poison_transform(spec, spec_args, is_normalized_input=is_normalized_input)
        model = load_model(find_model_path(attack_dir), spec_args, device)

        source_clean_all, source_payload_all = [], []
        a_clean = a_poison = None
        if args.plot_mode == "cross_domain":
            if use_torchvision_source:
                source_dataset = torchvision.datasets.CIFAR10(
                    root=str(REPO / "data" / "cifar10"),
                    train=True,
                    download=False,
                    transform=data_transform,
                )
            else:
                source_dataset = IMG_Dataset(str(clean_data_dir), str(clean_label_path), transforms=data_transform)
            source_clean_all, source_payload_all, _ = split_target_class_indices(attack_dir, spec)
            source_clean_idx = sample_list(source_clean_all, args.max_source_clean, args.seed + 100 + panel_idx)
            source_payload_idx = sample_list(source_payload_all, args.max_source_poison, args.seed + 200 + panel_idx)
            a_clean = collect_group(
                model, source_dataset, source_clean_idx, None, target_class, args.feature_layer,
                args.batch_size, device, "A_clean_target",
            )
            a_poison = collect_group(
                model, source_dataset, source_payload_idx, poison_transform, target_class, args.feature_layer,
                args.batch_size, device, "A_poison_target",
            )
        if args.target_clean_scope == "all" and args.plot_mode == "target_only":
            target_clean_idx = sample_list(list(range(len(target_dataset))), args.max_target_clean, args.seed + 300 + panel_idx)
            target_clean_group = ("B clean all", "target_clean_all")
        else:
            target_clean_idx = sample_indices_by_label(
                target_dataset, target_class, True, args.max_target_clean, args.seed + 300 + panel_idx
            )
            target_clean_group = ("B clean target", "target_clean_target")
        target_candidate_idx = sample_indices_by_label(
            target_dataset, target_class, False, args.target_poison_candidates, args.seed + 400 + panel_idx
        )

        b_clean = collect_group(
            model, target_dataset, target_clean_idx, None, target_class, args.feature_layer,
            args.batch_size, device, "B_clean_target",
        )
        b_poison_candidates = collect_group(
            model, target_dataset, target_candidate_idx, poison_transform, target_class, args.feature_layer,
            args.batch_size, device, "B_poison_target",
        )
        if args.target_poison_mode == "success":
            keep = np.where(b_poison_candidates.success)[0]
        else:
            keep = np.arange(b_poison_candidates.features.shape[0])
        if args.max_target_poison and keep.size > args.max_target_poison:
            rng = np.random.default_rng(args.seed + 500 + panel_idx)
            keep = np.sort(rng.choice(keep, size=args.max_target_poison, replace=False))
        b_poison_features = b_poison_candidates.features[keep]

        if args.plot_mode == "cross_domain":
            groups = [
                ("A clean target", "source_clean_target", a_clean.features),
                ("A poison target", "source_poison_target", a_poison.features),
                ("B clean target", "target_clean_target", b_clean.features),
                ("B poison target", "target_poison_target", b_poison_features),
            ]
        else:
            groups = [
                (target_clean_group[0], target_clean_group[1], b_clean.features),
                ("B poison target", "target_poison_target", b_poison_features),
            ]
        features = np.concatenate([g[2] for g in groups if g[2].shape[0] > 0], axis=0)
        z = panel_tsne(features, args.seed, args.perplexity)

        cursor = 0
        coords_by_group = {}
        for label, key, feats in groups:
            n = feats.shape[0]
            coords_by_group[key] = z[cursor:cursor + n]
            cursor += n
            for i in range(n):
                point_rows.append({
                    "attack": spec.key,
                    "attack_display": DISPLAY_NAMES.get(spec.key, spec.key),
                    "group": key,
                    "group_display": label,
                    "x_2d": float(coords_by_group[key][i, 0]),
                    "y_2d": float(coords_by_group[key][i, 1]),
                })

        transfer_success_all = int(b_poison_candidates.success.sum())
        transfer_candidate_total = int(b_poison_candidates.success.size)
        transfer_asr_candidate = float(b_poison_candidates.success.mean()) if transfer_candidate_total else float("nan")
        source_asr_payload = (
            float(a_poison.success.mean())
            if a_poison is not None and a_poison.success.size
            else float("nan")
        )
        row = {
            "attack": spec.key,
            "attack_display": DISPLAY_NAMES.get(spec.key, spec.key),
            "directory": spec.dirname,
            "source_clean_available": len(source_clean_all),
            "source_poison_available": len(source_payload_all),
            "target_clean_available": len(target_dataset) if target_clean_group[1] == "target_clean_all" else len(sample_indices_by_label(target_dataset, target_class, True, 0, args.seed)),
            "target_poison_candidate_total": transfer_candidate_total,
            "target_poison_success_total": transfer_success_all,
            "source_clean_plotted": int(a_clean.features.shape[0]) if a_clean is not None else 0,
            "source_poison_plotted": int(a_poison.features.shape[0]) if a_poison is not None else 0,
            "target_clean_plotted": int(b_clean.features.shape[0]),
            "target_poison_plotted": int(b_poison_features.shape[0]),
            "source_payload_asr": source_asr_payload,
            "target_candidate_transfer_asr": transfer_asr_candidate,
            "target_poison_mode": args.target_poison_mode,
            "target_clean_scope": args.target_clean_scope,
            "plot_mode": args.plot_mode,
            "feature_layer": args.feature_layer,
            "reducer": "t-SNE",
            "target_class": target_class,
            "note": "B poison target means triggered STL10 non-target samples predicted as target class when mode=success.",
        }
        count_rows.append(row)
        panels.append((spec, coords_by_group, row))
        print(
            f"[done] {spec.key}: A_clean={row['source_clean_plotted']}, "
            f"A_poison={row['source_poison_plotted']}, B_clean={row['target_clean_plotted']}, "
            f"B_poison={row['target_poison_plotted']}, transfer_candidate_asr={transfer_asr_candidate:.3f}"
        )

    if not panels:
        raise RuntimeError("No panels generated.")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.2))
    axes = axes.ravel()
    styles = {
        "source_clean_target": dict(c="#0072B2", marker=".", s=3.0, alpha=0.48, linewidths=0, label="A clean target"),
        "source_poison_target": dict(c="#D55E00", marker="x", s=16.0, alpha=0.90, linewidths=0.65, label="A poison target"),
        "target_clean_target": dict(facecolors="none", edgecolors="#009E73", marker="o", s=15.0, alpha=0.70, linewidths=0.55, label="B clean target"),
        "target_clean_all": dict(c="#9AA0A6", marker=".", s=2.2, alpha=0.38, linewidths=0, label="B clean all"),
        "target_poison_target": dict(c="#CC79A7", marker="^", s=30.0, alpha=0.92, linewidths=0, label="B poison target"),
    }
    for ax, (spec, coords_by_group, row) in zip(axes, panels):
        for key in ["source_clean_target", "source_poison_target", "target_clean_target", "target_clean_all", "target_poison_target"]:
            pts = coords_by_group.get(key)
            if pts is None or pts.shape[0] == 0:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], rasterized=True, **styles[key])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.set_title(DISPLAY_NAMES.get(spec.key, spec.key), fontsize=15, y=-0.17)
        if args.plot_mode == "cross_domain":
            count_text = (
                f"A clean {row['source_clean_plotted']}, A poison {row['source_poison_plotted']} | "
                f"B clean {row['target_clean_plotted']}, B poison {row['target_poison_plotted']}"
            )
        else:
            clean_label = "B clean all" if row["target_clean_scope"] == "all" and args.plot_mode == "target_only" else "B clean"
            count_text = f"{clean_label} {row['target_clean_plotted']}, B poison {row['target_poison_plotted']}"
        ax.text(
            0.5, -0.29,
            count_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#444444",
        )
    for ax in axes[len(panels):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.93), fontsize=11)
    if args.plot_mode == "cross_domain":
        fig.suptitle("Cross-Domain Target-Class Latent Separability | CIFAR10 -> STL10 | ResNet18", fontsize=20, y=0.985)
        caption = (
            "A = source CIFAR10, B = target STL10. Features are penultimate-layer activations from the CIFAR10 poisoned model, "
            "projected by t-SNE separately per attack panel.\n"
            "B poison target denotes triggered STL10 non-target samples that are predicted as the target class. "
            "This is a qualitative feature-space mechanism view, not a replacement for transfer ASR."
        )
        stem = "F_cross_domain_target_class_latent_tsne_cifar10_to_stl10_resnet18"
    else:
        title_scope = "All Clean Samples" if args.target_clean_scope == "all" else "Target-Class Clean Samples"
        fig.suptitle(f"Target-Domain Latent Separability | STL10 {title_scope} under CIFAR10 ResNet18", fontsize=20, y=0.985)
        clean_desc = "all clean STL10 test samples after CIFAR10 label mapping" if args.target_clean_scope == "all" else "clean STL10 samples mapped to CIFAR10 target class"
        caption = (
            f"Only target-domain STL10 samples are plotted. B clean denotes {clean_desc}; "
            "B poison target denotes triggered STL10 non-target samples predicted as the target class.\n"
            "Features are penultimate-layer activations from the source-trained CIFAR10 poisoned model, projected by t-SNE separately per attack panel."
        )
        stem_suffix = "all_clean" if args.target_clean_scope == "all" else "target_class"
        stem = f"F_target_domain_{stem_suffix}_latent_tsne_stl10_under_cifar10_resnet18"
    fig.text(0.05, 0.045, caption, fontsize=11)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.20, wspace=0.08, hspace=0.55)

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    csv_counts = output_dir / f"{stem}_counts.csv"
    csv_points = output_dir / f"{stem}_points.csv"
    md_path = output_dir / f"{stem}_README.md"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    write_csv(csv_counts, count_rows)
    write_csv(csv_points, point_rows)
    md_path.write_text(
        "# Cross-Domain Target-Class Latent t-SNE\n\n"
        f"This figure adapts target-class latent separability visualization to dataset transfer. Plot mode: `{args.plot_mode}`.\n\n"
        "- `A clean target`: source CIFAR10 clean target-class samples.\n"
        "- `A poison target`: source CIFAR10 payload poison samples.\n"
        "- `B clean target`: target STL10 clean target-class samples after CIFAR10 label mapping.\n"
        "- `B poison target`: triggered STL10 non-target samples predicted as the target class, by default.\n\n"
        f"Configuration: feature layer `{args.feature_layer}`, reducer t-SNE, perplexity `{args.perplexity}`, "
        f"target class `{target_class}`, target poison mode `{args.target_poison_mode}`, target clean scope `{args.target_clean_scope}`.\n\n"
        "This figure should be interpreted qualitatively. It shows whether transferred triggered samples occupy a "
        "feature-space region close to source poison or target clean samples, but transferability should still be "
        "reported using target-side transfer ASR.\n",
        encoding="utf-8",
    )
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")
    print(f"[saved] {csv_counts}")
    print(f"[saved] {csv_points}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
