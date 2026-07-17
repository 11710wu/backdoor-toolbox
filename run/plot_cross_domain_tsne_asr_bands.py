#!/usr/bin/env python3
"""Plot three cross-domain t-SNE galleries at source-ASR bands ≈90/80/60%.

Keeps the existing main figure untouched. For each band, selects the closest
available CIFAR-10 ResNet18 config per attack and annotates source/transfer ASR.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "run"))
sys.path.insert(0, str(REPO / "analysis-transfer-asr2" / "paper_analysis"))
sys.path.insert(0, str(REPO))  # prefer repo-root config.py with target_class

os.environ.setdefault("POISONED_TRAIN_SET_ROOT", "poisoned_train_set")

import config  # noqa: E402
assert hasattr(config, "target_class"), f"wrong config imported: {config}"
from make_target_class_latent_tsne import (  # noqa: E402
    find_model_path,
    get_poison_transform,
    load_model,
    make_args,
    split_target_class_indices,
)
from plot_cross_domain_target_class_latent_tsne import (  # noqa: E402
    panel_tsne,
    sample_list,
)
from plot_cross_domain_trigger_response_and_retention import (  # noqa: E402
    DISPLAY_NAMES,
    FilteredSTL10,
    collect_group,
    sample_indices_by_label,
)
from run_dataset_level_transfer_mechanism_multiconfig import (  # noqa: E402
    ATTACK_ORDER,
    spec_from_folder,
)

OUT_DIR = REPO / "analysis-transfer-asr2/paper_analysis_outputs/dataset_transfer_mechanism"
SEL_PATH = OUT_DIR / "tsne_asr_band_selection.json"
BANDS = [
    ("asr90", "≈90% source ASR", "F_cross_domain_target_class_latent_tsne_cifar10_to_stl10_resnet18_asr90"),
    ("asr80", "≈80% source ASR", "F_cross_domain_target_class_latent_tsne_cifar10_to_stl10_resnet18_asr80"),
    ("asr60", "≈60% source ASR", "F_cross_domain_target_class_latent_tsne_cifar10_to_stl10_resnet18_asr60"),
]


def ensure_symlink_root(band: str, band_sel: dict) -> Path:
    stage = OUT_DIR / f"_tsne_stage_{band}" / "cifar10"
    if stage.exists():
        # rebuild cleanly
        import shutil

        shutil.rmtree(stage.parent)
    stage.mkdir(parents=True)
    for atk in ATTACK_ORDER:
        info = band_sel[atk]
        src = REPO / info["root"] / "cifar10" / info["folder"]
        dst = stage / info["folder"]
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    return stage.parent  # poisoned root parent containing cifar10/


def run_band(band_key: str, band_title: str, stem: str, band_sel: dict, device: torch.device) -> None:
    poison_root = ensure_symlink_root(band_key, band_sel)
    poisoned_root = poison_root / "cifar10"
    target_class = int(config.target_class["cifar10"])
    seed = 2333
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    panels = []
    metric_lines = []
    for panel_idx, atk in enumerate(ATTACK_ORDER):
        info = band_sel[atk]
        folder = info["folder"]
        # map keys for SPECS-style parsing
        key = "wanet" if atk == "WaNet" else ("sig" if atk == "SIG" else atk)
        # spec_from_folder expects attack names as in multiconfig
        parse_atk = atk if atk not in {"WaNet", "SIG"} else atk
        if atk == "WaNet":
            parse_atk = "WaNet"
        elif atk == "SIG":
            parse_atk = "SIG"
        spec = spec_from_folder(parse_atk, folder)
        # normalize display key used by DISPLAY_NAMES
        if spec.key == "WaNet":
            display_key = "wanet"
        elif spec.key == "SIG":
            display_key = "sig"
        else:
            display_key = spec.key

        attack_dir = poisoned_root / folder
        from utils import supervisor

        spec_args = make_args(spec)
        data_transform = supervisor.get_transforms(spec_args)[1]
        skip_normalize = spec.poison_type in {"upgd", "belt"}
        target_dataset = FilteredSTL10(REPO / "data", skip_normalize=skip_normalize)
        is_normalized_input = spec.poison_type not in {"upgd", "belt"} and not spec_args.no_normalize
        poison_transform = get_poison_transform(spec, spec_args, is_normalized_input=is_normalized_input)
        if hasattr(poison_transform, "trigger_marks"):
            poison_transform.trigger_marks = [t.cpu() for t in poison_transform.trigger_marks]
            poison_transform.trigger_masks = [t.cpu() for t in poison_transform.trigger_masks]
        model = load_model(find_model_path(attack_dir), spec_args, device)

        source_dataset = torchvision.datasets.CIFAR10(
            root=str(REPO / "data" / "cifar10"),
            train=True,
            download=False,
            transform=data_transform,
        )
        source_clean_all, source_payload_all, _ = split_target_class_indices(attack_dir, spec)
        source_clean_idx = sample_list(source_clean_all, 2200, seed + 100 + panel_idx)
        source_payload_idx = sample_list(source_payload_all, 250, seed + 200 + panel_idx)
        a_clean = collect_group(
            model, source_dataset, source_clean_idx, None, target_class, "penultimate", 64, device, "A_clean_target"
        )
        a_poison = collect_group(
            model, source_dataset, source_payload_idx, poison_transform, target_class, "penultimate", 64, device, "A_poison_target"
        )

        target_clean_idx = sample_indices_by_label(target_dataset, target_class, True, 450, seed + 300 + panel_idx)
        target_poison_cand_idx = sample_indices_by_label(target_dataset, target_class, False, 1800, seed + 400 + panel_idx)
        b_clean = collect_group(
            model, target_dataset, target_clean_idx, None, target_class, "penultimate", 64, device, "B_clean_target"
        )
        b_poison_all = collect_group(
            model, target_dataset, target_poison_cand_idx, poison_transform, target_class, "penultimate", 64, device, "B_poison_cand"
        )
        # success-only B poison
        success_mask = b_poison_all.success
        success_idx = np.where(success_mask)[0]
        if len(success_idx) > 50:
            rng = random.Random(seed + 500 + panel_idx)
            success_idx = np.array(sorted(rng.sample(list(success_idx), 50)))
        b_poison_feat = b_poison_all.features[success_idx] if len(success_idx) else np.empty((0, a_clean.features.shape[1]))

        feats = [a_clean.features, a_poison.features, b_clean.features]
        if b_poison_feat.shape[0]:
            feats.append(b_poison_feat)
        all_feat = np.concatenate(feats, axis=0)
        z = panel_tsne(all_feat, seed + panel_idx, 35)
        n0 = a_clean.features.shape[0]
        n1 = a_poison.features.shape[0]
        n2 = b_clean.features.shape[0]
        coords = {
            "source_clean_target": z[:n0],
            "source_poison_target": z[n0 : n0 + n1],
            "target_clean_target": z[n0 + n1 : n0 + n1 + n2],
            "target_poison_target": z[n0 + n1 + n2 :],
        }
        src = float(info["source"])
        tr = float(info["transfer"]) if info.get("transfer") is not None else float("nan")
        st = float(info["stealth"]) if info.get("stealth") is not None else float("nan")
        panels.append((display_key, coords, src, tr, st, folder))
        metric_lines.append(f"{DISPLAY_NAMES.get(display_key, display_key)}: srcASR={src:.1%}, trASR={tr:.1%}, stealth={st:.3f}")
        print(f"[{band_key}] {display_key}: src={src:.3f} tr={tr:.3f}")

    # plot
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "figure.dpi": 300, "savefig.dpi": 300})
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6))
    axes = axes.ravel()
    styles = {
        "source_clean_target": dict(c="#0072B2", marker=".", s=3.0, alpha=0.48, linewidths=0, label="A clean target"),
        "source_poison_target": dict(c="#D55E00", marker="x", s=16.0, alpha=0.90, linewidths=0.65, label="A poison target"),
        "target_clean_target": dict(facecolors="none", edgecolors="#009E73", marker="o", s=15.0, alpha=0.70, linewidths=0.55, label="B clean target"),
        "target_poison_target": dict(c="#CC79A7", marker="^", s=30.0, alpha=0.92, linewidths=0, label="B poison target"),
    }
    for ax, (display_key, coords, src, tr, st, folder) in zip(axes, panels):
        for key, style in styles.items():
            pts = coords.get(key)
            if pts is None or pts.shape[0] == 0:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], rasterized=True, **style)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.set_title(DISPLAY_NAMES.get(display_key, display_key), fontsize=14, y=-0.14)
        ax.text(
            0.5,
            -0.26,
            f"srcASR {src:.1%} | trASR {tr:.1%} | stealth {st:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#222222",
            fontweight="bold",
        )
        ax.text(
            0.5,
            -0.38,
            Path(folder).name[:54] + ("…" if len(folder) > 54 else ""),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.5,
            color="#666666",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.935), fontsize=11)
    fig.suptitle(
        f"Cross-Domain Target-Class Latent Separability | CIFAR10→STL10 | ResNet18\nConfig band: {band_title}",
        fontsize=18,
        y=0.995,
    )
    caption = (
        "A=CIFAR10 source, B=STL10 target. Each panel uses the closest available config to the band target source ASR. "
        "ASR values under each panel are from master_results / matched test logs.\n"
        "Qualitative feature-space view only; not a substitute for transfer ASR tables."
    )
    fig.text(0.04, 0.035, caption, fontsize=10)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.20, wspace=0.08, hspace=0.62)
    out_png = OUT_DIR / f"{stem}.png"
    out_pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # ASR table sidecar
    table_path = OUT_DIR / f"{stem}_asr_table.md"
    lines = [f"# {band_title}", "", "| Attack | Source ASR | Transfer ASR | Stealthiness | Config |", "|---|---:|---:|---:|---|"]
    for display_key, _, src, tr, st, folder in panels:
        lines.append(
            f"| {DISPLAY_NAMES.get(display_key, display_key)} | {src:.1%} | {tr:.1%} | {st:.3f} | `{folder}` |"
        )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {out_png}")
    print(f"[saved] {table_path}")


def main() -> None:
    device = torch.device("cpu")
    # Keep adaptive_patch / belt triggers on CPU.
    _orig = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore
    try:
        sel = json.loads(SEL_PATH.read_text())
        for band_key, band_title, stem in BANDS:
            print(f"\n==== {band_key} ====")
            run_band(band_key, band_title, stem, sel[band_key], device)
    finally:
        torch.Tensor.cuda = _orig  # type: ignore
    print("\nKept original figure untouched:")
    print(OUT_DIR / "F_cross_domain_target_class_latent_tsne_cifar10_to_stl10_resnet18.png")


if __name__ == "__main__":
    main()
