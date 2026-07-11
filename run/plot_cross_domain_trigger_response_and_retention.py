#!/usr/bin/env python3
"""Plot dataset-level trigger-response transfer mechanism figures.

Figure A: Cross-domain Trigger Response Map.
Figure B: Source-Target Trigger Response Retention Map.

Default setting is CIFAR10 -> STL10 with ResNet18 medium-strength attacks from
``poisoned_train_set``. The figures are qualitative/mechanistic diagnostics and
do not replace target-side transfer ASR as the primary transferability metric.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PAPER_ANALYSIS = REPO / "analysis-transfer-asr2" / "paper_analysis"

import config  # noqa: E402
from utils import supervisor, tools  # noqa: E402
from utils.tools import IMG_Dataset  # noqa: E402

if str(PAPER_ANALYSIS) not in sys.path:
    sys.path.append(str(PAPER_ANALYSIS))

from make_target_class_latent_tsne import (  # noqa: E402
    AttackSpec,
    SPECS,
    find_model_path,
    get_poison_transform,
    load_model,
    make_args,
)


DISPLAY_NAMES = {
    "basic": "Basic",
    "blend": "Blend",
    "adaptive_blend": "Adaptive-Blend",
    "adaptive_patch": "Adaptive-Patch",
    "wanet": "WaNet",
    "sig": "SIG",
    "upgd": "UPGD",
    "belt": "BELT",
}


@dataclass
class GroupResult:
    group: str
    features: np.ndarray
    logits_clean: np.ndarray | None
    logits_triggered: np.ndarray | None
    pred_label: np.ndarray
    true_label: np.ndarray
    success: np.ndarray


class FilteredSTL10(Dataset):
    """STL10 test split mapped into CIFAR10 label ids and resized to 32x32."""

    LABEL_MAPPING = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 7, 7: -1, 8: 8, 9: 9}

    def __init__(self, root: Path, skip_normalize: bool):
        raw_transform = transforms.Compose([transforms.ToTensor()])
        base = torchvision.datasets.STL10(
            root=str(root),
            split="test",
            download=True,
            transform=raw_transform,
        )
        self.data = []
        self.targets = []
        self.skip_normalize = skip_normalize
        self.normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        for i in range(len(base)):
            x, y = base[i]
            mapped = self.LABEL_MAPPING[int(y)]
            if mapped == -1:
                continue
            self.data.append(x)
            self.targets.append(mapped)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        x = self.data[idx]
        y = self.targets[idx]
        if x.shape[-1] != 32:
            x = F.interpolate(x.unsqueeze(0), size=(32, 32), mode="bicubic", align_corners=False).squeeze(0)
        if not self.skip_normalize:
            x = self.normalize(x)
        return x, y


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(s)).strip("-")


def parse_attack_filter(value: str) -> set[str] | None:
    if value.lower() in {"all", "*"}:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def selected_specs(attack_filter: set[str] | None, poison_rate: float | None, strength: str | None) -> list[AttackSpec]:
    specs = [s for s in SPECS if s.key != "none"]
    if attack_filter is not None:
        specs = [s for s in specs if s.key in attack_filter or DISPLAY_NAMES.get(s.key, "").lower() in attack_filter]
    if poison_rate is not None:
        specs = [s for s in specs if abs(float(s.poison_rate) - poison_rate) < 1e-12]
    if strength:
        specs = [s for s in specs if strength in s.dirname]
    return specs


def sample_indices_by_label(dataset: Dataset, target_class: int, want_target: bool, max_n: int, seed: int) -> list[int]:
    idxs = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        is_target = int(y) == int(target_class)
        if is_target == want_target:
            idxs.append(i)
    rng = random.Random(seed)
    rng.shuffle(idxs)
    if max_n > 0:
        idxs = idxs[:max_n]
    return sorted(idxs)


def target_margin(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    target = logits[:, target_class]
    masked = logits.clone()
    masked[:, target_class] = -float("inf")
    other = masked.max(dim=1).values
    return target - other


def forward_with_feature(model, x: torch.Tensor, feature_layer: str):
    if feature_layer == "penultimate":
        out = model(x, return_hidden=True)
        if isinstance(out, tuple):
            return out[0], out[1]
        raise RuntimeError("Model does not support return_hidden=True.")

    # Minimal hook-based layer4 support for ResNet-like models.
    module = getattr(model, "module", model)
    if feature_layer != "layer4" or not hasattr(module, "layer4"):
        raise NotImplementedError(f"feature_layer={feature_layer} is not supported for this model.")
    box = {}

    def hook(_, __, output):
        box["feat"] = output.detach()

    handle = module.layer4.register_forward_hook(hook)
    try:
        logits = model(x)
    finally:
        handle.remove()
    feat = box.get("feat")
    if feat is None:
        raise RuntimeError("Failed to capture layer4 feature.")
    feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
    return logits, feat


def collect_group(
    model,
    dataset: Dataset,
    indices: list[int],
    poison_transform,
    target_class: int,
    feature_layer: str,
    batch_size: int,
    device: torch.device,
    group: str,
    clean_logits_for_margin: bool = False,
) -> GroupResult:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=tools.worker_init,
    )
    feats, logits_clean_all, logits_trig_all, preds, trues, successes = [], [], [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            if poison_transform is None:
                logits, h = forward_with_feature(model, x, feature_layer)
                logits_trig = None
            else:
                logits_clean = None
                if clean_logits_for_margin:
                    logits_clean, _ = forward_with_feature(model, x, feature_layer)
                x_trig, y_trig = poison_transform.transform(x, y.clone())
                logits, h = forward_with_feature(model, x_trig, feature_layer)
                logits_trig = logits
                if logits_clean is not None:
                    logits_clean_all.append(logits_clean.detach().cpu().float().numpy())
            pred = logits.argmax(dim=1)
            feats.append(h.detach().cpu().float().numpy())
            logits_trig_all.append(logits.detach().cpu().float().numpy())
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
            successes.append((pred == target_class).detach().cpu().numpy())
    return GroupResult(
        group=group,
        features=np.concatenate(feats, axis=0) if feats else np.empty((0, 1), dtype=np.float32),
        logits_clean=np.concatenate(logits_clean_all, axis=0) if logits_clean_all else None,
        logits_triggered=np.concatenate(logits_trig_all, axis=0) if logits_trig_all else None,
        pred_label=np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.int64),
        true_label=np.concatenate(trues, axis=0) if trues else np.empty((0,), dtype=np.int64),
        success=np.concatenate(successes, axis=0).astype(bool) if successes else np.empty((0,), dtype=bool),
    )


def reduce_features(features: np.ndarray, reducer: str, seed: int) -> np.ndarray:
    x = StandardScaler().fit_transform(features.astype(np.float64))
    x = normalize(x, norm="l2")
    if reducer == "umap":
        try:
            import umap

            return umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=seed).fit_transform(x)
        except Exception as exc:
            print(f"[warning] UMAP failed or unavailable ({exc}); falling back to t-SNE.")
    perplexity = min(30, max(5, (x.shape[0] - 1) // 5))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed, learning_rate="auto").fit_transform(x)


def write_points_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_cross_domain_map(
    rows: list[dict],
    title: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    groups = {
        "A_clean_t": dict(label="A clean target", marker="o", c="#009E73", s=36, alpha=0.85, edgecolors="none"),
        "A_trig_nt": dict(label="A triggered non-target", marker="o", c="#D55E00", s=28, alpha=0.75, edgecolors="none"),
        "B_clean_t": dict(label="B clean target", marker="o", facecolors="none", edgecolors="#009E73", s=52, alpha=0.85, linewidths=1.1),
        "B_clean_nt": dict(label="B clean non-target", marker=".", c="#9AA0A6", s=12, alpha=0.28, edgecolors="none"),
    }
    for group, style in groups.items():
        pts = [r for r in rows if r["sample_group"] == group]
        if not pts:
            continue
        ax.scatter([r["x_2d"] for r in pts], [r["y_2d"] for r in pts], **style)
    for success, label, face in [(True, "B triggered success", "#E69F00"), (False, "B triggered failure", "none")]:
        pts = [r for r in rows if r["sample_group"] == "B_trig_nt" and bool(r["is_transfer_success"]) == success]
        if not pts:
            continue
        ax.scatter(
            [r["x_2d"] for r in pts],
            [r["y_2d"] for r in pts],
            marker="^",
            s=48,
            facecolors=face,
            edgecolors="#E69F00",
            linewidths=1.0,
            alpha=0.86,
            label=label,
        )
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Joint latent dimension 1")
    ax.set_ylabel("Joint latent dimension 2")
    ax.legend(loc="best", frameon=True, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_retention_map(rows: list[dict], title: str, png_path: Path, pdf_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    attacks = sorted({r["attack"] for r in rows})
    colors = dict(zip(attacks, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#8C8C8C"]))
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for attack in attacks:
        pts = [r for r in rows if r["attack"] == attack]
        sizes = [35 + 180 * float(r["transfer_asr"]) for r in pts]
        ax.scatter(
            [r["R_A_median_delta_margin"] for r in pts],
            [r["R_B_median_delta_margin"] for r in pts],
            s=sizes,
            color=colors[attack],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.7,
            label=DISPLAY_NAMES.get(attack, attack),
        )
        if len(rows) <= 12:
            for r in pts:
                ax.annotate(DISPLAY_NAMES.get(attack, attack), (r["R_A_median_delta_margin"], r["R_B_median_delta_margin"]),
                            xytext=(4, 3), textcoords="offset points", fontsize=7.2, color="#333333")
    ax.axhline(0, color="#5F6B7A", linewidth=0.9, alpha=0.75)
    ax.axvline(0, color="#5F6B7A", linewidth=0.9, alpha=0.75)
    finite = np.array([[r["R_A_median_delta_margin"], r["R_B_median_delta_margin"]] for r in rows], dtype=float)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    pad = 0.08 * max(hi - lo, 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#666666", linestyle="--", linewidth=0.9, alpha=0.7, label="y = x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Source trigger response: median Delta target margin")
    ax.set_ylabel("Target trigger response: median Delta target margin")
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)


def spearman_safe(x: list[float], y: list[float]) -> float | None:
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(x, y, nan_policy="omit")
        if math.isnan(float(rho)):
            return None
        return float(rho)
    except Exception:
        return None


def write_captions(output_dir: Path) -> None:
    fig_a = """English caption:
This figure jointly visualizes source-domain target/trigger features and target-domain triggered features in the latent space of the source-trained poisoned model. Target-domain triggered samples are marked by transfer success or failure. The map shows where triggered target-domain samples land relative to source target features, source triggered features, and target-domain clean samples. This visualization is qualitative mechanism evidence. Successful transfer should not be interpreted as requiring target-domain triggered samples to fall into a single fixed cluster; high transfer may correspond to target-region alignment, source-trigger alignment, or a stable target-domain trigger-specific structure.

Chinese caption:
该图将 source 域 target/trigger 特征与 target 域 clean/triggered 特征共同投影到 source-trained poisoned model 的 latent space 中，并用实心/空心三角标记 B 域 triggered samples 是否被预测为目标类。该图用于观察 target-domain triggered samples 相对于 source target features、source triggered features 以及 target clean samples 的位置。该图是迁移性的定性机制可视化，不能将高迁移简单解释为 B triggered 必须落入某个固定 cluster；高迁移可能表现为靠近 target region、靠近 source-trigger region，或形成稳定的 target-domain trigger-specific structure。
"""
    fig_b = """English caption:
This figure summarizes cross-domain trigger-response retention. Each point represents one attack setting. The x-axis measures the median trigger-induced target-margin change on source-domain non-target samples, while the y-axis measures the same quantity on target-domain non-target samples. Points in the upper-right indicate that the source-domain trigger response is retained in the target domain, whereas points in the lower-right indicate source-effective but poorly transferable responses. Point size indicates target-side transfer ASR. This mechanism metric is used only to explain transferability and does not replace transfer ASR as the primary metric.

Chinese caption:
该图总结跨域触发响应的保留程度。每个点代表一个 attack setting。横轴表示 source 域 non-target samples 加 trigger 后的 target margin 中位提升，纵轴表示 target 域 non-target samples 加 trigger 后的 target margin 中位提升。右上角表示 source 上学到的 trigger response 能在 target 域中较好保留；右下角表示 source 上有效但 target 上响应衰减，可能对应 source ASR 高但 transfer ASR 低。点大小表示 target-side transfer ASR。该机制指标只用于解释迁移性，不替代 transfer ASR 作为主迁移性指标。
"""
    (output_dir / "cross_domain_trigger_response_map_caption.txt").write_text(fig_a, encoding="utf-8")
    (output_dir / "source_target_trigger_response_retention_map_caption.txt").write_text(fig_b, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_source", default="cifar10")
    ap.add_argument("--dataset_target", default="stl10")
    ap.add_argument("--attack", default="all", help="Comma-separated attack keys or all.")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--strength", default=None, help="Optional substring filter on experiment directory.")
    ap.add_argument("--poison_rate", type=float, default=None)
    ap.add_argument("--target_class", type=int, default=None)
    ap.add_argument("--model_path", default=None, help="Optional model path for a single attack setting.")
    ap.add_argument("--poison_root", default="poisoned_train_set")
    ap.add_argument("--output_dir", default=str(REPO / "analysis-transfer-asr2/paper_analysis_outputs/dataset_transfer_mechanism"))
    ap.add_argument("--max_samples_per_group", type=int, default=350)
    ap.add_argument("--retention_samples", type=int, default=800)
    ap.add_argument("--reducer", choices=["umap", "tsne"], default="umap")
    ap.add_argument("--feature_layer", choices=["penultimate", "layer4"], default="penultimate")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2333)
    args = ap.parse_args()

    if args.dataset_source != "cifar10" or args.dataset_target != "stl10":
        raise NotImplementedError("This first implementation supports CIFAR10 -> STL10.")
    os.environ["POISONED_TRAIN_SET_ROOT"] = args.poison_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_class = int(config.target_class[args.dataset_source] if args.target_class is None else args.target_class)
    specs = selected_specs(parse_attack_filter(args.attack), args.poison_rate, args.strength)
    if not specs:
        raise RuntimeError("No attack settings selected.")

    poisoned_root = REPO / args.poison_root / args.dataset_source
    clean_dir = poisoned_root / f"none_0.000_poison_seed=2333_arch=ResNet18_{args.dataset_source}"
    if not clean_dir.exists():
        raise FileNotFoundError(f"Missing clean source directory: {clean_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retention_rows: list[dict] = []
    all_point_rows: list[dict] = []

    for spec_idx, spec in enumerate(specs):
        attack_dir = poisoned_root / spec.dirname
        if not attack_dir.exists():
            print(f"[warning] skip missing attack directory: {attack_dir}")
            continue
        spec_args = make_args(spec)
        spec_args.dataset = args.dataset_source
        spec_args.model = args.arch
        if args.model_path:
            model_path = Path(args.model_path)
        else:
            try:
                model_path = find_model_path(attack_dir)
            except Exception as exc:
                print(f"[warning] skip {spec.key}: {exc}")
                continue

        data_transform = supervisor.get_transforms(spec_args)[1]
        source_dataset = IMG_Dataset(clean_dir / "data", clean_dir / "labels", transforms=data_transform)
        skip_normalize = spec.poison_type in {"upgd", "belt"}
        target_dataset = FilteredSTL10(REPO / "data", skip_normalize=skip_normalize)
        is_normalized_input = spec.poison_type not in {"upgd", "belt"} and not spec_args.no_normalize
        poison_transform = get_poison_transform(spec, spec_args, is_normalized_input=is_normalized_input)
        model = load_model(model_path, spec_args, device)

        a_clean_t_idx = sample_indices_by_label(source_dataset, target_class, True, args.max_samples_per_group, args.seed + 10 + spec_idx)
        a_clean_nt_idx = sample_indices_by_label(source_dataset, target_class, False, max(args.max_samples_per_group, args.retention_samples), args.seed + 20 + spec_idx)
        b_clean_t_idx = sample_indices_by_label(target_dataset, target_class, True, args.max_samples_per_group, args.seed + 30 + spec_idx)
        b_clean_nt_idx = sample_indices_by_label(target_dataset, target_class, False, max(args.max_samples_per_group, args.retention_samples), args.seed + 40 + spec_idx)

        # Figure A groups use the same non-target slice for clean and triggered B.
        a_trig_plot_idx = a_clean_nt_idx[:args.max_samples_per_group]
        b_clean_nt_plot_idx = b_clean_nt_idx[:args.max_samples_per_group]
        b_trig_plot_idx = b_clean_nt_idx[:args.max_samples_per_group]

        groups = [
            collect_group(model, source_dataset, a_clean_t_idx, None, target_class, args.feature_layer, args.batch_size, device, "A_clean_t"),
            collect_group(model, source_dataset, a_trig_plot_idx, poison_transform, target_class, args.feature_layer, args.batch_size, device, "A_trig_nt"),
            collect_group(model, target_dataset, b_clean_t_idx, None, target_class, args.feature_layer, args.batch_size, device, "B_clean_t"),
            collect_group(model, target_dataset, b_clean_nt_plot_idx, None, target_class, args.feature_layer, args.batch_size, device, "B_clean_nt"),
            collect_group(model, target_dataset, b_trig_plot_idx, poison_transform, target_class, args.feature_layer, args.batch_size, device, "B_trig_nt"),
        ]
        features = np.concatenate([g.features for g in groups], axis=0)
        z2d = reduce_features(features, args.reducer, args.seed)

        setting_id = f"{args.dataset_source}_to_{args.dataset_target}_{spec.key}_{args.arch}_{sanitize(spec.dirname)}"
        point_rows = []
        cursor = 0
        for g in groups:
            n = g.features.shape[0]
            coords = z2d[cursor:cursor + n]
            cursor += n
            for i in range(n):
                row = {
                    "setting_id": setting_id,
                    "dataset_source": args.dataset_source,
                    "dataset_target": args.dataset_target,
                    "attack": spec.key,
                    "strength": spec.dirname,
                    "poison_rate": spec.poison_rate,
                    "arch": args.arch,
                    "target_class": target_class,
                    "sample_group": g.group,
                    "is_transfer_success": bool(g.success[i]) if g.group == "B_trig_nt" else "",
                    "x_2d": float(coords[i, 0]),
                    "y_2d": float(coords[i, 1]),
                    "pred_label": int(g.pred_label[i]),
                    "true_label": int(g.true_label[i]),
                }
                point_rows.append(row)
                all_point_rows.append(row)

        source_asr = float(groups[1].success.mean()) if len(groups[1].success) else float("nan")
        transfer_asr = float(groups[4].success.mean()) if len(groups[4].success) else float("nan")
        name = f"{args.dataset_source}_to_{args.dataset_target}_{spec.key}_{args.arch}_{sanitize(spec.dirname)}"
        title = (
            f"Cross-domain Trigger Response Map | {args.dataset_source.upper()}->{args.dataset_target.upper()} | "
            f"{DISPLAY_NAMES.get(spec.key, spec.key)} | {args.arch} | target={target_class} | "
            f"source ASR={source_asr:.2f} | transfer ASR={transfer_asr:.2f}"
        )
        plot_cross_domain_map(
            point_rows,
            title,
            output_dir / f"F_cross_domain_trigger_response_map_{name}.png",
            output_dir / f"F_cross_domain_trigger_response_map_{name}.pdf",
        )
        write_points_csv(output_dir / f"cross_domain_trigger_response_map_points_{name}.csv", point_rows)

        # Figure B retention statistics use larger non-target slices.
        a_nt_ret = a_clean_nt_idx[:args.retention_samples]
        b_nt_ret = b_clean_nt_idx[:args.retention_samples]
        a_ret = collect_group(model, source_dataset, a_nt_ret, poison_transform, target_class, args.feature_layer, args.batch_size, device, "A_retention", clean_logits_for_margin=True)
        b_ret = collect_group(model, target_dataset, b_nt_ret, poison_transform, target_class, args.feature_layer, args.batch_size, device, "B_retention", clean_logits_for_margin=True)

        a_clean_logits = torch.from_numpy(a_ret.logits_clean)
        a_trig_logits = torch.from_numpy(a_ret.logits_triggered)
        b_clean_logits = torch.from_numpy(b_ret.logits_clean)
        b_trig_logits = torch.from_numpy(b_ret.logits_triggered)
        dm_a = (target_margin(a_trig_logits, target_class) - target_margin(a_clean_logits, target_class)).numpy()
        dm_b = (target_margin(b_trig_logits, target_class) - target_margin(b_clean_logits, target_class)).numpy()
        r_a = float(np.median(dm_a))
        r_b = float(np.median(dm_b))
        ratio = float(r_b / r_a) if abs(r_a) >= 1e-6 else float("nan")
        retention_rows.append({
            "setting_id": setting_id,
            "dataset_source": args.dataset_source,
            "dataset_target": args.dataset_target,
            "attack": spec.key,
            "strength": spec.dirname,
            "poison_rate": spec.poison_rate,
            "arch": args.arch,
            "target_class": target_class,
            "R_A_median_delta_margin": r_a,
            "R_B_median_delta_margin": r_b,
            "R_A_mean_delta_margin": float(np.mean(dm_a)),
            "R_B_mean_delta_margin": float(np.mean(dm_b)),
            "R_A_std_delta_margin": float(np.std(dm_a)),
            "R_B_std_delta_margin": float(np.std(dm_b)),
            "source_asr": float(a_ret.success.mean()) if len(a_ret.success) else float("nan"),
            "transfer_asr": float(b_ret.success.mean()) if len(b_ret.success) else float("nan"),
            "response_retention_ratio": ratio,
            "n_source_samples": int(len(dm_a)),
            "n_target_samples": int(len(dm_b)),
        })
        print(f"[done] {spec.key}: source_asr={source_asr:.3f}, transfer_asr={transfer_asr:.3f}, R_A={r_a:.3f}, R_B={r_b:.3f}")

    if not retention_rows:
        raise RuntimeError("No figures generated; all settings were skipped.")

    write_points_csv(output_dir / f"cross_domain_trigger_response_map_points_all_{args.dataset_source}_to_{args.dataset_target}_{args.arch}.csv", all_point_rows)
    retention_csv = output_dir / f"source_target_trigger_response_retention_{args.dataset_source}_to_{args.dataset_target}_{args.arch}.csv"
    write_points_csv(retention_csv, retention_rows)
    rho = spearman_safe([r["R_B_median_delta_margin"] for r in retention_rows], [r["transfer_asr"] for r in retention_rows])
    title = f"Source-Target Trigger Response Retention | {args.dataset_source.upper()}->{args.dataset_target.upper()} | {args.arch}"
    if rho is not None:
        title += f" | Spearman(R_B, transfer ASR)={rho:.2f}"
    plot_retention_map(
        retention_rows,
        title,
        output_dir / f"F_source_target_trigger_response_retention_map_{args.dataset_source}_to_{args.dataset_target}_{args.arch}.png",
        output_dir / f"F_source_target_trigger_response_retention_map_{args.dataset_source}_to_{args.dataset_target}_{args.arch}.pdf",
    )
    write_captions(output_dir)
    print(f"[saved] {output_dir}")


if __name__ == "__main__":
    main()
