#!/usr/bin/env python3
"""Frozen-model WaNet diagnostics.

This script writes only to a separate diagnostics directory.  It evaluates
existing checkpoints on paired clean/triggered inputs, STL-10 resize variants,
source-domain resampling variants, and non-matching smooth warp fields.  It
does not train models or modify the formal CTR assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "diagnostics_wanet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = {"cifar10": 0, "tiny_imagenet": 0}
MEAN = {"cifar10": [0.4914, 0.4822, 0.4465], "tiny_imagenet": [0.4802, 0.4481, 0.3975]}
STD = {"cifar10": [0.247, 0.243, 0.261], "tiny_imagenet": [0.2302, 0.2265, 0.2262]}
ARCH = {"ResNet18": "resnet18"}

CONFIGS = [
    ("tiny_imagenet", "ResNet18", 0.01, 0.6),
    ("tiny_imagenet", "ResNet18", 0.01, 0.8),
    ("tiny_imagenet", "ResNet18", 0.05, 0.4),
    ("cifar10", "ResNet18", 0.01, 0.8),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_dir(dataset: str, arch: str, rate: float, s: float) -> Path:
    return ROOT / "poisoned_train_set" / dataset / (
        f"WaNet_{rate:.3f}_cover={2*rate:.3f}_s={s:g}_k=4_poison_seed=2333_arch={arch}_{dataset}"
    )


def load_model(dataset: str, arch: str, checkpoint: Path):
    sys.path.insert(0, str(ROOT))
    from utils import supervisor
    model = supervisor.get_arch(SimpleNamespace(dataset=dataset, poison_type="WaNet", model=ARCH[arch]))(
        num_classes=10 if dataset == "cifar10" else 200
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def raw_source(dataset: str):
    split = ROOT / "clean_set" / dataset / "test_split"
    labels = torch.load(split / "labels", map_location="cpu", weights_only=False).long()
    imgs = []
    for i in range(len(labels)):
        with Image.open(split / "data" / f"{i}.png") as im:
            imgs.append(transforms.ToTensor()(im.convert("RGB")))
    return torch.stack(imgs), labels


def raw_stl():
    ds = datasets.STL10(str(ROOT / "data"), split="test", download=False)
    mapping = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 7, 7: -1, 8: 8, 9: 9}
    xs, ys = [], []
    for arr, y in zip(ds.data, ds.labels):
        y = mapping[int(y)]
        if y >= 0:
            xs.append(torch.from_numpy(arr.copy()).float().div(255))
            ys.append(y)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def raw_target_tiny():
    ds = datasets.ImageFolder(str(ROOT / "data/imagenetv2-matched-frequency-tiny-organized/test"), transform=transforms.ToTensor())
    return torch.stack([x for x, _ in ds], 0), torch.tensor([y for _, y in ds], dtype=torch.long)


def normalize(x: torch.Tensor, dataset: str) -> torch.Tensor:
    return transforms.Normalize(MEAN[dataset], STD[dataset])(x)


def resize_batch(x: torch.Tensor, size: int, mode: str, antialias: bool) -> torch.Tensor:
    kwargs = {"size": (size, size), "mode": mode, "align_corners": False}
    if mode in {"bilinear", "bicubic"}:
        kwargs["antialias"] = antialias
    return F.interpolate(x, **kwargs)


def trigger(x: torch.Tensor, dataset: str, grid: torch.Tensor) -> torch.Tensor:
    x = normalize(x, dataset)
    denorm = transforms.Normalize(
        [-m / s for m, s in zip(MEAN[dataset], STD[dataset])],
        [1 / s for s in STD[dataset]],
    )
    x = denorm(x)
    g = grid.to(x.device).repeat(x.shape[0], 1, 1, 1)
    return normalize(F.grid_sample(x, g, align_corners=True), dataset)


def alternative_grid(identity: torch.Tensor, noise: torch.Tensor, s: float, seed: int, add_noise: bool):
    g = identity + s * noise / identity.shape[1]
    if add_noise:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        ins = torch.rand((1, identity.shape[1], identity.shape[2], 2), generator=gen) * 2 - 1
        g = g + ins.to(g.device) / identity.shape[1]
    return g.clamp(-1, 1)


def independent_smooth_grid(identity: torch.Tensor, s: float, seed: int):
    """Generate an independent smooth displacement field at the model grid size."""
    h, w = identity.shape[1:3]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # A low-resolution random field, upsampled smoothly, is a simple
    # independent analogue of WaNet's smooth cover deformation.
    low = torch.rand((1, 2, 4, 4), generator=gen) * 2 - 1
    field = F.interpolate(low, size=(h, w), mode="bicubic", align_corners=True)
    field = field.permute(0, 2, 3, 1)
    field = field / field.abs().mean().clamp_min(1e-8)
    return (identity + s * field.to(identity.device) / h).clamp(-1, 1)


@torch.inference_mode()
def predict(model, x: torch.Tensor, bs: int, target: int):
    pred, prob, margin = [], [], []
    for i in range(0, len(x), bs):
        z = model(x[i:i+bs].to(DEVICE))
        p = z.softmax(1)
        vals, idx = p.topk(2, 1)
        other = p.clone(); other[:, target] = -1
        pred.append(idx[:, 0].cpu()); prob.append(p[:, target].cpu()); margin.append((p[:, target] - other.max(1).values).cpu())
    return torch.cat(pred), torch.cat(prob), torch.cat(margin)


def paired_rows(model, dataset, domain, x_raw, y, variants, grid, bs):
    target = TARGET[dataset]
    clean_raw = variants["clean_raw"]
    clean = normalize(clean_raw, dataset)
    cp, cpp, cpm = predict(model, clean, bs, target)
    rows = []
    for name, raw in variants.items():
        if name == "clean_raw":
            continue
        if name.startswith("warp_") or name.startswith("noise_warp_") or name.endswith("_trigger") or name == "trigger":
            grid_name = name.removesuffix("_trigger") if (name.startswith("warp_") or name.startswith("noise_warp_")) else "trigger"
            inp = trigger(raw, dataset, grid[grid_name])
        elif name == "trigger":
            inp = trigger(raw, dataset, grid["trigger"])
        else:
            inp = normalize(raw, dataset)
        tp, tpp, tpm = predict(model, inp, bs, target)
        for i in range(len(y)):
            rows.append({"dataset": dataset, "domain": domain, "index": i, "true_label": int(y[i]), "variant": name, "clean_pred": int(cp[i]), "pred": int(tp[i]), "clean_target_prob": float(cpp[i]), "target_prob": float(tpp[i]), "clean_target_margin": float(cpm[i]), "target_margin": float(tpm[i])})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ds, dom, var), g in df.groupby(["dataset", "domain", "variant"]):
        elig = g.true_label.ne(TARGET[ds])
        ctar, tar = g.clean_pred.eq(TARGET[ds]), g.pred.eq(TARGET[ds])
        base = elig & ~ctar
        is_trigger = var == "trigger" or var.endswith("_trigger") or var.startswith("warp_") or var.startswith("noise_warp_")
        out.append({"dataset": ds, "domain": dom, "variant": var, "n": len(g), "baseline_clean_acc": float(g.clean_pred.eq(g.true_label).mean()), "variant_clean_acc": float(g.pred.eq(g.true_label).mean()), "baseline_ctr": float(ctar[elig].mean()), "target_rate": float(tar[elig].mean()), "asr": float(tar[elig].mean()) if is_trigger else np.nan, "asr_minus_ctr": float(tar[elig].mean() - ctar[elig].mean()) if is_trigger else np.nan, "conditional_conversion": float(tar[base].mean()) if (base.any() and is_trigger) else np.nan, "clean_to_target": int((elig & ~ctar & tar).sum()), "target_stays_target": int((elig & ctar & tar).sum()), "target_leaves_target": int((elig & ctar & ~tar).sum()), "predicted_target": int(tar[elig].sum()), "eligible": int(elig.sum())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset, arch, rate, s in CONFIGS:
        d = model_dir(dataset, arch, rate, s)
        ckpt = d / f"{arch}_{dataset}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model = load_model(dataset, arch, ckpt)
        ident = torch.load(d / "identity_grid", map_location=DEVICE, weights_only=False)
        noise = torch.load(d / "noise_grid", map_location=DEVICE, weights_only=False)
        if dataset == "cifar10":
            x, y = raw_source(dataset); xt, yt = raw_stl(); domain_data = [("source", x, y)]
            for mode, aa in [("bicubic_aa", ("bicubic", True)), ("bilinear_noaa", ("bilinear", False)), ("bicubic_noaa", ("bicubic", False)), ("bilinear_aa", ("bilinear", True))]:
                domain_data.append(("target_" + mode, resize_batch(xt, 32, *aa), yt))
        else:
            x, y = raw_source(dataset); xt, yt = raw_target_tiny(); domain_data = [("source", x, y), ("target", xt, yt)]
        if args.limit:
            domain_data = [(n, xx[:args.limit], yy[:args.limit]) for n, xx, yy in domain_data]
        variants_by_domain = {}
        for domain, xx, yy in domain_data:
            variants = {"clean_raw": xx}
            variants["trigger"] = xx
            if domain == "source":
                variants["resample_x2"] = resize_batch(resize_batch(xx, xx.shape[-1]*2, "bicubic", True), xx.shape[-1], "bicubic", True)
                variants["resample_x4"] = resize_batch(resize_batch(xx, xx.shape[-1]*4, "bicubic", True), xx.shape[-1], "bicubic", True)
                variants["resample_x2_trigger"] = variants["resample_x2"]
                variants["resample_x4_trigger"] = variants["resample_x4"]
            if domain.startswith("target_"):
                variants["trigger"] = xx
            grids = {"trigger": alternative_grid(ident, noise, s, 0, False)}
            for k in range(10): grids[f"warp_{k}"] = independent_smooth_grid(ident, s, k)
            for k in range(10): grids[f"noise_warp_{k}"] = alternative_grid(ident, noise, s, k, True)
            for k in range(10): variants[f"warp_{k}"] = xx
            for k in range(10): variants[f"noise_warp_{k}"] = xx
            variants_by_domain[domain] = (variants, grids, yy)
        all_rows = []
        for domain, (variants, grids, yy) in variants_by_domain.items():
            all_rows.append(paired_rows(model, dataset, domain, variants["clean_raw"], yy, variants, grids, args.batch_size))
        raw = pd.concat(all_rows, ignore_index=True)
        tag = f"{dataset}_{rate:g}_s{s:g}"
        raw.to_csv(OUT / f"{tag}_paired.csv", index=False)
        summarize(raw).to_csv(OUT / f"{tag}_summary.csv", index=False)
        manifest.append({"tag": tag, "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt), "domains": sorted(raw.domain.unique()), "rows": len(raw)})
        del model
        torch.cuda.empty_cache()
    (OUT / "manifest.json").write_text(json.dumps({"device": str(DEVICE), "configs": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
