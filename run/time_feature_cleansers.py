#!/usr/bin/env python3
"""Wall-clock timing for SS / AC / SCAn / SPECTRE(python) on CIFAR-10 and Tiny-ImageNet."""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms

REPO = Path("/workspace/backdoor-toolbox-new1")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from cleansers_tool_box import activation_clustering, scan, spectral_signature, spectre_python
from utils import supervisor


class TinyTrainFolder(Dataset):
    """Tiny-ImageNet train folder with integer labels 0..199 in wnids order."""

    def __init__(self, root: Path, transform):
        self.transform = transform
        wnids = (root / "wnids.txt").read_text().strip().splitlines()
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}
        self.samples = []
        train = root / "train"
        for wnid in wnids:
            img_dir = train / wnid / "images"
            if not img_dir.is_dir():
                continue
            label = self.class_to_idx[wnid]
            for p in sorted(img_dir.glob("*.JPEG")):
                self.samples.append((p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = torchvision.datasets.folder.default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_model(dataset: str, ckpt: Path, num_classes: int):
    args = SimpleNamespace(dataset=dataset, model="resnet18", no_normalize=False)
    # supervisor.get_arch expects more fields for some paths; set minimally
    for k, v in {
        "poison_type": "basic",
        "poison_rate": 0.005,
        "cover_rate": 0.0,
        "alpha": 0.2,
        "trigger": "badnet_patch_32.png" if dataset == "cifar10" else "badnet_patch_64.png",
        "devices": "0",
        "seed": 2333,
    }.items():
        setattr(args, k, v)
    arch = supervisor.get_arch(args)
    model = arch(num_classes=num_classes)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model = nn.DataParallel(model).cuda()
    model.eval()
    return model, args


def load_inspection(dataset: str):
    if dataset == "cifar10":
        # Use torchvision CIFAR10 so labels are plain ints (SCAn is fragile to 0-d numpy labels).
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
            ]
        )
        ds = torchvision.datasets.CIFAR10(root=str(REPO / "data"), train=True, download=False, transform=tfm)
        note = f"torchvision CIFAR10 train (N={len(ds)})"
        return ds, note
    if dataset == "tiny_imagenet":
        root = REPO / "data/Tiny-imagenet/tiny-imagenet-200"
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        )
        ds = TinyTrainFolder(root, tfm)
        note = f"Tiny-ImageNet train folder (N={len(ds)})"
        return ds, note
    raise ValueError(dataset)


def load_clean(dataset: str):
    from utils import tools

    _, data_transform, _, _, _ = supervisor.get_transforms(
        SimpleNamespace(
            dataset=dataset,
            no_normalize=False,
            poison_type="basic",
            model="resnet18",
        )
    )
    clean_set_dir = REPO / "clean_set" / dataset / "clean_split"
    return tools.IMG_Dataset(
        data_dir=str(clean_set_dir / "data"),
        label_path=str(clean_set_dir / "clean_labels"),
        transforms=data_transform,
    )


def time_one(name, fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        out = fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n = len(out) if out is not None else -1
        return {"method": name, "ok": True, "seconds": dt, "n_suspicious": n, "error": ""}
    except Exception as e:
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return {"method": name, "ok": False, "seconds": dt, "n_suspicious": -1, "error": f"{type(e).__name__}: {e}"}


def main():
    configs = [
        (
            "cifar10",
            10,
            REPO
            / "poisoned_train_set/cifar10/basic_0.005_alpha=0.200_trigger=badnet_patch_32.png_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt",
        ),
        (
            "tiny_imagenet",
            200,
            REPO
            / "poisoned_train_set/tiny_imagenet/basic_0.005_alpha=0.200_trigger=badnet_patch_64.png_poison_seed=2333_arch=ResNet18_tiny_imagenet/ResNet18_tiny_imagenet.pt",
        ),
    ]

    results = []
    for dataset, num_classes, ckpt in configs:
        print("=" * 72)
        print(f"DATASET={dataset}  ckpt={ckpt.name}")
        inspection, note = load_inspection(dataset)
        print(f"inspection: {note}")
        clean = load_clean(dataset)
        print(f"clean split: N={len(clean)}")
        model, base_args = load_model(dataset, ckpt, num_classes)
        base_args.poison_rate = 0.005
        base_args.dataset = dataset

        jobs = []

        def run_ss():
            return spectral_signature.cleanser(inspection, model, num_classes, base_args)

        def run_ac():
            # cleansers AC raises for tiny_imagenet; threshold is unused in the loop.
            ac_args = SimpleNamespace(**vars(base_args))
            if dataset == "tiny_imagenet":
                ac_args.dataset = "imagenette"
            return activation_clustering.cleanser(inspection, model, num_classes, ac_args)

        def run_scan():
            return scan.cleanser(inspection, clean, model, num_classes)

        def run_spectre():
            # Official cleanser.py SPECTRE path needs Julia (not installed here).
            # Time the in-repo Python SPECTRE implementation instead.
            return spectre_python.cleanser(inspection, model, num_classes, base_args, oracle_clean_set=clean)

        jobs = [
            ("SS", run_ss),
            ("AC", run_ac),
            ("SCAn", run_scan),
            ("SPECTRE_python", run_spectre),
        ]

        for name, fn in jobs:
            print(f"\n>>> {dataset} / {name}")
            row = time_one(name, fn)
            row["dataset"] = dataset
            row["n_inspection"] = len(inspection)
            print(
                f"[{dataset}] {name}: {row['seconds']:.1f}s  ok={row['ok']}  "
                f"suspicious={row['n_suspicious']}  {row['error']}"
            )
            results.append(row)
            if not row["ok"]:
                traceback.print_exc()

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("SUMMARY (seconds)")
    print(f"{'dataset':<14}{'method':<16}{'seconds':>10}{'ok':>6}")
    for r in results:
        print(f"{r['dataset']:<14}{r['method']:<16}{r['seconds']:>10.1f}{str(r['ok']):>6}")

    out = REPO / "logs" / "cleanser_feature_defense_timing.txt"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        f.write("dataset\tmethod\tseconds\tok\tn_suspicious\tn_inspection\terror\n")
        for r in results:
            f.write(
                f"{r['dataset']}\t{r['method']}\t{r['seconds']:.3f}\t{r['ok']}\t"
                f"{r['n_suspicious']}\t{r['n_inspection']}\t{r['error']}\n"
            )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
