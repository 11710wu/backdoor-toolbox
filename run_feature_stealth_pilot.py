#!/usr/bin/env python3
"""Feature-level stealth pilot for existing backdoor checkpoints.

This script is intentionally standalone: it reads the paper-analysis master
table for ASR matching, selects a small ASR-stratified pilot set, extracts
victim-model penultimate features, trains a held-out linear probe per
checkpoint, and writes one JSON, two figures, and one Markdown report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_ORIG_TORCH_LOAD = torch.load


def torch_load(path: Path | str, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(path, *args, **kwargs)


torch.load = torch_load  # PyTorch >=2.6 compatibility for this trusted local repo.

from PIL import Image
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from utils import supervisor  # noqa: E402


ATTACKS = ["basic", "blend", "adaptive_blend", "adaptive_patch", "WaNet", "SIG", "upgd", "belt"]
ASR_BINS = ["Q1_low", "Q2_mid_low", "Q3_mid_high", "Q4_high"]
ARCH_TO_MODEL = {
    "ResNet18": "resnet18",
    "ResNet34": "resnet34",
    "ResNet50": "resnet50",
    "SmallCNN": "small_cnn",
    "mobilenetv2": "mobilenetv2",
    "vgg19_bn": "vgg19_bn",
    "densenet121": "densenet121",
}
NUM_CLASSES = {"cifar10": 10, "mnistm": 10, "tiny_imagenet": 200}
SOURCE_FAILURE_THRESHOLD = 0.05
ASR_RANGE_LIMIT = 0.05


@dataclass
class Candidate:
    experiment_id: str
    attack: str
    dataset: str
    architecture: str
    arch_base: str
    model_name: str
    result_dir: Path
    checkpoint_path: Path
    source_asr: float
    target_transfer_asr: float
    selection_asr_type: str
    selection_asr_value: float
    poison_rate: float
    cover_rate: float
    mask_rate: float
    label_mode: str
    strength_name: str
    strength_value: float
    folder_name: str
    target_class: int
    victim_seed: int = 2333
    asr_bin: str = ""
    selection_reason: str = ""
    attack_strength: str = ""
    args_dict: Dict[str, Any] = field(default_factory=dict)


class FeatureSubset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        probe_labels: Sequence[int],
        transform,
        actual_img_set: Optional[torch.Tensor] = None,
    ):
        self.base_dataset = base_dataset
        self.indices = list(map(int, indices))
        self.probe_labels = list(map(int, probe_labels))
        self.transform = transform
        self.actual_img_set = actual_img_set

    def __len__(self) -> int:
        return len(self.indices)

    def _to_pil(self, x):
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, torch.Tensor):
            if x.ndim == 2:
                x = x.unsqueeze(0)
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            return transforms.ToPILImage()(torch.clamp(x.cpu(), 0, 1))
        arr = np.asarray(x)
        return Image.fromarray(arr).convert("RGB")

    def __getitem__(self, item: int):
        idx = self.indices[item]
        if self.actual_img_set is not None:
            raw = self.actual_img_set[idx]
            img = self._to_pil(raw)
        else:
            img, _ = self.base_dataset[idx]
            img = self._to_pil(img)
        return self.transform(img), int(self.probe_labels[item]), idx


class MNISTMTrain(Dataset):
    def __init__(self, root: Path, mnist_root: Path, transform=None):
        self.data = np.load(root / "train.npy")
        mnist = datasets.MNIST(root=str(mnist_root), train=True, download=False)
        self.targets = [int(x) for x in mnist.targets]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.data[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if math.isnan(value) else value
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def safe_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def parse_numeric(pattern: str, text: str, default: float = float("nan")) -> float:
    m = re.search(pattern, text)
    return finite_float(m.group(1).rstrip(".")) if m else default


def parse_text(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else default


def infer_checkpoint_path(result_dir: Path, attack: str, arch: str, seed: int = 2333) -> Path:
    if attack == "belt":
        return result_dir / f"{arch}_belt_aug_model_seed={seed}.pt"
    return result_dir / f"{arch}.pt"


def make_candidate_args(row: pd.Series, data_root: Path) -> Dict[str, Any]:
    folder = str(row["folder_name"])
    attack = str(row["attack_type"])
    dataset = str(row["dataset"])
    arch_base = str(row["arch_base"])
    poison_rate = finite_float(row.get("poison_rate"))
    cover_rate = finite_float(row.get("cover_rate"))
    if not math.isfinite(cover_rate):
        cover_rate = 0.0
    mask_rate = finite_float(row.get("mask_rate"))
    if not math.isfinite(mask_rate):
        mask_rate = 0.2
    alpha = parse_numeric(r"(?:^|_)alpha=([0-9.]+)", folder, 0.2)
    delta = parse_numeric(r"(?:^|_)delta=([0-9.]+)", folder, 30.0)
    f = parse_numeric(r"(?:^|_)f=([0-9.]+)", folder, 6.0)
    s = parse_numeric(r"(?:^|_)s=([0-9.]+)", folder, 0.5)
    k = int(parse_numeric(r"(?:^|_)k=([0-9.]+)", folder, 4.0))
    eps = parse_numeric(r"(?:^|_)eps=([0-9.]+)", folder, 8.0)
    upgd_steps = int(parse_numeric(r"(?:^|_)steps=([0-9.]+)", folder, 100.0))
    upgd_mult = int(parse_numeric(r"(?:^|_)mult=([0-9.]+)", folder, 5.0))
    constraint = parse_text(r"(?:^|_)constraint=([^_]+)", folder, "Linf")
    trigger = parse_text(r"(?:^|_)trigger=([^_]+(?:_[^_]+)*?\.(?:png|jpg|jpeg))", folder, None)
    if trigger is None:
        trigger = config.trigger_default[dataset][attack]
    label_mode = str(row.get("label_mode") or "clean")
    if label_mode == "nan":
        label_mode = "clean"
    return {
        "dataset": dataset,
        "poison_type": attack,
        "poison_rate": poison_rate,
        "cover_rate": cover_rate,
        "alpha": alpha,
        "test_alpha": None,
        "label_mode": label_mode,
        "trigger": trigger,
        "no_aug": False,
        "no_normalize": attack in {"upgd", "belt"},
        "devices": "0",
        "seed": 2333,
        "s": s,
        "k": k,
        "delta": delta,
        "f": f,
        "eps": eps,
        "constraint": constraint,
        "upgd_steps": upgd_steps,
        "upgd_steps_multiplier": upgd_mult,
        "mask_rate": mask_rate,
        "model": ARCH_TO_MODEL.get(arch_base, arch_base.lower()),
        "model_path": None,
        "defense": None,
        "cleanser": None,
        "train_poison_dir": str(Path(row["result_dir"])),
        "data_root": str(data_root),
    }


def discover_experiments(data_root: Path, output_dir: Path) -> Tuple[List[Candidate], Dict[str, Any]]:
    master_path = REPO_ROOT / "analysis-transfer-asr2" / "paper_analysis_outputs" / "master_results.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master table: {master_path}")
    df = pd.read_csv(master_path)
    data_root_resolved = data_root.resolve()
    rows = []
    skipped = {"not_requested_attack": 0, "outside_data_root": 0, "missing_transfer": 0, "missing_checkpoint": 0}
    for _, row in df.iterrows():
        attack = str(row.get("attack_type"))
        if attack not in ATTACKS:
            skipped["not_requested_attack"] += 1
            continue
        result_dir = Path(str(row.get("result_dir", "")))
        try:
            result_dir.resolve().relative_to(data_root_resolved)
        except Exception:
            skipped["outside_data_root"] += 1
            continue
        if str(row.get("is_main_transfer_dataset")).lower() not in {"true", "1"}:
            continue
        target_asr = finite_float(row.get("transfer_asr"))
        if not math.isfinite(target_asr):
            skipped["missing_transfer"] += 1
            continue
        arch = str(row.get("arch"))
        ckpt = infer_checkpoint_path(result_dir, attack, arch)
        if not ckpt.exists():
            skipped["missing_checkpoint"] += 1
            continue
        dataset = str(row.get("dataset"))
        if dataset not in NUM_CLASSES:
            continue
        args_dict = make_candidate_args(row, data_root_resolved)
        strength_name = str(row.get("strength_name") or "")
        strength_value = finite_float(row.get("strength_value"))
        if strength_name and strength_name != "nan" and math.isfinite(strength_value):
            attack_strength = f"{strength_name}={strength_value:g}"
        else:
            attack_strength = ""
        rows.append(
            Candidate(
                experiment_id=f"{dataset}/{row.get('folder_name')}",
                attack=attack,
                dataset=dataset,
                architecture=arch,
                arch_base=str(row.get("arch_base")),
                model_name=args_dict["model"],
                result_dir=result_dir,
                checkpoint_path=ckpt,
                source_asr=finite_float(row.get("source_asr")),
                target_transfer_asr=target_asr,
                selection_asr_type="target_transfer_asr",
                selection_asr_value=target_asr,
                poison_rate=finite_float(row.get("poison_rate")),
                cover_rate=finite_float(row.get("cover_rate")),
                mask_rate=finite_float(row.get("mask_rate")),
                label_mode=str(row.get("label_mode") or "clean"),
                strength_name=strength_name,
                strength_value=strength_value,
                folder_name=str(row.get("folder_name")),
                target_class=int(config.target_class[dataset]),
                attack_strength=attack_strength,
                args_dict=args_dict,
            )
        )
    return rows, {"master_path": str(master_path), "skipped": skipped, "candidate_count": len(rows)}


def diversity_key(c: Candidate) -> Tuple[str, str, str, float, float, float, str]:
    return (
        str(c.victim_seed),
        c.attack_strength,
        str(c.poison_rate),
        finite_float(c.cover_rate),
        finite_float(c.mask_rate),
        c.dataset,
        c.arch_base,
    )


def choose_diverse(items: List[Candidate], k: int) -> List[Candidate]:
    if len(items) <= k:
        return list(items)
    picked: List[Candidate] = []
    seen_values: List[set] = [set() for _ in range(7)]
    pool = list(items)
    while pool and len(picked) < k:
        scored = []
        for item in pool:
            key = diversity_key(item)
            score = sum(1 for i, value in enumerate(key) if value not in seen_values[i])
            scored.append((score, abs(item.selection_asr_value - np.median([x.selection_asr_value for x in pool])), item))
        scored.sort(key=lambda x: (-x[0], x[1], x[2].experiment_id))
        item = scored[0][2]
        picked.append(item)
        for i, value in enumerate(diversity_key(item)):
            seen_values[i].add(value)
        pool.remove(item)
    return picked


def select_asr_stratified_checkpoints(
    candidates: List[Candidate],
    checkpoints_per_attack: int,
    samples_per_bin: int,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    selected: List[Candidate] = []
    summary: Dict[str, Any] = {}
    for attack in ATTACKS:
        group = sorted([c for c in candidates if c.attack == attack], key=lambda c: c.selection_asr_value)
        if not group:
            summary[attack] = {"candidate_count": 0, "selected_count": 0}
            continue
        values = np.array([c.selection_asr_value for c in group], dtype=float)
        chunks = np.array_split(group, 4)
        attack_pick: List[Candidate] = []
        per_bin_counts: Dict[str, int] = {}
        for bin_name, chunk in zip(ASR_BINS, chunks):
            chunk_list = list(chunk)
            for item in chunk_list:
                item.asr_bin = bin_name
            chosen = choose_diverse(chunk_list, samples_per_bin)
            for item in chosen:
                item.selection_reason = f"{bin_name}: ASR-stratified pilot selection by target_transfer_asr"
            attack_pick.extend(chosen)
            per_bin_counts[bin_name] = len(chosen)
        if len(attack_pick) < min(checkpoints_per_attack, len(group)):
            remaining = [c for c in group if c not in attack_pick]
            for item in choose_diverse(remaining, min(checkpoints_per_attack, len(group)) - len(attack_pick)):
                if not item.asr_bin:
                    rank = group.index(item)
                    item.asr_bin = ASR_BINS[min(3, int(rank / max(1, len(group)) * 4))]
                item.selection_reason = f"{item.asr_bin}: adjacent-bin fill after sparse bin"
                attack_pick.append(item)
                per_bin_counts[item.asr_bin] = per_bin_counts.get(item.asr_bin, 0) + 1
        attack_pick = sorted(attack_pick, key=lambda c: (ASR_BINS.index(c.asr_bin), c.selection_asr_value, c.experiment_id))
        attack_pick = attack_pick[: min(checkpoints_per_attack, len(group))]
        selected.extend(attack_pick)
        summary[attack] = {
            "candidate_count": len(group),
            "selected_count": len(attack_pick),
            "asr_min": float(np.min(values)),
            "asr_max": float(np.max(values)),
            "asr_median": float(np.median(values)),
            "asr_range_limited": bool(float(np.max(values) - np.min(values)) < ASR_RANGE_LIMIT),
            "bin_selected_counts": {b: sum(1 for c in attack_pick if c.asr_bin == b) for b in ASR_BINS},
            "candidate_bin_counts": {b: len(chunks[i]) for i, b in enumerate(ASR_BINS)},
            "distinct_victim_seeds": len({c.victim_seed for c in attack_pick}),
            "used_source_asr_fallback": any(c.selection_asr_type != "target_transfer_asr" for c in attack_pick),
        }
    return selected, summary


def sample_availability(candidate: Candidate) -> Tuple[int, int]:
    labels_path = candidate.result_dir / "labels"
    if not labels_path.exists():
        return 0, 0
    labels = torch.load(labels_path, map_location="cpu").long()
    cover = tensor_indices(candidate.result_dir / "cover_indices")
    if (candidate.result_dir / "pmarks").exists():
        pmarks = torch.load(candidate.result_dir / "pmarks", map_location="cpu")
        poison = {i for i, mark in enumerate(pmarks.cpu().view(-1).tolist()) if int(mark) == 1}
        cover.update({i for i, mark in enumerate(pmarks.cpu().view(-1).tolist()) if int(mark) == 2})
    else:
        poison = tensor_indices(candidate.result_dir / "poison_indices").difference(cover)
    clean = [
        i
        for i in range(len(labels))
        if int(labels[i]) == int(candidate.target_class) and i not in poison and i not in cover
    ]
    return len(clean), len(poison)


def select_dataset_stratified_checkpoints(
    candidates: List[Candidate],
    datasets_wanted: Sequence[str],
    checkpoints_per_dataset: int,
    samples_per_bin: int,
    feasibility_precheck: bool,
) -> Tuple[List[Candidate], Dict[str, Any], Dict[str, Any]]:
    selected: List[Candidate] = []
    dataset_summary: Dict[str, Any] = {}
    availability_cache: Dict[str, Tuple[int, int]] = {}
    for dataset in datasets_wanted:
        group = [c for c in candidates if c.dataset == dataset]
        feasible_group = []
        filtered = []
        for c in group:
            if feasibility_precheck:
                try:
                    availability_cache[c.experiment_id] = sample_availability(c)
                except Exception:
                    availability_cache[c.experiment_id] = (0, 0)
                n_clean, n_poison = availability_cache[c.experiment_id]
                if min(n_clean, n_poison) <= 0:
                    filtered.append(c)
                    continue
            feasible_group.append(c)
        group = sorted(feasible_group, key=lambda c: c.selection_asr_value)
        if not group:
            dataset_summary[dataset] = {
                "candidate_count": 0,
                "selected_count": 0,
                "filtered_infeasible_count": len(filtered),
            }
            continue
        values = np.array([c.selection_asr_value for c in group], dtype=float)
        chunks = np.array_split(group, 4)
        dataset_pick: List[Candidate] = []
        for bin_name, chunk in zip(ASR_BINS, chunks):
            chunk_list = list(chunk)
            for item in chunk_list:
                item.asr_bin = bin_name
            chosen = choose_diverse(chunk_list, samples_per_bin)
            for item in chosen:
                item.selection_reason = (
                    f"{dataset} {bin_name}: dataset-level ASR-stratified pilot "
                    "selection by target_transfer_asr"
                )
            dataset_pick.extend(chosen)
        target_n = min(checkpoints_per_dataset, len(group))
        if len(dataset_pick) < target_n:
            remaining = [c for c in group if c not in dataset_pick]
            fill = choose_diverse(remaining, target_n - len(dataset_pick))
            for item in fill:
                if not item.asr_bin:
                    rank = group.index(item)
                    item.asr_bin = ASR_BINS[min(3, int(rank / max(1, len(group)) * 4))]
                item.selection_reason = f"{dataset} {item.asr_bin}: adjacent-bin fill after feasibility filtering"
                dataset_pick.append(item)
        dataset_pick = sorted(dataset_pick, key=lambda c: (ASR_BINS.index(c.asr_bin), c.selection_asr_value, c.attack, c.experiment_id))
        dataset_pick = dataset_pick[:target_n]
        selected.extend(dataset_pick)
        dataset_summary[dataset] = {
            "candidate_count": len(feasible_group),
            "raw_candidate_count": len(feasible_group) + len(filtered),
            "filtered_infeasible_count": len(filtered),
            "selected_count": len(dataset_pick),
            "asr_min": float(np.min(values)),
            "asr_max": float(np.max(values)),
            "asr_median": float(np.median(values)),
            "asr_range_limited": bool(float(np.max(values) - np.min(values)) < ASR_RANGE_LIMIT),
            "bin_selected_counts": {b: sum(1 for c in dataset_pick if c.asr_bin == b) for b in ASR_BINS},
            "candidate_bin_counts": {b: len(chunks[i]) for i, b in enumerate(ASR_BINS)},
            "selected_attack_counts": {a: sum(1 for c in dataset_pick if c.attack == a) for a in ATTACKS},
            "filtered_infeasible_attack_counts": {a: sum(1 for c in filtered if c.attack == a) for a in ATTACKS},
            "selected_arch_counts": dict(pd.Series([c.arch_base for c in dataset_pick]).value_counts()) if dataset_pick else {},
        }
    attack_summary: Dict[str, Any] = {}
    for attack in ATTACKS:
        attack_candidates = [c for c in candidates if c.attack == attack and c.dataset in datasets_wanted]
        attack_selected = [c for c in selected if c.attack == attack]
        vals = [c.selection_asr_value for c in attack_candidates]
        attack_summary[attack] = {
            "candidate_count": len(attack_candidates),
            "selected_count": len(attack_selected),
            "asr_min": float(np.min(vals)) if vals else float("nan"),
            "asr_max": float(np.max(vals)) if vals else float("nan"),
            "asr_median": float(np.median(vals)) if vals else float("nan"),
            "asr_range_limited": bool((np.max(vals) - np.min(vals)) < ASR_RANGE_LIMIT) if vals else False,
            "bin_selected_counts": {b: sum(1 for c in attack_selected if c.asr_bin == b) for b in ASR_BINS},
            "distinct_victim_seeds": len({c.victim_seed for c in attack_selected}),
            "used_source_asr_fallback": any(c.selection_asr_type != "target_transfer_asr" for c in attack_selected),
        }
    return selected, attack_summary, dataset_summary


def value_counts_text(values: Sequence[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def poison_rate_label(candidate: Candidate) -> str:
    return "NA" if not math.isfinite(float(candidate.poison_rate)) else f"{float(candidate.poison_rate):g}"


def strength_label(candidate: Candidate) -> str:
    if candidate.attack_strength:
        return candidate.attack_strength
    if candidate.strength_name and candidate.strength_name != "nan" and math.isfinite(float(candidate.strength_value)):
        return f"{candidate.strength_name}={float(candidate.strength_value):g}"
    return "NA"


def coverage_asr_buckets(group: Sequence[Candidate]) -> Dict[str, str]:
    ordered = sorted(group, key=lambda c: (c.selection_asr_value, c.experiment_id))
    chunks = np.array_split(ordered, 4)
    mapping: Dict[str, str] = {}
    for name, chunk in zip(("low", "mid_low", "mid_high", "high"), chunks):
        for item in chunk:
            mapping[item.experiment_id] = name
    return mapping


def choose_coverage_diverse(items: List[Candidate], k: int) -> List[Candidate]:
    if len(items) <= k:
        return list(items)
    asr_bucket = coverage_asr_buckets(items)
    counts: Dict[str, Dict[str, int]] = {
        "attack": {},
        "poison_rate": {},
        "strength": {},
        "arch": {},
        "asr_bucket": {},
    }
    weights = {"attack": 4.0, "poison_rate": 3.0, "strength": 2.5, "arch": 1.0, "asr_bucket": 1.5}
    picked: List[Candidate] = []
    pool = sorted(items, key=lambda c: (c.dataset, c.attack, poison_rate_label(c), strength_label(c), c.arch_base, c.selection_asr_value, c.experiment_id))
    while pool and len(picked) < k:
        scored = []
        for item in pool:
            dims = {
                "attack": item.attack,
                "poison_rate": poison_rate_label(item),
                "strength": strength_label(item),
                "arch": item.arch_base,
                "asr_bucket": asr_bucket.get(item.experiment_id, "NA"),
            }
            score = 0.0
            for dim, value in dims.items():
                score += weights[dim] / (1.0 + counts[dim].get(value, 0))
            scored.append((-score, item.selection_asr_value, item.experiment_id, item, dims))
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, _, chosen, dims = scored[0]
        picked.append(chosen)
        for dim, value in dims.items():
            counts[dim][value] = counts[dim].get(value, 0) + 1
        pool.remove(chosen)
    return picked


def balanced_dataset_quotas(datasets_wanted: Sequence[str], total_checkpoints: int) -> Dict[str, int]:
    if not datasets_wanted:
        return {}
    base_n = total_checkpoints // len(datasets_wanted)
    rem = total_checkpoints % len(datasets_wanted)
    return {dataset: base_n + (1 if i < rem else 0) for i, dataset in enumerate(datasets_wanted)}


def select_coverage_balanced_checkpoints(
    candidates: List[Candidate],
    datasets_wanted: Sequence[str],
    total_checkpoints: int,
    feasibility_precheck: bool,
) -> Tuple[List[Candidate], Dict[str, Any], Dict[str, Any]]:
    selected: List[Candidate] = []
    dataset_summary: Dict[str, Any] = {}
    quotas = balanced_dataset_quotas(datasets_wanted, total_checkpoints)
    for dataset in datasets_wanted:
        group_raw = [c for c in candidates if c.dataset == dataset]
        feasible_group: List[Candidate] = []
        filtered: List[Candidate] = []
        for c in group_raw:
            if feasibility_precheck:
                try:
                    n_clean, n_poison = sample_availability(c)
                except Exception:
                    n_clean, n_poison = 0, 0
                if min(n_clean, n_poison) <= 0:
                    filtered.append(c)
                    continue
            feasible_group.append(c)
        quota = min(quotas.get(dataset, 0), len(feasible_group))
        dataset_pick = choose_coverage_diverse(feasible_group, quota)
        dataset_pick = sorted(dataset_pick, key=lambda c: (c.selection_asr_value, c.attack, poison_rate_label(c), strength_label(c), c.experiment_id))
        for item in dataset_pick:
            item.asr_bin = ""
            item.selection_reason = (
                "coverage-balanced pilot selection over dataset, attack, poison_rate, "
                "attack strength, architecture, and target_transfer_asr range"
            )
        selected.extend(dataset_pick)
        vals = [c.selection_asr_value for c in feasible_group]
        dataset_summary[dataset] = {
            "selection_mode": "coverage_balanced",
            "quota": quotas.get(dataset, 0),
            "raw_candidate_count": len(group_raw),
            "candidate_count": len(feasible_group),
            "filtered_infeasible_count": len(filtered),
            "selected_count": len(dataset_pick),
            "asr_min": float(np.min(vals)) if vals else float("nan"),
            "asr_max": float(np.max(vals)) if vals else float("nan"),
            "asr_median": float(np.median(vals)) if vals else float("nan"),
            "selected_attack_counts": {a: sum(1 for c in dataset_pick if c.attack == a) for a in ATTACKS},
            "filtered_infeasible_attack_counts": {a: sum(1 for c in filtered if c.attack == a) for a in ATTACKS},
            "selected_poison_rate_counts": value_counts_text([poison_rate_label(c) for c in dataset_pick]),
            "selected_strength_counts": value_counts_text([strength_label(c) for c in dataset_pick]),
            "selected_arch_counts": value_counts_text([c.arch_base for c in dataset_pick]),
        }
    attack_summary: Dict[str, Any] = {}
    for attack in ATTACKS:
        attack_candidates = [c for c in candidates if c.attack == attack and c.dataset in datasets_wanted]
        attack_selected = [c for c in selected if c.attack == attack]
        vals = [c.selection_asr_value for c in attack_candidates]
        attack_summary[attack] = {
            "candidate_count": len(attack_candidates),
            "selected_count": len(attack_selected),
            "asr_min": float(np.min(vals)) if vals else float("nan"),
            "asr_max": float(np.max(vals)) if vals else float("nan"),
            "asr_median": float(np.median(vals)) if vals else float("nan"),
            "distinct_victim_seeds": len({c.victim_seed for c in attack_selected}),
            "used_source_asr_fallback": any(c.selection_asr_type != "target_transfer_asr" for c in attack_selected),
        }
    selected = sorted(selected, key=lambda c: (c.dataset, c.selection_asr_value, c.attack, c.experiment_id))
    return selected[:total_checkpoints], attack_summary, dataset_summary


def load_base_train_dataset(dataset: str) -> Dataset:
    if dataset == "cifar10":
        return datasets.CIFAR10(str(REPO_ROOT / "data" / "cifar10"), train=True, download=False, transform=transforms.ToTensor())
    if dataset == "tiny_imagenet":
        return datasets.ImageFolder(str(Path(config.tiny_imagenet_dir) / "train"), transform=transforms.ToTensor())
    if dataset == "mnistm":
        return MNISTMTrain(Path(config.mnistm_dir), REPO_ROOT / "data" / "mnist", transform=transforms.ToTensor())
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_actual_img_set(result_dir: Path, dataset: str) -> Tuple[Optional[torch.Tensor], str]:
    for name in ("data", "imgs"):
        path = result_dir / name
        if path.is_file():
            return torch.load(path, map_location="cpu"), str(path)
    return None, "reconstructed_from_original_train_set_and_poison_artifacts"


def tensor_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    value = torch.load(path, map_location="cpu")
    if isinstance(value, torch.Tensor):
        return {int(x) for x in value.cpu().view(-1).tolist()}
    return {int(x) for x in list(value)}


def build_clean_poison_subset(candidate: Candidate, sample_seed: int, max_samples_per_class: int) -> Dict[str, Any]:
    labels_path = candidate.result_dir / "labels"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels: {labels_path}")
    labels = torch.load(labels_path, map_location="cpu").long()
    n_total = len(labels)
    cover = tensor_indices(candidate.result_dir / "cover_indices")
    if (candidate.result_dir / "pmarks").exists():
        pmarks = torch.load(candidate.result_dir / "pmarks", map_location="cpu")
        poison = {i for i, mark in enumerate(pmarks.cpu().view(-1).tolist()) if int(mark) == 1}
        cover.update({i for i, mark in enumerate(pmarks.cpu().view(-1).tolist()) if int(mark) == 2})
    else:
        poison = tensor_indices(candidate.result_dir / "poison_indices")
        poison = poison.difference(cover)
    clean = [
        i
        for i in range(n_total)
        if int(labels[i]) == int(candidate.target_class) and i not in poison and i not in cover
    ]
    poison_list = sorted(poison)
    stable_offset = int(hashlib.md5(candidate.experiment_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(sample_seed + stable_offset)
    rng.shuffle(clean)
    rng.shuffle(poison_list)
    n = min(len(clean), len(poison_list), int(max_samples_per_class))
    clean_used = sorted(clean[:n])
    poison_used = sorted(poison_list[:n])
    return {
        "clean_indices": clean_used,
        "poison_indices": poison_used,
        "cover_indices": sorted(cover),
        "n_clean_available": len(clean),
        "n_poison_available": len(poison_list),
        "n_clean_used": len(clean_used),
        "n_poison_used": len(poison_used),
    }


def extract_penultimate_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if model.__class__.__name__ == "VGG" and hasattr(model, "partial_forward"):
        feats = model.partial_forward(x)
    else:
        try:
            _, feats = model(x, return_hidden=True)
        except TypeError:
            feats = model.from_input_to_features(x)
            if feats.ndim > 2:
                feats = torch.flatten(feats, 1)
    if feats.ndim > 2:
        feats = torch.flatten(feats, 1)
    return feats


def load_model(candidate: Candidate, device: torch.device) -> torch.nn.Module:
    args = SimpleNamespace(**candidate.args_dict)
    arch = supervisor.get_arch(args)
    model = arch(num_classes=NUM_CLASSES[candidate.dataset])
    state = torch.load(candidate.checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(str(k).startswith("module.") for k in state.keys()):
        state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def get_poison_transform_for_candidate(candidate: Candidate, data_transform) -> Any:
    args = SimpleNamespace(**candidate.args_dict)
    args.train_poison_dir = str(candidate.result_dir)
    _, _, trigger_transform, _, _ = supervisor.get_transforms(args)
    is_normalized = False if candidate.attack in {"upgd", "belt"} else not getattr(args, "no_normalize", False)
    return supervisor.get_poison_transform(
        poison_type=candidate.attack,
        dataset_name=candidate.dataset,
        target_class=candidate.target_class,
        trigger_transform=trigger_transform,
        is_normalized_input=is_normalized,
        alpha=args.alpha,
        trigger_name=args.trigger,
        args=args,
    )


def extract_features_for_indices(
    candidate: Candidate,
    indices: Sequence[int],
    probe_labels: Sequence[int],
    actual_img_set: Optional[torch.Tensor],
    reconstruct_poison: bool,
    poison_index_set: set[int],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    args = SimpleNamespace(**candidate.args_dict)
    _, data_transform, _, _, _ = supervisor.get_transforms(args)
    base_dataset = load_base_train_dataset(candidate.dataset)
    dataset = FeatureSubset(base_dataset, indices, probe_labels, data_transform, actual_img_set=actual_img_set)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    model = load_model(candidate, device)
    poison_transform = None
    if reconstruct_poison:
        poison_transform = get_poison_transform_for_candidate(candidate, data_transform)
    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sample_ids: List[np.ndarray] = []
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device, non_blocking=True).float()
            if reconstruct_poison:
                poison_mask = torch.tensor([int(i) in poison_index_set for i in idx.tolist()], device=device)
                if poison_mask.any():
                    sub_x = x[poison_mask]
                    sub_y = torch.full((sub_x.shape[0],), candidate.target_class, dtype=torch.long, device=device)
                    poisoned, _ = poison_transform.transform(sub_x, sub_y)
                    x[poison_mask] = poisoned
            f = extract_penultimate_features(model, x)
            if not torch.isfinite(f).all():
                raise ValueError("Feature tensor contains NaN or Inf")
            feats.append(f.detach().cpu().numpy())
            labels.append(y.numpy())
            sample_ids.append(idx.numpy())
    feature_layer = "penultimate_before_final_classifier"
    return np.concatenate(feats), np.concatenate(labels), np.concatenate(sample_ids), feature_layer


def train_and_evaluate_probe(
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    split_seed: int,
    test_size: float,
    shuffle_labels: bool = False,
) -> Dict[str, Any]:
    labels = labels.astype(int).copy()
    if shuffle_labels:
        rng = np.random.default_rng(split_seed)
        labels = rng.permutation(labels)
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        random_state=split_seed,
        stratify=labels,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features[train_idx])
    x_test = scaler.transform(features[test_idx])
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=split_seed)
    clf.fit(x_train, labels[train_idx])
    classes = list(clf.classes_)
    if 1 not in classes:
        raise ValueError("Probe classifier did not see poison positive class")
    positive_col = classes.index(1)
    scores = clf.predict_proba(x_test)[:, positive_col]
    pred = clf.predict(x_test)
    auc = float(roc_auc_score(labels[test_idx], scores))
    ap = float(average_precision_score(labels[test_idx], scores))
    bal = float(balanced_accuracy_score(labels[test_idx], pred))
    return {
        "probe_auc_raw": auc,
        "feature_stealth": float(2.0 * (1.0 - max(0.5, auc))),
        "balanced_accuracy": bal,
        "average_precision": ap,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "split_method": "stratified_70_30_train_test_split",
        "grouped_split_used": True,
        "split_seed": split_seed,
        "auc_below_0_45_warning": bool(auc < 0.45),
    }


def clean_clean_check(
    candidate: Candidate,
    clean_indices: Sequence[int],
    actual_img_set: Optional[torch.Tensor],
    n_each: int,
    batch_size: int,
    split_seed: int,
    test_size: float,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    if len(clean_indices) < max(100, 2 * n_each):
        return {"status": "skipped", "reason": "not_enough_clean_target_class_samples"}
    rng = random.Random(split_seed + 17)
    pool = list(clean_indices)
    rng.shuffle(pool)
    a = sorted(pool[:n_each])
    b = sorted(pool[n_each : 2 * n_each])
    features, labels, sample_ids, _ = extract_features_for_indices(
        candidate,
        a + b,
        [0] * len(a) + [1] * len(b),
        actual_img_set=actual_img_set,
        reconstruct_poison=False,
        poison_index_set=set(),
        batch_size=batch_size,
        device=device,
    )
    out = train_and_evaluate_probe(features, labels, sample_ids, split_seed=split_seed + 101, test_size=test_size)
    out["status"] = "ok"
    return out


def run_checkpoint(
    candidate: Candidate,
    args: argparse.Namespace,
    run_sanity: bool,
    device: torch.device,
) -> Dict[str, Any]:
    started = time.time()
    base = {
        "experiment_id": candidate.experiment_id,
        "attack": candidate.attack,
        "dataset": candidate.dataset,
        "architecture": candidate.architecture,
        "attack_strength": candidate.attack_strength,
        "poison_rate": candidate.poison_rate,
        "cover_rate": candidate.cover_rate,
        "mask_rate": candidate.mask_rate,
        "label_mode": candidate.label_mode,
        "victim_seed": candidate.victim_seed,
        "target_class": candidate.target_class,
        "checkpoint_path": str(candidate.checkpoint_path),
        "poisoned_dataset_path": "",
        "source_asr": candidate.source_asr,
        "target_transfer_asr": candidate.target_transfer_asr,
        "selection_asr_type": candidate.selection_asr_type,
        "selection_asr_value": candidate.selection_asr_value,
        "asr_bin": candidate.asr_bin,
        "selection_reason": candidate.selection_reason,
        "feature_layer": "penultimate_before_final_classifier",
        "feature_dimension": None,
        "split_method": "stratified_70_30_train_test_split",
        "grouped_split_used": True,
        "split_seed": args.split_seed,
        "low_sample_warning": False,
        "exploratory_only": False,
        "source_attack_failure_warning": bool(is_finite(candidate.source_asr) and candidate.source_asr < SOURCE_FAILURE_THRESHOLD),
        "reliability_warning": False,
        "sanity_check_passed": None,
        "status": "failed",
        "error": "",
    }
    try:
        subset = build_clean_poison_subset(candidate, args.sample_seed, args.max_samples_per_class)
        base.update({k: v for k, v in subset.items() if k.startswith("n_")})
        n = min(subset["n_clean_used"], subset["n_poison_used"])
        base["low_sample_warning"] = bool(n < 100)
        base["exploratory_only"] = bool(n < 50)
        if n < 2:
            raise ValueError("No balanced clean/poison samples available")
        actual_img_set, dataset_path = load_actual_img_set(candidate.result_dir, candidate.dataset)
        base["poisoned_dataset_path"] = dataset_path
        reconstruct_poison = actual_img_set is None
        indices = subset["clean_indices"] + subset["poison_indices"]
        labels = [0] * len(subset["clean_indices"]) + [1] * len(subset["poison_indices"])
        features, probe_labels, sample_ids, feature_layer = extract_features_for_indices(
            candidate,
            indices,
            labels,
            actual_img_set=actual_img_set,
            reconstruct_poison=reconstruct_poison,
            poison_index_set=set(subset["poison_indices"]),
            batch_size=args.batch_size,
            device=device,
        )
        base["feature_layer"] = feature_layer
        base["feature_dimension"] = int(features.shape[1])
        probe = train_and_evaluate_probe(features, probe_labels, sample_ids, split_seed=args.split_seed, test_size=args.test_size)
        base.update(probe)
        sanity = {}
        if run_sanity:
            random_probe = train_and_evaluate_probe(
                features, probe_labels, sample_ids, split_seed=args.split_seed + 31, test_size=args.test_size, shuffle_labels=True
            )
            clean_probe = clean_clean_check(
                candidate,
                subset["clean_indices"],
                actual_img_set=actual_img_set,
                n_each=min(len(subset["clean_indices"]) // 2, max(50, n)),
                batch_size=args.batch_size,
                split_seed=args.split_seed,
                test_size=args.test_size,
                device=device,
            )
            sanity = {
                "random_label_auc": random_probe.get("probe_auc_raw"),
                "clean_clean_auc": None if clean_probe is None else clean_probe.get("probe_auc_raw"),
                "clean_clean_status": None if clean_probe is None else clean_probe.get("status"),
            }
            random_ok = abs(float(sanity["random_label_auc"]) - 0.5) <= 0.15
            clean_value = sanity["clean_clean_auc"]
            clean_ok = clean_value is None or abs(float(clean_value) - 0.5) <= 0.15
            base["sanity_check_passed"] = bool(random_ok and clean_ok)
            base["reliability_warning"] = bool(not base["sanity_check_passed"])
        if base.get("auc_below_0_45_warning"):
            base["reliability_warning"] = True
        base["sanity_checks"] = sanity
        base["runtime_seconds"] = time.time() - started
        base["status"] = "ok"
        return base
    except Exception as exc:
        base["runtime_seconds"] = time.time() - started
        base["status"] = "failed"
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 2333) -> Tuple[float, float]:
    if len(x) < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        vals.append(stats.spearmanr(x[idx], y[idx]).statistic)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_mean(values: Sequence[float], seed: int = 2333, n_boot: int = 1000) -> Tuple[float, float]:
    arr = np.array([v for v in values if is_finite(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(np.mean(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_results(data: Dict[str, Any]) -> None:
    results = data.get("checkpoint_results", [])
    for attack in ATTACKS:
        rows = [r for r in results if r.get("attack") == attack]
        ok = [r for r in rows if r.get("status") == "ok"]
        valid = [
            r
            for r in ok
            if r.get("selection_asr_type") == "target_transfer_asr"
            and not r.get("exploratory_only")
            and not r.get("reliability_warning")
        ]
        stealth = [float(r["feature_stealth"]) for r in ok if is_finite(r.get("feature_stealth"))]
        ci_low, ci_high = bootstrap_mean(stealth)
        q1 = [r for r in valid if r.get("asr_bin") == "Q1_low"]
        q4 = [r for r in valid if r.get("asr_bin") == "Q4_high"]
        q1_vals = [float(r["feature_stealth"]) for r in q1 if is_finite(r.get("feature_stealth"))]
        q4_vals = [float(r["feature_stealth"]) for r in q4 if is_finite(r.get("feature_stealth"))]
        x = np.array([float(r["target_transfer_asr"]) for r in valid], dtype=float)
        y = np.array([float(r["feature_stealth"]) for r in valid], dtype=float)
        if len(valid) >= 3 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
            rho, pval = stats.spearmanr(x, y)
        else:
            rho, pval = float("nan"), float("nan")
        data.setdefault("attack_level_summary", {})[attack] = {
            "success_count": len(ok),
            "failed_count": len([r for r in rows if r.get("status") != "ok"]),
            "asr_min": float(np.nanmin(x)) if len(x) else float("nan"),
            "asr_max": float(np.nanmax(x)) if len(x) else float("nan"),
            "bin_counts": {b: sum(1 for r in ok if r.get("asr_bin") == b) for b in ASR_BINS},
            "feature_stealth_mean": float(np.mean(stealth)) if stealth else float("nan"),
            "feature_stealth_median": float(np.median(stealth)) if stealth else float("nan"),
            "feature_stealth_std": float(np.std(stealth, ddof=1)) if len(stealth) > 1 else float("nan"),
            "feature_stealth_min": float(np.min(stealth)) if stealth else float("nan"),
            "feature_stealth_max": float(np.max(stealth)) if stealth else float("nan"),
            "feature_stealth_bootstrap_ci_low": ci_low,
            "feature_stealth_bootstrap_ci_high": ci_high,
            "q1_low_asr_feature_stealth_mean": float(np.mean(q1_vals)) if q1_vals else float("nan"),
            "q1_low_asr_feature_stealth_median": float(np.median(q1_vals)) if q1_vals else float("nan"),
            "q4_high_asr_feature_stealth_mean": float(np.mean(q4_vals)) if q4_vals else float("nan"),
            "q4_high_asr_feature_stealth_median": float(np.median(q4_vals)) if q4_vals else float("nan"),
            "high_minus_low_feature_stealth": (
                float(np.mean(q4_vals) - np.mean(q1_vals)) if q1_vals and q4_vals else float("nan")
            ),
            "spearman_rho": float(rho),
            "spearman_p_value": float(pval),
            "valid_correlation_n": len(valid),
        }
    valid = [
        r
        for r in results
        if r.get("status") == "ok"
        and r.get("selection_asr_type") == "target_transfer_asr"
        and not r.get("exploratory_only")
        and not r.get("reliability_warning")
    ]
    data["statistical_analysis"] = {"valid_n": len(valid)}
    for key, rows in [
        ("pooled", valid),
        ("pooled_excluding_source_attack_failure", [r for r in valid if not r.get("source_attack_failure_warning")]),
    ]:
        x = np.array([float(r["target_transfer_asr"]) for r in rows], dtype=float)
        y = np.array([float(r["feature_stealth"]) for r in rows], dtype=float)
        if len(rows) >= 3 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
            rho, pval = stats.spearmanr(x, y)
            lo, hi = bootstrap_spearman(x, y)
        else:
            rho, pval, lo, hi = float("nan"), float("nan"), float("nan"), float("nan")
        data["statistical_analysis"][key] = {
            "spearman_rho": float(rho),
            "spearman_p_value": float(pval),
            "spearman_bootstrap_ci_low": lo,
            "spearman_bootstrap_ci_high": hi,
            "n": len(rows),
        }
    data["dataset_level_summary"] = {}
    for dataset in sorted({r.get("dataset") for r in results if r.get("dataset")}):
        rows = [r for r in results if r.get("dataset") == dataset]
        ok = [r for r in rows if r.get("status") == "ok"]
        valid_rows = [
            r
            for r in ok
            if r.get("selection_asr_type") == "target_transfer_asr"
            and not r.get("exploratory_only")
            and not r.get("reliability_warning")
            and is_finite(r.get("target_transfer_asr"))
            and is_finite(r.get("feature_stealth"))
        ]
        x = np.array([float(r["target_transfer_asr"]) for r in valid_rows], dtype=float)
        y = np.array([float(r["feature_stealth"]) for r in valid_rows], dtype=float)
        if len(valid_rows) >= 3 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
            rho, pval = stats.spearmanr(x, y)
        else:
            rho, pval = float("nan"), float("nan")
        data["dataset_level_summary"][dataset] = {
            "selected_count": len(rows),
            "success_count": len(ok),
            "failed_count": len([r for r in rows if r.get("status") != "ok"]),
            "valid_correlation_n": len(valid_rows),
            "feature_stealth_mean": float(np.mean(y)) if len(y) else float("nan"),
            "target_transfer_asr_mean": float(np.mean(x)) if len(x) else float("nan"),
            "spearman_rho": float(rho),
            "spearman_p_value": float(pval),
            "bin_counts": {b: sum(1 for r in ok if r.get("asr_bin") == b) for b in ASR_BINS},
            "attack_counts": {a: sum(1 for r in ok if r.get("attack") == a) for a in ATTACKS},
        }


def plot_results(data: Dict[str, Any], output_dir: Path) -> None:
    rows = [r for r in data.get("checkpoint_results", []) if r.get("status") == "ok"]
    valid = [
        r
        for r in rows
        if r.get("selection_asr_type") == "target_transfer_asr"
        and is_finite(r.get("target_transfer_asr"))
        and is_finite(r.get("feature_stealth"))
    ]
    colors = dict(zip(ATTACKS, plt.cm.tab10(np.linspace(0, 1, len(ATTACKS)))))
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for attack in ATTACKS:
        sub = [r for r in valid if r.get("attack") == attack]
        if not sub:
            continue
        x = [r["target_transfer_asr"] for r in sub]
        y = [r["feature_stealth"] for r in sub]
        alpha = [0.35 if (r.get("low_sample_warning") or r.get("source_attack_failure_warning") or r.get("reliability_warning")) else 0.85 for r in sub]
        for xi, yi, ai in zip(x, y, alpha):
            ax.scatter(xi, yi, color=colors[attack], alpha=ai, s=34, edgecolor="black", linewidth=0.3)
        ax.scatter([], [], color=colors[attack], label=attack)
    x = np.array([r["target_transfer_asr"] for r in valid], dtype=float)
    y = np.array([r["feature_stealth"] for r in valid], dtype=float)
    if len(valid) >= 3:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="black", linewidth=1.5)
    pooled = data.get("statistical_analysis", {}).get("pooled", {})
    ax.text(
        0.02,
        0.03,
        f"Spearman rho={pooled.get('spearman_rho', float('nan')):.3f}, "
        f"p={pooled.get('spearman_p_value', float('nan')):.3g}\n"
        f"95% CI [{pooled.get('spearman_bootstrap_ci_low', float('nan')):.3f}, "
        f"{pooled.get('spearman_bootstrap_ci_high', float('nan')):.3f}], n={pooled.get('n', 0)}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    ax.set_xlabel("Target-side transfer ASR")
    ax.set_ylabel("Feature stealth")
    ax.set_title("Transfer ASR vs Feature Stealth")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "transfer_asr_vs_feature_stealth.png", dpi=220)
    plt.close(fig)

    if data.get("pilot_config", {}).get("selection_mode") == "coverage_balanced":
        dataset_markers = {"cifar10": "o", "mnistm": "s", "tiny_imagenet": "^"}
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        for i, attack in enumerate(ATTACKS):
            sub = [r for r in valid if r.get("attack") == attack]
            for r in sub:
                jitter_key = int(hashlib.md5(str(r.get("experiment_id")).encode("utf-8")).hexdigest()[:4], 16)
                jitter = ((jitter_key % 100) / 100.0 - 0.5) * 0.34
                ax.scatter(
                    i + jitter,
                    r["feature_stealth"],
                    marker=dataset_markers.get(r.get("dataset"), "o"),
                    color=colors.get(attack),
                    alpha=0.82,
                    s=38,
                    edgecolor="black",
                    linewidth=0.3,
                )
            ys = [r["feature_stealth"] for r in sub]
            if ys:
                ax.hlines(float(np.mean(ys)), i - 0.28, i + 0.28, color="black", linewidth=1.2)
        for dataset, marker in dataset_markers.items():
            ax.scatter([], [], marker=marker, color="#777777", label=dataset)
        ax.set_xticks(range(len(ATTACKS)))
        ax.set_xticklabels(ATTACKS, rotation=30, ha="right")
        ax.set_ylabel("Feature stealth")
        ax.set_title("Coverage-40 Feature Stealth by Attack and Dataset")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Dataset", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "feature_stealth_by_attack_coverage.png", dpi=220)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    offsets = {"Q1_low": -0.24, "Q2_mid_low": -0.08, "Q3_mid_high": 0.08, "Q4_high": 0.24}
    markers = {"Q1_low": "o", "Q2_mid_low": "s", "Q3_mid_high": "^", "Q4_high": "D"}
    for i, attack in enumerate(ATTACKS):
        q_means = {}
        for bin_name in ASR_BINS:
            sub = [r for r in valid if r.get("attack") == attack and r.get("asr_bin") == bin_name]
            ys = [r["feature_stealth"] for r in sub]
            xs = [i + offsets[bin_name]] * len(ys)
            ax.scatter(xs, ys, label=bin_name if i == 0 else None, marker=markers[bin_name], s=34, alpha=0.82)
            if ys:
                q_means[bin_name] = float(np.mean(ys))
        if "Q1_low" in q_means and "Q4_high" in q_means:
            ax.plot([i + offsets["Q1_low"], i + offsets["Q4_high"]], [q_means["Q1_low"], q_means["Q4_high"]], color="black", linewidth=1.2)
    ax.set_xticks(range(len(ATTACKS)))
    ax.set_xticklabels(ATTACKS, rotation=30, ha="right")
    ax.set_ylabel("Feature stealth")
    ax.set_title("Feature Stealth by Attack and ASR Bin")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="ASR bin", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_stealth_by_attack_and_asr_bin.png", dpi=220)
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    if not is_finite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def short_experiment_id(experiment_id: Any) -> str:
    text = str(experiment_id)
    if "/" not in text:
        return text
    dataset, folder = text.split("/", 1)
    folder = folder.replace("_poison_seed=2333", "")
    return f"{dataset}/{folder}"


def load_defense_stealth_lookup() -> Dict[str, Dict[str, float]]:
    master_path = REPO_ROOT / "analysis-transfer-asr2" / "paper_analysis_outputs" / "master_results.csv"
    if not master_path.exists():
        return {}
    df = pd.read_csv(master_path)
    if "is_main_transfer_dataset" in df.columns:
        df = df[df["is_main_transfer_dataset"].astype(str).str.lower().isin({"true", "1"})]
    wanted = ["stealthiness", "sentinet_tpr", "scaleup_tpr", "strip_tpr", "ibd_psc_tpr"]
    lookup: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        result_dir = str(row.get("result_dir", ""))
        if not result_dir:
            continue
        item = {col: finite_float(row.get(col)) for col in wanted}
        if not is_finite(item.get("stealthiness")):
            tprs = [item.get(col) for col in wanted[1:] if is_finite(item.get(col))]
            item["stealthiness"] = float(1.0 - np.mean(tprs)) if len(tprs) == 4 else float("nan")
        lookup[result_dir] = item
    return lookup


def format_count_map(counts: Dict[str, Any], max_items: int = 8) -> str:
    items = [(str(k), int(v)) for k, v in counts.items() if int(v)]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    shown = items[:max_items]
    text = ", ".join(f"{k}:{v}" for k, v in shown)
    if len(items) > max_items:
        text += f", +{len(items) - max_items} more"
    return text or "NA"


def write_report(data: Dict[str, Any], output_dir: Path) -> None:
    sel = data.get("selection_summary", {})
    dataset_sel = data.get("dataset_selection_summary", {})
    dataset_sum = data.get("dataset_level_summary", {})
    attack_sum = data.get("attack_level_summary", {})
    stat = data.get("statistical_analysis", {})
    cfg = data.get("pilot_config", {})
    results = data.get("checkpoint_results", [])
    defense_stealth = load_defense_stealth_lookup()
    lines: List[str] = []
    lines.append("# Feature-level Stealth Pilot Report\n")
    lines.append("## 1. 这次实验在做什么\n")
    lines.append(
        "本实验是 feature-level stealth pilot，不重新训练 victim model，只读取已有 poisoned training set 和已有 checkpoint。"
        "核心问题是：在固定 victim model 后，payload poison 样本和 clean target-class 样本在 penultimate feature 上是否容易被一个简单线性分类器区分。"
        "如果容易区分，说明触发器/中毒机制在特征层留下了明显痕迹；如果不容易区分，则 feature-level stealth 更高。\n"
    )
    lines.append("## 2. 本次选择范围\n")
    if cfg.get("selection_mode") == "coverage_balanced":
        lines.append(
            f"- 数据根目录：`{cfg.get('data_root')}`。\n"
            f"- 本次选择 `{', '.join(cfg.get('datasets', []))}`，总计 {cfg.get('total_checkpoints')} 个 victim checkpoint/config。"
            "这里的 config/checkpoint 不是 probe 训练用的图像样本数。\n"
            "- 选择方式为 coverage-balanced：先按 dataset 尽量均分名额，再在每个 dataset 内贪心覆盖 attack、poison rate、attack strength、architecture 和 target transfer ASR 范围。\n"
            "- 选择指标仍只用 target-side transfer ASR 做覆盖，不用 source ASR；source ASR 只作为源域攻击是否成功的 warning。\n"
            "- 选择前会检查 clean target-class 样本和 payload poison 样本是否都存在。Tiny-ImageNet 的 clean-label SIG/UPGD 在某些 poison_rate 下会把 target class 500 张图全 poison，这类配置会被过滤。\n"
        )
    else:
        lines.append(
            f"- 数据根目录：`{cfg.get('data_root')}`。\n"
            f"- 本次只选择 `{', '.join(cfg.get('datasets', [])) if cfg.get('datasets') else 'all'}`。"
            f"这里的“选取 20 个样本”指每个数据集选取 {cfg.get('checkpoints_per_dataset', 'NA')} 个 victim checkpoint/config，"
            "不是每个 checkpoint 只用 20 张图像。\n"
            "- checkpoint 选择按 target-side transfer ASR 覆盖低、中、高不同迁移强度，便于观察 transfer ASR 与 feature-level stealth 的关系。\n"
            "- 选择指标只用 target-side transfer ASR；source ASR 只作为 source attack 是否成功的 warning，不参与主选择和主迁移性定义。\n"
            "- 选择前会检查 clean target-class 样本和 payload poison 样本是否都存在。Tiny-ImageNet 的 clean-label SIG/UPGD 在某些 poison_rate 下会把 target class 500 张图全 poison，"
            "这种配置没有 clean target-class 对照，会被过滤；因此 Tiny 的 SIG/UPGD 被选中数量可以少一些。\n"
        )
    if dataset_sel:
        lines.append("### Dataset-level 选择\n")
        lines.append("| Dataset | quota | raw candidates | feasible candidates | filtered infeasible | selected | ASR min | ASR max | selected attacks | poison rates | strengths | filtered attacks |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
        for dataset, s in dataset_sel.items():
            lines.append(
                f"| {dataset} | {s.get('quota', cfg.get('checkpoints_per_dataset', 'NA'))} | "
                f"{s.get('raw_candidate_count', 0)} | {s.get('candidate_count', 0)} | "
                f"{s.get('filtered_infeasible_count', 0)} | {s.get('selected_count', 0)} | "
                f"{fmt(s.get('asr_min'))} | {fmt(s.get('asr_max'))} | "
                f"{format_count_map(s.get('selected_attack_counts', {}))} | "
                f"{format_count_map(s.get('selected_poison_rate_counts', {}))} | "
                f"{format_count_map(s.get('selected_strength_counts', {}))} | "
                f"{format_count_map(s.get('filtered_infeasible_attack_counts', {}))} |"
            )
        lines.append("")
    lines.append("### Attack-level 覆盖\n")
    lines.append("| Attack | candidates | selected | success | failed | source fallback |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for attack in ATTACKS:
        s = sel.get(attack, {})
        a = attack_sum.get(attack, {})
        lines.append(
            f"| {attack} | {s.get('candidate_count', 0)} | {s.get('selected_count', 0)} | "
            f"{a.get('success_count', 0)} | {a.get('failed_count', 0)} | "
            f"{s.get('used_source_asr_fallback', False)} |"
        )
    if cfg.get("selection_mode") == "coverage_balanced":
        lines.append("\n本次是 coverage-balanced 主动抽样，因此均值不能解释为完整结果总体中的自然均值；它更适合用来比较不同数据集、毒化率和强度覆盖下的指标行为。\n")
    else:
        lines.append("\n本次是主动分层抽样，因此均值不能解释为各攻击或各数据集在完整结果总体中的自然均值。\n")
    lines.append("## 3. 线性 probe 训练和测试指标\n")
    lines.append(
        "- clean 组：`final_label == target_class`，且不在 payload poison，也不在 cover。\n"
        "- poison 组：Basic/Blend/SIG/UPGD 等使用 `poison_indices`；Adaptive-Blend/Adaptive-Patch/WaNet 使用 `poison_indices` 但排除 cover；BELT 优先使用 `pmarks==1` 表示 payload poison，`pmarks==2` 表示 cover。\n"
        "- cover 样本不进入 clean 组，也不进入 poison 组。\n"
        f"- 每个 checkpoint 内 clean/poison 平衡抽样，每类最多 {cfg.get('max_samples_per_class')} 张图像；"
        "若少于 100 张会标记 `low_sample_warning`，少于 50 张会标记 `exploratory_only`。\n"
        "- 特征：victim model 的 penultimate feature。victim 参数固定，不参与训练。\n"
        "- 分类器：`StandardScaler` 只在 train split 上 fit，然后训练 L2 Logistic Regression。\n"
        "- split：stratified 70/30 train/test。所有最终指标都在 held-out test split 上算。\n"
        "- `probe_auc_raw`：测试集 AUROC，正类是 payload poison。0.5 表示线性不可分，1.0 表示几乎完全可分。\n"
        "- `feature_stealth = 2 * (1 - max(0.5, probe_auc_raw))`。越接近 1 表示越隐蔽；越接近 0 表示越容易被线性检测。\n"
        "- `defense_stealth` 是当前主论文指标中的防御侧隐蔽性：`1 - mean(TPR_SentiNet, TPR_ScaleUp, TPR_STRIP, TPR_IBD_PSC)`，越高表示越难被这四个防御检出。\n"
        "- 同时记录 `balanced_accuracy` 和 `average_precision` 作为辅助指标，但主 feature-level stealth 指标看 `feature_stealth`。\n"
        "- 注意区分两种 AUROC：主 probe AUROC 衡量 clean/poison 是否可分；sanity check 里的 random-label AUROC 和 clean-clean AUROC 应该接近 0.5，接近 0.5 说明没有明显流程泄漏。\n"
        "- 若目录中存在 `data`/`imgs` tensor，则直接使用保存的 poisoned dataset；若没有，则用原始训练集、`poison_indices` 和项目现有 `poison_transform` 重建 payload poison。"
        "UPGD/BELT 按项目原逻辑使用 raw `[0,1]`/`no_normalize` 兼容路径。\n"
    )
    lines.append("## 4. Sanity check\n")
    sanity_rows = [r for r in results if r.get("sanity_checks")]
    lines.append("| Attack | experiment | random-label AUROC | clean-clean AUROC | passed | warning |")
    lines.append("|---|---|---:|---:|---|---|")
    for r in sanity_rows:
        sc = r.get("sanity_checks", {})
        lines.append(
            f"| {r.get('attack')} | `{r.get('experiment_id')}` | "
            f"{fmt(sc.get('random_label_auc'))} | {fmt(sc.get('clean_clean_auc'))} | "
            f"{r.get('sanity_check_passed')} | {r.get('reliability_warning')} |"
        )
    if not sanity_rows:
        lines.append("| NA | NA | NA | NA | NA | NA |")
    lines.append("\nrandom-label 和 clean-clean 控制用于检查线性 probe 流程是否存在明显泄漏；异常结果在正式分析中以 reliability_warning 降权或排除。\n")
    lines.append("## 5. Dataset-level 结果\n")
    lines.append("| Dataset | selected | success | valid n | mean target transfer ASR | mean feature stealth | Spearman rho | p-value |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, s in dataset_sum.items():
        lines.append(
            f"| {dataset} | {s.get('selected_count', 0)} | {s.get('success_count', 0)} | {s.get('valid_correlation_n', 0)} | "
            f"{fmt(s.get('target_transfer_asr_mean'))} | {fmt(s.get('feature_stealth_mean'))} | "
            f"{fmt(s.get('spearman_rho'))} | {fmt(s.get('spearman_p_value'), 4)} |"
        )
    lines.append("")
    lines.append("## 6. 总体 ASR-Feature Stealth 结果\n")
    pooled = stat.get("pooled", {})
    sens = stat.get("pooled_excluding_source_attack_failure", {})
    lines.append(
        f"主分析有效样本 n={pooled.get('n', 0)}，pooled Spearman rho={fmt(pooled.get('spearman_rho'))}，"
        f"p={fmt(pooled.get('spearman_p_value'), 4)}，bootstrap 95% CI=[{fmt(pooled.get('spearman_bootstrap_ci_low'))}, {fmt(pooled.get('spearman_bootstrap_ci_high'))}]。"
        f"排除 source_attack_failure_warning 后 n={sens.get('n', 0)}，rho={fmt(sens.get('spearman_rho'))}，p={fmt(sens.get('spearman_p_value'), 4)}。\n"
    )
    lines.append("图 `transfer_asr_vs_feature_stealth.png` 展示每个 checkpoint 的 target transfer ASR 与 feature stealth。coverage 模式下另有 `feature_stealth_by_attack_coverage.png`，用于查看 attack/dataset 覆盖下的 feature stealth 分布。带 warning 的点透明度较低，因此总体趋势主要参考非 exploratory、非 reliability-warning 的点。\n")
    lines.append("### Checkpoint-level 明细\n")
    lines.append("下表把每个 checkpoint 的源域攻击成功度、迁移 ASR、防御侧隐蔽性和 feature-level 隐蔽性放在一起看。`Source ASR` 只用于确认源域攻击是否成立；主迁移性仍看 `Target transfer ASR`。\n")
    lines.append("| Dataset | Attack | Poison rate | Strength | Source ASR | Target transfer ASR | Defense stealth | Probe AUROC | Feature stealth | Bal Acc | AP | n clean/poison | Experiment |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    detail_rows = sorted(
        [r for r in results if r.get("status") == "ok"],
        key=lambda r: (str(r.get("dataset")), finite_float(r.get("target_transfer_asr")), str(r.get("attack"))),
    )
    for r in detail_rows:
        result_dir = str(Path(str(r.get("checkpoint_path", ""))).parent)
        stealth_row = defense_stealth.get(result_dir, {})
        lines.append(
            f"| {r.get('dataset')} | {r.get('attack')} | {fmt(r.get('poison_rate'))} | {r.get('attack_strength') or 'NA'} | "
            f"{fmt(r.get('source_asr'))} | {fmt(r.get('target_transfer_asr'))} | "
            f"{fmt(stealth_row.get('stealthiness'))} | "
            f"{fmt(r.get('probe_auc_raw'))} | {fmt(r.get('feature_stealth'))} | "
            f"{fmt(r.get('balanced_accuracy'))} | {fmt(r.get('average_precision'))} | "
            f"{r.get('n_clean_used', 'NA')}/{r.get('n_poison_used', 'NA')} | "
            f"`{short_experiment_id(r.get('experiment_id'))}` |"
        )
    lines.append("")
    lines.append("## 7. 各攻击内部结果\n")
    lines.append("| Attack | valid n | ASR range | mean feature stealth | Spearman rho | p-value | interpretation |")
    lines.append("|---|---:|---|---:|---:|---:|---|")
    for attack in ATTACKS:
        a = attack_sum.get(attack, {})
        valid_n = int(a.get("valid_correlation_n", 0) or 0)
        pval = a.get("spearman_p_value")
        if valid_n < 8:
            interp = "有效样本不足，只作为 pilot 线索"
        else:
            rho = a.get("spearman_rho")
            if is_finite(rho) and float(rho) < -0.1:
                interp = "方向为负：迁移 ASR 越高，feature stealth 越低"
            elif is_finite(rho) and float(rho) > 0.1:
                interp = "方向为正：存在反向/异质性信号"
            else:
                interp = "相关方向不明显"
        lines.append(
            f"| {attack} | {a.get('valid_correlation_n', 0)} | {fmt(a.get('asr_min'))}-{fmt(a.get('asr_max'))} | "
            f"{fmt(a.get('feature_stealth_mean'))} | {fmt(a.get('spearman_rho'))} | {fmt(pval, 4)} | {interp} |"
        )
    lines.append("\n每种攻击内部 Spearman 只有在有效 checkpoint 至少 8 个时才适合重点解释；低于该数量时更适合作为 pilot 线索。\n")
    lines.append("## 8. Source ASR 和配置混杂\n")
    fail_warn = [r for r in results if r.get("status") == "ok" and r.get("source_attack_failure_warning")]
    recon = [r for r in results if r.get("status") == "ok" and "reconstructed" in str(r.get("poisoned_dataset_path"))]
    lines.append(
        f"source_attack_failure_warning 的 checkpoint 数量为 {len(fail_warn)}。这些点不能简单解释为迁移失败，因为源域攻击本身可能也弱。"
        f"使用重建 payload poison 的成功 checkpoint 数量为 {len(recon)}；报告解释时应把它看作基于现有 artifacts 的 pilot 近似，而不是完整 poisoned tensor 的逐像素审计。\n"
    )
    if cfg.get("selection_mode") == "coverage_balanced":
        lines.append("coverage-balanced 主动抽样已经尽量覆盖 dataset、attack、poison rate、attack strength 和 architecture，但这些因素仍可能互相绑定；因此当前相关关系不能解释为 ASR 对 feature stealth 的因果作用。\n")
    else:
        lines.append("ASR 分层可能与 strength、poison rate、cover rate、dataset、architecture 同时变化，因此当前相关关系不能解释为 ASR 对 feature stealth 的因果作用。\n")
    lines.append("## 9. 异常结果分析\n")
    ok = [r for r in results if r.get("status") == "ok"]
    high_high = sorted([r for r in ok if is_finite(r.get("target_transfer_asr")) and is_finite(r.get("feature_stealth")) and r["target_transfer_asr"] >= 0.8 and r["feature_stealth"] >= 0.8], key=lambda r: -r["target_transfer_asr"])[:8]
    low_low = sorted([r for r in ok if is_finite(r.get("target_transfer_asr")) and is_finite(r.get("feature_stealth")) and r["target_transfer_asr"] <= 0.2 and r["feature_stealth"] <= 0.2], key=lambda r: r["target_transfer_asr"])[:8]
    near_one_auc_all = sorted([r for r in ok if is_finite(r.get("probe_auc_raw")) and r["probe_auc_raw"] >= 0.95], key=lambda r: -r["probe_auc_raw"])
    near_one_auc = near_one_auc_all[:8]
    lines.append(f"- 高 ASR 且高 feature stealth 的点数: {len(high_high)}。代表例: " + (", ".join(f"`{r['experiment_id']}`" for r in high_high[:3]) or "无") + "\n")
    lines.append(f"- 低 ASR 且低 feature stealth 的点数: {len(low_low)}。代表例: " + (", ".join(f"`{r['experiment_id']}`" for r in low_low[:3]) or "无") + "\n")
    lines.append(f"- AUROC >= 0.95 的点数: {len(near_one_auc_all)}。这类点表示 payload poison 在 penultimate feature 中几乎可线性分离，需要结合 attack/strength/source ASR 和样本量解释。\n")
    failures = [r for r in results if r.get("status") != "ok"]
    if failures:
        lines.append("失败 checkpoint 保留在 JSON 中。最常见失败原因示例：\n")
        for r in failures[:10]:
            lines.append(f"- `{r.get('experiment_id')}`: {r.get('error')}\n")
    lines.append("## 10. 当前结论\n")
    rho = pooled.get("spearman_rho")
    if is_finite(rho) and float(rho) < -0.1:
        overall = "总体呈负相关，初步支持 transfer ASR 越高、feature stealth 越低的 tradeoff。"
    elif is_finite(rho) and float(rho) > 0.1:
        overall = "总体呈正相关，不支持简单 tradeoff，并提示攻击机制异质性或配置混杂。"
    else:
        overall = "总体关系接近零或证据不足，当前 pilot 不支持强 tradeoff 结论。"
    if cfg.get("selection_mode") == "coverage_balanced":
        lines.append(overall + "但本次是 coverage-balanced pilot，各 attack 的 selected n 仍不相同，且 11 个点有 source_attack_failure_warning；是否扩大实验，应优先看 sanity checks、每攻击有效样本数、以及跨数据集/毒化率/强度覆盖后相关方向是否稳定。\n")
    else:
        lines.append(overall + "但本次按 dataset 分层后，各攻击覆盖不均衡，SIG 只有 1 个有效点，BELT 还出现反向信号；是否扩大实验，应优先看 sanity checks、每攻击有效样本数、以及跨数据集/跨攻击的相关方向是否稳定。\n")
    lines.append("## 11. 当前限制\n")
    lines.append(
        "本次是主动抽样 pilot，不代表原始总体比例；相关关系不能解释为因果；不同配置可能产生混杂；feature stealth 不等于现有 detector-evasion stealth；"
        "dirty-label 攻击可能包含原始类别语义差异；单次 70/30 split 有随机性；部分目录若缺 poisoned tensor，payload poison 由现有 artifacts 重建，适合作为 pilot 而非最终像素级复现实验。\n"
    )
    (output_dir / "feature_stealth_pilot_report.md").write_text("\n".join(lines), encoding="utf-8")


def initial_results(
    args: argparse.Namespace,
    selection_summary: Dict[str, Any],
    discovery: Dict[str, Any],
    dataset_selection_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "pilot_config": {
            "data_root": str(Path(args.data_root).resolve()),
            "datasets": getattr(args, "datasets_list", []),
            "selection_mode": args.selection_mode,
            "checkpoints_per_attack": args.checkpoints_per_attack,
            "checkpoints_per_dataset": args.checkpoints_per_dataset,
            "total_checkpoints": args.total_checkpoints,
            "selection_metric": args.selection_metric,
            "asr_bins": args.asr_bins,
            "samples_per_bin": args.samples_per_bin,
            "max_samples_per_class": args.max_samples_per_class,
            "train_ratio": 0.7,
            "test_ratio": args.test_size,
            "sample_seed": args.sample_seed,
            "split_seed": args.split_seed,
            "sample_feasibility_precheck": args.sample_feasibility_precheck,
            "cross_validation": False,
            "discovery": discovery,
        },
        "selection_summary": selection_summary,
        "dataset_selection_summary": dataset_selection_summary or {},
        "checkpoint_results": [],
        "dataset_level_summary": {},
        "attack_level_summary": {},
        "statistical_analysis": {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature-level stealth pilot.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoints-per-attack", type=int, default=20)
    parser.add_argument("--datasets", default="", help="Comma-separated dataset filter, e.g. cifar10,tiny_imagenet.")
    parser.add_argument("--checkpoints-per-dataset", type=int, default=None)
    parser.add_argument("--selection-mode", default="asr_stratified", choices=["asr_stratified", "coverage_balanced"])
    parser.add_argument("--total-checkpoints", type=int, default=40)
    parser.add_argument("--selection-metric", default="target_transfer_asr", choices=["target_transfer_asr"])
    parser.add_argument("--asr-bins", type=int, default=4)
    parser.add_argument("--samples-per-bin", type=int, default=5)
    parser.add_argument("--max-samples-per-class", type=int, default=3000)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--run-sanity-checks", action="store_true")
    parser.add_argument("--sample-feasibility-precheck", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    args.datasets_list = []
    if args.datasets.strip():
        args.datasets_list = [x.strip().replace("-", "_") for x in args.datasets.split(",") if x.strip()]
        unknown = [x for x in args.datasets_list if x not in NUM_CLASSES]
        if unknown:
            raise ValueError(f"Unknown datasets in --datasets: {unknown}")
    return args


def main() -> None:
    args = parse_args()
    if args.asr_bins != 4:
        raise ValueError("This pilot expects --asr-bins 4")
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["POISONED_TRAIN_SET_ROOT"] = str(data_root)
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates, discovery = discover_experiments(data_root, output_dir)
    if args.datasets_list:
        candidates = [c for c in candidates if c.dataset in args.datasets_list]
    if args.selection_mode == "coverage_balanced":
        if not args.datasets_list:
            raise ValueError("--selection-mode coverage_balanced requires --datasets")
        selected, selection_summary, dataset_selection_summary = select_coverage_balanced_checkpoints(
            candidates,
            datasets_wanted=args.datasets_list,
            total_checkpoints=args.total_checkpoints,
            feasibility_precheck=args.sample_feasibility_precheck,
        )
    elif args.checkpoints_per_dataset is not None:
        if not args.datasets_list:
            raise ValueError("--checkpoints-per-dataset requires --datasets")
        selected, selection_summary, dataset_selection_summary = select_dataset_stratified_checkpoints(
            candidates,
            datasets_wanted=args.datasets_list,
            checkpoints_per_dataset=args.checkpoints_per_dataset,
            samples_per_bin=args.samples_per_bin,
            feasibility_precheck=args.sample_feasibility_precheck,
        )
    else:
        selected, selection_summary = select_asr_stratified_checkpoints(
            candidates,
            checkpoints_per_attack=args.checkpoints_per_attack,
            samples_per_bin=args.samples_per_bin,
        )
        dataset_selection_summary = {}
    selected_keys = {c.experiment_id for c in selected}
    result_path = output_dir / "feature_stealth_pilot_results.json"
    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        existing = {r.get("experiment_id") for r in data.get("checkpoint_results", []) if r.get("status") == "ok"}
        data["selection_summary"] = selection_summary
        data["dataset_selection_summary"] = dataset_selection_summary
        data["pilot_config"]["discovery"] = discovery
        data["pilot_config"]["datasets"] = args.datasets_list
        data["pilot_config"]["selection_mode"] = args.selection_mode
        data["pilot_config"]["checkpoints_per_dataset"] = args.checkpoints_per_dataset
        data["pilot_config"]["total_checkpoints"] = args.total_checkpoints
        data["pilot_config"]["sample_feasibility_precheck"] = args.sample_feasibility_precheck
    else:
        data = initial_results(args, selection_summary, discovery, dataset_selection_summary)
        existing = set()
    sanity_targets = set()
    if args.run_sanity_checks:
        for attack in ATTACKS:
            attack_sel = [c for c in selected if c.attack == attack]
            if args.selection_mode == "coverage_balanced":
                if attack_sel:
                    sanity_targets.add(min(attack_sel, key=lambda c: c.selection_asr_value).experiment_id)
                    sanity_targets.add(max(attack_sel, key=lambda c: c.selection_asr_value).experiment_id)
            else:
                for bin_name in ("Q1_low", "Q4_high"):
                    bin_items = [c for c in attack_sel if c.asr_bin == bin_name]
                    if bin_items:
                        sanity_targets.add(bin_items[0].experiment_id)
    safe_write_json(result_path, data)

    for i, candidate in enumerate(selected, 1):
        if candidate.experiment_id in existing:
            print(f"[{i}/{len(selected)}] skip existing {candidate.experiment_id}")
            continue
        print(f"[{i}/{len(selected)}] {candidate.attack} {candidate.asr_bin} ASR={candidate.selection_asr_value:.4f} {candidate.experiment_id}", flush=True)
        run_sanity = candidate.experiment_id in sanity_targets
        result = run_checkpoint(candidate, args, run_sanity=run_sanity, device=device)
        data["checkpoint_results"] = [r for r in data.get("checkpoint_results", []) if r.get("experiment_id") != candidate.experiment_id]
        data["checkpoint_results"].append(result)
        summarize_results(data)
        safe_write_json(result_path, data)

    summarize_results(data)
    plot_results(data, output_dir)
    write_report(data, output_dir)
    safe_write_json(result_path, data)
    print(f"[done] {result_path}")
    print(f"[done] {output_dir / 'transfer_asr_vs_feature_stealth.png'}")
    if args.selection_mode == "coverage_balanced":
        print(f"[done] {output_dir / 'feature_stealth_by_attack_coverage.png'}")
    else:
        print(f"[done] {output_dir / 'feature_stealth_by_attack_and_asr_bin.png'}")
    print(f"[done] {output_dir / 'feature_stealth_pilot_report.md'}")


if __name__ == "__main__":
    main()
