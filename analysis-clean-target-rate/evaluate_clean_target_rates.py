#!/usr/bin/env python3
"""Evaluate clean target-label rates for every paper benchmark configuration.

For domain d, CTR_d is the fraction of clean non-target-label samples that the
frozen backdoored model predicts as the attack target label.  This script does
not modify any existing experiment result or paper asset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms


SCRIPT_DIR = Path(__file__).resolve().parent
TOOLBOX_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = Path(
    "/workspace/tifs_backdoor_paper_restructure_20260815/"
    "data/primary_baseline_belt_completed.csv"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "clean_target_rates_all_configurations.csv"
TINY_TARGET_ROOT = (
    TOOLBOX_ROOT / "data/imagenetv2-matched-frequency-tiny-organized"
)
ACCURACY_TOLERANCE_PP = 0.2
TARGET_CLASS = {"cifar10": 0, "tiny_imagenet": 0}
NUM_CLASSES = {"cifar10": 10, "tiny_imagenet": 200}
MODEL_NAME = {
    "ResNet18": "resnet18",
    "vgg19_bn": "vgg19_bn",
    "mobilenetv2": "mobilenetv2",
}
MODEL_FILE_STEM = {
    ("cifar10", "ResNet18"): "ResNet18_cifar10",
    ("cifar10", "vgg19_bn"): "vgg19_bn_cifar10",
    ("cifar10", "mobilenetv2"): "mobilenetv2_cifar10",
    ("tiny_imagenet", "ResNet18"): "ResNet18_tiny_imagenet",
    ("tiny_imagenet", "vgg19_bn"): "vgg19_bn_tiny_imagenet",
    ("tiny_imagenet", "mobilenetv2"): "mobilenetv2_tiny_imagenet",
}
IDENTITY_COLUMNS = [
    "strict_config_key",
    "result_group",
    "dataset",
    "transfer_dataset",
    "arch_base",
    "attack_type",
    "poison_rate",
    "strength_name",
    "strength_value",
    "cover_rate",
    "mask_rate",
    "label_mode",
    "input_noise_type",
    "input_noise_level",
    "folder_name",
]
RESULT_COLUMNS = IDENTITY_COLUMNS + [
    "manifest_model_path",
    "resolved_model_path",
    "target_class",
    "source_clean_correct",
    "source_clean_total",
    "source_ctr_numerator",
    "source_ctr_denominator",
    "source_ctr",
    "target_clean_correct",
    "target_clean_total",
    "target_ctr_numerator",
    "target_ctr_denominator",
    "target_ctr",
    "recorded_source_clean_acc",
    "recomputed_source_clean_acc",
    "source_clean_acc_diff_pp",
    "source_accuracy_valid",
    "recorded_target_clean_acc",
    "recomputed_target_clean_acc",
    "target_clean_acc_diff_pp",
    "target_accuracy_valid",
    "status",
    "error",
    "evaluated_at_utc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Evaluate only the first N pending configurations (for smoke tests).",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry rows already recorded with a non-ok status.",
    )
    return parser.parse_args()


def scalar_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def assign_stable_configuration_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill legacy missing keys without depending on manifest row order."""
    frame = frame.copy()
    missing = frame["strict_config_key"].isna() | frame["strict_config_key"].eq("")
    if missing.any():
        legacy = frame.loc[missing]
        if not legacy["dataset"].eq("tiny_imagenet").all() or not legacy[
            "attack_type"
        ].eq("belt").all():
            raise ValueError("missing strict_config_key outside legacy Tiny-ImageNet BELT rows")
        if legacy["folder_name"].isna().any() or legacy["folder_name"].duplicated().any():
            raise ValueError("legacy BELT rows require unique, non-empty folder_name values")
        frame.loc[missing, "strict_config_key"] = (
            "legacy_belt|" + legacy["folder_name"].astype(str)
        )
    if frame["strict_config_key"].isna().any() or frame["strict_config_key"].duplicated().any():
        raise ValueError("configuration keys must be present and unique")
    return frame


def deduplicate_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        path = path.resolve()
        if path not in result:
            result.append(path)
    return result


def relocate_to_toolbox(path: Path) -> Path | None:
    if not path.is_absolute():
        return TOOLBOX_ROOT / path
    for marker in ("backdoor-toolbox-new1", "backdoor-toolbox"):
        if marker in path.parts:
            index = path.parts.index(marker)
            return TOOLBOX_ROOT.joinpath(*path.parts[index + 1 :])
    return None


def resolve_model_path(row: pd.Series) -> Path:
    candidates: list[Path] = []
    recorded = scalar_text(row.get("recorded_model_path"))
    if recorded:
        recorded_path = Path(recorded)
        candidates.append(recorded_path)
        relocated = relocate_to_toolbox(recorded_path)
        if relocated is not None:
            candidates.append(relocated)

    directories: list[Path] = []
    for column in ("source_checkpoint_id", "result_dir"):
        value = scalar_text(row.get(column))
        if not value:
            continue
        path = Path(value)
        relocated = relocate_to_toolbox(path)
        directories.extend([path] + ([relocated] if relocated is not None else []))

    stem = MODEL_FILE_STEM[(row["dataset"], row["arch_base"])]
    if row["attack_type"] == "belt":
        filename = f"{stem}_belt_aug_model_seed=2333.pt"
    else:
        filename = f"{stem}.pt"
    candidates.extend(directory / filename for directory in directories)

    existing = [path for path in deduplicate_paths(candidates) if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected exactly one model file, found {len(existing)}; "
            f"candidates={[str(path) for path in deduplicate_paths(candidates)]}"
        )
    return existing[0]


def load_source_images(dataset_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    split = TOOLBOX_ROOT / "clean_set" / dataset_name / "test_split"
    labels = torch.as_tensor(
        torch.load(split / "labels", map_location="cpu", weights_only=False),
        dtype=torch.long,
    )
    image_dir = split / "data"
    to_tensor = transforms.ToTensor()
    images = []
    for index in range(len(labels)):
        with Image.open(image_dir / f"{index}.png") as image:
            images.append(to_tensor(image.convert("RGB")))
    return torch.stack(images), labels


def load_stl10_images() -> tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.STL10(
        root=str(TOOLBOX_ROOT / "data"), split="test", download=False
    )
    mapping = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 7, 7: -1, 8: 8, 9: 9}
    images = []
    labels = []
    for array, label in zip(dataset.data, dataset.labels):
        mapped = mapping[int(label)]
        if mapped == -1:
            continue
        image = torch.from_numpy(array.copy()).float().div_(255.0)
        image = F.interpolate(
            image.unsqueeze(0),
            size=(32, 32),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        images.append(image)
        labels.append(mapped)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def validate_tiny_class_mapping(dataset: datasets.ImageFolder) -> None:
    mapping_path = TINY_TARGET_ROOT / "class_to_idx.json"
    with mapping_path.open("r", encoding="utf-8") as handle:
        saved_mapping = json.load(handle)
    if saved_mapping != dataset.class_to_idx:
        raise ValueError("ImageNetV2 class_to_idx.json does not match ImageFolder")

    train_root = TOOLBOX_ROOT / "data/Tiny-imagenet/tiny-imagenet-200/train"
    train_mapping = {
        path.name: index
        for index, path in enumerate(sorted(path for path in train_root.iterdir() if path.is_dir()))
    }
    if train_mapping != dataset.class_to_idx:
        raise ValueError("ImageNetV2 class mapping does not match Tiny-ImageNet train")


def load_tiny_target_images() -> tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.ImageFolder(
        str(TINY_TARGET_ROOT / "test"), transform=transforms.ToTensor()
    )
    validate_tiny_class_mapping(dataset)
    images = []
    labels = []
    for image, label in dataset:
        images.append(image)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def normalized_copy(images: torch.Tensor, dataset_name: str) -> torch.Tensor:
    if dataset_name == "cifar10":
        mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    else:
        mean, std = [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
    normalizer = transforms.Normalize(mean, std)
    return normalizer(images.clone())


def prepare_datasets() -> dict[tuple[str, str, bool], tuple[torch.Tensor, torch.Tensor]]:
    source_cifar = load_source_images("cifar10")
    source_tiny = load_source_images("tiny_imagenet")
    target_cifar = load_stl10_images()
    target_tiny = load_tiny_target_images()

    raw = {
        ("source", "cifar10"): source_cifar,
        ("source", "tiny_imagenet"): source_tiny,
        ("target", "cifar10"): target_cifar,
        ("target", "tiny_imagenet"): target_tiny,
    }
    prepared = {}
    for (domain, dataset_name), (images, labels) in raw.items():
        prepared[(domain, dataset_name, True)] = (images, labels)
        prepared[(domain, dataset_name, False)] = (
            normalized_copy(images, dataset_name),
            labels,
        )
    return prepared


def validate_dataset_denominators(
    prepared: dict[tuple[str, str, bool], tuple[torch.Tensor, torch.Tensor]]
) -> None:
    expected = {
        ("source", "cifar10"): (8000, 7193),
        ("target", "cifar10"): (7200, 6400),
        ("source", "tiny_imagenet"): (8000, 7962),
        ("target", "tiny_imagenet"): (2000, 1990),
    }
    for (domain, dataset_name), (total_expected, eligible_expected) in expected.items():
        _, labels = prepared[(domain, dataset_name, True)]
        eligible = labels.ne(TARGET_CLASS[dataset_name]).sum().item()
        if len(labels) != total_expected or eligible != eligible_expected:
            raise ValueError(
                f"unexpected {domain}/{dataset_name} counts: "
                f"total={len(labels)}, non_target={eligible}; "
                f"expected total={total_expected}, non_target={eligible_expected}"
            )


def build_model(row: pd.Series, model_path: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(TOOLBOX_ROOT)) if str(TOOLBOX_ROOT) not in sys.path else None
    from utils import supervisor

    args = SimpleNamespace(
        dataset=row["dataset"],
        poison_type=row["attack_type"],
        model=MODEL_NAME[row["arch_base"]],
    )
    architecture = supervisor.get_arch(args)
    model = architecture(num_classes=NUM_CLASSES[row["dataset"]])
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {type(state)!r}")
    if any(key.startswith("module.") for key in state):
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def evaluate_clean(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, int | float]:
    clean_correct = 0
    ctr_numerator = 0
    ctr_denominator = 0
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        batch_images = images[start:stop].to(device, non_blocking=True)
        batch_labels = labels[start:stop].to(device, non_blocking=True)
        predictions = model(batch_images).argmax(dim=1)
        eligible = batch_labels.ne(target_class)
        clean_correct += predictions.eq(batch_labels).sum().item()
        ctr_numerator += predictions[eligible].eq(target_class).sum().item()
        ctr_denominator += eligible.sum().item()
    if ctr_denominator == 0:
        raise RuntimeError("clean non-target denominator is zero")
    return {
        "clean_correct": clean_correct,
        "clean_total": len(labels),
        "clean_acc": clean_correct / len(labels),
        "ctr_numerator": ctr_numerator,
        "ctr_denominator": ctr_denominator,
        "ctr": ctr_numerator / ctr_denominator,
    }


def atomic_write(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output)


def base_result(row: pd.Series) -> dict[str, object]:
    result = {column: row.get(column, "") for column in IDENTITY_COLUMNS}
    result.update(
        {
            "manifest_model_path": scalar_text(row.get("recorded_model_path")),
            "recorded_source_clean_acc": row["clean_acc"],
            "recorded_target_clean_acc": row["transfer_acc"],
        }
    )
    return result


def evaluate_row(
    row: pd.Series,
    prepared: dict[tuple[str, str, bool], tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    result = base_result(row)
    result["evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        model_path = resolve_model_path(row)
        result["resolved_model_path"] = str(model_path)
        target_class = TARGET_CLASS[row["dataset"]]
        result["target_class"] = target_class
        raw_input = row["attack_type"] in ("upgd", "belt")
        model = build_model(row, model_path, device)

        source = evaluate_clean(
            model,
            *prepared[("source", row["dataset"], raw_input)],
            target_class,
            batch_size,
            device,
        )
        target = evaluate_clean(
            model,
            *prepared[("target", row["dataset"], raw_input)],
            target_class,
            batch_size,
            device,
        )
        result.update(
            {
                "source_clean_correct": source["clean_correct"],
                "source_clean_total": source["clean_total"],
                "source_ctr_numerator": source["ctr_numerator"],
                "source_ctr_denominator": source["ctr_denominator"],
                "source_ctr": source["ctr"],
                "target_clean_correct": target["clean_correct"],
                "target_clean_total": target["clean_total"],
                "target_ctr_numerator": target["ctr_numerator"],
                "target_ctr_denominator": target["ctr_denominator"],
                "target_ctr": target["ctr"],
                "recomputed_source_clean_acc": source["clean_acc"],
                "recomputed_target_clean_acc": target["clean_acc"],
            }
        )
        source_diff = abs(source["clean_acc"] - float(row["clean_acc"])) * 100
        target_diff = abs(target["clean_acc"] - float(row["transfer_acc"])) * 100
        source_valid = source_diff < ACCURACY_TOLERANCE_PP
        target_valid = target_diff < ACCURACY_TOLERANCE_PP
        result.update(
            {
                "source_clean_acc_diff_pp": source_diff,
                "source_accuracy_valid": source_valid,
                "target_clean_acc_diff_pp": target_diff,
                "target_accuracy_valid": target_valid,
                "status": "ok" if source_valid and target_valid else "failed_accuracy_mismatch",
                "error": "" if source_valid and target_valid else (
                    f"clean accuracy difference reaches or exceeds {ACCURACY_TOLERANCE_PP} pp"
                ),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # continue and preserve the failed configuration
        result["status"] = "failed_exception"
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return result


def load_existing(output: Path, manifest_keys: set[str]) -> tuple[pd.DataFrame, bool]:
    if not output.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS), False
    existing = pd.read_csv(output)
    if "strict_config_key" not in existing or "status" not in existing:
        raise ValueError(f"existing output has an incompatible schema: {output}")
    legacy = existing["strict_config_key"].astype(str).str.match(r"^manifest_row_\d{4}$")
    migrated = bool(legacy.any())
    if migrated:
        rows = existing.loc[legacy]
        if not rows["dataset"].eq("tiny_imagenet").all() or not rows[
            "attack_type"
        ].eq("belt").all():
            raise ValueError("row-number keys found outside legacy Tiny-ImageNet BELT rows")
        if rows["folder_name"].isna().any() or rows["folder_name"].duplicated().any():
            raise ValueError("cannot migrate legacy keys without unique folder_name values")
        existing.loc[legacy, "strict_config_key"] = (
            "legacy_belt|" + rows["folder_name"].astype(str)
        )
    if existing["strict_config_key"].duplicated().any():
        raise ValueError("existing output contains duplicate strict_config_key values")
    unknown = set(existing["strict_config_key"]) - manifest_keys
    if unknown:
        raise ValueError(f"existing output contains {len(unknown)} keys absent from manifest")
    return existing.reindex(columns=RESULT_COLUMNS), migrated


def final_verification(frame: pd.DataFrame, expected_count: int) -> None:
    problems = []
    if len(frame) != expected_count:
        problems.append(f"rows={len(frame)} (expected {expected_count})")
    if frame["strict_config_key"].nunique() != expected_count:
        problems.append(
            f"unique configurations={frame['strict_config_key'].nunique()} "
            f"(expected {expected_count})"
        )
    ok = frame[frame["status"] == "ok"]
    for column in ("source_ctr", "target_ctr"):
        if not ok[column].between(0, 1, inclusive="both").all():
            problems.append(f"{column} contains values outside [0,1]")
    for prefix in ("source", "target"):
        if not (
            ok[f"{prefix}_ctr_numerator"] <= ok[f"{prefix}_ctr_denominator"]
        ).all():
            problems.append(f"{prefix} CTR numerator exceeds denominator")
    print("\nCompletion summary")
    print(f"  rows: {len(frame)}/{expected_count}")
    print(f"  unique configurations: {frame['strict_config_key'].nunique()}/{expected_count}")
    print("  status counts:")
    print(frame["status"].value_counts(dropna=False).to_string())
    if len(frame):
        print("  source denominator counts:")
        print(frame.groupby("dataset")["source_ctr_denominator"].unique().to_string())
        print("  target denominator counts:")
        print(frame.groupby("dataset")["target_ctr_denominator"].unique().to_string())
    if problems:
        raise RuntimeError("; ".join(problems))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    manifest = assign_stable_configuration_keys(pd.read_csv(args.manifest))
    if len(manifest) != 889 or manifest["strict_config_key"].nunique() != 889:
        raise ValueError(
            f"manifest must contain 889 rows and keys; got rows={len(manifest)}, "
            f"keys={manifest['strict_config_key'].nunique()}"
        )
    if set(manifest["dataset"]) != set(TARGET_CLASS):
        raise ValueError(f"unexpected datasets: {sorted(manifest['dataset'].unique())}")

    existing, migrated = load_existing(args.output, set(manifest["strict_config_key"]))
    if migrated:
        atomic_write(existing, args.output)
        print("Migrated legacy row-number configuration keys to stable folder-based keys.")
    completed_statuses = {"ok"}
    if not args.retry_failures:
        completed_statuses.update({"failed_accuracy_mismatch", "failed_exception"})
    completed = set(
        existing.loc[existing["status"].isin(completed_statuses), "strict_config_key"]
    )
    pending = manifest[~manifest["strict_config_key"].isin(completed)]
    if args.max_configs is not None:
        pending = pending.head(args.max_configs)

    print(f"Manifest: {args.manifest}")
    print(f"Output: {args.output}")
    print(f"Existing rows: {len(existing)}; pending this run: {len(pending)}")
    if len(pending):
        print("Loading and validating the four clean evaluation datasets...")
        prepared = prepare_datasets()
        validate_dataset_denominators(prepared)
        print("Dataset denominators verified.")
    else:
        prepared = {}

    rows = {
        row["strict_config_key"]: row.to_dict()
        for _, row in existing.iterrows()
    }
    total_pending = len(pending)
    for position, (_, row) in enumerate(pending.iterrows(), start=1):
        key = row["strict_config_key"]
        print(
            f"[{position}/{total_pending}] {row['dataset']} {row['arch_base']} "
            f"{row['attack_type']} | {key}",
            flush=True,
        )
        result = evaluate_row(row, prepared, args.batch_size, device)
        rows[key] = result
        ordered = pd.DataFrame(rows.values()).reindex(columns=RESULT_COLUMNS)
        order = {key: index for index, key in enumerate(manifest["strict_config_key"])}
        ordered["_order"] = ordered["strict_config_key"].map(order)
        ordered = ordered.sort_values("_order").drop(columns="_order")
        atomic_write(ordered, args.output)
        print(
            f"  status={result['status']} "
            f"CTR_s={result.get('source_ctr', '')} CTR_t={result.get('target_ctr', '')}",
            flush=True,
        )

    final = pd.read_csv(args.output) if args.output.exists() else existing
    final_verification(final, len(manifest))


if __name__ == "__main__":
    main()
