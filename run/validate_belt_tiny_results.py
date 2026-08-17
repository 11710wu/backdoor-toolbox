#!/usr/bin/env python3

"""Read-only completeness and BELT-rate validation for the 3x3x6 Tiny grid."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_FILES = (
    "labels",
    "poison_indices",
    "cover_indices",
    "pmarks",
    "belt_trigger.pt",
    "train_results_seed=2333.json",
    "test_results_seed=2333.json",
    "test_tiny_target_domain_results.txt",
    "sentinet_defense_results.json",
    "strip_defense_results.json",
    "scaleup_defense_results.json",
    "ibd_psc_defense_results.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--include-existing-0p5", action="store_true")
    return parser.parse_args()


def expected_rows(include_existing_0p5: bool) -> list[dict[str, str]]:
    rates = (("0.020", "1%"), ("0.100", "5%"))
    if include_existing_0p5:
        rates = (("0.010", "0.5%"),) + rates

    rows = []
    for internal_rate, paper_rate in rates:
        for arch, model in (
            ("ResNet18", "resnet18"),
            ("mobilenetv2", "mobilenetv2"),
            ("vgg19_bn", "vgg19_bn"),
        ):
            for alpha in ("0.100", "0.150", "0.200", "0.250", "0.300", "0.350"):
                rows.append(
                    {
                        "internal_rate": internal_rate,
                        "paper_rate": paper_rate,
                        "arch": arch,
                        "model": model,
                        "alpha": alpha,
                        "config_name": (
                            f"belt_{internal_rate}_alpha={alpha}_cover=0.500_mask=0.200_"
                            f"poison_seed=2333_arch={arch}_tiny_imagenet"
                        ),
                    }
                )
    return rows


def load_tensor(path: Path):
    import sys

    import numpy
    import torch

    # NumPy 2.x pickles refer to numpy._core, while some experiment servers
    # still use NumPy 1.x. Register narrow read-time aliases for those arrays.
    if "numpy._core" not in sys.modules:
        try:
            import numpy._core  # type: ignore[import-not-found]  # noqa: F401
        except ModuleNotFoundError:
            sys.modules["numpy._core"] = numpy.core
            sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
            sys.modules["numpy._core.numeric"] = numpy.core.numeric

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> int:
    args = parse_args()
    output_root = args.repo_root / "poisoned_train_set" / "tiny_imagenet"
    rows = expected_rows(args.include_existing_0p5)

    errors: list[str] = []
    summaries = Counter()
    expected_train_size = 100_000

    for row in rows:
        config_dir = output_root / row["config_name"]
        summaries[(row["paper_rate"], row["arch"])] += 1
        if not config_dir.is_dir():
            errors.append(f"missing config directory: {config_dir}")
            continue

        for name in REQUIRED_FILES:
            if not (config_dir / name).exists():
                errors.append(f"missing {name}: {config_dir}")

        main_models = list(config_dir.glob("*_belt_aug_model_seed=2333.pt"))
        best_models = list(config_dir.glob("*_belt_aug_model_seed=2333_best.pt"))
        if len(main_models) != 1:
            errors.append(f"expected one final checkpoint, found {len(main_models)}: {config_dir}")
        if len(best_models) != 1:
            errors.append(f"expected one best checkpoint, found {len(best_models)}: {config_dir}")

        train_json = config_dir / "train_results_seed=2333.json"
        if train_json.exists():
            record = json.loads(train_json.read_text(encoding="utf-8"))
            if record.get("checkpoint_selection") != "final_epoch":
                errors.append(f"non-final checkpoint policy: {config_dir}")
            if record.get("final_epoch") != record.get("epochs"):
                errors.append(f"final_epoch != epochs: {config_dir}")

        tensor_paths = {
            name: config_dir / name for name in ("pmarks", "poison_indices", "cover_indices")
        }
        if all(path.exists() for path in tensor_paths.values()):
            pmarks = load_tensor(tensor_paths["pmarks"])
            poison_indices = load_tensor(tensor_paths["poison_indices"])
            cover_indices = load_tensor(tensor_paths["cover_indices"])
            counts = Counter(int(value) for value in pmarks.tolist())
            selected = int(expected_train_size * float(row["internal_rate"]))
            expected_cover = int(selected * 0.5)
            expected_full_poison = selected - expected_cover
            if len(pmarks) != expected_train_size:
                errors.append(f"pmark length {len(pmarks)} != {expected_train_size}: {config_dir}")
            if len(poison_indices) != selected:
                errors.append(f"selected-index count {len(poison_indices)} != {selected}: {config_dir}")
            if len(cover_indices) != expected_cover:
                errors.append(f"cover count {len(cover_indices)} != {expected_cover}: {config_dir}")
            if counts[1] != expected_full_poison or counts[2] != expected_cover:
                errors.append(
                    f"pmark counts full={counts[1]}, cover={counts[2]} expected "
                    f"full={expected_full_poison}, cover={expected_cover}: {config_dir}"
                )

    print(f"validated configurations: {len(rows)}")
    for key, count in sorted(summaries.items()):
        print(f"  paper_rate={key[0]} arch={key[1]} configs={count}")
    if errors:
        print(f"FAILED: {len(errors)} issue(s)")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("PASS: artifact completeness, final-checkpoint policy, and BELT rate mapping are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
