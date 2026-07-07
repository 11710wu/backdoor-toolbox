#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


DATASETS = {"cifar10", "mnistm", "tiny_imagenet", "mnist", "gtsrb", "imagenette", "imagenet"}

UPGD_RE = re.compile(
    r"^upgd_(?P<rate>[0-9.]+)_eps=(?P<eps>[^_]+)_constraint=(?P<constraint>[^_]+)"
    r"_steps=(?P<steps>[0-9]+)_mode=(?P<label_mode>[^_]+)_mult=(?P<mult>[^_]+)"
    r"(?:_noise=(?P<noise_type>.+?)_level=(?P<noise_level>[0-9.]+))?"
    r"_poison_seed=(?P<seed>[0-9]+)_arch=(?P<arch>.+)$"
)

BELT_RE = re.compile(
    r"^belt_(?P<rate>[0-9.]+)_alpha=(?P<alpha>[0-9.]+)_cover=(?P<cover>[0-9.]+)_mask=(?P<mask>[0-9.]+)"
    r"(?:_noise=(?P<noise_type>.+?)_level=(?P<noise_level>[0-9.]+))?"
    r"_poison_seed=(?P<seed>[0-9]+)_arch=(?P<arch>.+)$"
)


def model_from_arch(arch: str) -> str:
    low = arch.lower()
    if low.startswith("resnet18"):
        return "resnet18"
    if low.startswith("resnet34"):
        return "resnet34"
    if low.startswith("resnet50"):
        return "resnet50"
    if low.startswith("vgg19_bn"):
        return "vgg19_bn"
    if low.startswith("mobilenetv2"):
        return "mobilenetv2"
    if low.startswith("smallcnn"):
        return "small_cnn"
    if low.startswith("densenet121"):
        return "densenet121"
    raise ValueError(f"Cannot map arch to -model: {arch}")


def root_env_value(repo: Path, poisoned_root: Path) -> str:
    try:
        return poisoned_root.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(poisoned_root)


def iter_attack_dirs(root: Path):
    if root.name in DATASETS:
        dataset_dirs = [(root.name, root)]
        poisoned_root = root.parent
    else:
        dataset_dirs = [(p.name, p) for p in sorted(root.iterdir()) if p.is_dir() and p.name in DATASETS]
        poisoned_root = root

    for dataset, dataset_dir in dataset_dirs:
        for p in sorted(dataset_dir.iterdir()):
            if not p.is_dir():
                continue
            name = p.name
            if name.startswith("upgd_raw_base_"):
                continue
            if name.startswith("upgd_0.") or name.startswith("belt_"):
                yield poisoned_root, dataset, p


def parse_dir(dataset: str, path: Path, supports_input_noise: bool):
    name = path.name
    m = UPGD_RE.match(name)
    if m:
        d = m.groupdict()
        cmd = [
            sys.executable, "other_defense.py",
            f"-dataset={dataset}",
            "-poison_type=upgd",
            f"-poison_rate={d['rate']}",
            "-defense=SentiNet",
            f"-model={model_from_arch(d['arch'])}",
            f"-eps={d['eps']}",
            f"-constraint={d['constraint']}",
            f"-upgd_steps={d['steps']}",
            f"-upgd_steps_multiplier={d['mult']}",
            f"-label_mode={d['label_mode']}",
            "-trigger=none",
        ]
        if supports_input_noise and d.get("noise_type"):
            cmd.extend([f"-input_noise_type={d['noise_type']}", f"-input_noise_level={d['noise_level']}"])
        return "upgd", cmd

    m = BELT_RE.match(name)
    if m:
        d = m.groupdict()
        cmd = [
            sys.executable, "other_defense.py",
            f"-dataset={dataset}",
            "-poison_type=belt",
            f"-poison_rate={d['rate']}",
            "-defense=SentiNet",
            f"-model={model_from_arch(d['arch'])}",
            f"-alpha={d['alpha']}",
            f"-cover_rate={d['cover']}",
            f"-mask_rate={d['mask']}",
            "-trigger=none",
        ]
        if supports_input_noise and d.get("noise_type"):
            cmd.extend([f"-input_noise_type={d['noise_type']}", f"-input_noise_level={d['noise_level']}"])
        return "belt", cmd

    raise ValueError(f"Unrecognized UPGD/BELT directory name: {path}")


def has_required_files(path: Path, attack: str):
    if attack == "upgd" and not (path / "upgd_0.pth").exists() and not (path / "upgd_2.pth").exists():
        return False, "missing upgd_0.pth/upgd_2.pth"
    model_files = list(path.glob("*.pt"))
    model_files = [p for p in model_files if not p.name.startswith("meta_info_") and not p.name.startswith("belt_trigger")]
    if not model_files:
        return False, "missing model .pt"
    return True, ""


def main():
    ap = argparse.ArgumentParser(description="Rerun SentiNet for existing UPGD/BELT experiment directories.")
    ap.add_argument("--roots", nargs="+", default=["poisoned_train_set", "poisoned_train_set2", "poisoned_train_set4"],
                    help="Poisoned-set roots or dataset directories to scan.")
    ap.add_argument("--attacks", nargs="+", choices=["upgd", "belt"], default=["upgd", "belt"])
    ap.add_argument("--devices", default=os.environ.get("DEVICES", "0"))
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("NUM_SHARDS", "1")))
    ap.add_argument("--shard-index", type=int, default=int(os.environ.get("SHARD_INDEX", "0")))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-existing-sentinet", action="store_true",
                    help="Only rerun directories that already have sentinet_defense_results.json.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    supports_input_noise = "input_noise_type" in (repo / "other_defense.py").read_text(errors="ignore")

    tasks = []
    for root_arg in args.roots:
        root = Path(root_arg)
        if not root.is_absolute():
            root = repo / root
        if not root.exists():
            print(f"[skip root] {root} does not exist")
            continue
        for poisoned_root, dataset, path in iter_attack_dirs(root):
            attack, cmd = parse_dir(dataset, path, supports_input_noise)
            if attack not in args.attacks:
                continue
            if args.only_existing_sentinet and not (path / "sentinet_defense_results.json").exists():
                continue
            ok, reason = has_required_files(path, attack)
            if not ok:
                print(f"[skip] {path}: {reason}")
                continue
            tasks.append((root_env_value(repo, poisoned_root), dataset, attack, path, cmd))

    tasks.sort(key=lambda x: str(x[3]))
    if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--num-shards must be positive and --shard-index must be in [0, num_shards).")
    tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard_index]
    if args.limit > 0:
        tasks = tasks[:args.limit]

    log_dir = repo / "logs" / "rerun_sentinet_upgd_belt"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] repo={repo}")
    print(f"[info] shard={args.shard_index}/{args.num_shards}, devices={args.devices}, tasks={len(tasks)}")

    for idx, (poisoned_root, dataset, attack, path, cmd) in enumerate(tasks, 1):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.devices)
        env["POISONED_TRAIN_SET_ROOT"] = poisoned_root
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = "Agg"
        env["QT_QPA_PLATFORM"] = "offscreen"
        env.pop("DISPLAY", None)
        run_cmd = cmd + [f"-devices={args.devices}"]
        log_name = path.name.replace("/", "_") + ".log"
        log_path = log_dir / log_name
        print(f"[{idx}/{len(tasks)}] {path}")
        print(f"  POISONED_TRAIN_SET_ROOT={poisoned_root}")
        print(f"  CUDA_VISIBLE_DEVICES={args.devices}")
        print("  " + " ".join(run_cmd))
        print(f"  log: {log_path}")
        if args.dry_run:
            continue
        with log_path.open("w") as f:
            f.write(f"# cwd={repo}\n# POISONED_TRAIN_SET_ROOT={poisoned_root}\n# CUDA_VISIBLE_DEVICES={args.devices}\n")
            f.write("# MPLBACKEND=Agg\n# QT_QPA_PLATFORM=offscreen\n# DISPLAY=<unset>\n")
            f.write("# " + " ".join(run_cmd) + "\n")
            f.write("# Output is streamed to the terminal so SentiNet progress is visible.\n")
        rc = subprocess.run(run_cmd, cwd=repo, env=env).returncode
        if rc != 0:
            print(f"[error] failed with code {rc}: {path}")
            print(f"[error] see log: {log_path}")
            sys.exit(rc)


if __name__ == "__main__":
    main()
