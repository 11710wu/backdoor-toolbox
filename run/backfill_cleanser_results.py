#!/usr/bin/env python3
"""Backfill AC / SS / SPECTRE cleanser JSON results for existing poisoned sets.

Scope (by repo):
  backdoor-toolbox-new1:
    poisoned_train_set, poisoned_train_set2, poisoned_train_set3
    cifar10: AC + SS + SPECTRE
    tiny_imagenet: AC + SS
  backdoor-toolbox-noise:
    poisoned_train_set
    cifar10: AC + SS + SPECTRE

Excluded roots (by design):
  new1: poisoned_train_set4, poisoned_train_set5
  noise: poisoned_train_set1

Usage:
  conda activate backtool
  export PATH=/workspace/tools/julia-1.7.2/bin:$PATH
  python run/backfill_cleanser_results.py --dry-run
  python run/backfill_cleanser_results.py --devices 0 --num-shards 4 --shard-index 0
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

import config  # noqa: E402
from utils import supervisor  # noqa: E402

DATASETS = {"cifar10", "tiny_imagenet"}
SKIP_PREFIXES = ("none_", "upgd_raw_base_")
TAIL_RE = re.compile(
    r"^(?P<body>.*?)"
    r"(?:_noise=(?P<noise_type>gaussian|uniform|salt_pepper|speckle)_level=(?P<noise_level>[0-9.]+))?"
    r"_poison_seed=(?P<seed>\d+)_arch=(?P<arch>.+)$"
)
PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "basic",
        re.compile(
            r"^basic_(?P<rate>[0-9.]+)_alpha=(?P<alpha>[0-9.]+)_trigger=(?P<trigger>.+)$"
        ),
    ),
    (
        "blend",
        re.compile(
            r"^blend_(?P<rate>[0-9.]+)_alpha=(?P<alpha>[0-9.]+)_trigger=(?P<trigger>.+)$"
        ),
    ),
    (
        "adaptive_blend",
        re.compile(
            r"^adaptive_blend_(?P<rate>[0-9.]+)_alpha=(?P<alpha>[0-9.]+)_cover=(?P<cover>[0-9.]+)_trigger=(?P<trigger>.+)$"
        ),
    ),
    (
        "adaptive_patch",
        re.compile(
            r"^adaptive_patch_(?P<rate>[0-9.]+)_alpha=(?P<alpha>[0-9.]+)_cover=(?P<cover>[0-9.]+)$"
        ),
    ),
    (
        "SIG",
        re.compile(
            r"^SIG_(?P<rate>[0-9.]+)_delta=(?P<delta>[0-9.]+)_f=(?P<f>[0-9.]+)(?:_mode=(?P<label_mode>clean|all2one))?$"
        ),
    ),
    (
        "WaNet",
        re.compile(
            r"^WaNet_(?P<rate>[0-9.]+)_cover=(?P<cover>[0-9.]+)_s=(?P<s>[0-9.]+)_k=(?P<k>[0-9]+)$"
        ),
    ),
    (
        "upgd",
        re.compile(
            r"^upgd_(?P<rate>[0-9.]+)_eps=(?P<eps>[^_]+)_constraint=(?P<constraint>[^_]+)"
            r"_steps=(?P<steps>[0-9]+)_mode=(?P<label_mode>clean|all2one)_mult=(?P<mult>[0-9]+)$"
        ),
    ),
    (
        "belt",
        re.compile(
            r"^belt_(?P<rate>[0-9.]+)_alpha=(?P<alpha>[0-9.]+)_cover=(?P<cover>[0-9.]+)_mask=(?P<mask>[0-9.]+)$"
        ),
    ),
]

CLEANSER_JSON = {
    "AC": "ac_cleanser_results.json",
    "SS": "ss_cleanser_results.json",
    "SPECTRE": "spectre_cleanser_results.json",
}


def is_noise_repo() -> bool:
    return REPO.name.endswith("noise")


def default_roots() -> list[str]:
    if is_noise_repo():
        return ["poisoned_train_set"]
    return ["poisoned_train_set", "poisoned_train_set2", "poisoned_train_set3"]


def model_from_arch(arch: str) -> str:
    low = arch.lower()
    mapping = [
        ("resnet18", "resnet18"),
        ("resnet34", "resnet34"),
        ("resnet50", "resnet50"),
        ("vgg19_bn", "vgg19_bn"),
        ("mobilenetv2", "mobilenetv2"),
        ("smallcnn", "small_cnn"),
        ("densenet121", "densenet121"),
    ]
    for prefix, model in mapping:
        if low.startswith(prefix):
            return model
    raise ValueError(f"Cannot map arch to -model: {arch}")


def parse_config_name(dataset: str, name: str) -> dict | None:
    tail = TAIL_RE.match(name)
    if not tail:
        return None
    body = tail.group("body")
    fields = {
        "dataset": dataset,
        "seed": int(tail.group("seed")),
        "model": model_from_arch(tail.group("arch")),
    }
    if tail.group("noise_type"):
        fields["input_noise_type"] = tail.group("noise_type")
        fields["input_noise_level"] = float(tail.group("noise_level"))
    for poison_type, pattern in PATTERNS:
        m = pattern.match(body)
        if not m:
            continue
        fields["poison_type"] = poison_type
        g = m.groupdict()
        fields["poison_rate"] = float(g["rate"])
        if poison_type in {"basic", "blend", "adaptive_blend"}:
            fields["alpha"] = float(g["alpha"])
            fields["trigger"] = g["trigger"]
        if poison_type in {"adaptive_blend", "adaptive_patch", "WaNet", "belt"}:
            fields["cover_rate"] = float(g["cover"])
        if poison_type == "adaptive_patch":
            fields["alpha"] = float(g["alpha"])
        if poison_type == "SIG":
            fields["delta"] = float(g["delta"])
            fields["f"] = float(g["f"])
            fields["label_mode"] = g.get("label_mode") or "clean"
        if poison_type == "WaNet":
            fields["s"] = float(g["s"])
            fields["k"] = int(g["k"])
        if poison_type == "upgd":
            fields["eps"] = float(g["eps"])
            fields["constraint"] = g["constraint"]
            fields["upgd_steps"] = int(g["steps"])
            fields["upgd_steps_multiplier"] = int(g["mult"])
            fields["label_mode"] = g["label_mode"]
        if poison_type == "belt":
            fields["alpha"] = float(g["alpha"])
            fields["mask_rate"] = float(g["mask"])
        return fields
    return None


def build_args(fields: dict) -> SimpleNamespace:
    args = SimpleNamespace(
        dataset=fields["dataset"],
        poison_type=fields["poison_type"],
        poison_rate=fields["poison_rate"],
        cover_rate=fields.get("cover_rate", 0.0),
        alpha=fields.get("alpha", 0.2),
        test_alpha=None,
        label_mode=fields.get("label_mode", "clean"),
        trigger=fields.get("trigger"),
        no_aug=False,
        model=fields["model"],
        model_path=None,
        no_normalize=False,
        devices="0",
        seed=fields.get("seed", config.poison_seed),
        s=fields.get("s", 0.5),
        k=fields.get("k", 4),
        delta=fields.get("delta", 30),
        f=fields.get("f", 6),
        eps=fields.get("eps", 8.0),
        constraint=fields.get("constraint", "Linf"),
        upgd_steps=fields.get("upgd_steps", 100),
        upgd_steps_multiplier=fields.get("upgd_steps_multiplier", 5),
        mask_rate=fields.get("mask_rate", 0.2),
        input_noise_type=fields.get("input_noise_type", "none"),
        input_noise_level=fields.get("input_noise_level", 0.0),
        input_noise_seed=fields.get("seed", config.poison_seed),
        spectre_jobs=4,
    )
    if args.trigger is None:
        args.trigger = config.trigger_default[args.dataset][args.poison_type]
    return args


def has_required_files(cfg_dir: Path) -> tuple[bool, str]:
    data_ok = (cfg_dir / "data").exists() or (cfg_dir / "imgs").exists()
    if not data_ok:
        return False, "missing data/imgs"
    model_files = [
        p
        for p in cfg_dir.glob("*.pt")
        if not p.name.startswith("meta_info_")
        and "belt_trigger" not in p.name
        and "nc_detection" not in p.name
    ]
    if not model_files:
        return False, "missing model .pt"
    return True, ""


def ensure_lookup_path(args: SimpleNamespace, actual_dir: Path, create_symlink: bool) -> tuple[bool, str]:
    expected = Path(supervisor.get_poison_set_dir(args))
    if expected.resolve() == actual_dir.resolve():
        return True, ""
    if expected.exists():
        if expected.is_symlink() and expected.resolve() == actual_dir.resolve():
            return True, ""
        return False, f"expected path exists but differs: {expected}"
    if not create_symlink:
        return False, f"path mismatch (dry-run): expected {expected}"
    expected.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(actual_dir, expected.parent)
    expected.symlink_to(rel)
    return True, f"symlink {expected.name} -> {rel}"


def build_cleanser_cmd(cleanser: str, args: SimpleNamespace, devices: str, spectre_jobs: int) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO / "cleanser.py"),
        f"-dataset={args.dataset}",
        f"-poison_type={args.poison_type}",
        f"-poison_rate={args.poison_rate}",
        f"-cleanser={cleanser}",
        f"-seed={args.seed}",
        f"-devices={devices}",
        f"-model={args.model}",
        f"-label_mode={args.label_mode}",
    ]
    if args.poison_type in {"basic", "blend", "adaptive_blend"}:
        cmd.extend([f"-alpha={args.alpha}", f"-trigger={args.trigger}"])
        if args.poison_type == "adaptive_blend":
            cmd.append(f"-cover_rate={args.cover_rate}")
    elif args.poison_type == "adaptive_patch":
        cmd.extend(
            [
                f"-alpha={args.alpha}",
                f"-cover_rate={args.cover_rate}",
                f"-s={args.s}",
                f"-k={args.k}",
            ]
        )
    elif args.poison_type == "WaNet":
        cmd.extend(
            [
                f"-cover_rate={args.cover_rate}",
                f"-s={args.s}",
                f"-k={args.k}",
            ]
        )
    elif args.poison_type == "SIG":
        cmd.extend([f"-delta={args.delta}", f"-f={args.f}"])
    elif args.poison_type == "upgd":
        cmd.extend(
            [
                f"-eps={args.eps}",
                f"-constraint={args.constraint}",
                f"-upgd_steps={args.upgd_steps}",
                f"-upgd_steps_multiplier={args.upgd_steps_multiplier}",
            ]
        )
    elif args.poison_type == "belt":
        cmd.extend(
            [
                f"-alpha={args.alpha}",
                f"-cover_rate={args.cover_rate}",
                f"-mask_rate={args.mask_rate}",
            ]
        )
    if is_noise_repo() and args.input_noise_type not in (None, "none") and args.input_noise_level:
        cmd.extend(
            [
                f"-input_noise_type={args.input_noise_type}",
                f"-input_noise_level={args.input_noise_level}",
                f"-input_noise_seed={args.input_noise_seed}",
            ]
        )
    if cleanser == "SPECTRE":
        cmd.append(f"-spectre_jobs={spectre_jobs}")
    return cmd


@dataclass
class Task:
    poison_root: str
    dataset: str
    cfg_dir: Path
    cleanser: str
    cmd: list[str]
    note: str = ""


def iter_tasks(
    roots: list[str],
    force: bool,
    devices: str,
    spectre_jobs: int,
    create_symlink: bool,
    datasets: list[str] | None = None,
) -> tuple[list[Task], dict[str, int]]:
    stats = {
        "configs_seen": 0,
        "skipped_prefix": 0,
        "skipped_parse": 0,
        "skipped_files": 0,
        "skipped_existing": 0,
        "skipped_path": 0,
        "tasks": 0,
    }
    tasks: list[Task] = []
    dataset_list = datasets or ["cifar10", "tiny_imagenet"]

    for root_name in roots:
        root = REPO / root_name
        if not root.exists():
            continue
        for dataset in dataset_list:
            ds_dir = root / dataset
            if not ds_dir.is_dir():
                continue
            cleansers = ["AC", "SS", "SPECTRE"] if dataset == "cifar10" else ["AC", "SS"]
            for cfg_dir in sorted(ds_dir.iterdir()):
                if not cfg_dir.is_dir():
                    continue
                name = cfg_dir.name
                if name.startswith(SKIP_PREFIXES):
                    stats["skipped_prefix"] += 1
                    continue
                stats["configs_seen"] += 1
                fields = parse_config_name(dataset, name)
                if fields is None:
                    stats["skipped_parse"] += 1
                    print(f"[skip parse] {cfg_dir}")
                    continue
                ok, reason = has_required_files(cfg_dir)
                if not ok:
                    stats["skipped_files"] += 1
                    print(f"[skip files] {cfg_dir}: {reason}")
                    continue
                args = build_args(fields)
                ok, note = ensure_lookup_path(args, cfg_dir, create_symlink=create_symlink)
                if not ok:
                    stats["skipped_path"] += 1
                    print(f"[skip path] {cfg_dir}: {note}")
                    continue
                for cleanser in cleansers:
                    result_json = cfg_dir / CLEANSER_JSON[cleanser]
                    if result_json.exists() and not force:
                        stats["skipped_existing"] += 1
                        continue
                    cmd = build_cleanser_cmd(cleanser, args, devices, spectre_jobs)
                    tasks.append(
                        Task(
                            poison_root=root_name,
                            dataset=dataset,
                            cfg_dir=cfg_dir,
                            cleanser=cleanser,
                            cmd=cmd,
                            note=note,
                        )
                    )
                    stats["tasks"] += 1
    return tasks, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill cleanser JSON results for existing poisoned sets.")
    ap.add_argument("--roots", nargs="+", default=default_roots(), help="Poisoned-set roots under repo.")
    ap.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=None,
        help="Only scan these datasets (default: cifar10 + tiny_imagenet).",
    )
    ap.add_argument("--devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--spectre-jobs", type=int, default=int(os.environ.get("SPECTRE_JOBS", "4")))
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("NUM_SHARDS", "1")))
    ap.add_argument("--shard-index", type=int, default=int(os.environ.get("SHARD_INDEX", "0")))
    ap.add_argument("--force", action="store_true", help="Re-run even if *_cleanser_results.json exists.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-symlink", action="store_true", help="Do not create SIG legacy symlinks.")
    args = ap.parse_args()

    if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--num-shards must be > 0 and --shard-index must be in range.")

    tasks, stats = iter_tasks(
        roots=args.roots,
        force=args.force,
        devices=str(args.devices),
        spectre_jobs=args.spectre_jobs,
        create_symlink=not args.no_symlink and not args.dry_run,
        datasets=args.datasets,
    )
    tasks.sort(key=lambda t: (t.poison_root, t.dataset, str(t.cfg_dir), t.cleanser))
    tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard_index]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    log_dir = REPO / "logs" / "backfill_cleanser"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] repo={REPO}")
    print(f"[info] roots={args.roots}")
    print(f"[info] datasets={args.datasets or sorted(DATASETS)}")
    print(f"[info] shard={args.shard_index}/{args.num_shards} devices={args.devices} spectre_jobs={args.spectre_jobs}")
    print(
        "[info] scan: configs_seen={configs_seen} tasks_selected={task_count} "
        "skipped_existing={skipped_existing} skipped_parse={skipped_parse} skipped_files={skipped_files} "
        "skipped_path={skipped_path}".format(task_count=len(tasks), **stats)
    )

    # count by cleanser for ETA
    from collections import Counter
    c = Counter(t.cleanser for t in tasks)
    ds = Counter((t.dataset, t.cleanser) for t in tasks)
    print(f"[info] cleanser counts: {dict(c)}")
    print(f"[info] dataset x cleanser: {dict(ds)}")

    # ETA seconds (single GPU, from smoke benchmarks on RTX 5090)
    sec_per = {"AC": 33, "SS": 30, "SPECTRE": 165}
    tiny_mult = {"AC": 4.3, "SS": 4.9}  # tiny_imagenet relative to cifar10
    eta = 0.0
    for t in tasks:
        base = sec_per[t.cleanser]
        if t.dataset == "tiny_imagenet":
            base = int(base * tiny_mult[t.cleanser])
        eta += base
    print(f"[info] estimated runtime this shard: {eta/3600:.1f} h ({eta/60:.0f} min) on one GPU")

    failures = 0
    for idx, task in enumerate(tasks, 1):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.devices)
        env["POISONED_TRAIN_SET_ROOT"] = task.poison_root
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = "Agg"
        log_path = log_dir / f"{task.cfg_dir.name}__{task.cleanser}.log"
        print(f"[{idx}/{len(tasks)}] {task.cleanser} {task.cfg_dir}")
        if task.note:
            print(f"  note: {task.note}")
        print(f"  POISONED_TRAIN_SET_ROOT={task.poison_root}")
        print("  " + " ".join(task.cmd))
        if args.dry_run:
            continue
        with log_path.open("w") as f:
            f.write(f"# POISONED_TRAIN_SET_ROOT={task.poison_root}\n")
            f.write("# " + " ".join(task.cmd) + "\n")
        rc = subprocess.run(task.cmd, cwd=REPO, env=env).returncode
        if rc != 0:
            failures += 1
            print(f"  [error] exit={rc} log={log_path}")
    if failures:
        print(f"[done] failures={failures}")
        return 1
    print("[done] success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
