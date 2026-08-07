#!/usr/bin/env python3
"""Recreate missing poison tensors, then backfill AC / SS / SPECTRE JSON.

Typical case: model .pt exists but `data`/`imgs` was deleted to save disk.
Cleanser needs both; training is NOT repeated.

Per config:
  1. create_poisoned_set.py  (if data/imgs missing)
  2. cleanser.py for each missing JSON (AC/SS/SPECTRE on cifar10)

Usage:
  conda activate backtool
  export PATH=/workspace/tools/julia-1.7.2/bin:$PATH
  cd /workspace/backdoor-toolbox-new1

  # preview poisoned_train_set / cifar10 only
  python run/recreate_and_backfill_cleanser.py --roots poisoned_train_set --datasets cifar10 --dry-run

  # stream mode: delete ~586MB data after each config (saves disk)
  python run/recreate_and_backfill_cleanser.py --roots poisoned_train_set --datasets cifar10 --delete-data-after

  # keep data on disk (needs ~300GB for 511 cifar10 configs in poisoned_train_set)
  python run/recreate_and_backfill_cleanser.py --roots poisoned_train_set --datasets cifar10 --devices 0
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from run.backfill_cleanser_results import (  # noqa: E402
    CLEANSER_JSON,
    DATASETS,
    SKIP_PREFIXES,
    build_args,
    build_cleanser_cmd,
    default_roots,
    ensure_lookup_path,
    has_required_files,
    is_noise_repo,
    parse_config_name,
)

ARCH_TAIL_RE = re.compile(r"_arch=(?P<arch>[^/]+)$")


def arch_from_dirname(name: str) -> str | None:
    m = ARCH_TAIL_RE.search(name)
    return m.group("arch") if m else None


def upgd_model_path(poison_root: str, dataset: str, cfg_name: str) -> Path:
    arch = arch_from_dirname(cfg_name)
    if not arch:
        raise ValueError(f"Cannot parse arch from dirname: {cfg_name}")
    base_dir = REPO / poison_root / dataset / f"upgd_raw_base_0.000_poison_seed=2333_arch={arch}"
    candidates = sorted(base_dir.glob("upgd_raw_base_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"UPGD base model not found under {base_dir}")
    return candidates[0]


def has_data(cfg_dir: Path) -> bool:
    return (cfg_dir / "data").exists() or (cfg_dir / "imgs").exists()


def has_model(cfg_dir: Path) -> bool:
    ok, _ = has_required_files(cfg_dir)
    if ok:
        return True
    return any(
        p.suffix == ".pt"
        and not p.name.startswith("meta_info_")
        and "belt_trigger" not in p.name
        and "nc_detection" not in p.name
        for p in cfg_dir.glob("*.pt")
    )


def missing_cleansers(cfg_dir: Path, dataset: str, force: bool) -> list[str]:
    cleansers = ["AC", "SS", "SPECTRE"] if dataset == "cifar10" else ["AC", "SS"]
    if force:
        return cleansers
    return [c for c in cleansers if not (cfg_dir / CLEANSER_JSON[c]).exists()]


def build_create_cmd(args: SimpleNamespace, devices: str, poison_root: str, cfg_name: str) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO / "create_poisoned_set.py"),
        f"-dataset={args.dataset}",
        f"-poison_type={args.poison_type}",
        f"-poison_rate={args.poison_rate}",
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
        base = upgd_model_path(poison_root, args.dataset, cfg_name)
        cmd.extend(
            [
                f"-eps={args.eps}",
                f"-constraint={args.constraint}",
                f"-upgd_steps={args.upgd_steps}",
                f"-upgd_steps_multiplier={args.upgd_steps_multiplier}",
                f"-upgd_model_path={base}",
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
    return cmd


def delete_data_files(cfg_dir: Path) -> None:
    for name in ("data", "imgs", "labels", "poison_indices"):
        p = cfg_dir / name
        if p.exists():
            p.unlink()


def arch_matches(name: str, fields: dict, arches: list[str] | None) -> bool:
    if not arches:
        return True
    arch = (arch_from_dirname(name) or "").lower()
    model = str(fields.get("model", "")).lower()
    for a in arches:
        key = a.lower().replace("-", "").replace("_", "")
        if key in arch.replace("_", "").replace("-", "") or key in model.replace("_", "").replace("-", ""):
            return True
    return False


def iter_configs(
    roots: list[str],
    datasets: list[str],
    force: bool,
    recreate_only: bool,
    arches: list[str] | None = None,
) -> list[dict]:
    items: list[dict] = []
    for root_name in roots:
        root = REPO / root_name
        if not root.is_dir():
            continue
        for dataset in datasets:
            ds_dir = root / dataset
            if not ds_dir.is_dir():
                continue
            for cfg_dir in sorted(ds_dir.iterdir()):
                if not cfg_dir.is_dir():
                    continue
                name = cfg_dir.name
                if name.startswith(SKIP_PREFIXES):
                    continue
                if not has_model(cfg_dir):
                    continue
                fields = parse_config_name(dataset, name)
                if fields is None:
                    print(f"[skip parse] {cfg_dir}")
                    continue
                if not arch_matches(name, fields, arches):
                    continue
                need_create = not has_data(cfg_dir)
                need_cleanse = bool(missing_cleansers(cfg_dir, dataset, force))
                if recreate_only and not need_create:
                    continue
                # Normal mode: only process configs that still need cleanser JSON.
                # Do NOT recreate+delete for already-complete configs.
                if not recreate_only and not need_cleanse:
                    continue
                if not need_create and not need_cleanse:
                    continue
                items.append(
                    {
                        "poison_root": root_name,
                        "dataset": dataset,
                        "cfg_dir": cfg_dir,
                        "fields": fields,
                        "need_create": need_create,
                        "need_cleanse": need_cleanse,
                        "cleansers": missing_cleansers(cfg_dir, dataset, force) if need_cleanse else [],
                    }
                )
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="Recreate poison data then backfill cleanser JSON.")
    ap.add_argument("--roots", nargs="+", default=default_roots())
    ap.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["cifar10"])
    ap.add_argument(
        "--arch",
        nargs="+",
        default=None,
        help="Only configs whose dirname arch matches (e.g. ResNet18). Case-insensitive substring.",
    )
    ap.add_argument("--devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--spectre-jobs", type=int, default=int(os.environ.get("SPECTRE_JOBS", "4")))
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="Re-run cleansers even if JSON exists.")
    ap.add_argument("--recreate-only", action="store_true", help="Only run create_poisoned_set.py.")
    ap.add_argument(
        "--delete-data-after",
        action="store_true",
        help="Delete data/imgs/labels/poison_indices after cleansers finish (saves ~600MB/config).",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-symlink", action="store_true")
    args = ap.parse_args()

    if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--num-shards must be > 0 and --shard-index must be in range.")

    items = iter_configs(args.roots, args.datasets, args.force, args.recreate_only, arches=args.arch)
    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_index]
    if args.limit > 0:
        items = items[: args.limit]

    n_create = sum(1 for it in items if it["need_create"])
    n_cleanse = sum(1 for it in items if it["need_cleanse"])
    n_tasks = sum(len(it["cleansers"]) for it in items)

    print(f"[info] repo={REPO}")
    print(f"[info] roots={args.roots} datasets={args.datasets} arch={args.arch or 'ALL'}")
    print(f"[info] shard={args.shard_index}/{args.num_shards} devices={args.devices}")
    print(
        f"[info] configs={len(items)} need_create={n_create} need_cleanse={n_cleanse} "
        f"cleanser_runs={n_tasks} delete_data_after={args.delete_data_after}"
    )
    if items and args.delete_data_after:
        print("[info] stream mode: peak disk ~600MB/config instead of keeping all data")
    elif n_create:
        print(f"[info] keeping data: rough disk +{n_create * 0.586:.0f} GB")

  # ETA: create varies; cleanser ~228s/config on cifar10
    sec = {"AC": 33, "SS": 30, "SPECTRE": 165}
    create_est = {"basic": 120, "blend": 120, "SIG": 180, "WaNet": 180, "adaptive_patch": 300,
                  "adaptive_blend": 300, "belt": 240, "upgd": 900}
    eta = 0.0
    for it in items:
        if it["need_create"]:
            eta += create_est.get(it["fields"]["poison_type"], 300)
        for c in it["cleansers"]:
            eta += sec[c]
    print(f"[info] estimated runtime this shard: {eta/3600:.1f} h ({eta/60:.0f} min)")

    log_dir = REPO / "logs" / "recreate_and_backfill"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for idx, it in enumerate(items, 1):
        cfg_dir = it["cfg_dir"]
        fields = it["fields"]
        ns = build_args(fields)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.devices)
        env["POISONED_TRAIN_SET_ROOT"] = it["poison_root"]
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = "Agg"

        ok, note = ensure_lookup_path(
            ns, cfg_dir, create_symlink=not args.no_symlink and not args.dry_run
        )
        if not ok:
            print(f"[skip path] {cfg_dir}: {note}")
            failures += 1
            continue
        if note:
            print(f"[symlink] {note}")

        print(f"[{idx}/{len(items)}] {cfg_dir.name}")
        if it["need_create"]:
            create_cmd = build_create_cmd(ns, str(args.devices), it["poison_root"], cfg_dir.name)
            print("  [create] " + " ".join(create_cmd))
            if not args.dry_run:
                log_path = log_dir / f"{cfg_dir.name}__create.log"
                rc = subprocess.run(create_cmd, cwd=REPO, env=env).returncode
                if rc != 0:
                    failures += 1
                    print(f"  [error] create exit={rc} (see {log_path})")
                    continue
                if not has_data(cfg_dir):
                    failures += 1
                    print("  [error] create finished but data/imgs still missing")
                    continue

        if not it["need_cleanse"]:
            if args.delete_data_after and has_data(cfg_dir) and not args.dry_run:
                delete_data_files(cfg_dir)
                print("  [cleanup] deleted data (recreate-only)")
            continue

        for cleanser in it["cleansers"]:
            cmd = build_cleanser_cmd(cleanser, ns, str(args.devices), args.spectre_jobs)
            print(f"  [{cleanser}] " + " ".join(cmd))
            if args.dry_run:
                continue
            rc = subprocess.run(cmd, cwd=REPO, env=env).returncode
            if rc != 0:
                failures += 1
                print(f"  [error] {cleanser} exit={rc}")
                break

        if args.delete_data_after and not args.dry_run and has_data(cfg_dir):
            delete_data_files(cfg_dir)
            print("  [cleanup] deleted data after cleansers")

    if failures:
        print(f"[done] failures={failures}")
        return 1
    print("[done] success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
