#!/usr/bin/env python3
"""Archive and rerun BadNets/Adaptive-Patch SentiNet results on eight GPUs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MANIFEST_PATHS = (
    SCRIPT_DIR / "poisoned_train_set.tsv",
    SCRIPT_DIR / "poisoned_train_set3.tsv",
)
EXPECTED_TASKS = 288
EXPECTED_ARCHIVED_RESULTS = 386
REQUIRED_RESULT_KEYS = ("tpr", "fpr", "auc")
MANIFEST_FIELDS = (
    "task_id",
    "gpu_slot",
    "root",
    "dataset",
    "poison_type",
    "poison_rate",
    "alpha",
    "cover_rate",
    "trigger",
    "model",
    "arch_name",
    "relative_dir",
    "checkpoint_name",
)


@dataclass(frozen=True)
class Task:
    task_id: str
    gpu_slot: int
    root: str
    dataset: str
    poison_type: str
    poison_rate: str
    alpha: str
    cover_rate: str
    trigger: str
    model: str
    arch_name: str
    relative_dir: str
    checkpoint_name: str

    @property
    def directory(self) -> Path:
        return REPO_ROOT / self.relative_dir

    @property
    def result_path(self) -> Path:
        return self.directory / "sentinet_defense_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, archive, or rerun the 288 CIFAR-10/Tiny-ImageNet "
            "BadNets and Adaptive-Patch SentiNet jobs."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="validate manifests, paths, checkpoints, and GPU balance")

    archive_parser = subparsers.add_parser(
        "archive", help="rename existing SentiNet results with an _old suffix"
    )
    archive_parser.add_argument(
        "--dry-run", action="store_true", help="validate and report renames without changing files"
    )
    archive_parser.add_argument(
        "--audit-log",
        type=Path,
        default=None,
        help="write the rename/hash audit TSV to this path",
    )

    run_parser = subparsers.add_parser("run", help="run the manifests with eight GPU workers")
    run_parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="eight comma-separated physical GPU IDs (default: 0,1,2,3,4,5,6,7)",
    )
    run_parser.add_argument("--python", default=sys.executable, help="Python executable for other_defense.py")
    run_parser.add_argument("--seed", type=int, default=2333)
    run_parser.add_argument("--force", action="store_true", help="rerun valid new results")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="print all 288 commands without starting experiments"
    )
    run_parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="log directory (default: logs/sentinet_badnet_adaptive_patch_rerun/<timestamp>)",
    )
    return parser.parse_args()


def load_tasks() -> list[Task]:
    tasks: list[Task] = []
    for manifest_path in MANIFEST_PATHS:
        if not manifest_path.is_file():
            raise RuntimeError(f"missing manifest: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                raise RuntimeError(
                    f"unexpected columns in {manifest_path}: {reader.fieldnames}; "
                    f"expected {MANIFEST_FIELDS}"
                )
            for row in reader:
                try:
                    tasks.append(
                        Task(
                            task_id=row["task_id"],
                            gpu_slot=int(row["gpu_slot"]),
                            root=row["root"],
                            dataset=row["dataset"],
                            poison_type=row["poison_type"],
                            poison_rate=row["poison_rate"],
                            alpha=row["alpha"],
                            cover_rate=row["cover_rate"],
                            trigger=row["trigger"],
                            model=row["model"],
                            arch_name=row["arch_name"],
                            relative_dir=row["relative_dir"],
                            checkpoint_name=row["checkpoint_name"],
                        )
                    )
                except (KeyError, ValueError) as error:
                    raise RuntimeError(f"invalid row in {manifest_path}: {row}") from error
    return tasks


def reconstruct_directory(task: Task) -> Path:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from utils import supervisor

    args = SimpleNamespace(
        dataset=task.dataset,
        poison_type=task.poison_type,
        poison_rate=float(task.poison_rate),
        alpha=float(task.alpha),
        cover_rate=float(task.cover_rate),
        trigger=task.trigger or None,
        model=task.model,
        model_path=None,
        defense="SentiNet",
        no_aug=False,
        sample_cap=None,
    )
    previous_root = os.environ.get("POISONED_TRAIN_SET_ROOT")
    os.environ["POISONED_TRAIN_SET_ROOT"] = str(REPO_ROOT / task.root)
    try:
        return Path(supervisor.get_poison_set_dir(args)).resolve()
    finally:
        if previous_root is None:
            os.environ.pop("POISONED_TRAIN_SET_ROOT", None)
        else:
            os.environ["POISONED_TRAIN_SET_ROOT"] = previous_root


def validate_tasks(tasks: list[Task]) -> None:
    errors: list[str] = []
    if not (REPO_ROOT / "other_defense.py").is_file():
        errors.append(f"missing defense entrypoint: {REPO_ROOT / 'other_defense.py'}")
    if len(tasks) != EXPECTED_TASKS:
        errors.append(f"expected {EXPECTED_TASKS} tasks, found {len(tasks)}")

    ids = [task.task_id for task in tasks]
    directories = [task.relative_dir for task in tasks]
    if len(ids) != len(set(ids)):
        errors.append("task_id values are not unique")
    if len(directories) != len(set(directories)):
        errors.append("relative_dir values are not unique")

    expected_counts = {
        "root": {"poisoned_train_set": 252, "poisoned_train_set3": 36},
        "dataset": {"cifar10": 144, "tiny_imagenet": 144},
        "poison_type": {"basic": 162, "adaptive_patch": 126},
        "gpu_slot": {slot: 36 for slot in range(8)},
    }
    actual_counts = {
        "root": Counter(task.root for task in tasks),
        "dataset": Counter(task.dataset for task in tasks),
        "poison_type": Counter(task.poison_type for task in tasks),
        "gpu_slot": Counter(task.gpu_slot for task in tasks),
    }
    for label, expected in expected_counts.items():
        if dict(actual_counts[label]) != expected:
            errors.append(f"unexpected {label} counts: {dict(actual_counts[label])}; expected {expected}")

    for slot in range(8):
        slot_tasks = [task for task in tasks if task.gpu_slot == slot]
        dataset_counts = Counter(task.dataset for task in slot_tasks)
        arch_counts = Counter(task.arch_name for task in slot_tasks)
        if dataset_counts != {"cifar10": 18, "tiny_imagenet": 18}:
            errors.append(f"GPU slot {slot} dataset imbalance: {dict(dataset_counts)}")
        expected_arch_counts = {
            "ResNet18_cifar10": 6,
            "ResNet18_tiny_imagenet": 6,
            "mobilenetv2_cifar10": 6,
            "mobilenetv2_tiny_imagenet": 6,
            "vgg19_bn_cifar10": 6,
            "vgg19_bn_tiny_imagenet": 6,
        }
        if dict(arch_counts) != expected_arch_counts:
            errors.append(f"GPU slot {slot} architecture imbalance: {dict(arch_counts)}")

    for task in tasks:
        directory = task.directory
        checkpoint = directory / task.checkpoint_name
        if not directory.is_dir():
            errors.append(f"missing task directory: {directory}")
            continue
        if not checkpoint.is_file():
            errors.append(f"missing checkpoint: {checkpoint}")
        try:
            reconstructed = reconstruct_directory(task)
        except Exception as error:  # report every bad task in one validation pass
            errors.append(f"could not reconstruct {task.relative_dir}: {error}")
        else:
            if reconstructed != directory.resolve():
                errors.append(
                    f"path mismatch for {task.task_id}: supervisor={reconstructed}, manifest={directory.resolve()}"
                )

    if errors:
        detail = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeError(f"validation failed with {len(errors)} error(s):\n{detail}")


def old_name(source: Path) -> Path:
    return source.with_name(f"{source.stem}_old{source.suffix}")


def discover_archive_pairs(tasks: list[Task]) -> list[tuple[Path, Path]]:
    logical_pairs: dict[Path, Path] = {}
    for task in tasks:
        directory = task.directory
        base_source = directory / "sentinet_defense_results.json"
        base_old = old_name(base_source)
        if base_source.exists() or base_old.exists():
            logical_pairs[base_source] = base_old

        for path in directory.glob("sentinet_defense_results_test_alpha=*.json"):
            if path.name.endswith("_old.json"):
                source = path.with_name(path.name[: -len("_old.json")] + ".json")
                logical_pairs[source] = path
            else:
                logical_pairs[path] = old_name(path)
    return sorted(logical_pairs.items(), key=lambda pair: str(pair[0]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_results(tasks: list[Task], dry_run: bool, audit_log: Path | None) -> None:
    pairs = discover_archive_pairs(tasks)
    if len(pairs) != EXPECTED_ARCHIVED_RESULTS:
        raise RuntimeError(
            f"expected {EXPECTED_ARCHIVED_RESULTS} historical result files, found {len(pairs)} logical files"
        )

    collisions = [(source, destination) for source, destination in pairs if source.exists() and destination.exists()]
    if collisions:
        examples = "\n".join(f"  - {source} and {destination}" for source, destination in collisions[:10])
        raise RuntimeError(
            "archive refused because both the unsuffixed and _old result exist; "
            f"this protects rerun results from being overwritten:\n{examples}"
        )

    records: list[dict[str, str | int]] = []
    moves: list[tuple[Path, Path, str, int]] = []
    for source, destination in pairs:
        current = source if source.exists() else destination
        if not current.is_file():
            raise RuntimeError(f"archive pair has no file: {source} -> {destination}")
        file_hash = sha256(current)
        size = current.stat().st_size
        action = "would_move" if source.exists() else "already_archived"
        records.append(
            {
                "action": action,
                "source": str(source.relative_to(REPO_ROOT)),
                "destination": str(destination.relative_to(REPO_ROOT)),
                "size": size,
                "sha256": file_hash,
            }
        )
        if source.exists():
            moves.append((source, destination, file_hash, size))

    if dry_run:
        print(f"Archive dry run passed: {len(pairs)} historical files, {len(moves)} pending renames.")
        return

    if audit_log is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_log = REPO_ROOT / "logs" / "sentinet_badnet_adaptive_patch_rerun" / f"archive_{timestamp}.tsv"
    elif not audit_log.is_absolute():
        audit_log = REPO_ROOT / audit_log
    if audit_log.exists():
        raise RuntimeError(f"audit log already exists: {audit_log}")

    for source, destination, expected_hash, expected_size in moves:
        source.rename(destination)
        if destination.stat().st_size != expected_size or sha256(destination) != expected_hash:
            raise RuntimeError(f"content verification failed after renaming {source} to {destination}")

    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with audit_log.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("action", "source", "destination", "size", "sha256"),
            delimiter="\t",
        )
        writer.writeheader()
        for record in records:
            if record["action"] == "would_move":
                record["action"] = "moved"
            writer.writerow(record)

    remaining_sources = [source for source, _ in pairs if source.exists()]
    archived = [destination for _, destination in pairs if destination.is_file()]
    if remaining_sources or len(archived) != EXPECTED_ARCHIVED_RESULTS:
        raise RuntimeError(
            f"archive post-check failed: {len(remaining_sources)} sources remain and {len(archived)} old files exist"
        )
    print(f"Archived {len(moves)} files; {len(archived)} _old files verified. Audit: {audit_log}")


def validate_archive_state(tasks: list[Task]) -> None:
    pairs = discover_archive_pairs(tasks)
    old_files = [destination for _, destination in pairs if destination.is_file()]
    unsuffixed_test_results = [
        source
        for source, destination in pairs
        if "_test_alpha=" in source.name and source.is_file() and not destination.is_file()
    ]
    if len(old_files) != EXPECTED_ARCHIVED_RESULTS or unsuffixed_test_results:
        raise RuntimeError(
            "historical results are not fully archived; run the 'archive' subcommand before 'run' "
            f"(found {len(old_files)} _old files, expected {EXPECTED_ARCHIVED_RESULTS})"
        )


def parse_gpus(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if len(gpus) != 8 or len(set(gpus)) != 8:
        raise RuntimeError(f"--gpus must contain exactly eight unique IDs; got {raw!r}")
    return gpus


def build_command(task: Task, gpu: str, python_bin: str, seed: int) -> list[str]:
    command = [
        python_bin,
        str(REPO_ROOT / "other_defense.py"),
        "-defense=SentiNet",
        f"-dataset={task.dataset}",
        f"-poison_type={task.poison_type}",
        f"-poison_rate={task.poison_rate}",
        f"-alpha={task.alpha}",
        f"-model={task.model}",
        f"-devices={gpu}",
        f"-seed={seed}",
    ]
    if task.poison_type == "adaptive_patch":
        command.append(f"-cover_rate={task.cover_rate}")
    if task.trigger:
        command.append(f"-trigger={task.trigger}")
    return command


def valid_result(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and all(key in value for key in REQUIRED_RESULT_KEYS)


def run_tasks(args: argparse.Namespace, tasks: list[Task]) -> None:
    validate_archive_state(tasks)
    gpus = parse_gpus(args.gpus)
    python_bin = str(Path(args.python).expanduser()) if os.sep in args.python else args.python

    if args.dry_run:
        for task in sorted(tasks, key=lambda item: (item.gpu_slot, item.task_id)):
            gpu = gpus[task.gpu_slot]
            root = REPO_ROOT / task.root
            command = build_command(task, gpu, python_bin, args.seed)
            print(
                f"GPU_SLOT={task.gpu_slot} GPU={gpu} "
                f"POISONED_TRAIN_SET_ROOT={shlex.quote(str(root))} {shlex.join(command)}"
            )
        print(f"Dry run passed: {len(tasks)} commands, 36 commands per GPU.")
        return

    if args.log_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_root = REPO_ROOT / "logs" / "sentinet_badnet_adaptive_patch_rerun" / timestamp
    else:
        log_root = args.log_root if args.log_root.is_absolute() else REPO_ROOT / args.log_root
    log_root.mkdir(parents=True, exist_ok=False)

    print_lock = threading.Lock()
    failure_lock = threading.Lock()
    failures: list[dict[str, str | int]] = []
    counts = Counter()

    def report(message: str) -> None:
        with print_lock:
            print(message, flush=True)

    def worker(slot: int) -> None:
        gpu = gpus[slot]
        slot_tasks = sorted((task for task in tasks if task.gpu_slot == slot), key=lambda item: item.task_id)
        for index, task in enumerate(slot_tasks, start=1):
            if not args.force and valid_result(task.result_path):
                with failure_lock:
                    counts["skipped"] += 1
                report(f"[GPU {gpu}] skip valid result {index}/36: {task.relative_dir}")
                continue

            log_file = (
                log_root
                / f"gpu{gpu}"
                / task.root
                / task.dataset
                / f"{Path(task.relative_dir).name}.log"
            )
            log_file.parent.mkdir(parents=True, exist_ok=True)
            command = build_command(task, gpu, python_bin, args.seed)
            environment = os.environ.copy()
            environment.pop("CUDA_VISIBLE_DEVICES", None)
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment["POISONED_TRAIN_SET_ROOT"] = str(REPO_ROOT / task.root)
            report(f"[GPU {gpu}] run {index}/36: {task.relative_dir}")

            started = datetime.now().astimezone().isoformat()
            launch_error = None
            with log_file.open("w", encoding="utf-8") as log_handle:
                log_handle.write(f"Started: {started}\n")
                log_handle.write(f"POISONED_TRAIN_SET_ROOT={environment['POISONED_TRAIN_SET_ROOT']}\n")
                log_handle.write(f"Command: {shlex.join(command)}\n")
                log_handle.flush()
                try:
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    return_code = completed.returncode
                except OSError as error:
                    launch_error = str(error)
                    return_code = -1
                    log_handle.write(f"Launch error: {launch_error}\n")
                log_handle.write(f"\nFinished: {datetime.now().astimezone().isoformat()}\n")
                log_handle.write(f"Exit status: {return_code}\n")

            if return_code == 0 and valid_result(task.result_path):
                with failure_lock:
                    counts["completed"] += 1
                continue

            reason = f"exit={return_code}, result_valid={valid_result(task.result_path)}"
            if launch_error:
                reason += f", launch_error={launch_error}"
            with failure_lock:
                counts["failed"] += 1
                failures.append(
                    {
                        "task_id": task.task_id,
                        "gpu": gpu,
                        "relative_dir": task.relative_dir,
                        "reason": reason,
                        "log": str(log_file.relative_to(REPO_ROOT)),
                    }
                )
            report(f"[GPU {gpu}] FAILED: {task.relative_dir} ({reason})")

    threads = [threading.Thread(target=worker, args=(slot,), name=f"gpu-slot-{slot}") for slot in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failure_path = log_root / "failures.tsv"
    with failure_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_id", "gpu", "relative_dir", "reason", "log"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(sorted(failures, key=lambda item: str(item["task_id"])))

    print(
        f"Run finished: completed={counts['completed']}, skipped={counts['skipped']}, "
        f"failed={counts['failed']}. Logs: {log_root}"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} task(s) failed; see {failure_path}")


def main() -> int:
    args = parse_args()
    try:
        tasks = load_tasks()
        validate_tasks(tasks)
        if args.command == "check":
            print(
                "Validation passed: 288 unique tasks; roots=252+36; datasets=144+144; "
                "each GPU slot has 36 tasks, 18 per dataset, and 6 per dataset/architecture."
            )
        elif args.command == "archive":
            archive_results(tasks, args.dry_run, args.audit_log)
        elif args.command == "run":
            run_tasks(args, tasks)
        return 0
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
