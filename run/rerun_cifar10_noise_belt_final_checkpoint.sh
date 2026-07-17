#!/usr/bin/env bash

# Rerun CIFAR10 input-noise BELT results with final-epoch checkpoints.
#
# The script reads old configurations from the backup produced before rerun,
# and writes fresh results back to poisoned_train_set/cifar10.
#
# Examples:
#   DRY_RUN=1 bash run/rerun_cifar10_noise_belt_final_checkpoint.sh
#   PHASE=all PARALLEL=1 GPU_IDS="6 7" bash run/rerun_cifar10_noise_belt_final_checkpoint.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/root/anaconda3/envs/backtool/bin/python" ]; then
    PYTHON_BIN="/root/anaconda3/envs/backtool/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
export PYTHON_BIN

if [ -z "${BACKUP_ROOT:-}" ]; then
  BACKUP_ROOT="$(ls -dt /workspace/data1/belt_final_checkpoint_rerun_*/backdoor-toolbox-noise/poisoned_train_set/cifar10 2>/dev/null | head -n 1 || true)"
fi

PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
PARALLEL="${PARALLEL:-0}"
GPU_IDS="${GPU_IDS:-${DEVICES:-0}}"
MAX_JOBS="${MAX_JOBS:-}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DEFENSE_LIST="${DEFENSE_LIST:-SentiNet STRIP ScaleUp IBD_PSC}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/rerun_cifar10_noise_belt_final_checkpoint_$(date +%Y%m%d_%H%M%S).log}"

case "$PHASE" in
  all|create|train|source|test|transfer|target|defense|defenses)
    ;;
  *)
    echo "Unsupported PHASE=${PHASE}. Use all|create|train|source|transfer|defense." >&2
    exit 2
    ;;
esac

if [ -z "$BACKUP_ROOT" ] || [ ! -d "$BACKUP_ROOT" ]; then
  echo "BACKUP_ROOT not found. Set BACKUP_ROOT to the backed-up noise cifar10 path." >&2
  exit 2
fi

gpu_for_index() {
  local index="$1"
  local ids=($GPU_IDS)
  local n="${#ids[@]}"
  if [ "$n" -eq 0 ]; then
    echo "0"
  else
    echo "${ids[$((index % n))]}"
  fi
}

max_jobs() {
  local ids=($GPU_IDS)
  if [ "$PARALLEL" = "1" ]; then
    if [ -n "$MAX_JOBS" ]; then
      echo "$MAX_JOBS"
    else
      echo "${#ids[@]}"
    fi
  else
    echo "1"
  fi
}

run_command() {
  local cmd="$1"
  local desc="$2"
  local tmp_out
  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/belt_noise_$$_${RANDOM}.out")"
  echo
  echo ">>> ${desc}"
  echo "$cmd"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] skipped"
    rm -f "$tmp_out"
    return 0
  fi
  eval "$cmd" 2>&1 | tee "$tmp_out"
  local exit_code="${PIPESTATUS[0]}"
  if [ "$exit_code" -ne 0 ]; then
    {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] command failed with exit code ${exit_code}"
      echo "description: ${desc}"
      echo "command: ${cmd}"
      cat "$tmp_out" 2>/dev/null
      echo "---"
    } >> "$ERROR_LOG"
    rm -f "$tmp_out"
    if [ "$STOP_ON_FAIL" = "1" ]; then
      exit "$exit_code"
    fi
    return "$exit_code"
  fi
  rm -f "$tmp_out"
}

wait_phase_batch() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
    exit 1
  fi
  return "$failed"
}

generate_commands() {
  local requested_phase="$1"
  "$PYTHON_BIN" - "$REPO_ROOT" "$BACKUP_ROOT" "$requested_phase" "$DEFENSE_LIST" "$SKIP_EXISTING" <<'PY'
import glob
import os
import re
import shlex
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
backup_root = Path(sys.argv[2]).resolve()
phase_arg = sys.argv[3]
defenses = sys.argv[4].split()
skip_existing = sys.argv[5] == "1"
py = os.environ.get("PYTHON_BIN", "python")
out_root = repo / "poisoned_train_set" / "cifar10"

def shell_join(parts):
    return " ".join(shlex.quote(str(p)) for p in parts)

def model_from_arch(arch):
    suffix = "_cifar10"
    base = arch[:-len(suffix)] if arch.endswith(suffix) else arch
    mapping = {
        "ResNet18": "resnet18",
        "ResNet34": "resnet34",
        "ResNet50": "resnet50",
        "vgg19_bn": "vgg19_bn",
        "mobilenetv2": "mobilenetv2",
        "SmallCNN": "small_cnn",
        "MicroCNN": "micro_cnn",
    }
    return mapping.get(base, base)

def parse_config(path):
    m = re.match(
        r"^belt_([0-9.]+)_alpha=([0-9.]+)_cover=([0-9.]+)_mask=([0-9.]+)_noise=([^_]+(?:_[^_]+)?)_level=([0-9.]+)_poison_seed=([0-9]+)_arch=(.+)$",
        path.name,
    )
    if not m:
        return None
    rate, alpha, cover, mask, noise_type, noise_level, seed, arch = m.groups()
    return {
        "dataset": "cifar10",
        "rate": rate,
        "alpha": alpha,
        "cover": cover,
        "mask": mask,
        "noise_type": noise_type,
        "noise_level": noise_level,
        "seed": seed,
        "arch": arch,
        "model": model_from_arch(arch),
        "out_dir": out_root / path.name,
        "name": path.name,
    }

def wanted(phase):
    if phase_arg == "all":
        return True
    if phase_arg == "test":
        return phase == "source"
    if phase_arg == "target":
        return phase == "transfer"
    if phase_arg == "defenses":
        return phase == "defense"
    return phase == phase_arg

def complete(cfg, phase, defense=None):
    d = cfg["out_dir"]
    if not skip_existing:
        return False
    if phase == "create":
        return (d / "labels").exists() and (d / "poison_indices").exists() and (d / "pmarks").exists()
    if phase == "train":
        return bool(glob.glob(str(d / "*_belt_aug_model_seed=2333.pt"))) and (d / "train_results_seed=2333.json").exists()
    if phase == "source":
        return bool(glob.glob(str(d / "test_results_seed=2333*.json")))
    if phase == "transfer":
        return bool(glob.glob(str(d / "test_stl10_results*.txt")))
    if phase == "defense":
        files = {
            "SentiNet": "sentinet_defense_results.json",
            "STRIP": "strip_defense_results.json",
            "ScaleUp": "scaleup_defense_results.json",
            "IBD_PSC": "ibd_psc_defense_results.json",
            "NC": "nc_defense_results.json",
        }
        out = files.get(defense)
        return bool(out and (d / out).exists())
    return False

def emit(phase, cfg, desc, parts):
    print("\t".join([phase, cfg["dataset"], cfg["arch"], str(cfg["out_dir"]), desc, shell_join(parts)]))

configs = []
for path in sorted(backup_root.glob("belt_*")):
    if path.is_dir():
        cfg = parse_config(path)
        if cfg:
            configs.append(cfg)

for cfg in configs:
    base = [py, "__SCRIPT__", "-dataset=cifar10", f"-model={cfg['model']}", "-devices=__GPU__"]
    noise_args = [
        f"-input_noise_type={cfg['noise_type']}",
        f"-input_noise_level={cfg['noise_level']}",
    ]
    create_args = [
        "-poison_type=belt",
        f"-poison_rate={cfg['rate']}",
        f"-cover_rate={cfg['cover']}",
        f"-mask_rate={cfg['mask']}",
        f"-alpha={cfg['alpha']}",
    ] + noise_args
    raw_args = create_args + ["-no_normalize"]
    desc = f"cifar10 {cfg['arch']} noise={cfg['noise_type']} level={cfg['noise_level']} rate={cfg['rate']} alpha={cfg['alpha']} mask={cfg['mask']}"

    if wanted("create") and not complete(cfg, "create"):
        parts = base.copy()
        parts[1] = "create_poisoned_set.py"
        emit("create", cfg, desc, parts + create_args)

    if wanted("train") and not complete(cfg, "train"):
        parts = base.copy()
        parts[1] = "train_on_poisoned_set.py"
        emit("train", cfg, desc, parts + raw_args)

    if wanted("source") and not complete(cfg, "source"):
        parts = base.copy()
        parts[1] = "test_model.py"
        emit("source", cfg, desc, parts + raw_args)

    if wanted("transfer") and not complete(cfg, "transfer"):
        parts = base.copy()
        parts[1] = "test_stl10.py"
        emit("transfer", cfg, desc, parts + raw_args)

    if wanted("defense"):
        for defense in defenses:
            if complete(cfg, "defense", defense):
                continue
            parts = base.copy()
            parts[1] = "other_defense.py"
            emit("defense", cfg, f"{defense} {desc}", parts + [f"-defense={defense}"] + raw_args)
PY
}

summarize_configs() {
  "$PYTHON_BIN" - "$BACKUP_ROOT" <<'PY'
import re
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
counts = Counter()
for path in root.glob("belt_*"):
    if not path.is_dir():
        continue
    m = re.search(r"_noise=([^_]+(?:_[^_]+)?)_level=([0-9.]+).*_arch=(.+)$", path.name)
    if not m:
        continue
    noise_type, noise_level, arch = m.groups()
    counts[(noise_type, noise_level, arch)] += 1
print(f"Total configs: {sum(counts.values())}")
for (noise_type, noise_level, arch), count in sorted(counts.items()):
    print(f"  {noise_type} level={noise_level} {arch}: {count}")
PY
}

run_generated_commands() {
  local requested_phase="$1"
  local idx=0
  local active=0
  local max
  local pids=()
  local phase dataset arch path desc cmd gpu
  max="$(max_jobs)"

  while IFS=$'\t' read -r phase dataset arch path desc cmd; do
    [ -n "${cmd:-}" ] || continue
    gpu="$(gpu_for_index "$idx")"
    idx=$((idx + 1))
    cmd="${cmd//__GPU__/${gpu}}"

    if [ "$max" -gt 1 ] && [ "$DRY_RUN" != "1" ]; then
      (run_command "$cmd" "${phase}: ${desc}") &
      pids+=("$!")
      active=$((active + 1))
      if [ "$active" -ge "$max" ]; then
        wait_phase_batch "${pids[@]}"
        pids=()
        active=0
      fi
    else
      run_command "$cmd" "${phase}: ${desc}"
    fi
  done < <(generate_commands "$requested_phase")

  if [ "$active" -gt 0 ]; then
    wait_phase_batch "${pids[@]}"
  fi
  echo
  echo "Generated commands for phase ${requested_phase}: ${idx}"
}

echo "============================================================"
echo "CIFAR10 noise BELT final-checkpoint rerun"
echo "============================================================"
echo "repo          : ${REPO_ROOT}"
echo "backup root   : ${BACKUP_ROOT}"
echo "phase         : ${PHASE}"
echo "gpu ids       : ${GPU_IDS}"
echo "parallel      : ${PARALLEL}"
echo "skip existing : ${SKIP_EXISTING}"
echo "defenses      : ${DEFENSE_LIST}"
echo "dry run       : ${DRY_RUN}"
echo "error log     : ${ERROR_LOG}"
echo "============================================================"
summarize_configs

if [ "$PHASE" = "all" ]; then
  for phase in create train source transfer defense; do
    echo
    echo "----- ${phase} -----"
    run_generated_commands "$phase"
  done
else
  run_generated_commands "$PHASE"
fi

echo
echo "Finished CIFAR10 noise BELT final-checkpoint rerun."
