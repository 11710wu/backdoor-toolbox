#!/usr/bin/env bash

# Complete poisoned_train_set4 backfill runner.
#
# What this script does:
#   1) Optionally reruns the corrected BELT set4 grid using final-epoch checkpoints.
#   2) Scans existing non-BELT poisoned_train_set4 result directories and reruns only
#      missing source/transfer/Qwen/defense artifacts.
#
# It never deletes result directories. Use DRY_RUN=1 first.
#
# Examples:
#   DRY_RUN=1 bash run/rerun_set4_complete_missing_and_belt.sh
#   PHASE=all PARALLEL=1 GPU_IDS="0 1 2 3" bash run/rerun_set4_complete_missing_and_belt.sh
#   RUN_BELT=0 PHASE=defense PARALLEL=1 GPU_IDS="0 1 2 3" bash run/rerun_set4_complete_missing_and_belt.sh
#
# Phase meanings for non-BELT backfill:
#   source    -> missing test_results_seed=2333*.json
#   transfer  -> missing STL10 or ImageNetV2 tiny-domain transfer result
#   qwen      -> missing tiny-domain Qwen transfer result
#   defense   -> missing SentiNet/STRIP/ScaleUp/IBD_PSC JSON
#   all       -> all of the above

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

export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"
PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
PARALLEL="${PARALLEL:-0}"
GPU_IDS="${GPU_IDS:-${DEVICES:-0}}"
MAX_JOBS="${MAX_JOBS:-}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
RUN_BELT="${RUN_BELT:-1}"
RUN_EXISTING_MISSING="${RUN_EXISTING_MISSING:-1}"
RUN_QWEN="${RUN_QWEN:-1}"
DEFENSE_LIST="${DEFENSE_LIST:-SentiNet STRIP ScaleUp IBD_PSC}"
TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/imagenetv2-matched-frequency-tiny-organized}"
QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/tiny-target-domain-qwen-full-organized}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/rerun_set4_complete_missing_and_belt_$(date +%Y%m%d_%H%M%S).log}"

case "$PHASE" in
  all|source|test|transfer|target|qwen|defense|defenses|belt)
    ;;
  *)
    echo "Unsupported PHASE=${PHASE}. Use all|source|transfer|qwen|defense|belt." >&2
    exit 2
    ;;
esac

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
  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/set4_missing_$$_${RANDOM}.out")"
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

run_generated_commands() {
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
  done < <(generate_missing_commands)

  if [ "$active" -gt 0 ]; then
    wait_phase_batch "${pids[@]}"
  fi

  if [ "$idx" -eq 0 ]; then
    echo
    echo "No non-BELT missing commands for PHASE=${PHASE}."
  else
    echo
    echo "Generated non-BELT missing commands: ${idx}"
  fi
}

generate_missing_commands() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$POISONED_TRAIN_SET_ROOT" "$PHASE" "$RUN_QWEN" "$DEFENSE_LIST" "$TARGET_DOMAIN_DIR" "$QWEN_TARGET_DOMAIN_DIR" <<'PY'
import glob
import os
import re
import shlex
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
root = repo / sys.argv[2]
phase_arg = sys.argv[3]
run_qwen = sys.argv[4] == "1"
defenses = sys.argv[5].split()
target_domain = sys.argv[6]
qwen_domain = sys.argv[7]

py = os.environ.get("PYTHON_BIN", "python")

def shell_join(parts):
    return " ".join(shlex.quote(str(p)) for p in parts)

def model_from_arch(arch):
    mapping = {
        "MicroCNN_cifar10": "micro_cnn",
        "SmallCNN_cifar10": "small_cnn",
        "ResNet18_cifar10": "resnet18",
        "ResNet34_cifar10": "resnet34",
        "ResNet50_cifar10": "resnet50",
        "ResNet34_tiny_imagenet": "resnet34",
        "ResNet50_tiny_imagenet": "resnet50",
        "DenseNet121_tiny_imagenet": "densenet121",
    }
    return mapping.get(arch)

def parse_attack_args(name):
    # Return (poison_type, extra_args, raw_input_required) or None if unsupported.
    m = re.match(r"^basic_([0-9.]+)_alpha=([0-9.]+)_trigger=(.+?)_poison_seed=", name)
    if m:
        rate, alpha, trigger = m.groups()
        return "basic", ["-poison_type=basic", f"-poison_rate={rate}", f"-alpha={alpha}", f"-trigger={trigger}"], False

    m = re.match(r"^blend_([0-9.]+)_alpha=([0-9.]+)_trigger=(.+?)_poison_seed=", name)
    if m:
        rate, alpha, trigger = m.groups()
        return "blend", ["-poison_type=blend", f"-poison_rate={rate}", f"-alpha={alpha}", f"-trigger={trigger}"], False

    m = re.match(r"^adaptive_blend_([0-9.]+)_alpha=([0-9.]+)_cover=([0-9.]+)_trigger=(.+?)_poison_seed=", name)
    if m:
        rate, alpha, cover, trigger = m.groups()
        return "adaptive_blend", ["-poison_type=adaptive_blend", f"-poison_rate={rate}", f"-cover_rate={cover}", f"-alpha={alpha}", f"-trigger={trigger}"], False

    m = re.match(r"^adaptive_patch_([0-9.]+)_alpha=([0-9.]+)_cover=([0-9.]+)_poison_seed=", name)
    if m:
        rate, alpha, cover = m.groups()
        return "adaptive_patch", ["-poison_type=adaptive_patch", f"-poison_rate={rate}", f"-cover_rate={cover}", f"-alpha={alpha}"], False

    m = re.match(r"^WaNet_([0-9.]+)_cover=([0-9.]+)_s=([0-9.]+)_k=([0-9]+)_poison_seed=", name)
    if m:
        rate, cover, s, k = m.groups()
        return "WaNet", ["-poison_type=WaNet", f"-poison_rate={rate}", f"-cover_rate={cover}", f"-s={s}", f"-k={k}"], False

    m = re.match(r"^SIG_([0-9.]+)_delta=([0-9.]+)_f=([0-9.]+)_mode=([^_]+)_poison_seed=", name)
    if m:
        rate, delta, freq, mode = m.groups()
        return "SIG", ["-poison_type=SIG", f"-poison_rate={rate}", f"-delta={delta}", f"-f={freq}", f"-label_mode={mode}"], False

    m = re.match(r"^upgd_([0-9.]+)_eps=([0-9.]+)_constraint=([^_]+)_steps=([0-9]+)_mode=([^_]+)_mult=([0-9]+)_poison_seed=", name)
    if m:
        rate, eps, constraint, steps, mode, mult = m.groups()
        return "upgd", ["-poison_type=upgd", f"-poison_rate={rate}", f"-eps={eps}", f"-constraint={constraint}", f"-upgd_steps={steps}", f"-label_mode={mode}", f"-upgd_steps_multiplier={mult}"], True

    return None

def wanted(phase):
    if phase_arg == "all":
        return phase in {"source", "transfer", "qwen", "defense"}
    if phase_arg in {"test"}:
        return phase == "source"
    if phase_arg in {"target"}:
        return phase == "transfer"
    if phase_arg in {"defenses"}:
        return phase == "defense"
    return phase == phase_arg

def emit(phase, dataset, arch, path, desc, parts):
    print("\t".join([phase, dataset, arch, str(path), desc, shell_join(parts)]))

for dataset_dir in sorted(root.glob("*")):
    if not dataset_dir.is_dir():
        continue
    dataset = dataset_dir.name
    if dataset not in {"cifar10", "tiny_imagenet"}:
        continue

    for d in sorted(dataset_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith(("belt_", "none_", "upgd_raw_base_")):
            continue
        if "_arch=" not in name:
            continue
        arch = name.rsplit("_arch=", 1)[1]
        model = model_from_arch(arch)
        parsed = parse_attack_args(name)
        if model is None or parsed is None:
            continue

        poison_type, attack_args, raw_required = parsed
        raw_args = ["-no_normalize"] if raw_required else []
        common = [py, "__SCRIPT__", f"-dataset={dataset}", f"-model={model}", "-devices=__GPU__"]
        attack = attack_args + raw_args
        desc = f"{dataset} {arch} {name}"

        if wanted("source") and not glob.glob(str(d / "test_results_seed=2333*.json")):
            parts = common.copy()
            parts[1] = "test_model.py"
            emit("source", dataset, arch, d, desc, parts + attack)

        if wanted("transfer"):
            if dataset == "cifar10":
                missing = not glob.glob(str(d / "test_stl10_results*.txt"))
                if missing:
                    parts = common.copy()
                    parts[1] = "test_stl10.py"
                    emit("transfer", dataset, arch, d, desc, parts + attack)
            else:
                missing = not (d / "test_tiny_target_domain_results.txt").exists()
                if missing:
                    parts = common.copy()
                    parts[1] = "test_tiny_target_domain.py"
                    emit("transfer", dataset, arch, d, desc, parts + [f"-source_dataset={dataset}", f"-target_domain_dir={target_domain}"] + attack)

        if run_qwen and wanted("qwen") and dataset == "tiny_imagenet":
            if not (d / "test_tiny_target_domain_qwen_results.txt").exists():
                parts = common.copy()
                parts[1] = "test_tiny_target_domain_qwen.py"
                emit("qwen", dataset, arch, d, desc, parts + [f"-source_dataset={dataset}", f"-target_domain_dir={qwen_domain}"] + attack)

        if wanted("defense"):
            defense_files = {
                "SentiNet": "sentinet_defense_results.json",
                "STRIP": "strip_defense_results.json",
                "ScaleUp": "scaleup_defense_results.json",
                "IBD_PSC": "ibd_psc_defense_results.json",
            }
            for defense in defenses:
                out = defense_files.get(defense)
                if not out:
                    continue
                if not (d / out).exists():
                    parts = common.copy()
                    parts[1] = "other_defense.py"
                    emit("defense", dataset, arch, d, f"{defense} {desc}", parts + [f"-defense={defense}"] + attack)
PY
}

summarize_missing() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$POISONED_TRAIN_SET_ROOT" "$RUN_QWEN" "$DEFENSE_LIST" <<'PY'
import glob
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
root = repo / sys.argv[2]
run_qwen = sys.argv[3] == "1"
defenses = sys.argv[4].split()

def supported(name):
    if name.startswith(("belt_", "none_", "upgd_raw_base_")):
        return False
    return "_arch=" in name and any(name.startswith(prefix) for prefix in (
        "basic_", "blend_", "adaptive_blend_", "adaptive_patch_", "WaNet_", "SIG_", "upgd_"
    ))

missing_counter = Counter()
arch_counter = Counter()
dir_count = 0

for dataset_dir in sorted(root.glob("*")):
    if not dataset_dir.is_dir() or dataset_dir.name not in {"cifar10", "tiny_imagenet"}:
        continue
    dataset = dataset_dir.name
    for d in sorted(dataset_dir.iterdir()):
        if not d.is_dir() or not supported(d.name):
            continue
        arch = d.name.rsplit("_arch=", 1)[1]
        misses = []
        if not glob.glob(str(d / "test_results_seed=2333*.json")):
            misses.append("source")
        if dataset == "cifar10":
            if not glob.glob(str(d / "test_stl10_results*.txt")):
                misses.append("transfer:stl10")
        else:
            if not (d / "test_tiny_target_domain_results.txt").exists():
                misses.append("transfer:imagenetv2")
            if run_qwen and not (d / "test_tiny_target_domain_qwen_results.txt").exists():
                misses.append("transfer:qwen")
        files = {
            "SentiNet": "sentinet_defense_results.json",
            "STRIP": "strip_defense_results.json",
            "ScaleUp": "scaleup_defense_results.json",
            "IBD_PSC": "ibd_psc_defense_results.json",
        }
        for defense in defenses:
            out = files.get(defense)
            if out and not (d / out).exists():
                misses.append(f"defense:{defense}")
        if misses:
            dir_count += 1
            arch_counter[(dataset, arch)] += 1
            for m in misses:
                missing_counter[m] += 1

print(f"Existing non-BELT dirs with missing artifacts: {dir_count}")
if missing_counter:
    print("Missing artifact counts:")
    for key, value in sorted(missing_counter.items()):
        print(f"  {key}: {value}")
    print("Missing dirs by dataset/arch:")
    for (dataset, arch), value in sorted(arch_counter.items()):
        print(f"  {dataset} {arch}: {value}")
PY
}

echo "============================================================"
echo "poisoned_train_set4 complete missing-result backfill"
echo "============================================================"
echo "repo                  : ${REPO_ROOT}"
echo "python                : ${PYTHON_BIN}"
echo "result root           : ${POISONED_TRAIN_SET_ROOT}"
echo "phase                 : ${PHASE}"
echo "gpu ids               : ${GPU_IDS}"
echo "parallel              : ${PARALLEL}"
echo "run BELT              : ${RUN_BELT}"
echo "run existing missing  : ${RUN_EXISTING_MISSING}"
echo "run qwen              : ${RUN_QWEN}"
echo "defenses              : ${DEFENSE_LIST}"
echo "dry run               : ${DRY_RUN}"
echo "error log             : ${ERROR_LOG}"
echo "============================================================"

echo
echo "Current non-BELT missing summary:"
summarize_missing

if [ "$RUN_BELT" = "1" ] && { [ "$PHASE" = "all" ] || [ "$PHASE" = "belt" ]; }; then
  echo
  echo "----- BELT corrected full rerun -----"
  PHASE=all \
  DRY_RUN="$DRY_RUN" \
  PARALLEL="$PARALLEL" \
  GPU_IDS="$GPU_IDS" \
  MAX_JOBS="$MAX_JOBS" \
  STOP_ON_FAIL="$STOP_ON_FAIL" \
  SKIP_EXISTING="${BELT_SKIP_EXISTING:-1}" \
  RUN_QWEN="$RUN_QWEN" \
  DEFENSE_LIST="$DEFENSE_LIST" \
  TARGET_DOMAIN_DIR="$TARGET_DOMAIN_DIR" \
  QWEN_TARGET_DOMAIN_DIR="$QWEN_TARGET_DOMAIN_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash run/rerun_set4_belt_final_checkpoint_full.sh
fi

if [ "$RUN_EXISTING_MISSING" = "1" ] && [ "$PHASE" != "belt" ]; then
  echo
  echo "----- Existing non-BELT missing-result backfill -----"
  run_generated_commands
fi

echo
echo "Finished poisoned_train_set4 complete missing-result backfill."
