#!/usr/bin/env bash

# Qwen-only backfill for missing Tiny-ImageNet target-domain transfer results.
#
# Scope:
#   poisoned_train_set2:
#     - Tiny-ImageNet SIG missing Qwen transfer
#   poisoned_train_set3:
#     - Tiny-ImageNet WaNet/adaptive_blend/adaptive_patch missing Qwen transfer
#   poisoned_train_set4:
#     - Tiny-ImageNet WaNet/adaptive_blend/adaptive_patch/belt/blend missing Qwen transfer
#
# This script does not recreate poisoned sets, retrain models, rerun source tests,
# rerun ImageNetV2 transfer, or run defenses.
#
# Usage:
#   cd /workspace/backdoor-toolbox-new1
#   DEVICES=0 bash run/backfill_missing_tiny_qwen_poisoned_train_sets.sh
#
# Useful overrides:
#   DRY_RUN=1 bash run/backfill_missing_tiny_qwen_poisoned_train_sets.sh
#   ROOTS="poisoned_train_set4" bash run/backfill_missing_tiny_qwen_poisoned_train_sets.sh
#   STOP_ON_FAIL=1 bash run/backfill_missing_tiny_qwen_poisoned_train_sets.sh

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
ROOTS="${ROOTS:-poisoned_train_set2 poisoned_train_set3 poisoned_train_set4}"
QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR:-/workspace/backdoor-toolbox-new1/data/tiny-target-domain-qwen-full-organized}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${LOG_DIR}/backfill_missing_tiny_qwen_poisoned_train_sets_${TIMESTAMP}.log"

run_command() {
  local cmd="$1"
  local description="$2"
  local tmp_out
  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")"

  echo
  echo ">>> ${description}"
  echo "${cmd}"

  if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY_RUN] skipped"
    rm -f "${tmp_out}"
    return 0
  fi

  eval "${cmd}" 2>&1 | tee "${tmp_out}"
  local exit_code="${PIPESTATUS[0]}"
  if [ "${exit_code}" -ne 0 ]; then
    {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] command failed with exit code ${exit_code}"
      echo "command: ${cmd}"
      echo "description: ${description}"
      echo "--- stdout/stderr ---"
      cat "${tmp_out}" 2>/dev/null
      echo "---"
    } >> "${ERROR_LOG}"

    if [ "${STOP_ON_FAIL}" = "1" ]; then
      rm -f "${tmp_out}"
      exit "${exit_code}"
    fi
  fi

  rm -f "${tmp_out}"
  return "${exit_code}"
}

echo "============================================================"
echo "Backfill missing Tiny-ImageNet Qwen transfer results"
echo "============================================================"
echo "repo root    : ${REPO_ROOT}"
echo "python       : ${PYTHON_BIN}"
echo "devices      : ${DEVICES}"
echo "roots        : ${ROOTS}"
echo "qwen domain  : ${QWEN_TARGET_DOMAIN_DIR}"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

if [ ! -d "${QWEN_TARGET_DOMAIN_DIR}/test" ] && [ ! -d "${QWEN_TARGET_DOMAIN_DIR}/images" ]; then
  echo "Qwen target-domain directory must contain test/ or images/: ${QWEN_TARGET_DOMAIN_DIR}" | tee -a "${ERROR_LOG}"
  exit 1
fi

while IFS=$'\t' read -r description command; do
  [ -n "${command}" ] || continue
  run_command "${command}" "${description}"
done < <(
  ROOTS="${ROOTS}" DEVICES="${DEVICES}" QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
import re
import shlex
from pathlib import Path

roots = os.environ["ROOTS"].split()
devices = os.environ["DEVICES"]
target_domain = os.environ["QWEN_TARGET_DOMAIN_DIR"]

allowed_attacks = {
    "poisoned_train_set2": {"SIG"},
    "poisoned_train_set3": {"WaNet", "adaptive_blend", "adaptive_patch"},
    "poisoned_train_set4": {"WaNet", "adaptive_blend", "adaptive_patch", "belt", "blend"},
}

model_map = {
    "ResNet18": "resnet18",
    "ResNet34": "resnet34",
    "mobilenetv2": "mobilenetv2",
    "vgg19_bn": "vgg19_bn",
    "densenet121": "densenet121",
}


def attack_type(name):
    for attack in ["adaptive_blend", "adaptive_patch", "WaNet", "SIG", "belt", "blend"]:
        if name.startswith(f"{attack}_"):
            return attack
    return None


def arch_to_model(name):
    match = re.search(r"_arch=(.+?)_tiny_imagenet$", name)
    if not match:
        raise ValueError(f"cannot parse arch from {name}")
    arch = match.group(1)
    if arch not in model_map:
        raise ValueError(f"unsupported arch {arch} in {name}")
    return model_map[arch]


def core_name(name):
    return name.split("_poison_seed=", 1)[0]


def parse_args(name, attack):
    core = core_name(name)
    if attack == "SIG":
        m = re.fullmatch(r"SIG_([0-9.]+)_delta=([0-9.]+)_f=([0-9.]+)_mode=([^_]+)", core)
        if not m:
            raise ValueError(f"cannot parse SIG folder {name}")
        rate, delta, f_value, mode = m.groups()
        return rate, ["-f", f_value, "-delta", delta, "-label_mode", mode]

    if attack == "WaNet":
        m = re.fullmatch(r"WaNet_([0-9.]+)_cover=([0-9.]+)_s=([0-9.]+)_k=([0-9]+)", core)
        if not m:
            raise ValueError(f"cannot parse WaNet folder {name}")
        rate, cover, s_value, k_value = m.groups()
        return rate, ["-cover_rate", cover, "-s", s_value, "-k", k_value]

    if attack == "adaptive_blend":
        m = re.fullmatch(r"adaptive_blend_([0-9.]+)_alpha=([0-9.]+)_cover=([0-9.]+)_trigger=(.+)", core)
        if not m:
            raise ValueError(f"cannot parse adaptive_blend folder {name}")
        rate, alpha, cover, trigger = m.groups()
        return rate, ["-cover_rate", cover, "-alpha", alpha, "-trigger", trigger]

    if attack == "adaptive_patch":
        m = re.fullmatch(r"adaptive_patch_([0-9.]+)_alpha=([0-9.]+)_cover=([0-9.]+)", core)
        if not m:
            raise ValueError(f"cannot parse adaptive_patch folder {name}")
        rate, alpha, cover = m.groups()
        return rate, ["-cover_rate", cover, "-alpha", alpha]

    if attack == "belt":
        m = re.fullmatch(r"belt_([0-9.]+)_alpha=([0-9.]+)_cover=([0-9.]+)_mask=([0-9.]+)", core)
        if not m:
            raise ValueError(f"cannot parse belt folder {name}")
        rate, alpha, cover, mask = m.groups()
        return rate, ["-cover_rate", cover, "-mask_rate", mask, "-alpha", alpha]

    if attack == "blend":
        m = re.fullmatch(r"blend_([0-9.]+)_alpha=([0-9.]+)_trigger=(.+)", core)
        if not m:
            raise ValueError(f"cannot parse blend folder {name}")
        rate, alpha, trigger = m.groups()
        return rate, ["-alpha", alpha, "-trigger", trigger]

    raise ValueError(f"unsupported attack {attack}")


def q(value):
    return shlex.quote(str(value))


commands = []
for root in roots:
    dataset_root = Path(root) / "tiny_imagenet"
    if not dataset_root.exists():
        continue
    allowed = allowed_attacks.get(root, set())
    for folder in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        attack = attack_type(folder.name)
        if attack not in allowed:
            continue
        if list(folder.glob("test_tiny_target_domain_qwen_results*.txt")):
            continue
        model = arch_to_model(folder.name)
        rate, extra_args = parse_args(folder.name, attack)
        env = "POISONED_TRAIN_SET_ROOT=" + q(root)
        args = [
            "python",
            "test_tiny_target_domain_qwen.py",
            "-dataset=tiny_imagenet",
            f"-model={model}",
            f"-devices={devices}",
            "-source_dataset=tiny_imagenet",
            f"-poison_type={attack}",
            f"-poison_rate={rate}",
        ]
        args.extend(extra_args)
        args.append("-target_domain_dir=" + target_domain)
        command = " ".join([env] + [q(part) for part in args])
        description = f"{root} tiny_imagenet {model} {attack} {folder.name}"
        commands.append((description, command))

for description, command in commands:
    print(f"{description}\t{command}")
PY
)

echo
echo "============================================================"
echo "Qwen backfill finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
