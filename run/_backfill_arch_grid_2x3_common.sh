#!/usr/bin/env bash
#
# Shared helpers for the 2x3 architecture-grid backfill (4-GPU split).
# Sourced by backfill_arch_grid_2x3_part{0,1,2,3}.sh — do not run directly.

: "${PYTHON_BIN:=python}"
: "${DRY_RUN:=0}"
: "${STOP_ON_FAIL:=0}"
: "${CLEAN_OLD:=1}"
: "${SKIP_EXISTING:=0}"
: "${RUN_CLEAN_PREP:=1}"
: "${POISONED_TRAIN_SET_ROOT:=poisoned_train_set4}"
: "${QWEN_TARGET_DOMAIN_DIR:=/workspace/data/tiny-target-domain-qwen-full-organized}"

export POISONED_TRAIN_SET_ROOT

DEFENSES=(
  "SentiNet"
  "STRIP"
  "ScaleUp"
  "IBD_PSC"
)

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/backfill_arch_grid_2x3_${PART_SLUG}_${TIMESTAMP}.log}"

run_command() {
  local cmd="$1"
  local description="$2"
  local tmp_out
  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")"

  echo
  echo ">>> ${description}"
  echo "${cmd}"

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
      echo "command: ${cmd}"
      echo "description: ${description}"
      echo "--- stdout/stderr ---"
      cat "$tmp_out" 2>/dev/null
      echo "---"
    } >> "$ERROR_LOG"

    if [ "$STOP_ON_FAIL" = "1" ]; then
      rm -f "$tmp_out"
      exit "$exit_code"
    fi
  fi

  rm -f "$tmp_out"
  return "$exit_code"
}

base_args() {
  echo "-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES}"
}

transfer_command() {
  local attack="$1"
  local rate="$2"
  local args="$3"

  if [ "$TRANSFER_MODE" = "imagenetv2" ]; then
    echo "${PYTHON_BIN} test_tiny_target_domain.py $(base_args) -source_dataset=${DATASET} -poison_type=${attack} -poison_rate=${rate} ${args}"
  elif [ "$TRANSFER_MODE" = "stl10" ]; then
    echo "${PYTHON_BIN} test_stl10.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args}"
  else
    echo "Unsupported TRANSFER_MODE: ${TRANSFER_MODE}" >&2
    return 1
  fi
}

qwen_command() {
  local attack="$1"
  local rate="$2"
  local args="$3"
  echo "${PYTHON_BIN} test_tiny_target_domain_qwen.py $(base_args) -source_dataset=${DATASET} -poison_type=${attack} -poison_rate=${rate} ${args} -target_domain_dir=${QWEN_TARGET_DOMAIN_DIR}"
}

config_result_complete() {
  local attack="$1"
  local rate="$2"
  local label="$3"
  local result_root="${POISONED_TRAIN_SET_ROOT}/${DATASET}"

  if [ ! -d "$result_root" ]; then
    return 1
  fi

  local pattern
  case "$attack" in
    belt)
      pattern="${attack}_${rate}_${label}_cover=0.500_mask=0.200_*_arch=${ARCH_NAME}_${DATASET}"
      ;;
    adaptive_blend)
      pattern="${attack}_${rate}_${label}_cover=${rate}_*_arch=${ARCH_NAME}_${DATASET}"
      ;;
    adaptive_patch|WaNet)
      local cover
      case "$rate" in
        0.005|0.001) cover="0.010" ;;
        0.01) cover="0.020" ;;
        0.02) cover="0.040" ;;
        *) return 1 ;;
      esac
      pattern="${attack}_${rate}_${label}_cover=${cover}_*_arch=${ARCH_NAME}_${DATASET}"
      ;;
    *)
      pattern="${attack}_${rate}_${label}_*_arch=${ARCH_NAME}_${DATASET}"
      ;;
  esac

  local dir
  for dir in "$result_root"/$pattern; do
    if [ -d "$dir" ] && [ -f "$dir/strip_defense_results.json" ]; then
      return 0
    fi
  done
  return 1
}

cleanup_unmatched_arch_dirs() {
  if [ "$CLEAN_OLD" != "1" ]; then
    echo "[CLEAN_OLD=0] skip unmatched-dir cleanup"
    return 0
  fi

  echo
  echo "----- Cleanup: remove off-grid dirs for ${ARCH_NAME} / ${DATASET} -----"

  "${PYTHON_BIN}" - <<'PY' "${POISONED_TRAIN_SET_ROOT}" "${DATASET}" "${ARCH_NAME}" "${DRY_RUN}"
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset = sys.argv[2]
arch = sys.argv[3]
dry_run = sys.argv[4] == "1"

TARGET = {
    ("cifar10", "SmallCNN"): {
        "basic": {"prs": [0.005, 0.01], "sn": "alpha", "strengths": [0.2, 0.5, 1.0]},
        "blend": {"prs": [0.005, 0.01], "sn": "alpha", "strengths": [0.05, 0.15, 0.30]},
        "adaptive_blend": {"prs": [0.005, 0.01], "sn": "alpha", "strengths": [0.05, 0.15, 0.25]},
        "adaptive_patch": {"prs": [0.005, 0.01], "sn": "alpha", "strengths": [0.1, 0.2, 0.3]},
        "SIG": {"prs": [0.005, 0.01], "sn": "delta", "strengths": [20, 28, 36]},
        "WaNet": {"prs": [0.005, 0.01], "sn": "s", "strengths": [0.4, 0.6, 0.8]},
        "belt": {"prs": [0.01, 0.02], "sn": "alpha", "strengths": [0.1, 0.2, 0.3]},
        "upgd": {"prs": [0.005, 0.01], "sn": "eps", "strengths": [4, 8, 12]},
    },
    ("tiny_imagenet", "ResNet34"): {
        "basic": {"prs": [0.005, 0.01], "sn": "alpha", "strengths": [0.2, 0.5, 1.0]},
        "blend": {"prs": [0.005, 0.01], "sn": "alpha", "strengths": [0.05, 0.15, 0.30]},
        "adaptive_blend": {"prs": [0.005, 0.01], "sn": "alpha", "strengths": [0.05, 0.15, 0.25]},
        "adaptive_patch": {"prs": [0.005, 0.01], "sn": "alpha", "strengths": [0.1, 0.2, 0.3]},
        "SIG": {"prs": [0.001, 0.005], "sn": "delta", "strengths": [20, 28, 36]},
        "WaNet": {"prs": [0.005, 0.01], "sn": "s", "strengths": [0.4, 0.6, 0.8]},
        "belt": {"prs": [0.01, 0.02], "sn": "alpha", "strengths": [0.1, 0.2, 0.3]},
        "upgd": {"prs": [0.001, 0.005], "sn": "eps", "strengths": [4, 8, 12]},
    },
}

sys.path.insert(0, "/workspace/backdoor-toolbox-new1/analysis-transfer-asr2/paper_analysis")
from parsing_utils import parse_folder_name

def cover_for(atk, pr):
    if atk in ("adaptive_patch", "WaNet"):
        return 0.01 if abs(pr - 0.005) < 1e-9 or abs(pr - 0.001) < 1e-9 else 0.02
    if atk == "adaptive_blend":
        return pr
    if atk == "belt":
        return 0.5
    return None

def in_target(p):
    spec_all = TARGET.get((dataset, arch))
    if not spec_all:
        return False
    atk = p["attack_type"]
    if atk not in spec_all:
        return False
    spec = spec_all[atk]
    if not any(abs(p["poison_rate"] - pr) < 1e-9 for pr in spec["prs"]):
        return False
    if p["strength_name"] != spec["sn"]:
        return False
    if not any(abs(p["strength_value"] - sv) < 1e-6 for sv in spec["strengths"]):
        return False
    cr = cover_for(atk, p["poison_rate"])
    if cr is not None:
        if abs(p.get("cover_rate", float("nan")) - cr) > 1e-9:
            return False
    if atk == "belt" and abs(p.get("mask_rate", float("nan")) - 0.2) > 1e-9:
        return False
    return True

base = root / dataset
if not base.is_dir():
    print(f"[cleanup] missing dataset root: {base}")
    raise SystemExit(0)

deleted = []
kept = []
for path in sorted(base.iterdir()):
    if not path.is_dir():
        continue
    parsed = parse_folder_name(path.name, dataset)
    if parsed["arch_base"] != arch:
        continue
    if parsed["attack_type"] == "none":
        kept.append(path.name)
        continue
    if in_target(parsed):
        kept.append(path.name)
        continue
    deleted.append(path)
    tag = "[DRY_RUN]" if dry_run else "[DELETE]"
    print(f"{tag} {path.name}")
    if not dry_run:
        import shutil
        shutil.rmtree(path)

print(f"[cleanup] kept={len(kept)} delete={len(deleted)}")
PY
}

cleanup_config_dirs() {
  local attack="$1"
  local rate="$2"
  local label="$3"
  local result_root="${POISONED_TRAIN_SET_ROOT}/${DATASET}"
  local patterns=()

  case "$attack" in
    belt)
      patterns+=("${attack}_${rate}_${label}_cover=0.500_mask=0.200_*_arch=${ARCH_NAME}_${DATASET}")
      ;;
    adaptive_blend)
      patterns+=("${attack}_${rate}_${label}_cover=${rate}_*_arch=${ARCH_NAME}_${DATASET}")
      ;;
    adaptive_patch|WaNet)
      local cover
      case "$rate" in
        0.005|0.001) cover="0.010" ;;
        0.01) cover="0.020" ;;
        0.02) cover="0.040" ;;
        *) return 0 ;;
      esac
      patterns+=("${attack}_${rate}_${label}_cover=${cover}_*_arch=${ARCH_NAME}_${DATASET}")
      patterns+=("${attack}_${rate}_alpha=0.200_cover=${cover}_*_arch=${ARCH_NAME}_${DATASET}")
      ;;
    *)
      patterns+=("${attack}_${rate}_${label}_*_arch=${ARCH_NAME}_${DATASET}")
      ;;
  esac

  local pattern path
  for pattern in "${patterns[@]}"; do
    for path in "$result_root"/$pattern; do
      if [ -d "$path" ]; then
        if [ "$DRY_RUN" = "1" ]; then
          echo "[DRY_RUN] would delete ${path}"
        else
          echo "[DELETE] ${path}"
          rm -rf "$path"
        fi
      fi
    done
  done
}

run_backfill_pipeline() {
  echo "============================================================"
  echo "${PART_TITLE}"
  echo "============================================================"
  echo "python       : ${PYTHON_BIN}"
  echo "dataset      : ${DATASET}"
  echo "model        : ${MODEL}"
  echo "arch name    : ${ARCH_NAME}"
  echo "devices      : ${DEVICES}"
  echo "result root  : ${POISONED_TRAIN_SET_ROOT}"
  echo "configs      : ${#CONFIGS[@]} backfill configs"
  echo "dry run      : ${DRY_RUN}"
  echo "clean old    : ${CLEAN_OLD}"
  echo "skip existing: ${SKIP_EXISTING}"
  echo "error log    : ${ERROR_LOG}"
  echo "============================================================"

  cleanup_unmatched_arch_dirs

  if [ "$RUN_CLEAN_PREP" = "1" ]; then
    echo
    echo "----- clean model preparation -----"
    run_command \
      "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
      "Create clean set/model dir"
    run_command \
      "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
      "Train clean model"
  fi

  local phases
  if [ "$TRANSFER_MODE" = "imagenetv2" ]; then
    phases=(create train source imagenetv2 qwen defenses)
  else
    phases=(create train source transfer defenses)
  fi

  for phase in "${phases[@]}"; do
    echo
    echo "----- ${phase} -----"
    for item in "${CONFIGS[@]}"; do
      IFS="|" read -r attack rate label args <<< "$item"

      if [ "$SKIP_EXISTING" = "1" ] && config_result_complete "$attack" "$rate" "$label"; then
        echo "[SKIP_EXISTING] ${attack} rate=${rate} ${label}"
        continue
      fi

      if [ "$phase" = "create" ] && [ "$CLEAN_OLD" = "1" ]; then
        cleanup_config_dirs "$attack" "$rate" "$label"
      fi

      case "$phase" in
        create)
          run_command \
            "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
            "Create: ${attack}, rate=${rate}, ${label}"
          ;;
        train)
          run_command \
            "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
            "Train: ${attack}, rate=${rate}, ${label}"
          ;;
        source)
          run_command \
            "${PYTHON_BIN} test_model.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
            "Source test: ${attack}, rate=${rate}, ${label}"
          ;;
        transfer)
          run_command \
            "$(transfer_command "$attack" "$rate" "$args")" \
            "Transfer: ${attack}, rate=${rate}, ${label}"
          ;;
        imagenetv2)
          run_command \
            "$(transfer_command "$attack" "$rate" "$args")" \
            "ImageNetV2-tiny transfer: ${attack}, rate=${rate}, ${label}"
          ;;
        qwen)
          run_command \
            "$(qwen_command "$attack" "$rate" "$args")" \
            "Qwen transfer: ${attack}, rate=${rate}, ${label}"
          ;;
        defenses)
          local defense
          for defense in "${DEFENSES[@]}"; do
            run_command \
              "${PYTHON_BIN} other_defense.py $(base_args) -defense=${defense} -poison_type=${attack} -poison_rate=${rate} ${args}" \
              "Defense ${defense}: ${attack}, rate=${rate}, ${label}"
          done
          ;;
      esac
    done
  done

  echo
  echo "============================================================"
  echo "${PART_TITLE} finished."
  echo "Check ${ERROR_LOG} for failures."
  echo "============================================================"
}
