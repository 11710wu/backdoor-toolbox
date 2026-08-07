#!/usr/bin/env bash
# Wait for the current ResNet18 feature-cleanser job to finish, then backfill
# MobileNetV2 and VGG19_BN on poisoned_train_set / CIFAR-10.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

export PATH="/workspace/tools/julia-1.7.2/bin:${PATH:-}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

PYTHON="${PYTHON:-/root/anaconda3/envs/backtool/bin/python}"
DEVICES="${DEVICES:-0}"
SPECTRE_JOBS="${SPECTRE_JOBS:-4}"
LOG_DIR="${REPO}/logs/recreate_and_backfill"
mkdir -p "${LOG_DIR}"
CHAIN_LOG="${LOG_DIR}/chain_other_arches_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${CHAIN_LOG}"; }

resnet18_running() {
  pgrep -af 'recreate_and_backfill_cleanser.py' 2>/dev/null | grep -E -- '--arch( |=)ResNet18|arch ResNet18' >/dev/null \
    || pgrep -af 'resume_resnet18_feature_cleansers\.sh' >/dev/null
}

run_arch() {
  local arch="$1"
  local stamp logf arch_tag
  stamp="$(date +%Y%m%d_%H%M%S)"
  arch_tag="$(echo "${arch}" | tr '[:upper:]' '[:lower:]')"
  logf="${LOG_DIR}/resume_${arch_tag}_feature_${stamp}.log"
  log "START arch=${arch} log=${logf}"
  set +e
  "${PYTHON}" -u run/recreate_and_backfill_cleanser.py \
    --roots poisoned_train_set \
    --datasets cifar10 \
    --arch "${arch}" \
    --delete-data-after \
    --devices "${DEVICES}" \
    --spectre-jobs "${SPECTRE_JOBS}" \
    2>&1 | tee -a "${logf}"
  local rc=${PIPESTATUS[0]}
  set -e
  log "FINISH arch=${arch} exit=${rc}"
  return "${rc}"
}

log "chain started; waiting for ResNet18 job to finish"
while resnet18_running; do
  log "ResNet18 still running; sleep 120s"
  sleep 120
done
log "ResNet18 job not running"

# Sweep any leftover ResNet18 missing JSON first (e.g. SPECTRE-only leftovers).
log "sweep leftover ResNet18"
run_arch ResNet18 || log "ResNet18 sweep had failures (continuing)"

log "backfill MobileNetV2 (~25h est.)"
run_arch MobileNetV2 || log "MobileNetV2 finished with failures"

log "backfill VGG19_BN (~25h est.)"
run_arch VGG19_BN || log "VGG19_BN finished with failures"

log "chain complete"
