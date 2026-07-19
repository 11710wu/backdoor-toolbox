#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif python -c "import torch" >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif [[ -x /root/anaconda3/envs/backtool/bin/python ]]; then
  PYTHON_BIN="/root/anaconda3/envs/backtool/bin/python"
else
  echo "Cannot find a Python interpreter with PyTorch; set PYTHON_BIN explicitly." >&2
  exit 2
fi
DEVICE="${DEVICE:-cuda:0}"
SAMPLES="${SAMPLES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WORKERS="${WORKERS:-2}"
SCENARIOS="${SCENARIOS:-cifar_resnet18,cifar_microcnn,tiny_resnet18}"
ATTACKS="${ATTACKS:-basic,blend,adaptive_blend,adaptive_patch,wanet,sig,upgd}"
STRENGTHS="${STRENGTHS:-0,0.25,0.5,0.75,1.0,1.25,1.5}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis-transfer-asr2/paper_analysis_outputs/trigger_strength_response_e6}"

args=(
  analysis-transfer-asr2/paper_analysis/run_trigger_strength_response_e6.py
  --device "$DEVICE"
  --samples "$SAMPLES"
  --batch-size "$BATCH_SIZE"
  --workers "$WORKERS"
  --scenarios "$SCENARIOS"
  --attacks "$ATTACKS"
  --strengths "$STRENGTHS"
  --output-dir "$OUTPUT_DIR"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi
if [[ "${REPORT_ONLY:-0}" == "1" ]]; then
  args+=(--report-only)
fi

"$PYTHON_BIN" "${args[@]}"
