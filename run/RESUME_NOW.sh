#!/usr/bin/env bash
# Run this in a shell where `nvidia-smi` already works (e.g. your GPU screen).
#
# Default: only backfill missing non-BELT artifacts (no BELT rerun).
# To also run BELT later:
#   RUN_BELT=1 bash run/RESUME_NOW.sh
set -euo pipefail
cd /workspace/backdoor-toolbox-new1
echo "Checking GPU..."
nvidia-smi | head -20
echo
echo "Starting single-GPU resume (RUN_BELT=${RUN_BELT:-0}, skip existing, no deletes)..."
PARALLEL=0 GPU_IDS=0 \
  RUN_BELT="${RUN_BELT:-0}" \
  RUN_EXISTING_MISSING="${RUN_EXISTING_MISSING:-1}" \
  bash run/resume_set4_backfill.sh
