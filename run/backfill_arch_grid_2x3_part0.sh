#!/usr/bin/env bash
#
# Architecture 2x3 grid backfill — Script 1/4 (GPU0)
# CIFAR-10 / SmallCNN: 11 missing configs (~11h @ 1h/dir)
#
#   DEVICES=0 bash run/backfill_arch_grid_2x3_part0.sh
#
# Covers: SIG delta=28/36, WaNet s=0.6, belt pr=0.02, adaptive_patch alpha=0.2
#
# Note: adaptive_patch alpha=0.2 requires ResNet18 baseline rerun to finish first.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

PART_SLUG="part0_smallcnn"
PART_TITLE="Arch grid backfill 1/4 — CIFAR-10 SmallCNN (11 configs)"
DEVICES="${DEVICES:-0}"

DATASET="cifar10"
MODEL="small_cnn"
ARCH_NAME="SmallCNN"
TRANSFER_MODE="stl10"
RUN_CLEAN_PREP="${RUN_CLEAN_PREP:-1}"

CONFIGS=(
  "SIG|0.005|delta=28|-f 6 -delta 28 -label_mode clean"
  "SIG|0.005|delta=36|-f 6 -delta 36 -label_mode clean"
  "SIG|0.010|delta=28|-f 6 -delta 28 -label_mode clean"
  "SIG|0.010|delta=36|-f 6 -delta 36 -label_mode clean"
  "WaNet|0.005|s=0.6|-cover_rate 0.010 -s 0.6 -k 4"
  "WaNet|0.010|s=0.6|-cover_rate 0.020 -s 0.6 -k 4"
  "belt|0.020|alpha=0.10|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.10"
  "belt|0.020|alpha=0.20|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.20"
  "belt|0.020|alpha=0.30|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.30"
  "adaptive_patch|0.005|alpha=0.2|-cover_rate 0.010 -alpha 0.2"
  "adaptive_patch|0.010|alpha=0.2|-cover_rate 0.020 -alpha 0.2"
)

source "${SCRIPT_DIR}/_backfill_arch_grid_2x3_common.sh"
run_backfill_pipeline
