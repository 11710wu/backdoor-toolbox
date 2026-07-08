#!/usr/bin/env bash
#
# Architecture 2x3 grid backfill — Script 4/4 (GPU3)
# Tiny-ImageNet / ResNet34: 9 configs (~27h @ 3h/dir)
#
#   DEVICES=3 bash run/backfill_arch_grid_2x3_part3.sh
#
# Covers: WaNet s=0.4/0.6/0.8 @ pr=0.01, belt pr=0.01/0.02 (all 3 alphas)

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

PART_SLUG="part3_resnet34_wanet_belt"
PART_TITLE="Arch grid backfill 4/4 — Tiny-ImageNet ResNet34 WaNet/belt (9 configs)"
DEVICES="${DEVICES:-3}"

DATASET="tiny_imagenet"
MODEL="resnet34"
ARCH_NAME="ResNet34"
TRANSFER_MODE="imagenetv2"
RUN_CLEAN_PREP="${RUN_CLEAN_PREP:-0}"

CONFIGS=(
  "WaNet|0.010|s=0.4|-cover_rate 0.020 -s 0.4 -k 4"
  "WaNet|0.010|s=0.6|-cover_rate 0.020 -s 0.6 -k 4"
  "WaNet|0.010|s=0.8|-cover_rate 0.020 -s 0.8 -k 4"
  "belt|0.010|alpha=0.10|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.10"
  "belt|0.010|alpha=0.20|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.20"
  "belt|0.010|alpha=0.30|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.30"
  "belt|0.020|alpha=0.10|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.10"
  "belt|0.020|alpha=0.20|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.20"
  "belt|0.020|alpha=0.30|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.30"
)

source "${SCRIPT_DIR}/_backfill_arch_grid_2x3_common.sh"
run_backfill_pipeline
