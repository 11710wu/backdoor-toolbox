#!/usr/bin/env bash
#
# Architecture 2x3 grid backfill — Script 3/4 (GPU2)
# Tiny-ImageNet / ResNet34: 9 configs (~27h @ 3h/dir)
#
#   DEVICES=2 bash run/backfill_arch_grid_2x3_part2.sh
#
# Covers: SIG delta=28/36, WaNet s=0.6, adaptive_patch (4 cells)
#
# Note: adaptive_patch alpha=0.2 requires ResNet18 baseline rerun to finish first.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

PART_SLUG="part2_resnet34_sig_wanet_patch"
PART_TITLE="Arch grid backfill 3/4 — Tiny-ImageNet ResNet34 SIG/WaNet/adaptive_patch (9 configs)"
DEVICES="${DEVICES:-2}"

DATASET="tiny_imagenet"
MODEL="resnet34"
ARCH_NAME="ResNet34"
TRANSFER_MODE="imagenetv2"
RUN_CLEAN_PREP="${RUN_CLEAN_PREP:-0}"

CONFIGS=(
  "SIG|0.001|delta=28|-f 6 -delta 28 -label_mode clean"
  "SIG|0.001|delta=36|-f 6 -delta 36 -label_mode clean"
  "SIG|0.005|delta=28|-f 6 -delta 28 -label_mode clean"
  "SIG|0.005|delta=36|-f 6 -delta 36 -label_mode clean"
  "WaNet|0.005|s=0.6|-cover_rate 0.010 -s 0.6 -k 4"
  "adaptive_patch|0.005|alpha=0.2|-cover_rate 0.010 -alpha 0.2"
  "adaptive_patch|0.010|alpha=0.1|-cover_rate 0.020 -alpha 0.1"
  "adaptive_patch|0.010|alpha=0.2|-cover_rate 0.020 -alpha 0.2"
  "adaptive_patch|0.010|alpha=0.3|-cover_rate 0.020 -alpha 0.3"
)

source "${SCRIPT_DIR}/_backfill_arch_grid_2x3_common.sh"
run_backfill_pipeline
