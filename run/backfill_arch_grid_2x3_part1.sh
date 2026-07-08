#!/usr/bin/env bash
#
# Architecture 2x3 grid backfill — Script 2/4 (GPU1)
# Tiny-ImageNet / ResNet34: 9 configs (~27h @ 3h/dir)
#
#   DEVICES=1 bash run/backfill_arch_grid_2x3_part1.sh
#
# Covers: basic/blend/adaptive_blend @ pr=0.01 (all 3 strengths each)

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

PART_SLUG="part1_resnet34_blend"
PART_TITLE="Arch grid backfill 2/4 — Tiny-ImageNet ResNet34 basic/blend/adaptive_blend (9 configs)"
DEVICES="${DEVICES:-1}"

DATASET="tiny_imagenet"
MODEL="resnet34"
ARCH_NAME="ResNet34"
TRANSFER_MODE="imagenetv2"
RUN_CLEAN_PREP="${RUN_CLEAN_PREP:-1}"

CONFIGS=(
  "basic|0.010|alpha=0.2|-alpha 0.2"
  "basic|0.010|alpha=0.5|-alpha 0.5"
  "basic|0.010|alpha=1.0|-alpha 1.0"
  "blend|0.010|alpha=0.05|-alpha 0.05"
  "blend|0.010|alpha=0.15|-alpha 0.15"
  "blend|0.010|alpha=0.30|-alpha 0.30"
  "adaptive_blend|0.010|alpha=0.05|-cover_rate 0.010 -alpha 0.05"
  "adaptive_blend|0.010|alpha=0.15|-cover_rate 0.010 -alpha 0.15"
  "adaptive_blend|0.010|alpha=0.25|-cover_rate 0.010 -alpha 0.25"
)

source "${SCRIPT_DIR}/_backfill_arch_grid_2x3_common.sh"
run_backfill_pipeline
