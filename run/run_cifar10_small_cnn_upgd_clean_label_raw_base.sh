#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="cifar10"
export MODEL="small_cnn"
export LABEL_MODE="${LABEL_MODE:-clean}"
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"
export POISON_RATES="${POISON_RATES:-0.01 0.005}"
export EPS_VALUES="${EPS_VALUES:-4 8 12}"
export DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"

echo "output root: ${POISONED_TRAIN_SET_ROOT}"
echo "output dataset/model: ${POISONED_TRAIN_SET_ROOT}/${DATASET} (${MODEL})"
echo "output note: this wrapper delegates to run_upgd_clean_label_raw_base.sh; poisoned data, raw-base model, trained model, test/transfer files, and defense outputs are written below that root"

exec bash "$SCRIPT_DIR/run_upgd_clean_label_raw_base.sh" "$@"
