#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

output_root="analysis-transfer-asr2/paper_analysis_outputs/syn_svhn_full_cifar_grid"
launcher_logs="$output_root/launcher_logs"
mkdir -p "$launcher_logs"

exec 9>"$output_root/launcher.lock"
if ! flock -n 9; then
    echo "Another four-GPU SYN full-grid launcher is already active." >&2
    exit 1
fi

if pgrep -f '[r]un/run_syn_svhn_pilot.py' >/dev/null; then
    echo "An existing SYN pilot is running and may write the same result directories." >&2
    echo "Wait for it to finish or stop it explicitly before launching the full grid." >&2
    exit 1
fi
if pgrep -f '[r]un/run_syn_svhn_full.py' >/dev/null; then
    echo "One or more SYN full-grid workers are already active." >&2
    exit 1
fi

for device in 4 5 6 7; do
    if ! nvidia-smi -i "$device" --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
        echo "GPU $device is not available." >&2
        exit 1
    fi
done

if [[ ! -f clean_set/syn/clean_split/clean_labels || ! -f clean_set/syn/test_split/labels ]]; then
    python create_clean_set.py -dataset syn
fi

python run/run_syn_svhn_full.py --device 4 \
    --attacks basic blend >"$launcher_logs/gpu4_basic_blend.log" 2>&1 &
pid4=$!
python run/run_syn_svhn_full.py --device 5 \
    --attacks adaptive_blend adaptive_patch >"$launcher_logs/gpu5_adaptive.log" 2>&1 &
pid5=$!
python run/run_syn_svhn_full.py --device 6 \
    --attacks wanet sig >"$launcher_logs/gpu6_wanet_sig.log" 2>&1 &
pid6=$!
python run/run_syn_svhn_full.py --device 7 --prepare-clean \
    --attacks upgd belt >"$launcher_logs/gpu7_upgd_belt.log" 2>&1 &
pid7=$!

status=0
for pid in "$pid4" "$pid5" "$pid6" "$pid7"; do
    if ! wait "$pid"; then
        status=1
    fi
done

python analysis-transfer-asr2/paper_analysis/syn_svhn_full/collect_syn_svhn_full.py
exit "$status"
