#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python run/run_syn_svhn_pilot.py --attack adaptive_patch "$@"
