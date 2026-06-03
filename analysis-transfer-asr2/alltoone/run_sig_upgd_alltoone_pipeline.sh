#!/usr/bin/env bash
set -euo pipefail

# 可复现执行清单：
# 1) 双根目录提取
# 2) 提取结果校验
# 3) 对比统计（pairwise/group/unmatched）
# 4) 构建 Markdown 专项报告

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python analysis/alltoone/extract_sig_upgd_alltoone.py \
  --new-root poisoned_train_set2 \
  --baseline-root poisoned_train_set1 \
  --output-csv analysis/alltoone/data_sig_upgd_alltoone_raw.csv

python analysis/alltoone/validate_sig_upgd_alltoone.py \
  --input-csv analysis/alltoone/data_sig_upgd_alltoone_raw.csv \
  --output-json analysis/alltoone/validation_sig_upgd_alltoone.json

python analysis/alltoone/compare_sig_upgd_alltoone.py \
  --input-csv analysis/alltoone/data_sig_upgd_alltoone_raw.csv \
  --output-dir analysis/alltoone/report_tables \
  --prefix sig_upgd_alltoone

python analysis/alltoone/analyze_mode_effects.py \
  --project-root . \
  --output-dir analysis/alltoone

python analysis/alltoone/build_sig_upgd_alltoone_report.py \
  --mode-summary-csv analysis/alltoone/mode_transfer_stealth_summary.csv \
  --mode-delta-csv analysis/alltoone/mode_transfer_stealth_delta.csv \
  --defense-summary-csv analysis/alltoone/mode_defense_dataset_summary.csv \
  --output-md analysis/alltoone/sig_upgd_alltoone_comparison_report.md

echo "SIG/UPGD all-to-one 对比分析流程已完成。"
