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
  --new-root poisoned_train_set \
  --baseline-root poisoned_train_set1 \
  --output-csv analysis/data_sig_upgd_alltoone_raw.csv

python analysis/alltoone/validate_sig_upgd_alltoone.py \
  --input-csv analysis/data_sig_upgd_alltoone_raw.csv \
  --output-json analysis/validation_sig_upgd_alltoone.json

python analysis/alltoone/compare_sig_upgd_alltoone.py \
  --input-csv analysis/data_sig_upgd_alltoone_raw.csv \
  --output-dir analysis/report_tables \
  --prefix sig_upgd_alltoone

python analysis/alltoone/build_sig_upgd_alltoone_report.py \
  --validation-json analysis/validation_sig_upgd_alltoone.json \
  --pairwise-csv analysis/report_tables/sig_upgd_alltoone_pairwise_comparison.csv \
  --group-csv analysis/report_tables/sig_upgd_alltoone_group_summary.csv \
  --unmatched-csv analysis/report_tables/sig_upgd_alltoone_unmatched_cases.csv \
  --output-md analysis/sig_upgd_alltoone_comparison_report.md

echo "SIG/UPGD all-to-one 对比分析流程已完成。"
