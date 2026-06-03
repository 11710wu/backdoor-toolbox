# analysis-transfer-asr2

在 `analysis-testASR` 相同流水线基础上，将**迁移性指标**改为：

\[
\text{transfer\_rate} = \frac{\text{ASRt}^2}{\text{ASRs}}
\]

| 字段 | 含义 |
|------|------|
| `transfer_asr` | 目标域 ASR（ASRt），从跨域测试结果 txt 解析 |
| `asr` | 源域 ASR（ASRs），从 `test_results_seed=*.json` |
| `transfer_rate` | ASRt² / ASRs（分析中使用的迁移性） |

隐蔽性、NC、`S_stealth` 等与 `analysis-testASR` 一致。

## 使用

```bash
cd /workspace/backdoor-toolbox-new1

# 1. 提取 CSV/JSON（输出到本目录）
python analysis-transfer-asr2/extract_all_results.py --mode all

# 2. 校验抽样
python analysis-transfer-asr2/validate_extraction.py --mode no_nc --sample-size 30

# 3. 统计图与报告（与 analysis-testASR 相同参数）
python analysis-transfer-asr2/analyze_stats.py \
  --data-dir analysis-transfer-asr2 --data-suffix _no_nc --all-plots \
  --output-dir analysis-transfer-asr2/outputs_no_nc

python analysis-transfer-asr2/build_non_nc_richer_report.py \
  --analysis-dir analysis-transfer-asr2

python analysis-transfer-asr2/generate_complete_tradeoff_report.py \
  --analysis-dir analysis-transfer-asr2
```

## 与 analysis-testASR 的差异

| 项目 | analysis-testASR | analysis-transfer-asr2 |
|------|------------------|------------------------|
| 迁移性 | 目标域 ASR（ASRt） | ASRt² / ASRs |
| CSV 列 | `transfer_rate`, `asr` | 增加 `transfer_asr`，`transfer_rate` 为派生指标 |
| 缺源域 ASR | 仍可有 transfer | 跳过该点（无法计算） |

## 目录

- `transfer_metric.py`：指标定义与计算
- `extract_all_results.py`：主提取脚本
- `validate_extraction.py`、`analyze_stats.py`、`build_*_report.py`：与 testASR 版对应
- `alltoone/`：SIG/UPGD 模式对比（已同步新指标）
