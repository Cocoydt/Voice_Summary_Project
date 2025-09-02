# Baseline Summary for Voice Project

## 1. 概览
本分支记录项目的 baseline 表现，基于转录后的语音文本生成摘要和情感标签。

- 输入：`data/CHI智能耳机_数据表.csv`（包含转录文本与参考摘要）
- 输出：`data/baseline_results_debug.jsonl` + `data/evaluation.json`

## 2. 文件说明
| 文件 | 描述 |
|------|------|
| `api_baseline.py` | baseline 生成摘要的主要脚本 |
| `api_baseline_debug.py` | 调试版 baseline，用于测试和 JSON 格式验证 |
| `evaluate_quality.py` | 计算 ROUGE 指标和情感准确率 |
| `data/CHI智能耳机_数据表.csv` | 输入 CSV，包含语音转录文本与参考摘要 |
| `data/baseline_results_debug.jsonl` | 模型生成的 JSON 摘要结果 |
| `data/evaluation.json` | baseline 质量评估结果 |

## 3. Baseline 指标
```json
{
  "samples_evaluated": 915,
  "avg_rouge1": 0.0598,
  "avg_rougeL": 0.0593,
  "emotion_accuracy": 0.9989
}