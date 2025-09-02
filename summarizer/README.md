# Baseline Summary for Voice Project

（更新版）
## 1. 概览
本分支记录项目的 baseline 表现，基于转录后的语音文本生成摘要和情感标签。

- 输入：`generated_audios`(音频）（包含转录文本与参考摘要）
- 输出：`data/baseline_results_debug.jsonl` + `data/evaluation.json`


### Workflow

音频文件 → 语音转写(Whisper) → 分类(msg_type) → 情感识别 → 定制化摘要 → 保存 JSONL → 可选质量评估

- 输入：音频文件（.mp3）
- 输出：JSONL 文件，每条包含字段：
  - `ID`：音频文件名
  - `summary`：摘要内容
  - `msg_type`：任务/通知/闲聊
  - `emotion`：情感标签
  - `emphasis_kept`：保留的重点词
  - `quality_flags`：质量标识（可扩展）

---
## 2. 文件说明
| 文件 | 描述                             |
|------|--------------------------------|
| `api_baseline.py` | baseline 生成摘要的主要脚本（文字转录稿）      |
| `api_baseline_debug.py` | 调试版 baseline，用于测试和 JSON 格式验证   |
| `workflow_baseline.py` | 端到端 baseline 脚本（生成 JSONL）-音频输入 |
| `generated_audios/` | 示例音频文件夹（.mp3 文件）               |
| `workflow_baseline_results_20.jsonl` | 输出的 20 条测试摘要结果                 |
| `data/baseline_results_debug.jsonl` | 模型生成的 JSON 摘要结果                |
| `data/evaluation.json` | baseline 质量评估结果                |
| `evaluate_quality.py` | 单独的质量评估脚本，可计算 ROUGE 和情感准确率     |
| `CHI智能耳机_数据表.csv` | 参考摘要，用于质量评估                    |

---

## 3. 生成摘要（JSONL）
运行前 20 条音频示例：
- 输入：音频文件（.mp3）
- 输出：JSONL 文件，每条包含字段：
```bash
python summarizer/workflow_baseline.py \
    --audio_folder generated_audios \
    --output workflow_baseline_results_20.jsonl \
    --limit 20
   ```
  - `ID`：音频文件名
  - `summary`：摘要内容
  - `msg_type`：任务/通知/闲聊
  - `emotion`：情感标签
  - `emphasis_kept`：保留的重点词
  - `quality_flags`：质量标识（可扩展）

---

4. 质量评估

使用独立脚本 evaluate_quality.py 进行评估：

python summarizer/evaluate_quality.py \
    --baseline workflow_baseline_results_20.jsonl \
    --reference CHI智能耳机_数据表.csv

	•	输出：
	•	samples_evaluated：评估样本数量
	•	avg_rouge1 / avg_rougeL：ROUGE 分数
	•	emotion_accuracy：情感匹配准确率

示例输出

{
  "samples_evaluated": 20,
  "avg_rouge1": 0.05,
  "avg_rougeL": 0.05,
  "emotion_accuracy": 0.05
}

⚠️ 当前为测试小样本（20 条音频），完整 baseline 建议使用全部音频（如 1000 条）生成 JSONL 再评估。

⸻

5. 使用建议
	1.	测试流程：先用少量音频验证脚本运行正确
	2.	生成完整 baseline：可扩展 --limit 或去掉限制
	3.	质量评估：独立运行 evaluate_quality.py，保证与参考摘要对齐
	4.	提交团队：JSONL + 评估报告可作为 baseline 数据共享给队友

⸻

6. 注意事项
	•	需有效 OpenAI API Key（Whisper + GPT 模型）
	•	输出 JSONL 可直接用于模型微调或进一步分析
	•	workflow_baseline.py 与 evaluate_quality.py 分离，确保灵活使用

---

说明：
- README 保留了**最新 20 条测试示例**  
- 明确区分了 **生成摘要** 和 **质量评估**  
- 提醒了小样本 vs 完整 baseline 的区别  

