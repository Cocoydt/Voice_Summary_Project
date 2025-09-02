# summarizer/api_baseline_qwen.py
import json
import csv
import requests

# === 配置 ===
INPUT_CSV = "CHI智能耳机_数据表.csv"  # 1000 条数据
OUTPUT_JSONL = "baseline_results_qwen.jsonl"
MODEL_NAME = "qwen-max"  # 可选：qwen-max, qwen-plus
CURRENT_DATE = "2025-09-01"
API_KEY = "YOUR_QWEN_API_KEY"  # need to be revised
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

PROMPTS = {
    "task": """你是一名智能助理，负责从一段语音转写文本中提取关键信息并生成结构化摘要。

输入信息：
- 类型：任务
- 情感：{emotion}
- 重点词：{emphasis_words}
- 文本：{text}

输出 JSON：
{{
  "summary": "用简洁书面语总结任务",
  "key_actions": ["主要行动1","主要行动2"],
  "due_time": "YYYY-MM-DD HH:MM 或 原文（如 '下周末'）或 null",
  "urgency": "high|medium|low",
  "need_confirmation": true/false,
  "msg_type": "task",
  "emotion": "{emotion}",
  "emphasis_kept": {emphasis_words},
  "quality_flags": []
}}

规则：
1. 标准时间 → YYYY-MM-DD HH:MM；模糊时间 → 保留原文；没有时间 → null。
2. 紧急程度：情感为“愤怒/严肃” → high；“害怕/迟疑” → need_confirmation=true。
3. 仅返回 JSON。
""",
    "notice": """你是一名智能助理，负责从一段通知类语音文本中提取主要信息并生成摘要。

输入信息：
- 类型：通知
- 情感：{emotion}
- 重点词：{emphasis_words}
- 文本：{text}

输出 JSON：
{{
  "summary": "一句简洁书面语的通知",
  "key_points": ["要点1","要点2"],
  "effective_time": "YYYY-MM-DD HH:MM 或 原文 或 null",
  "msg_type": "notice",
  "emotion": "{emotion}",
  "emphasis_kept": {emphasis_words},
  "quality_flags": []
}}

规则：
1. 主要提取信息要点，不强加行动。
2. 包含生效时间则填写，没有则 null。
3. 仅返回 JSON。
""",
    "chitchat": """你是一名智能助手，需要从一段朋友/同事/家人的闲聊语音文本中生成简短摘要。

输入信息：
- 类型：闲聊
- 情感：{emotion}
- 重点词：{emphasis_words}
- 文本：{text}

输出 JSON：
{{
  "summary": "一句简洁的闲聊内容总结，如果有询问语句需要包含",
  "highlight": ["关键信息1","关键信息2"],
  "sentiment_summary": "表达了什么情绪/观点",
  "msg_type": "chitchat",
  "emotion": "{emotion}",
  "emphasis_kept": {emphasis_words},
  "quality_flags": []
}}

规则：
1. 强调情感与观点，时间信息如有则包含。
2. 不做任务拆解。
3. 仅返回 JSON。
"""
}

def call_qwen(prompt: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    resp = requests.post(API_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    return json.loads(text)

def main():
    results = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= 10:  # 仅处理前 10 条
                break

            msg_type = row["msg_type"]
            emotion = row.get("emotion", "neutral")
            text = row["transcript_clean"]
            emphasis_words = row["emphasized_words"].split(",") if row.get("emphasized_words") else []

            prompt = PROMPTS[msg_type].format(
                emotion=emotion,
                emphasis_words=emphasis_words,
                text=text
            )

            try:
                summary_json = call_qwen(prompt)
                summary_json["id"] = row["id"]
                summary_json["prosody_version"] = 1
                summary_json["emphasis_timestamps"] = []  # 未来可填充
                results.append(summary_json)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for item in results:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"已生成 {len(results)} 条摘要，保存在 {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()