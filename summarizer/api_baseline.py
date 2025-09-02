## 调用 OpenAI API 按行生成摘要，支持 --limit 参数。
import csv
import json
import argparse
from openai import OpenAI

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='REDACTED_OPENAI_KEY',
)

def build_prompt(row):
    msg_type = row["msg_type"]
    emotion = row["emotions"]
    text = row["transcript_clean"]
    emphasis_words = row.get("emphasized_words", "[]")

    if msg_type == "task":
        return f"""
你是一名智能助理，负责从一段语音转写文本中提取关键信息并生成结构化摘要。

输入信息：
- 类型：任务
- 情感：{emotion}
- 重点词：{emphasis_words}
- 文本：“{text}”

请输出 JSON 格式摘要，包含以下字段：
{{
  "summary": "用简洁书面语总结任务",
  "key_actions": ["主要行动1","主要行动2"],
  "due_time": "YYYY-MM-DD HH:MM 或 null",
  "msg_type": "task",
  "emotion": "{emotion}",
  "emphasis_kept": ["保留的重音词"],
  "quality_flags": []
}}
仅返回 JSON，不要额外解释。
"""
    elif msg_type == "notice":
        return f"""
你是一名智能助理，负责从一段通知类语音文本中提取主要信息并生成摘要。

输入信息：
- 类型：通知
- 情感：{emotion}
- 重点词：{emphasis_words}
- 文本：“{text}”

输出 JSON：
{{
  "summary": "一句简洁书面语的通知",
  "key_points": ["要点1","要点2"],
  "effective_time": "YYYY-MM-DD HH:MM 或 null",
  "msg_type": "notice",
  "emotion": "{emotion}",
  "emphasis_kept": ["保留的重音词"],
  "quality_flags": []
}}
仅返回 JSON，不要额外解释。
"""
    else:  # chitchat
        return f"""
你是一名智能助手，需要从一段朋友/同事/家人的闲聊语音文本中生成简短摘要。

输入信息：
- 类型：闲聊
- 情感：{emotion}
- 重点词：{emphasis_words}
- 文本：“{text}”

输出 JSON：
{{
  "summary": "一句简洁的闲聊内容总结，如果有询问语句需要把内容涵盖在闲聊内容",
  "highlight": ["关键信息1","关键信息2"],
  "msg_type": "chitchat",
  "emotion": "{emotion}",
  "emphasis_kept": ["保留的重音词"],
  "quality_flags": []
}}
仅返回 JSON，不要额外解释。
"""

def call_model(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="CHI智能耳机_数据表.csv")
    parser.add_argument("--output", type=str, default="baseline_results.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
        print("检测到列名：", reader[0].keys())

    results = []
    for i, row in enumerate(reader[:args.limit]):
        print(f"\n=== 处理第 {i+1} 条（{row['msg_type']}）===")
        prompt = build_prompt(row)
        raw_output = call_model(prompt)

        print("模型原始返回：\n", raw_output)

        try:
            parsed = json.loads(raw_output)
            parsed["ID"] = row["ID"]
            results.append(parsed)
        except Exception as e:
            print(f"JSON 解析失败（第 {i+1} 条）：{e}")
            continue

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"已生成 {len(results)} 条摘要，保存在 {args.output}")

if __name__ == "__main__":
    main()