import csv
import json
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='REDACTED_OPENAI_KEY',  # 你的 API key
)

MODEL_NAME = "gpt-4o-mini"

def build_prompt(row):
    """根据 msg_type 生成不同的 prompt"""
    msg_type = row["msg_type"]
    emotion = row.get("emotions", "中立")
    text = row.get("transcript_clean", "")
    emphasis_words = row.get("emphasized_words", "")

    if msg_type == "task":
        return f"""
你是一名智能助理，负责从一段语音转写文本中提取关键信息并生成结构化摘要。

输入信息：
- 类型：任务
- 情感：{emotion}
- 重点词：{emphasis_words}
- 文本：{text}

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
- 文本：{text}

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
你是一名智能助手，需要从一段闲聊语音文本中生成简短摘要。

输入信息：
- 类型：闲聊
- 情感：{emotion}
- 重点词：{emphasis_words}
- 文本：{text}

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

def main():
    input_csv = "CHI智能耳机_数据表.csv"  # 你的数据文件
    output_file = "baseline_results_debug.jsonl"

    results = []
    with open(input_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        print(f"检测到列名：{reader.fieldnames}")

        for idx, row in enumerate(reader):
            print(f"\n=== 正在处理第 {idx+1} 条（{row['msg_type']}）===")
            prompt = build_prompt(row)
            print(f"Prompt:\n{prompt}")

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "你是一个专业的语音摘要生成助手，只能输出 JSON。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                raw_output = response.choices[0].message.content
                print("模型原始返回：\n", raw_output)

                try:
                    parsed = json.loads(raw_output)
                    results.append({
                        "ID": row["ID"],
                        "baseline_summary": parsed
                    })
                except json.JSONDecodeError as e:
                    print(f"JSON 解析失败（第 {idx+1} 条）：{e}")
                    print("原始文本：", raw_output)
            except Exception as e:
                print(f"调用 API 出错（第 {idx+1} 条）：{e}")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"已生成 {len(results)} 条摘要，保存在 {output_file}")

if __name__ == "__main__":
    main()