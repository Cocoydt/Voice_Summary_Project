import json
import pandas as pd
from openai import OpenAI
import argparse

# 初始化 API 客户端
client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='REDACTED_OPENAI_KEY',
)

def build_prompt(row):
    """
    构建提示词：生成总摘要 + 分段摘要（含原文片段）
    """
    return f"""
你是一名智能助理，需要根据以下语音转写文本生成 **总摘要** 和 **分段摘要**。

输入信息：
- 类型：{row['msg_type']}
- 情感：{row['emotions']}
- 重点词：{row['emphasized_words']}
- 文本：{row['transcript_clean']}

请输出 JSON 格式，格式如下：
{{
  "overall_summary": "整段语音的简洁书面摘要",
  "segments": [
    {{
      "time_range": "0:00-0:20",
      "segment_summary": "分段摘要内容",
      "original_fragment": "对应的原文片段"
    }},
    {{
      "time_range": "0:20-0:40",
      "segment_summary": "分段摘要内容",
      "original_fragment": "对应的原文片段"
    }}
  ],
  "msg_type": "{row['msg_type']}",
  "emotion": "{row['emotions']}",
  "emphasis_kept": {row['emphasized_words']}
}}

规则：
1. 根据文本长度决定分 2 段或 3 段，每段需覆盖主要信息。
2. 原文片段请从输入文本中截取相关内容（不必精确时间对齐，模拟即可）。
3. 时间范围按顺序均分（如 0:00-0:20、0:20-0:40）。
4. 仅返回 JSON，不要任何解释文字。
"""

def call_model(prompt):
    """
    调用 OpenAI 模型
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API 调用错误: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/compare_text.csv") #"CHI智能耳机_数据表.csv"
    parser.add_argument("--output", type=str, default="data/baseline_segments_openai.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    print("检测到列名：", df.columns)

    results = []
    for idx, row in df.head(args.limit).iterrows():
        print(f"\n=== 处理第 {idx+1} 条（{row['msg_type']}）===")
        prompt = build_prompt(row)
        raw_output = call_model(prompt)
        print("模型原始返回：\n", raw_output)

        try:
            parsed = json.loads(raw_output)
            parsed["ID"] = row["ID"]
            results.append(parsed)
        except json.JSONDecodeError:
            print(f"JSON 解析失败（第 {idx+1} 条）")
            continue

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n已生成 {len(results)} 条摘要，保存在 {args.output}")

if __name__ == "__main__":
    main()