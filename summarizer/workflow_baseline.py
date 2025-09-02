import os
import re
import csv
import json
import argparse
from openai import OpenAI

# ---------------------------
# 初始化 OpenAI 客户端
# ---------------------------
client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='REDACTED_OPENAI_KEY'
)

# ---------------------------
# 工具函数：清理模型输出
# ---------------------------
def clean_model_output(raw_output: str) -> str:
    """去掉模型输出中的 Markdown 代码块包裹"""
    return re.sub(r"```(json)?\s*|\s*```", "", raw_output).strip()

# ---------------------------
# 语音转写
# ---------------------------
def transcribe_audio(file_path: str) -> str:
    """将音频转写为文本"""
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

# ---------------------------
# 分类（任务/通知/闲聊）
# ---------------------------
def classify_message(transcript: str) -> str:
    """简单分类，可替换为真实分类模型"""
    if "通知" in transcript or "注意" in transcript:
        return "notice"
    elif "提醒" in transcript or "需要" in transcript:
        return "task"
    else:
        return "chitchat"

# ---------------------------
# 情感识别
# ---------------------------
def detect_emotion(transcript: str) -> str:
    """情感识别，可替换为情感分析模型"""
    if any(word in transcript for word in ["开心", "高兴", "棒"]):
        return "积极"
    elif any(word in transcript for word in ["气", "怒", "服了"]):
        return "愤怒"
    else:
        return "中立"

# ---------------------------
# 构建摘要 Prompt
# ---------------------------
def build_prompt(msg_type: str, emotion: str, transcript: str) -> str:
    """根据分类与情感构建定制化摘要 Prompt"""
    if msg_type == "task":
        return f"""
你是一名智能助理，负责从一段语音转写文本中提取任务信息并生成摘要。

输入：
- 类型：任务
- 情感：{emotion}
- 文本：{transcript}

输出 JSON：
{{
  "summary": "简洁任务摘要",
  "key_actions": ["行动1", "行动2"],
  "due_time": "YYYY-MM-DD HH:MM 或 null",
  "msg_type": "task",
  "emotion": "{emotion}",
  "emphasis_kept": [],
  "quality_flags": []
}}
"""
    elif msg_type == "notice":
        return f"""
你是一名智能助理，负责从通知类语音文本中提取主要信息并生成摘要。

输入：
- 类型：通知
- 情感：{emotion}
- 文本：{transcript}

输出 JSON：
{{
  "summary": "一句简洁书面语的通知",
  "key_points": ["要点1", "要点2"],
  "effective_time": "YYYY-MM-DD HH:MM 或 null",
  "msg_type": "notice",
  "emotion": "{emotion}",
  "emphasis_kept": [],
  "quality_flags": []
}}
"""
    else:
        return f"""
你是一名智能助手，需要从一段闲聊语音文本中生成简短摘要。

输入：
- 类型：闲聊
- 情感：{emotion}
- 文本：{transcript}

输出 JSON：
{{
  "summary": "一句简洁的闲聊内容总结",
  "highlight": ["关键信息1", "关键信息2"],
  "msg_type": "chitchat",
  "emotion": "{emotion}",
  "emphasis_kept": [],
  "quality_flags": []
}}
"""

# ---------------------------
# 调用模型生成摘要
# ---------------------------
def call_model(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# ---------------------------
# 主流程
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_folder", type=str, default="audio_files", help="音频文件夹路径")
    parser.add_argument("--output", type=str, default="workflow_baseline_results.jsonl", help="输出文件")
    parser.add_argument("--limit", type=int, default=10, help="处理音频数量")
    args = parser.parse_args()

    audio_files = [f for f in os.listdir(args.audio_folder) if f.endswith(".mp3")]
    results = []

    for idx, audio_file in enumerate(audio_files[:args.limit]):
        audio_path = os.path.join(args.audio_folder, audio_file)
        print(f"\n=== 处理第 {idx+1} 条音频：{audio_path} ===")

        transcript = transcribe_audio(audio_path)
        msg_type = classify_message(transcript)
        emotion = detect_emotion(transcript)

        prompt = build_prompt(msg_type, emotion, transcript)
        raw_output = call_model(prompt)
        cleaned_output = clean_model_output(raw_output)

        try:
            parsed = json.loads(cleaned_output)
            parsed["ID"] = os.path.splitext(audio_file)[0]
            results.append(parsed)
            print(f"摘要内容：{parsed['summary']}")
        except json.JSONDecodeError:
            print("JSON 解析失败，跳过该条音频。")
            print("原始输出：", raw_output)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n已保存 {len(results)} 条结果到 {args.output}")

if __name__ == "__main__":
    main()