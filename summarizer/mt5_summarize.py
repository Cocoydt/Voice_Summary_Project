# summarizer/mt5_summarize.py

import sys
import os
import torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.remove_fillers import clean_fillers

# 定义模型路径和基础模型名称
MODEL_BASE = "google/mt5-small"
LORA_OUT_PATH = "./summarizer/lora_out"

def load_models():
    """加载基础MT5模型和LoRA微调权重"""
    if not os.path.exists(LORA_OUT_PATH):
        raise FileNotFoundError(f"模型文件未找到，请先运行 'train_lora.py' 来训练模型。")
    print("正在加载基础 MT5 模型...")
    base_model = MT5ForConditionalGeneration.from_pretrained(MODEL_BASE, torch_dtype=torch.float32)
    print("正在加载 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, LORA_OUT_PATH)
    print("正在加载分词器...")
    tokenizer = T5Tokenizer.from_pretrained(LORA_OUT_PATH)
    model.eval()
    return model, tokenizer

import json

def summarize_with_mt5(model, tokenizer, text: str, msg_type: str):
    clean_text = clean_fillers(text)

    # 根据消息类型选择不同的 Prompt 和输出示例
    if msg_type == "notice":
        prompt = (
            f"你是一名高效的语义分析助手。请将以下微信语音转录稿总结为一份结构化的JSON摘要，重点突出通知内容和要点。\n"
            f"请严格遵循以下JSON格式输出：{{\"type\": \"notice\", \"notice_content\": [\"...\"], \"bullets\": [\"...\"]}}\n"
            f"原文：{clean_text}\n"
            f"摘要：")
    elif msg_type == "task":
        prompt = (
            f"你是一名高效的语义分析助手。请从以下文本中提取主语、动词和名词。请将以下微信语音转录稿总结为一份结构化的JSON摘要，重点突出待办任务和关键信息（人物、时间）。\n"
            f"请严格遵循以下JSON格式输出：{{\"type\": \"task\", \"tasks\": [\"...\"], \"mentions\": [\"...\"]}}\n"
            f"原文：{clean_text}\n"
            f"摘要：")
    elif msg_type == "chitchat":
        prompt = (
            f"你是一名高效的语义分析助手。请从以下文本中提取主语、动词和名词。请将以下微信语音转录稿总结为一份结构化的JSON摘要，记录核心事件或情绪。\n"
            f"请严格遵循以下JSON格式输出：{{\"type\": \"chitchat\", \"event\": [\"...\"], \"emotion\": \"...\"}}\n"
            f"原文：{clean_text}\n"
            f"摘要：")
    else:
        prompt = (
            f"你是一名高效的语义分析助手。请从以下文本中提取主语、动词和名词。请将以下微信语音转录稿总结为一份结构化的JSON摘要。\n"
            f"请严格遵循以下JSON格式输出：{{\"type\": \"unknown\", \"summary\": \"...\"}}\n"
            f"原文：{clean_text}\n"
            f"摘要：")

    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        model = model.to("cuda")
        inputs = inputs.to("cuda")

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    summary_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        summary_json = json.loads(summary_str)
        return summary_json
    except json.JSONDecodeError:
        print("警告: 无法解析 JSON 格式的摘要，返回原始文本。")
        return summary_str