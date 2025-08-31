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

def summarize_with_mt5(model, tokenizer, text: str, msg_type: str):
    """
    根据文本和消息类型生成摘要。
    """
    clean_text = clean_fillers(text)
    prompt = f"消息类型是【{msg_type}】。请将以下微信语音转录稿总结为一份简洁、重点突出的摘要。\n原文：{clean_text}\n摘要："
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
    return summary_str