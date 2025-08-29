import sys
import os
import torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 假设 LoRA 权重保存在这个目录
LORA_OUT_PATH = "./summarizer/lora_out"
MODEL_BASE = "google/mt5-base"


def load_summarization_model():
    """
    加载基础 MT5 模型和 LoRA 微调权重。
    这个函数只需要在脚本开始时运行一次。
    """
    if not os.path.exists(LORA_OUT_PATH):
        raise FileNotFoundError(f"LoRA model not found at {LORA_OUT_PATH}. Please run train_lora.py first.")

    print("正在加载基础 MT5 模型...")
    base_model = MT5ForConditionalGeneration.from_pretrained(MODEL_BASE, torch_dtype=torch.float32)

    print("正在加载 LoRA 微调权重...")
    model = PeftModel.from_pretrained(base_model, LORA_OUT_PATH)

    print("正在加载分词器...")
    tokenizer = T5Tokenizer.from_pretrained(LORA_OUT_PATH)

    model.eval()  # 设置模型为评估模式
    return model, tokenizer


def summarize_with_mt5(model, tokenizer, text: str, msg_type: str):
    """
    根据文本和消息类型生成摘要。

    Args:
        model: 训练好的 PEFT 模型。
        tokenizer: 对应的分词器。
        text: 需要生成摘要的文本。
        msg_type: 消息类型标签。

    Returns:
        生成的摘要文本。
    """
    prompt = f"消息类型是【{msg_type}】。请将以下微信语音转录稿总结为一份简洁、重点突出的摘要，去除无用的口语化词语和重复信息。\n原文：{text}\n摘要："

    inputs = tokenizer(prompt, return_tensors="pt")

    # 将模型和输入数据移动到 GPU (如果可用)
    if torch.cuda.is_available():
        model = model.to("cuda")
        inputs = inputs.to("cuda")

    # 生成摘要
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


if __name__ == "__main__":
    # 这个 __main__ 块用于单独测试这个模块
    # 在实际的 run_pipeline.py 中，你只会调用 summarize_with_mt5 函数
    try:
        model, tokenizer = load_summarization_model()
        sample_text = "小李，把项目报告下周一之前发给我。"
        sample_type = "task"
        summary = summarize_with_mt5(model, tokenizer, sample_text, sample_type)
        print(f"原文: {sample_text}")
        print(f"摘要: {summary}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 'python train_lora.py' 来生成模型文件。")