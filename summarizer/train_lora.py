import sys
import os
import torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel

# 这行代码让 Python 能够找到你项目根目录下的其他文件夹
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入口语化处理函数，用于处理输入文本
from preprocess.remove_fillers import clean_fillers

# 定义模型路径和基础模型名称
LORA_OUT_PATH = "./summarizer/lora_out"
MODEL_BASE = "google/mt5-base"


# 这个函数只在需要时加载模型，并返回一个可用的模型和分词器
def load_models():
    """加载基础MT5模型和LoRA微调权重"""
    # 检查模型文件是否存在，如果不存在就报错
    if not os.path.exists(LORA_OUT_PATH):
        raise FileNotFoundError(f"模型文件未找到，请先运行 'train_lora.py' 来训练模型。")

    print("正在加载基础 MT5 模型...")
    base_model = MT5ForConditionalGeneration.from_pretrained(MODEL_BASE, torch_dtype=torch.float32)

    print("正在加载 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, LORA_OUT_PATH)

    print("正在加载分词器...")
    tokenizer = T5Tokenizer.from_pretrained(LORA_OUT_PATH)

    model.eval()  # 将模型设置为评估模式
    return model, tokenizer


# 这是你的核心摘要函数，它将由 run_pipeline.py 调用
def summarize_with_mt5(model, tokenizer, text: str, msg_type: str):
    """
    根据文本和消息类型生成摘要。

    Args:
        model: 训练好的PEFT模型。
        tokenizer: 对应的分词器。
        text: 需要生成摘要的文本。
        msg_type: 消息类型标签。

    Returns:
        生成的摘要文本。
    """
    # 清理文本中的口语化词语，确保输入是干净的
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

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


if __name__ == "__main__":
    # 这个代码块用于单独测试这个文件，不会被 run_pipeline.py 调用
    try:
        loaded_model, loaded_tokenizer = load_models()
        sample_text = "小李啊，你那个项目报告，嗯，下周一之前发给我。"
        sample_type = "task"
        summary = summarize_with_mt5(loaded_model, loaded_tokenizer, sample_text, sample_type)
        print(f"原文: {sample_text}")
        print(f"摘要: {summary}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 'train_lora.py' 来生成模型文件。")