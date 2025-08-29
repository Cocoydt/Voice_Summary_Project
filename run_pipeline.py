# run_pipeline.py

# 导入所有必要的模块
from asr.transcribe import transcribe
from preprocess.remove_fillers import clean_fillers
# 注意：暂时不导入未完成的模块
# from classifier.inference import predict_msg_type
# from summarizer.mt5_summarize import summarize_with_mt5

# 假设你的摘要模型已经训练并保存在summarizer/lora_out目录
# 我们在这里直接导入并使用，而不是重新训练
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel


def load_models():
    """加载摘要模型和分词器"""
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    # 加载你的LoRA微调权重
    model = PeftModel.from_pretrained(model, "./summarizer/lora_out")
    tokenizer = T5Tokenizer.from_pretrained("./summarizer/lora_out")
    return model, tokenizer


def summarize_text(model, tokenizer, text, msg_type):
    """根据文本和消息类型生成摘要"""
    # 简化版Prompt，只包含消息类型
    prompt = f"消息类型是【{msg_type}】。请将以下微信语音转录稿总结为一份简洁、重点突出的摘要，去除无用的口语化词语和重复信息。\n原文：{text}\n摘要："

    inputs = tokenizer(prompt, return_tensors="pt")

    # 使用GPU加速
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


def run_pipeline(audio_path: str, model, tokenizer):
    print("--- 1. 语音转写 ---")
    raw_text, _ = transcribe(audio_path)
    print("原始转写:", raw_text)

    print("\n--- 2. 口语化处理 ---")
    clean_text = clean_fillers(raw_text)
    print("清理后文本:", clean_text)

    print("\n--- 3. 消息类型分类 ---")
    # 暂时使用硬编码的"task"，直到你完成分类器
    msg_type = "task"
    print("消息类型:", msg_type)

    print("\n--- 4. 摘要生成 ---")
    summary = summarize_text(model, tokenizer, clean_text, msg_type)
    print("最终摘要:", summary)

    return {
        "transcript": raw_text,
        "transcript_clean": clean_text,
        "summary": summary
    }


if __name__ == "__main__":
    # 加载模型和分词器，只需加载一次
    import torch

    model, tokenizer = load_models()

    # 确保你的data/raw_audio/sample.wav文件存在
    run_pipeline("data/raw_audio/sample.wav", model, tokenizer)