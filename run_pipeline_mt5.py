# run_pipeline.py

import sys
import os
import torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel

# 将项目根目录添加到 Python 路径，确保能够找到所有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入所有必要的模块
from asr.transcribe import transcribe
from preprocess.remove_fillers import clean_fillers
from classifier.inference import load_classifier_model, predict_message_type
from summarizer.mt5_summarize import load_models as load_summarizer_models, summarize_with_mt5

def run_pipeline(audio_path: str, classifier_models, summarizer_models):
    """
    运行完整的端到端管道，从音频转写到摘要生成。
    """
    classifier_model, classifier_tokenizer = classifier_models
    summarizer_model, summarizer_tokenizer = summarizer_models

    print("--- 1. 语音转写 ---")
    raw_text, _ = transcribe(audio_path)
    print("原始转写:", raw_text)

    print("\n--- 2. 口语化处理 ---")
    clean_text = clean_fillers(raw_text)
    print("清理后文本:", clean_text)

    print("\n--- 3. 消息类型分类 ---")
    predicted_msg_type = predict_message_type(classifier_model, classifier_tokenizer, clean_text)
    print("预测消息类型:", predicted_msg_type)

    print("\n--- 4. 摘要生成 ---")
    summary = summarize_with_mt5(
        summarizer_model,
        summarizer_tokenizer,
        clean_text,
        predicted_msg_type
    )
    print("最终摘要:", summary)

    return {
        "transcript": raw_text,
        "transcript_clean": clean_text,
        "predicted_type": predicted_msg_type,
        "summary": summary
    }

if __name__ == "__main__":
    try:
        print("正在加载所有模型，这可能需要一些时间...")
        classifier_models = load_classifier_model()
        summarizer_models = load_summarizer_models()
        audio_file_path = "generated_audios/6.mp3"
        print("模型加载完成。")
        run_pipeline(audio_file_path, classifier_models, summarizer_models)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保所有模型文件和音频文件都已正确保存。")