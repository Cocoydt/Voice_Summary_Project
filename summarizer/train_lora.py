# summarizer/train_lora.py

import sys
import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model
)

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.remove_fillers import clean_fillers

# 定义模型路径和基础模型名称
MODEL_BASE = "google/mt5-small"
LORA_OUT_PATH = "./summarizer/lora_out"


def preprocess(example, tokenizer):
    """
    处理数据集，为 mt5 模型训练做准备。
    """
    text = example.get("transcript_clean") or clean_fillers(example["transcript_raw"])
    json_summary = json.dumps(example["summary_ref"], ensure_ascii=False)

    # mt5 模型使用特定的输入格式
    # 根据消息类型选择不同的 Prompt 和输出示例
    if example.get("msg_type", "unknown") == "notice":
        prompt = (
            f"你是一名高效的语义分析助手。请从以下文本中提取主语、动词和名词。请将以下微信语音转录稿总结为一份结构化的JSON摘要，重点突出通知内容和要点。\n"
            f"请严格遵循以下JSON格式输出：{{\"type\": \"notice\", \"notice_content\": [\"...\"], \"bullets\": [\"...\"]}}\n"
            f"原文：{text}\n"
            f"摘要：")
    elif example.get("msg_type", "unknown") == "task":
        prompt = (
            f"你是一名高效的语义分析助手。请从以下文本中提取主语、动词和名词。请将以下微信语音转录稿总结为一份结构化的JSON摘要，重点突出待办任务和关键信息（人物、时间）。\n"
            f"请严格遵循以下JSON格式输出：{{\"type\": \"task\", \"tasks\": [\"...\"], \"mentions\": [\"...\"]}}\n"
            f"原文：{text}\n"
            f"摘要：")
    elif example.get("msg_type", "unknown") == "chitchat":
        prompt = (
            f"你是一名高效的摘要助手。请将以下微信语音转录稿总结为一份结构化的JSON摘要，记录核心事件或情绪。\n"
            f"请严格遵循以下JSON格式输出：{{\"type\": \"chitchat\", \"event\": [\"...\"], \"emotion\": \"...\"}}\n"
            f"原文：{text}\n"
            f"摘要：")
    else:
        prompt = (
            f"你是一名高效的摘要助手。请将以下微信语音转录稿总结为一份结构化的JSON摘要。\n"
            f"请严格遵循以下JSON格式输出：{{\"type\": \"unknown\", \"summary\": \"...\"}}\n"
            f"原文：{text}\n"
            f"摘要：")

    model_inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(json_summary, truncation=True, padding="max_length", max_length=512)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    """
    主函数：加载数据、配置模型、启动训练。
    """
    ds = load_dataset("json", data_files="data/labels.jsonl")
    ds = ds['train'].train_test_split(test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)

    # 数据预处理
    tokenized_datasets = ds.map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=ds["train"].column_names
    )

    # 加载 mt5 模型
    model = MT5ForConditionalGeneration.from_pretrained(
        MODEL_BASE,
        trust_remote_code=True
    )

    # 启用 LoRA 训练
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],  # T5模型的目标模块是"q"和"v"
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir=LORA_OUT_PATH,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,  # 减少为 1 个周期
        learning_rate=2e-5,
        eval_strategy="no",  # 跳过评估
        save_strategy="no",  # 训练结束后不保存模型
        logging_dir="./logs",
        fp16=False,
        report_to="none"
    )
    """
    # 训练参数
    args = TrainingArguments(
        output_dir=LORA_OUT_PATH,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        fp16=False,
        report_to="none"
    )
    """
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=None,  # T5 模型不需要特殊的 data collator
        tokenizer=tokenizer
    )

    trainer.train()

    print("Training complete! Saving model...")
    trainer.save_model(LORA_OUT_PATH)
    tokenizer.save_pretrained(LORA_OUT_PATH)
    print(f"Model and tokenizer saved to {LORA_OUT_PATH}")


if __name__ == "__main__":
    main()