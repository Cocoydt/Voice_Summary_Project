# summarizer/train_lora.py

import sys
import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
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
    prompt = f"消息类型是【{example.get('msg_type', 'unknown')}】。请将以下微信语音转录稿总结为一份简洁、重点突出的摘要。\n原文：{text}\n摘要："

    model_inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    labels = tokenizer(json_summary, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

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
    model = AutoModelForCausalLM.from_pretrained(
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
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

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