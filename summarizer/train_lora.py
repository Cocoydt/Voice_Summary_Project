import sys
import os
import torch
import json
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    MT5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.remove_fillers import clean_fillers

MODEL_BASE = "google/mt5-small"
LORA_OUT_PATH = "./summarizer/lora_out"

def preprocess(example, tokenizer):
    """
    处理数据集，为 mt5 模型训练做准备。
    """
    text = example.get("transcript_clean") or clean_fillers(example["transcript_raw"])
    # Retrieve the summary text. We now handle the case where it might not be a string.
    summary_data = example.get("summary_ref", "这是一个空摘要")

    # Ensure summary_text is a string before tokenizing
    if isinstance(summary_data, dict):
        # If the data is a dictionary (JSON object), convert it to a string
        summary_text = json.dumps(summary_data, ensure_ascii=False)
    else:
        # Otherwise, assume it's a string
        summary_text = str(summary_data)

    # mt5 模型使用特定的输入格式
    prompt = f"消息类型是【{example.get('msg_type', 'unknown')}】。请将以下微信语音转录稿总结为一份简洁、重点突出的摘要。\n原文：{text}\n摘要："

    # 将文本和摘要分别编码，并返回字典
    model_inputs = tokenizer(prompt, max_length=512, truncation=True)
    labels = tokenizer(text_target=summary_text, max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



def main():
    """
    主函数：加载数据、配置模型、启动训练。
    """
    ds = load_dataset("json", data_files="data/labels.jsonl")
    ds = ds['train'].train_test_split(test_size=0.1)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)

    # 加载 mt5 模型
    model = MT5ForConditionalGeneration.from_pretrained(
        MODEL_BASE,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    # 启用 LoRA 训练
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)

    # 数据预处理
    tokenized_datasets = ds.map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=ds["train"].column_names
    )

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

    # 修正：使用 DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,  # <-- 关键修正
        tokenizer=tokenizer
    )

    trainer.train()

    print("Training complete! Saving model...")
    trainer.save_model(LORA_OUT_PATH)
    tokenizer.save_pretrained(LORA_OUT_PATH)
    print(f"Model and tokenizer saved to {LORA_OUT_PATH}")


if __name__ == "__main__":
    main()