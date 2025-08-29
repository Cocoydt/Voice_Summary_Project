from datasets import load_dataset
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model

from datasets import load_dataset
from transformers import MT5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# summarizer/train_lora.py
from datasets import load_dataset
from transformers import MT5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import sys
import os

# Add the parent directory (project root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from transformers import MT5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

from preprocess.remove_fillers import clean_fillers

MODEL_NAME = "google/mt5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(example, tokenizer):
    # 如果有干净转写就用，没有就清洗
    text = example.get("transcript_clean") or clean_fillers(example["transcript_raw"])

    msg_type = example.get("msg_type", "unknown")
    emotion = example.get("emotion", "neutral")
    emphasis = ",".join(example.get("prosody", {}).get("emphasis_tokens", [])) or "none"

    prefix = f"[TYPE={msg_type}] [EMOTION={emotion}] [EMPHASIS={emphasis}]"
    input_text = f"{prefix} 原文：{text}"

    model_inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(example["summary_ref"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    # 加载原始数据
    ds = load_dataset("json", data_files="data/labels.jsonl")
    ds = ds['train'].train_test_split(test_size=0.1)  # 添加这行，用于创建 train 和 test 集
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    # 数据预处理
    tokenized_datasets = ds.map(lambda x: preprocess(x, tokenizer), remove_columns=ds["train"].column_names)

    # 模型 & LoRA 配置
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    # 替换 prepare_model_for_int8_training 为 prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q", "v"],
        lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)

    # 训练参数
    args = TrainingArguments(
        output_dir="./summarizer/lora_out",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        eval_strategy="epoch",  # 修正为 epoch
        save_strategy="epoch",
        logging_dir="./logs",
        fp16=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    # 保存模型和分词器
    model.save_pretrained("./summarizer/lora_out")
    tokenizer.save_pretrained("./summarizer/lora_out")


if __name__ == "__main__":
    main()



'''
MODEL_NAME = "google/mt5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess(example, tokenizer):
    msg_type = example.get("msg_type", "unknown")
    emotion = example.get("emotion", "neutral")
    emphasis = ",".join(example.get("prosody", {}).get("emphasis_tokens", [])) or "none"

    prefix = f"[TYPE={msg_type}] [EMOTION={emotion}] [EMPHASIS={emphasis}]"
    input_text = f"{prefix} 原文：{example['transcript_clean']}"

    model_inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(example["summary_ref"], truncation=True, padding="max_length", max_length=128)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    # 加载数据集
    ds = load_dataset("json", data_files="data/labels_cleaned.jsonl")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    ds = ds.map(lambda x: preprocess(x, tokenizer), remove_columns=ds["train"].column_names)

    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir="./summarizer/lora_out",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_dir="./logs",
        fp16=torch.cuda.is_available()
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained("./summarizer/lora_out")
    tokenizer.save_pretrained("./summarizer/lora_out")

if __name__ == "__main__":
    main()


MODEL_NAME = "google/mt5-base"

def main():
    ds = load_dataset("json", data_files="data/labels.jsonl")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    def preprocess(ex):
        ex["input_ids"] = tokenizer(f"总结以下内容: {ex['transcript_clean']}",
                                    truncation=True, padding="max_length", max_length=512).input_ids
        ex["labels"] = tokenizer(ex["summary_ref"], truncation=True, padding="max_length",
                                 max_length=128).input_ids
        return ex

    def preprocess(ex):
        msg_type = ex.get("msg_type", "unknown")
        emotion = ex.get("emotion", "neutral")
        emphasis = ",".join(ex.get("prosody", {}).get("emphasis_tokens", [])) or "none"

        prefix = f"[TYPE={msg_type}] [EMOTION={emotion}] [EMPHASIS={emphasis}]"
        input_text = f"{prefix} 原文：{ex['transcript_clean']}"

        ex["input_ids"] = tokenizer(input_text, truncation=True, padding="max_length", max_length=512).input_ids
        ex["labels"] = tokenizer(ex["summary_ref"], truncation=True, padding="max_length", max_length=128).input_ids
        return ex



    ds = ds.map(preprocess)

    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q","v"])
    model = get_peft_model(model, lora_config)

    model.train()
    # 此处可用 Trainer 训练，略

if __name__ == "__main__":
    main()
'''