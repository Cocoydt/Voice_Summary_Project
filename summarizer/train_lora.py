from datasets import load_dataset
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model

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

    ds = ds.map(preprocess)

    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q","v"])
    model = get_peft_model(model, lora_config)

    model.train()
    # 此处可用 Trainer 训练，略

if __name__ == "__main__":
    main()