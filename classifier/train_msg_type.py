from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"

def main():
    # 加载自标注数据
    ds = load_dataset("json", data_files="data/labels.jsonl")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(ex):
        tokens = tokenizer(ex["transcript_clean"], truncation=True, padding="max_length", max_length=128)
        ex.update(tokens)
        ex["label"] = {"notice":0,"task":1,"chitchat":2}[ex["msg_type"]]
        return ex

    ds = ds.map(preprocess)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    args = TrainingArguments(
        output_dir="./clf_out", evaluation_strategy="epoch",
        learning_rate=2e-5, per_device_train_batch_size=8,
        num_train_epochs=3, weight_decay=0.01
    )

    trainer = Trainer(model=model, args=args,
                      train_dataset=ds["train"],
                      eval_dataset=ds.get("validation", ds["train"]),
                      tokenizer=tokenizer)
    trainer.train()

if __name__ == "__main__":
    main()