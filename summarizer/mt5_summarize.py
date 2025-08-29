from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch

MODEL_NAME = "google/mt5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Summarizer:
    def __init__(self, model_path="./summarizer/lora_out"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)

    def summarize(self, text: str, msg_type="unknown", emotion="neutral", emphasis=None, max_length=128):
        if emphasis is None or len(emphasis) == 0:
            emphasis = "none"
        else:
            emphasis = ",".join(emphasis)

        prefix = f"[TYPE={msg_type}] [EMOTION={emotion}] [EMPHASIS={emphasis}]"
        input_text = f"{prefix} 原文：{text}"

        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        output = self.model.generate(**inputs, max_length=max_length, num_beams=4, do_sample=False)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    s = Summarizer()
    summary = s.summarize(
        text="明天下午三点我们去客户那边，记得带资料。",
        msg_type="task",
        emotion="angry",
        emphasis=["三点", "客户"]
    )
    print("摘要：", summary)