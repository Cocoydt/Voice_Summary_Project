import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# 模型：IEMOCAP 微调版（可换更合适的中文模型）
MODEL_NAME = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

class EmotionRecognizer:
    def __init__(self, model_name=MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.labels = ["neutral", "happy", "sad", "angry"]

    def predict(self, audio_path: str) -> str:
        waveform, sr = torchaudio.load(audio_path)
        inputs = self.extractor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(**inputs.to(self.device)).logits
            pred = torch.argmax(logits, dim=-1).item()
        return self.labels[pred]

if __name__ == "__main__":
    er = EmotionRecognizer()
    print(er.predict("data/raw_audio/sample.wav"))