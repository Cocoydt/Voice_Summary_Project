# speech_processor/emotion_recognizer.py
from transformers import pipeline

class EmotionRecognizer:
    def __init__(self):
        # 加载预训练模型，用于语音情感识别
        self.recognizer = pipeline("audio-classification", model="huyan/chinese-speech-emotion-recognition")

    def predict(self, audio_path: str) -> str:
        # 对音频文件进行情感预测
        result = self.recognizer(audio_path)
        # 提取得分最高的标签，例如 '高兴', '生气'
        emotion = result[0]['label']
        return emotion