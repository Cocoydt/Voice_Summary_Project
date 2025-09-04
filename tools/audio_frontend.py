# -*- coding: utf-8 -*-
"""
audio_frontend.py
功能：音频前端（ASR + SER）
- ASR: 使用 OpenAI Chat Completions / Audio Transcriptions 兼容接口（走你的 base_url 和 api_key）
- SER: 使用 speechbrain 的预训练情感模型。失败/不可用时返回 None

依赖：
pip install openai pydub librosa torch torchaudio speechbrain

注意：
- 模型：ASR 默认 'whisper-1'（
- SER 模型：speechbrain/emotion-recognition-wav2vec2-IEMOCAP
"""

import os
import io
import json
from typing import Optional, Tuple

import torch
import torchaudio

# OpenAI 兼容 SDK
from openai import OpenAI

# SER
try:
    from speechbrain.inference.interfaces import foreign_class
    _SER_AVAILABLE = True
except Exception:
    _SER_AVAILABLE = False


class AudioFrontend:
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        asr_model: str = "whisper-1",
        ser_enabled: bool = True,
    ):
        self.client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1"),
            api_key=api_key or os.getenv("OPENAI_API_KEY", "")
        )
        self.asr_model = asr_model
        self.ser_enabled = ser_enabled and _SER_AVAILABLE
        if self.ser_enabled:
            try:
                self.ser_model = foreign_class(
                    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                    pymodule_file="custom_interface.py",
                    classname="CustomEncoderWav2vec2Classifier",
                )
            except Exception:
                # 某些环境会拉不下自定义接口文件，回退到官方接口
                from speechbrain.pretrained import EncoderClassifier
                self.ser_model = EncoderClassifier.from_hparams(
                    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
                )

    # -------- ASR ----------
    def transcribe(self, audio_path: str) -> str:
        """
        使用 OpenAI 兼容的音频转写接口。
        如果你的代理不支持 audio.transcriptions，可以退回用 GPT 对音频不支持的情况。
        """
        # 优先尝试新版 responses API 之前的 /audio/transcriptions 兼容写法
        try:
            with open(audio_path, "rb") as f:
                # 一些兼容代理仍然用 client.audio.transcriptions.create
                transcript = self.client.audio.transcriptions.create(
                    model=self.asr_model,
                    file=f,
                    response_format="text"
                )
            if isinstance(transcript, str):
                return transcript.strip()
            # 某些代理会返回对象
            return getattr(transcript, "text", "").strip()
        except Exception as e:
            # 简单兜底：返回空串让上游决定是否跳过
            print(f"[ASR] 调用失败：{e}")
            return ""

    # -------- SER ----------
    def predict_emotion(self, audio_path: str) -> Optional[str]:
        """
        使用 SpeechBrain 预测音频情感。返回 {angry, happy, sad, neutral, disgust, fear, surprise} 中之一
        """
        if not self.ser_enabled:
            return None
        try:
            # 加载音频
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
                sr = 16000
            # SpeechBrain 的接口
            if hasattr(self.ser_model, "classify_batch"):
                out = self.ser_model.classify_batch(waveform)
                probs = out[0].squeeze().detach().cpu().tolist()
                # 类别名称
                if hasattr(self.ser_model, "hparams") and hasattr(self.ser_model.hparams, "label_encoder"):
                    labels = self.ser_model.hparams.label_encoder.decode_ndim(torch.arange(len(probs)))
                    labels = [str(l) for l in labels]
                else:
                    labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
                pred = labels[int(torch.tensor(probs).argmax().item())]
                return pred
            else:
                # 另一种接口
                out = self.ser_model.classify_file(audio_path)
                # out 可能是字典或列表，做稳健解析
                if isinstance(out, dict):
                    return out.get("prediction", None)
                return str(out)
        except Exception as e:
            print(f"[SER] 预测失败：{e}")
            return None