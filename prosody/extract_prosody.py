import librosa
import numpy as np

def extract_emphasis(audio_path: str, transcript: str, timestamps: list, threshold_db: float = -25.0):
    """
    根据音频能量和时间戳，找到可能被重读/强调的词
    timestamps: [(start, end), ...] 与转写词对应
    """
    y, sr = librosa.load(audio_path, sr=None)
    S, _ = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S).flatten()
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    emphasized_words = []
    for (word, (start, end)) in zip(transcript.split(), timestamps):
        start_frame = int(start * sr / 512)
        end_frame = int(end * sr / 512)
        avg_db = np.mean(rms_db[start_frame:end_frame])
        if avg_db > threshold_db:
            emphasized_words.append(word)
    return emphasized_words

if __name__ == "__main__":
    words = extract_emphasis("data/raw_audio/sample.wav",
                             "明天 下午 三点 我们 去 客户 那边",
                             [(0.0,0.2),(0.2,0.5),(0.5,0.8),(0.8,1.0),(1.0,1.3),(1.3,1.7)])
    print("重音词：", words)