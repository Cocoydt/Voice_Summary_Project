# from classifier.inference import predict_msg_type
# from summarizer.mt5_summarize import summarize
from asr.transcribe import transcribe
from preprocess.remove_fillers import clean_fillers
from emotion.emotion_detect import EmotionRecognizer
from prosody.extract_prosody import extract_emphasis
from summarizer.mt5_summarize import Summarizer
# from classifier.inference import predict_msg_type

emotion_recognizer = EmotionRecognizer()
summarizer = Summarizer()

def pipeline(audio_path):
    # 1. 语音转写
    text, timestamps = transcribe(audio_path)
    clean_text = clean_fillers(text)

    # 2. 情感识别
    emotion = emotion_recognizer.predict(audio_path)

    # 3. 重音提取
    emphasis_words = extract_emphasis(audio_path, text, timestamps)

    # 4. 消息类型（此处默认"task"，可替换分类器结果）
    msg_type = "task"

    # 5. 摘要生成
    summary = summarizer.summarize(
        text=clean_text,
        msg_type=msg_type,
        emotion=emotion,
        emphasis=emphasis_words
    )

    print("转写:", text)
    print("去口头语:", clean_text)
    print("情感:", emotion)
    print("重音词:", emphasis_words)
    print("摘要:", summary)

if __name__ == "__main__":
    pipeline("data/raw_audio/sample.wav")

'''
def pipeline(audio_path):
    text, _ = transcribe(audio_path)
    clean_text = clean_fillers(text)
    # msg_type = predict_msg_type(clean_text)
    # summary = summarize(clean_text, msg_type)
    print("转写:", text)
    print("去口头语:", clean_text)
    # print("消息类型:", msg_type)
    # print("摘要:", summary)

if __name__ == "__main__":
    pipeline("data/raw_audio/sample.wav")
'''
'''
emotion_recognizer = EmotionRecognizer()

def pipeline(audio_path):
    # 1. 语音转写
    text, timestamps = transcribe(audio_path)
    clean_text = clean_fillers(text)

    # 2. 语音情感识别（端到端）
    emotion = emotion_recognizer.predict(audio_path)

    # 3. 重音提取
    emphasis_words = extract_emphasis(audio_path, text, timestamps)

    # 4. 分类器 & 摘要
    # msg_type = predict_msg_type(clean_text)
    # summary = summarize(clean_text, msg_type, emotion=emotion, emphasis=emphasis_words)

    print("转写:", text)
    print("去口头语:", clean_text)
    print("情感:", emotion)
    print("重音词:", emphasis_words)
    # print("摘要:", summary)

if __name__ == "__main__":
    pipeline("data/raw_audio/sample.wav")
'''