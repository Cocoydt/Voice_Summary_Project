from asr.transcribe import transcribe
from preprocess.remove_fillers import clean_fillers
# from classifier.inference import predict_msg_type
# from summarizer.mt5_summarize import summarize

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