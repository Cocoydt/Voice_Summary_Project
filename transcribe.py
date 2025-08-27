import whisper

def transcribe_audio(audio_path, model_name="base"):
    print("正在加载 Whisper 模型，这可能需要一些时间...")
    model = whisper.load_model(model_name)

    print(f"正在转写音频文件：{audio_path}")
    result = model.transcribe(audio_path, language="zh")

    return result['text']

# 替换成你的音频文件路径
audio_file_path = "task1_高管普通话.mp3"

try:
    transcribed_text = transcribe_audio(audio_file_path)
    print("\n--- 转写结果 ---")
    print(transcribed_text)
except Exception as e:
    print(f"\n出现错误：{e}")
    print("请检查你的文件路径是否正确，或者重新安装相关库。")