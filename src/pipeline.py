from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

# 1. 初始化模型
# 我们使用预训练好的中文文本分类模型作为占位符
# 注意：这些模型可能不是专门用于你特定任务的，但足以验证流程
print("正在加载模型...")
# 消息类型分类器：使用一个通用的中文情感分析模型来模拟，因为它能理解文本的意图
# Old code classifier = pipeline("text-classification", model="uer/roberta-base-chinese-ext-ft-emotion")
# Old code classifier = pipeline("text-classification", model="IDEA-CCNL/Erlangshen-RoBERTa-110M-Emotion")
# Old code classifier = pipeline("text-classification", model="IDEA-CCNL/Erlangshen-RoBERTa-110M-Clue")
# Old code classifier = pipeline("text-classification", model="IDEA-CCNL/Erlangshen-RoBERTa-110M-Chinese")


# New code (replace with this line)
classifier = pipeline("text-classification", model="google-bert/bert-base-chinese")

# mT5 摘要模型
# 你已经安装了相关库，这里直接加载小版本的 mT5
tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
model = T5ForConditionalGeneration.from_pretrained("google/mt5-small")
print("模型加载完成。")


# 2. 定义函数
def get_message_type_and_emotion(text):
    """
    使用预训练模型来模拟消息类型分类和情感识别。
    """
    # 模拟消息类型分类：用情感分类器的结果来近似判断
    # 真实项目中需要用你自己的分类器
    result = classifier(text)[0]
    label = result['label']
    score = result['score']

    # 简化处理：根据情感判断消息类型
    # 比如：正面情感更可能是闲聊，负面情感可能是任务
    message_type = "闲聊" if "positive" in label else "任务" if "negative" in label else "通知"

    print(f"文本: '{text}'")
    print(f"分类器结果: 情感 '{label}', 分数 {score:.2f}")
    print(f"推断消息类型: {message_type}")

    return message_type, label


def generate_summary(text, message_type, emotion):
    """
    使用 mT5 模型生成摘要。
    """
    # 构建 Prompt，将所有信息传给模型
    prompt = f"消息类型是【{message_type}】，说话人情绪是【{emotion}】。请将以下微信语音转录稿总结为一份简洁、重点突出的摘要，去除无用的口语化词语和重复信息。\n原始文本：{text}\n摘要："

    inputs = tokenizer(prompt, return_tensors="pt")

    # 生成摘要
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary


# 3. 完整流程演示
def run_full_pipeline(transcribed_text):
    """
    整合整个流程：从文本到摘要。
    """
    # 1. 消息类型分类和情感识别
    message_type, emotion = get_message_type_and_emotion(transcribed_text)

    # 2. 生成摘要
    summary = generate_summary(transcribed_text, message_type, emotion)

    print("\n--- 摘要生成结果 ---")
    print(summary)

    return summary


# 4. 运行示例
if __name__ == "__main__":
    # 示例文本，模拟 Whisper 的输出
    # 这里包含了口语化的“呃”、“啊”、“嗯”，还有倒装句
    sample_texts = [
        "啊，小李啊，那个，你，嗯，把那个项目报告，嗯，下周一之前发给我，那个，对，那个，把数据整理好。",
        "哇，今天天气真好，我们，呃，下午去逛逛街怎么样啊？",
        "各位同事，啊，明天上午九点半，那个，项目例会，嗯，请大家准时参加。"
    ]

    for text in sample_texts:
        run_full_pipeline(text)
        print("\n" + "=" * 50 + "\n")