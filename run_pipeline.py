# run_pipeline.py

import sys
import os
import torch
import json
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel

# 将项目根目录添加到 Python 路径，确保能够找到所有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入所有必要的模块
from preprocess.remove_fillers import clean_fillers
from classifier.inference import load_classifier_model, predict_message_type
from summarizer.mt5_summarize import load_models as load_summarizer_models, summarize_with_mt5


# 这个函数现在用于处理文本，而不是音频
def run_text_pipeline(text: str, classifier_models, summarizer_models):
    classifier_model, classifier_tokenizer = classifier_models
    summarizer_model, summarizer_tokenizer = summarizer_models

    print("\n--- 1. 文本输入 ---")
    print("原始文本:", text)

    print("\n--- 2. 口语化处理 ---")
    clean_text = clean_fillers(text)
    print("清理后文本:", clean_text)

    print("\n--- 3. 消息类型分类 ---")
    predicted_msg_type = predict_message_type(classifier_model, classifier_tokenizer, clean_text)
    print("预测消息类型:", predicted_msg_type)

    print("\n--- 4. 摘要生成 ---")
    # 传递所有必要的参数
    summary = summarize_with_mt5(
        summarizer_model,
        summarizer_tokenizer,
        clean_text,
        predicted_msg_type,
        #emotion="neutral",  # 暂时硬编码
        #emphasis=[]  # 暂时硬编码
    )
    print("最终摘要:", summary)

    print("\n" + "=" * 50 + "\n")

    return {
        "transcript": text,
        "transcript_clean": clean_text,
        "predicted_type": predicted_msg_type,
        "summary": summary
    }


if __name__ == "__main__":
    try:
        # 在脚本开始时只加载一次模型
        print("正在加载所有模型，这可能需要一些时间...")
        classifier_models = load_classifier_model()
        summarizer_models = load_summarizer_models()
        print("模型加载完成。")

        # 多个文本示例，你可以根据需要进行修改
        sample_texts = [
            "哎,大家注意了啊,我真是服了,那个办公楼下的停车场,就刚才又被不知道谁的车给堵了出口啊,这都第几次了,嗯,我现在非常非常生气,麻烦这位车主,立刻,马上去把车挪开,否则后果自负,就这样。",
            "啊，小李啊，那个，你，嗯，把那个项目报告，嗯，下周一之前发给我，那个，对，那个，把数据整理好。",
            "哇，今天天气真好，我们，呃，下午去逛逛街怎么样啊？"
            "小李，这个项目的风险评估报告，嗯...需要更加详细一些。啊...特别是数据安全部分，呃...请补充完善后再提交"
            "呃，大家注意一下，我刚刚接到物业通知，说我们这栋楼，嗯，下午可能会临时检修电路，呃，会有几次短暂的停电，啊，请大家务必及时保存电脑上的工作资料，避免丢失。"
        ]

        # 遍历所有文本，并生成摘要
        for text in sample_texts:
            run_text_pipeline(text, classifier_models, summarizer_models)

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保所有模型文件都已正确保存。")