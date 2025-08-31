# classifier/inference.py

import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# 定义模型路径和名称
MODEL_PATH = "./clf_out"
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"

# 定义标签映射，用于将模型的输出（0, 1, 2）映射回你定义的标签
# 这里的顺序必须和你在 train_msg_type.py 中定义的映射顺序一致
LABEL_MAP = {0: "notice", 1: "task", 2: "chitchat"}


def load_classifier_model():
    """
    加载训练好的分类器模型和分词器。
    这个函数只需要在脚本开始时运行一次。
    """
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件未找到，请先运行 'train_msg_type.py' 来训练模型。")

    print("正在加载分类器模型...")
    # 加载你的微调模型
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    return model, tokenizer


def predict_message_type(model, tokenizer, text: str) -> str:
    """
    对给定的文本进行消息类型分类。

    Args:
        model: 训练好的分类模型。
        tokenizer: 对应的分词器。
        text: 需要分类的文本。

    Returns:
        预测的消息类型标签（'notice', 'task', 或 'chitchat'）。
    """
    # 创建一个 Hugging Face pipeline，用于简化推理过程
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,  # 使用 GPU 或 CPU
    )

    # 预测并返回结果
    # 这里的 top_k=None 确保返回所有标签的得分
    result = classifier(text, top_k=None)

    # 找出得分最高的标签
    # 结果是一个列表，例如 [{'label': 'LABEL_1', 'score': 0.95}, ...]
    predicted_label = result[0]['label']

    # 这是一个关键步骤：将 'LABEL_0', 'LABEL_1' 这样的标签，映射回我们的自定义标签
    # 例如：'LABEL_0' -> 'notice'
    # 确保这里的映射关系正确
    label_index = int(predicted_label.split('_')[-1])
    predicted_type = LABEL_MAP.get(label_index, "unknown")

    return predicted_type


if __name__ == "__main__":
    # 这个 __main__ 块用于单独测试这个文件，不会被 run_pipeline.py 调用
    try:
        loaded_model, loaded_tokenizer = load_classifier_model()

        sample_text_task = "小李，把项目报告下周一之前发给我。"
        sample_text_notice = "各位同事，明天上午九点半项目例会，请准时参加。"
        sample_text_chitchat = "周末我去北京玩了，故宫人好多，不过挺值得的。"

        print(f"文本: '{sample_text_task}'")
        print(f"预测类型: {predict_message_type(loaded_model, loaded_tokenizer, sample_text_task)}")

        print(f"\n文本: '{sample_text_notice}'")
        print(f"预测类型: {predict_message_type(loaded_model, loaded_tokenizer, sample_text_notice)}")

        print(f"\n文本: '{sample_text_chitchat}'")
        print(f"预测类型: {predict_message_type(loaded_model, loaded_tokenizer, sample_text_chitchat)}")

    except FileNotFoundError as e:
        print(f"错误: {e}")