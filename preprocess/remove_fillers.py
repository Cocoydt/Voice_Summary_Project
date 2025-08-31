# preprocess/remove_fillers.py
import re

# 常见口头语和填充词
FILLERS = ["嗯", "呃", "额", "那个", "就是", "然后", "emmm", "啊", "哎", "对对对"]

def clean_fillers(text: str):
    """
    去除口头语和重复词，并清理多余的标点和空格。
    """
    # 1. 去除常见口头语
    # 使用正则表达式，通过在词语前后添加空格来确保不误删
    pattern_fillers = "|".join([f"\\b{re.escape(f)}" for f in FILLERS])
    cleaned_text = re.sub(pattern_fillers, "", text)

    # 2. 去除重复词
    # 查找连续重复的词语，例如 "我我我"
    cleaned_text = re.sub(r'(\b\w+\b)\s+\1', r'\1', cleaned_text)

    # 3. 清理多余的标点和空格
    # 去除连续的标点，例如 ",," -> ","
    cleaned_text = re.sub(r'([，。、？！])\1+', r'\1', cleaned_text)
    # 去除句首句尾的标点和空格
    cleaned_text = cleaned_text.strip(' ，。、？！')
    # 将多个空格替换为一个空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text


if __name__ == "__main__":
    sample1 = "嗯那个就是我们明天然后去客户那边啊"
    sample2 = "我我我我去了趟北京，然后然后去故宫了。"
    sample3 = "呃，，，你明天，嗯，有空吗？"

    print("原文1:", sample1)
    print("去口头语后1:", remove_fillers(sample1))

    print("\n原文2:", sample2)
    print("去口头语后2:", remove_fillers(sample2))

    print("\n原文3:", sample3)
    print("去口头语后3:", remove_fillers(sample3))