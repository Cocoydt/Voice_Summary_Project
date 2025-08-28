import re

# 常见口头语
FILLERS = ["嗯", "呃", "额", "那个", "就是", "然后", "你知道吧", "emmm", "啊", "哎", "对对对"]

def clean_fillers(text: str):
    pattern = "|".join(map(re.escape, FILLERS))
    return re.sub(pattern, "", text)

if __name__ == "__main__":
    sample = "嗯那个就是我们明天然后去客户那边啊"
    print("原文:", sample)
    print("去口头语后:", clean_fillers(sample))