import pandas as pd
import requests
import os
import random
import time
import json
import codecs

# --- 配置信息 ---
# 替换成你的 MiniMax API 密钥和 Group ID
API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLmtbfonrrnlKjmiLdfNDE2NjU4MTExNTAyOTEzNTM4IiwiVXNlck5hbWUiOiLmtbfonrrnlKjmiLdfNDE2NjU4MTExNTAyOTEzNTM4IiwiQWNjb3VudCI6IiIsIlN1YmplY3RJRCI6IjE5NTk4NjE2MTA3NTEyMDU5MzQiLCJQaG9uZSI6IjE3NzIyODQ0OTEzIiwiR3JvdXBJRCI6IjE5NTk4NjE2MTA3NDI4MTczMjYiLCJQYWdlTmFtZSI6IiIsIk1haWwiOiIiLCJDcmVhdGVUaW1lIjoiMjAyNS0wOC0zMCAxODoyODowOCIsIlRva2VuVHlwZSI6MSwiaXNzIjoibWluaW1heCJ9.cfbMCk0OvTzBHZGo5ubCFiRsSNkQaWIsf3b6PwYOptQhmG-K18NCjrL3nekOvTx7iHTVVvWzYUJPjuSA0WBxPwh9UfgjopN3L70db5ZHYMSn4p4nIlqBtqGTFbC3lgsEThO780u6IZlW0D5gdIGgIipRSkeSz1z-yvx-OWeeQ-NCqXWVq71dEUdo5r43MVu_kr5GrFuRzVT43qb_Y1Ru6puYesPbvQD_5vMyIUnyMo7_IL_A-tK9hUd82zB4TWJFxTX0mhUh8szLhHn9hWqzo5xxHeWPwg_fauMbAR5E5Adg7Sf3NaMae1w1WgJr8YhcThdUkVlS7b1zzwsOzIZeJA"
GROUP_ID = "1959861610742817326"

# 更新为 MiniMax T2A V2 接口的正确地址
API_URL = f"https://api.minimaxi.com/v1/t2a_v2?GroupId={GROUP_ID}"

# 你的 CSV 文件路径
CSV_FILE = "CHI智能耳机_数据表.csv"

# 音频文件保存目录
OUTPUT_DIR = "generated_audios"

# 文本列名称和情感列名称
TEXT_COLUMN = 'transcript_raw'
EMOTION_COLUMN = 'emotions'

# 可供选择的多种音色列表
VOICE_NAMES = [
    "male-qn-jingying",
    "female-chengshu",
    "male-qn-badao-jingpin",
    "clever_boy",
    "lovely_girl",
    "junlang_nanyou",
    "female-tianmei-jingpin",
]

# 情绪映射字典
EMOTION_MAP = {
    "愤怒": "angry",
    "生气": "angry",
    "开心": "happy",
    "惊讶": "surprised",
    "惊喜": "surprised",
    "害怕": "fearful",
    "焦急": "fearful",
    "伤心":"sad",
    "悲伤":"sad",
    "厌恶":"disgusted",
    "严肃":"calm",
    # 根据文档调整映射关系...
}

# 检查并创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 主程序逻辑 ---
try:
    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(CSV_FILE)

    # 遍历 DataFrame 的每一行
    for index, row in df.iterrows():
        # 获取 ID、文本内容和情绪
        file_id = row['ID']

        # 检查 ID 是否为 nan（空值），如果是，则跳过
        if pd.isna(file_id):
            print(f"警告：第 {index + 2} 行的 ID 为空，跳过处理。")
            continue

        # 将 ID 转换为整数，以便命名文件
        file_id = int(file_id)

        # 文本列名称，根据你的需求选择
        text = row[TEXT_COLUMN]
        csv_emotion = row[EMOTION_COLUMN]

        # 随机选择一个音色
        selected_voice = random.choice(VOICE_NAMES)

        # 根据 CSV 中的情绪查找对应的 MiniMax 情绪参数
        minimax_emotion = EMOTION_MAP.get(csv_emotion, "neutral")

        print(f"正在处理 ID: {file_id}, 音色: {selected_voice}, 情绪: {minimax_emotion}")

        # 构造符合 MiniMax v2 接口的请求体
        payload = {
            "model": "speech-2.5-hd-preview",
            "text": text,
            "stream": False,
            "language_boost": "auto",
            "output_format": "hex",  # 保持 hex 格式，以便获取音频数据
            "voice_setting": {
                "voice_id": selected_voice,
                "speed": 1,
                "vol": 1,
                "pitch": 0,
                "emotion": minimax_emotion
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1
            }
        }

        # 构造请求头
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            # 发送请求
            response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            response.raise_for_status()  # 如果请求失败，抛出异常

            # 解析 JSON 响应
            response_json = response.json()

            # 从响应中提取十六进制编码的音频数据
            hex_audio_data = response_json['data']['audio']

            # 将十六进制数据解码为二进制数据
            audio_data = codecs.decode(hex_audio_data, 'hex')

            # 文件名格式化为 ID.mp3
            output_file_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp3")

            # 将返回的二进制音频数据写入文件
            with open(output_file_path, "wb") as f:
                f.write(audio_data)

            print(f"成功生成并保存文件: {output_file_path}")
            time.sleep(1)  # 增加延迟，避免触发API的频率限制

        except requests.exceptions.RequestException as e:
            print(f"处理 ID: {file_id} 时请求失败, 错误: {e}")
            if 'response' in locals() and response.text:
                print(f"响应内容: {response.text}")
            continue  # 跳过当前行，继续下一行
        except KeyError as e:
            print(f"处理 ID: {file_id} 时解析响应失败, 缺少键: {e}")
            print(f"完整的响应内容: {response.text if 'response' in locals() else '无'}")
            continue

except FileNotFoundError:
    print(f"错误：找不到文件 '{CSV_FILE}'，请检查文件路径是否正确。")
except Exception as e:
    print(f"发生未知错误: {e}")

print("--- 所有任务已完成 ---")