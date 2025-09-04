好的 ✅ 下面是一个简短版的 README（Markdown 格式），只保留队友快速上手需要的步骤：

# CHIProject Demo Pipeline

## 环境准备
```bash
conda activate voice_project
pip install openai pydub
export OPENAI_BASE_URL="https://api.openai-proxy.org/v1"
export OPENAI_API_KEY="你的 sk- 开头密钥"
```
使用方法
	1.	将待处理音频文件放到 data/raw_audio/
	2.	修改 segment_pipeline.py 配置：

AUDIO_PATH = "data/raw_audio/你的音频文件.m4a"
OUTPUT_PATH = "demo_result.json"


	3.	运行：
```bash
python segment_pipeline.py
```


输出示例

结果保存在 demo_result.json：

{
  "overall_summary": "整段语音的摘要",
  "segments": [
    {
      "time_range": "00:00-00:13",
      "segment_summary": "分段摘要",
      "original_fragment": "对应原文"
    }
  ],
  "msg_type": "task",
  "emotion": "期待"
}

