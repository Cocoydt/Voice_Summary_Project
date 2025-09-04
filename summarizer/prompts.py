# summarizer/prompts.py
# -*- coding: utf-8 -*-

PROMPT_TASK = """你是一名智能助理，负责从一段语音转写文本中提取关键信息并生成结构化摘要。

输入信息：
- 类型：任务
- 情感：{emotion}
- 重点词（可能有重音）：{emphasis_words}
- 文本：{cleaned_transcript}

请输出 JSON 格式摘要，包含以下字段：
{{
  "overall_summary": "整段语音的简洁书面摘要",
  "segments": [
    {{"time_range": "占位，由系统替换", "segment_summary": "分段摘要内容", "original_fragment": "对应的原文片段"}},
    {{"time_range": "占位，由系统替换", "segment_summary": "分段摘要内容", "original_fragment": "对应的原文片段"}}
  ],
  "msg_type": "task",
  "emotion": "{emotion}",
  "emphasis_kept": {emphasis_words},
  "quality_flags": []
}}

规则：
1. 根据文本长度决定分 2 段或 3 段。
2. 每个分段“segment_summary”下方要附上“original_fragment”，请从输入文本中截取对应片段。
3. 时间范围由系统替换，你只需保证段落顺序正确。
4. 仅返回 JSON，不要任何解释。
5. 不要改写或润色 original_fragment；它必须与输入片段逐字一致，仅对其做摘要。
"""

PROMPT_NOTICE = """你是一名智能助理，负责从一段通知类语音文本中提取主要信息并生成摘要。

输入信息：
- 类型：通知
- 情感：{emotion}
- 重点词：{emphasis_words}
- 文本：{cleaned_transcript}

输出 JSON：
{{
  "overall_summary": "整段语音的简洁书面摘要",
  "segments": [
    {{"time_range": "占位，由系统替换", "segment_summary": "分段摘要内容", "original_fragment": "对应的原文片段"}},
    {{"time_range": "占位，由系统替换", "segment_summary": "分段摘要内容", "original_fragment": "对应的原文片段"}}
  ],
  "msg_type": "notice",
  "emotion": "{emotion}",
  "emphasis_kept": {emphasis_words},
  "quality_flags": []
}}

规则：
1. 通知不一定有任务动作，主要提取信息要点。尽量客观。
2. 如有开始/生效时间请保留到分段或总摘要中。
3. 仅返回 JSON。
4. 不要改写或润色 original_fragment；它必须与输入片段逐字一致，仅对其做摘要。
"""

PROMPT_CHITCHAT = """你是一名智能助手，需要从一段朋友/同事/家人的闲聊语音文本中生成简短摘要。

输入信息：
- 类型：闲聊
- 情感：{emotion}
- 重点词：{emphasis_words}
- 文本：{cleaned_transcript}

输出 JSON：
{{
  "overall_summary": "整段语音的简洁书面摘要，覆盖主要话题和情绪",
  "segments": [
    {{"time_range": "占位，由系统替换", "segment_summary": "分段摘要内容", "original_fragment": "对应的原文片段"}},
    {{"time_range": "占位，由系统替换", "segment_summary": "分段摘要内容", "original_fragment": "对应的原文片段"}}
  ],
  "msg_type": "chitchat",
  "emotion": "{emotion}",
  "emphasis_kept": {emphasis_words},
  "quality_flags": []
}}

规则：
1. 不拆任务，只提要点；需要体现情绪与观点。
2. 有时间词可保留到分段中。
3. 仅返回 JSON。
4. 不要改写或润色 original_fragment；它必须与输入片段逐字一致，仅对其做摘要。
"""

PROMPT_CLASSIFY = """请判断下面文本的消息类型，只返回一个标签：["task","notice","chitchat"]
文本：
{cleaned_transcript}
仅返回三选一的英文小写单词，不要额外解释。"""

PROMPT_TEXT_EMOTION = """请判断下面文本的主要情绪，只在这些里选一个：
["开心","愤怒","伤心","难过","生气","害怕","厌恶","惊讶","严肃","疲惫","疑惑","期待","鼓励","失望","中立"]
文本：
{cleaned_transcript}
只返回一个词，不要解释。"""