# segment_pipeline.py
# -*- coding: utf-8 -*-
import os, json
from datetime import datetime
from openai import OpenAI

from summarizer.prompts import (
    PROMPT_TASK, PROMPT_NOTICE, PROMPT_CHITCHAT,
    PROMPT_CLASSIFY, PROMPT_TEXT_EMOTION
)
from summarizer.segmenter import (
    vad_segments, equal_splits,
    allocate_text_by_spans, build_segment_prompt, s2mmss,
    try_asr_with_segments, coalesce_segments
)


# ===== 默认配置（可直接改）=====
AUDIO_PATH   = "data/raw_audio/study3/任务类-长语音4.m4a"
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1")
API_KEY  = os.getenv("OPENAI_API_KEY", "")
ASR_MODEL    = "whisper-1"
MODEL_NAME   = "gpt-4o-mini"
OUTPUT_PATH  = "demo_result.json"
EMPHASIS_JSON = "[]"     # 后续可替换为重音检测结果
LEN_THRESH   = 120       # 小于该字数 → 2 段，否则 3 段
NSEG_SHORT   = 2
MIN_CHARS_PER_SEG = 20   # 每段最少多少个字（避免太短）
MIN_SEC_PER_SEG   = 3.0  # 每段最少多少秒（避免太短）

def llm_call(client: OpenAI, model: str, prompt: str, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def clean_filler(text: str) -> str:
    fillers = ["啊","嗯","呃","那个","这个","emmm","哎","唉","就是"]
    for f in fillers:
        text = text.replace(f, "")
    return " ".join(text.split())

def pick_prompt(msg_type: str) -> str:
    if msg_type == "task":
        return PROMPT_TASK
    if msg_type == "notice":
        return PROMPT_NOTICE
    return PROMPT_CHITCHAT

# ==== 片段合并规则（避免碎片、空摘要）====
MIN_CHARS_PER_SEG = 12      # 少于该字数认为太短，尝试合并
MIN_SEC_PER_SEG   = 2.5     # 少于该时长认为太短，尝试合并

def _parse_mmss_to_sec(mmss: str) -> float:
    mmss = (mmss or "").strip()
    if not mmss:
        return 0.0
    parts = mmss.split(":")
    try:
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(s)
        elif len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return 0.0
    return 0.0

def _seg_duration(tr: str) -> float:
    if "-" not in tr:
        return 0.0
    a, b = tr.split("-", 1)
    return max(0.0, _parse_mmss_to_sec(a) - 0.0) and max(0.0, _parse_mmss_to_sec(b) - _parse_mmss_to_sec(a))

def merge_short_fragments(segments_payload: list,
                          min_chars: int = MIN_CHARS_PER_SEG,
                          min_sec: float = MIN_SEC_PER_SEG) -> list:
    """
    将过短的片段（文本过短或时长过短）自动并到相邻片段，尽量消除“空摘要”或碎片。
    规则：
      - 优先并入前一段；如果是第一段，则并入后一段。
      - 合并后 time_range 取首尾；original_fragment 连接（加空格）。
    """
    if not segments_payload:
        return []

    # 先复制一份，避免原地修改
    segs = [dict(x) for x in segments_payload]

    # 一轮前向合并（尽量并到前一段）
    merged = []
    for i, seg in enumerate(segs):
        frag = (seg.get("original_fragment") or "").strip()
        tr = seg.get("time_range") or ""
        dur = _seg_duration(tr)
        too_short = (len(frag) < min_chars) or (dur > 0 and dur < min_sec)

        if too_short and merged:
            # 并入前一段
            prev = merged[-1]
            # 合并 time_range：prev 的开始 + seg 的结束
            left = prev.get("time_range", "")
            right = tr
            if "-" in left:
                left_l = left.split("-", 1)[0]
            else:
                left_l = left
            if "-" in right:
                right_r = right.split("-", 1)[1]
            else:
                right_r = right or left.split("-", 1)[-1] if "-" in left else left
            prev["time_range"] = f"{left_l}-{right_r}".strip("-")
            # 合并文本
            prev_text = (prev.get("original_fragment") or "").strip()
            new_text = (prev_text + " " + frag).strip() if frag else prev_text
            prev["original_fragment"] = new_text
        else:
            merged.append(seg)

    # 若第一段仍然很短（前向无法合并），尝试与后一段合并
    if len(merged) >= 2:
        first = merged[0]
        frag0 = (first.get("original_fragment") or "").strip()
        dur0 = _seg_duration(first.get("time_range") or "")
        if (len(frag0) < min_chars) or (dur0 > 0 and dur0 < min_sec):
            second = merged[1]
            # 合并成一段
            left = first.get("time_range", "")
            right = second.get("time_range", "")
            if "-" in left:
                left_l = left.split("-", 1)[0]
            else:
                left_l = left
            if "-" in right:
                right_r = right.split("-", 1)[1]
            else:
                right_r = right
            new_tr = f"{left_l}-{right_r}".strip("-")
            new_text = " ".join([(first.get("original_fragment") or "").strip(),
                                 (second.get("original_fragment") or "").strip()]).strip()
            merged = [{
                "time_range": new_tr,
                "segment_summary": "",
                "original_fragment": new_text
            }] + merged[2:]

    return merged

def process_audio(audio_path: str,
                  base_url: str = BASE_URL,
                  api_key: str = API_KEY,
                  asr_model: str = ASR_MODEL,
                  model_name: str = MODEL_NAME,
                  emphasis_json: str = EMPHASIS_JSON) -> dict:
    """输入音频路径 → 返回结构化 JSON（总 + 分摘要）"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    client = OpenAI(base_url=base_url, api_key=api_key)

    # 1) ASR：优先拿句边界
    full_text, segs = try_asr_with_segments(client, audio_path, asr_model)
    if not full_text:
        raise RuntimeError("ASR 未得到文本")

    transcript_clean = clean_filler(full_text)

    # 2) 分类与文本情感
    try:
        pred_type = llm_call(client, model_name, PROMPT_CLASSIFY.format(cleaned_transcript=transcript_clean), 0.0)
        msg_type = pred_type if pred_type in ["task","notice","chitchat"] else "task"
    except Exception:
        msg_type = "task"

    try:
        emotion = llm_call(client, model_name, PROMPT_TEXT_EMOTION.format(cleaned_transcript=transcript_clean), 0.0)
    except Exception:
        emotion = "中立"

    # 3) 组装片段（优先 ASR → VAD → 均分）
    # 先决定目标段数：短语音 2 段、长语音 3 段（按清洗后字符数阈值）
    n_segments = NSEG_SHORT if len(transcript_clean) < LEN_THRESH else 3

    if segs:
        segments_payload = []
        for s in segs:
            start = s.get("start", 0.0) if isinstance(s, dict) else getattr(s, "start", 0.0)
            end   = s.get("end",   0.0) if isinstance(s, dict) else getattr(s, "end",   0.0)
            text  = s.get("text",  "")  if isinstance(s, dict) else getattr(s, "text",  "")
            segments_payload.append({
                "time_range": f"{s2mmss(start)}-{s2mmss(end)}",
                "segment_summary": "",
                "original_fragment": text
            })
    else:
        spans = vad_segments(audio_path)  # 可能为空
        if spans:
            segments_payload = allocate_text_by_spans(transcript_clean, spans)
        else:
            from pydub import AudioSegment
            try:
                total_sec = len(AudioSegment.from_file(audio_path)) / 1000.0
            except Exception:
                total_sec = 60.0
            # 注意：这里的 n_segments 现在已在上面定义好
            segments_payload = equal_splits(transcript_clean, total_sec, n_segments)

    # 3.a) 先把过短片段并到相邻片段，减少碎片
    segments_payload = merge_short_fragments(segments_payload)

    # 3.b) 控制最终段数（短=2，长=3）
    n_segments = NSEG_SHORT if len(transcript_clean) < LEN_THRESH else 3
    # 3.x) 控制最终段数：无论前面给了多少细段，统一合并为 2 或 3 段
    segments_payload = coalesce_segments(segments_payload, n_segments)

    # 4) 逐段总结
    base_prompt = pick_prompt(msg_type)
    prompt = build_segment_prompt(
        base_prompt, msg_type, emotion, emphasis_json, transcript_clean, segments_payload
    )
    raw = llm_call(client, model_name, prompt, 0.2)
    try:
        parsed = json.loads(raw)
    except Exception:
        raw2 = llm_call(client, model_name, prompt + "\n仅返回 JSON。", 0.0)
        try:
            parsed = json.loads(raw2)
        except Exception:
            parsed = {"overall_summary": raw[:200], "segments": []}

    # 5) 对齐 time_range & original_fragment
    returned = parsed.get("segments", [])
    if len(returned) < len(segments_payload):
        returned += [{"time_range":"", "segment_summary":"", "original_fragment":""} for _ in range(len(segments_payload)-len(returned))]
    elif len(returned) > len(segments_payload):
        returned = returned[:len(segments_payload)]
    for i in range(len(segments_payload)):
        returned[i]["time_range"] = segments_payload[i]["time_range"]
        if not returned[i].get("original_fragment"):
            returned[i]["original_fragment"] = segments_payload[i]["original_fragment"]
    parsed["segments"] = returned

    # 6) 元信息补齐
    parsed.setdefault("msg_type", msg_type)
    parsed.setdefault("emotion", emotion)
    try:
        parsed.setdefault("emphasis_kept", json.loads(emphasis_json))
    except Exception:
        parsed.setdefault("emphasis_kept", [])
    parsed["ID"] = os.path.splitext(os.path.basename(audio_path))[0] or datetime.now().strftime("%Y%m%d%H%M%S")

    return parsed

# 直接运行
if __name__ == "__main__":
    result = process_audio(AUDIO_PATH)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if OUTPUT_PATH:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n已保存：{OUTPUT_PATH}")