# summarizer/segmenter.py
# -*- coding: utf-8 -*-
"""
提供四部分能力：
1) try_asr_with_segments：调用 OpenAI ASR，返回全文 text + 每段的 {start,end,text}
2) vad_segments：基于 webrtcvad 或能量阈值的端点检测（可选）
3) equal_splits / allocate_text_by_spans：均分兜底 & 按时间片比例分配 transcript
4) build_segment_prompt：把片段清单（带时间戳+原文片段）附加到提示词末尾

注意：
- OpenAI 新版 SDK 的 ASR 返回 TranscriptionVerbose 对象，不是 dict，因此不能 .get()。
  此处统一转换为 Python dict，后续流程保持不变。
"""

from __future__ import annotations
import io
import math
from typing import List, Dict, Tuple

# 允许“按需依赖”，没装就优雅降级
try:
    import webrtcvad
except Exception:
    webrtcvad = None

try:
    import librosa
    import numpy as np
except Exception:
    librosa = None
    np = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None


# ===================== 工具：时间格式 =====================

def s2mmss(sec: float) -> str:
    """秒 -> mm:ss，四舍五入到最近的秒"""
    sec = max(0, float(sec))
    m = int(sec // 60)
    s = int(round(sec - 60 * m))
    if s == 60:
        m += 1
        s = 0
    return f"{m:02d}:{s:02d}"


# ===================== 1) ASR：拿文本 + 分段 =====================
# 保底别名：旧代码调用 try_asr_with_segments 也能跑

def try_asr_with_segments(client, audio_path: str, asr_model: str) -> Tuple[str, List[Dict]]:
    """
    调用 OpenAI ASR 拿文本 + 分段（若支持）
    返回 (全文文本, [ {start, end, text}, ... ])
    兼容新版 SDK：TranscriptionVerbose / TranscriptionSegment 对象
    """
    try:
        with open(audio_path, "rb") as f:
            # 优先请求 verbose_json（含 segments）
            tr = client.audio.transcriptions.create(
                model=asr_model,
                file=f,
                response_format="verbose_json"
            )
    except Exception as e:
        # 某些代理或旧接口可能不支持 verbose_json，则退回 text
        print(f"[ASR segments] 失败：{e}")
        try:
            with open(audio_path, "rb") as f2:
                tr2 = client.audio.transcriptions.create(
                    model=asr_model,
                    file=f2,
                    response_format="text"
                )
            text_only = tr2 if isinstance(tr2, str) else getattr(tr2, "text", "")
            return text_only, []
        except Exception as e2:
            print(f"[ASR] 失败：{e2}")
            return "", []

    # 解析 TranscriptionVerbose -> 统一成 dict list
    try:
        text = getattr(tr, "text", "")
        segs_obj = getattr(tr, "segments", []) or []
        segs: List[Dict] = []
        for seg in segs_obj:
            # seg 是 TranscriptionSegment 对象
            segs.append({
                "start": float(getattr(seg, "start", 0.0)),
                "end": float(getattr(seg, "end", 0.0)),
                "text": getattr(seg, "text", "")
            })
        return text, segs
    except Exception as e:
        print(f"[ASR segments] 解析失败：{e}")
        return getattr(tr, "text", ""), []


# ===================== 2) 分段策略：VAD =====================

def _vad_with_webrtcvad(audio: AudioSegment, frame_ms: int = 30, aggressiveness: int = 2) -> List[Tuple[float, float]]:
    """
    使用 webrtcvad 对 AudioSegment 做端点检测，返回 [(start_sec, end_sec), ...]
    - 需要 16k、单声道、16-bit PCM
    """
    if webrtcvad is None:
        return []

    wav = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    raw = wav.raw_data
    vad = webrtcvad.Vad(aggressiveness)

    sample_rate = 16000
    bytes_per_sample = 2
    frame_size = int(sample_rate * frame_ms / 1000) * bytes_per_sample

    frames = [raw[i:i + frame_size] for i in range(0, len(raw), frame_size)]
    voiced = [vad.is_speech(f, sample_rate) if len(f) == frame_size else False for f in frames]

    spans = []
    in_voiced = False
    start_idx = 0
    for i, v in enumerate(voiced):
        if v and not in_voiced:
            in_voiced = True
            start_idx = i
        if not v and in_voiced:
            in_voiced = False
            end_idx = i
            spans.append((start_idx * frame_ms / 1000.0, end_idx * frame_ms / 1000.0))
    if in_voiced:
        end_idx = len(voiced)
        spans.append((start_idx * frame_ms / 1000.0, end_idx * frame_ms / 1000.0))

    merged = []
    for st, ed in spans:
        if not merged:
            merged.append([st, ed])
        else:
            if st - merged[-1][1] < 0.25:
                merged[-1][1] = ed
            else:
                merged.append([st, ed])
    return [(round(x[0], 2), round(x[1], 2)) for x in merged]


def _vad_energy_based(audio: AudioSegment, win_ms: int = 30, hop_ms: int = 15, thr_db: float = -40.0) -> List[Tuple[float, float]]:
    """
    简易能量阈值法（当 webrtcvad 不可用时）。需要 librosa & numpy。
    """
    if librosa is None or np is None:
        return []

    sr = 16000
    mono = audio.set_channels(1).set_frame_rate(sr)
    samples = np.frombuffer(mono.raw_data, dtype=np.int16).astype(np.float32) / 32768.0

    win = int(sr * win_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    n_frames = max(1, 1 + (len(samples) - win) // hop)

    spans = []
    in_voiced = False
    start_t = 0.0
    eps = 1e-8
    for i in range(n_frames):
        st = i * hop
        ed = min(len(samples), st + win)
        frame = samples[st:ed]
        rms = np.sqrt(np.mean(frame ** 2) + eps)
        db = 20 * np.log10(rms + eps)
        is_voiced = db > thr_db
        t_st = st / sr
        t_ed = ed / sr
        if is_voiced and not in_voiced:
            in_voiced = True
            start_t = t_st
        if (not is_voiced) and in_voiced:
            in_voiced = False
            spans.append((start_t, t_ed))
    if in_voiced:
        spans.append((start_t, len(samples) / sr))

    merged = []
    for st, ed in spans:
        if not merged:
            merged.append([st, ed])
        else:
            if st - merged[-1][1] < 0.25:
                merged[-1][1] = ed
            else:
                merged.append([st, ed])
    return [(round(x[0], 2), round(x[1], 2)) for x in merged]


def vad_segments(audio_path: str) -> List[Dict]:
    """
    返回 [{'start': float, 'end': float}]；失败时返回 []
    """
    if AudioSegment is None:
        return []
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception:
        return []

    spans = _vad_with_webrtcvad(audio)
    if not spans:
        spans = _vad_energy_based(audio)

    spans = [(st, ed) for (st, ed) in spans if (ed - st) > 0.4]
    return [{"start": float(st), "end": float(ed)} for (st, ed) in spans]


# ===================== 3) 均分 / 比例分配 =====================

def equal_splits(transcript: str, total_sec: float, n_segments: int) -> List[Dict]:
    """
    把全文按字符均分成 n 段，时间也均分。
    """
    text = transcript.strip()
    if not text:
        return []
    n_segments = max(1, int(n_segments))

    L = len(text)
    cuts = [round(i * L / n_segments) for i in range(n_segments + 1)]
    segtime = total_sec / n_segments if total_sec and total_sec > 0 else 0.0

    payload = []
    for i in range(n_segments):
        start_t = i * segtime
        end_t = (i + 1) * segtime
        frag = text[cuts[i]:cuts[i + 1]].strip()
        if not frag:
            continue
        payload.append({
            "time_range": f"{s2mmss(start_t)}-{s2mmss(end_t)}",
            "segment_summary": "",
            "original_fragment": frag
        })
    return payload


def allocate_text_by_spans(transcript: str, spans: List[Dict]) -> List[Dict]:
    """
    有 spans（start/end 秒）但没有词级时间戳：按“时间长度比例”把 transcript 均匀分配到每个 span。
    """
    text = transcript.strip()
    if not text or not spans:
        return []

    clean_spans = [(float(s["start"]), float(s["end"])) for s in spans if float(s["end"]) > float(s["start"])]
    if not clean_spans:
        return []

    total_dur = sum(ed - st for (st, ed) in clean_spans)
    if total_dur <= 0:
        return equal_splits(transcript, 0.0, len(clean_spans))

    L = len(text)
    payload = []
    pos = 0
    for idx, (st, ed) in enumerate(clean_spans):
        ratio = (ed - st) / total_dur
        take = int(round(L * ratio))
        frag = text[pos:pos + take].strip()
        pos += take
        if idx == len(clean_spans) - 1:
            frag = (text[pos - take:]).strip() if frag == "" else (frag + " " + text[pos:]).strip()
        payload.append({
            "time_range": f"{s2mmss(st)}-{s2mmss(ed)}",
            "segment_summary": "",
            "original_fragment": frag
        })
    payload = [p for p in payload if p["original_fragment"]]
    return payload

# ============== 段合并工具：把多段合并为 2/3 段 ==============
def _parse_mmss(mmss: str) -> float:
    """把 'MM:SS' 或 'H:MM:SS' 转秒，失败返回 0."""
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
        else:
            return 0.0
    except Exception:
        return 0.0


def coalesce_segments(segments_payload: List[Dict], n_target: int) -> List[Dict]:
    """
    把很多细碎的 segments_payload 合并成 n_target 段（2 或 3 段）
    优先用 time_range 的时长做均衡；若没有有效时长，就按文本长度均衡。
    """
    segments = [s for s in segments_payload if s.get("original_fragment")]
    if not segments:
        return []

    n_target = max(1, int(n_target))
    if len(segments) <= n_target:
        # 段数本来就 <= 目标，直接返回（不足时由上层决定是否增补）
        return segments

    # 先尝试按时长合并
    times = []
    total_dur = 0.0
    for s in segments:
        tr = s.get("time_range", "")
        if "-" in tr:
            a, b = tr.split("-", 1)
            st = _parse_mmss(a)
            ed = _parse_mmss(b)
            dur = max(0.0, ed - st)
        else:
            st = ed = 0.0
            dur = 0.0
        times.append((st, ed, dur))
        total_dur += dur

    use_duration = total_dur > 0.0

    buckets = []
    cur_items = []
    cur_start = None
    cur_end = None
    cur_dur = 0.0
    cur_chars = 0
    target_dur = total_dur / n_target if use_duration else None
    total_chars = sum(len(s.get("original_fragment", "")) for s in segments)
    target_chars = total_chars / n_target if not use_duration else None

    def flush_bucket():
        nonlocal cur_items, cur_start, cur_end, cur_dur, cur_chars
        if not cur_items:
            return
        merged_text = " ".join(x["original_fragment"].strip() for x in cur_items if x.get("original_fragment"))
        # time_range：如果有时长，取首尾；否则保留第一段的开始和最后段的结束（可能为空）
        if use_duration and cur_start is not None and cur_end is not None and cur_end > cur_start:
            tr = f"{s2mmss(cur_start)}-{s2mmss(cur_end)}"
        else:
            # 回退：拼接第一个和最后一个的 time_range 边界字符串
            first_tr = cur_items[0].get("time_range", "")
            last_tr = cur_items[-1].get("time_range", "")
            if "-" in first_tr and "-" in last_tr:
                tr = f"{first_tr.split('-',1)[0]}-{last_tr.split('-',1)[1]}"
            else:
                tr = first_tr or last_tr or ""
        buckets.append({
            "time_range": tr,
            "segment_summary": "",
            "original_fragment": merged_text
        })
        # reset
        cur_items = []
        cur_start = None
        cur_end = None
        cur_dur = 0.0
        cur_chars = 0

    remain = len(segments)
    for idx, s in enumerate(segments):
        cur_items.append(s)
        if use_duration:
            st, ed, dur = times[idx]
            if cur_start is None or (st and st < cur_start):
                cur_start = st
            if cur_end is None or (ed and ed > cur_end):
                cur_end = ed
            cur_dur += dur
            remain -= 1
            # 贪心：够到目标就切，但要确保后面还能凑出剩余桶数
            if (len(buckets) < n_target - 1) and (cur_dur >= target_dur) and (remain >= (n_target - len(buckets) - 1)):
                flush_bucket()
        else:
            cur_chars += len(s.get("original_fragment", ""))
            remain -= 1
            if (len(buckets) < n_target - 1) and (cur_chars >= target_chars) and (remain >= (n_target - len(buckets) - 1)):
                flush_bucket()

    # 把尾巴刷掉
    flush_bucket()

    # 极端情况下可能多于 n_target（比如时长全为 0），截断
    if len(buckets) > n_target:
        # 合并最后几段到第 n_target 段
        head = buckets[:n_target-1]
        tail = buckets[n_target-1:]
        merged_tail_text = " ".join(x["original_fragment"] for x in tail if x.get("original_fragment"))
        # time_range 合并
        first_tr = head[-1]["time_range"] if head else (tail[0]["time_range"] if tail else "")
        last_tr = tail[-1]["time_range"] if tail else ""
        if "-" in first_tr and "-" in last_tr:
            left = first_tr.split("-", 1)[0]
            right = last_tr.split("-", 1)[1]
            tr = f"{left}-{right}"
        else:
            tr = tail[0]["time_range"] if tail else ""
        head.append({"time_range": tr, "segment_summary": "", "original_fragment": merged_tail_text})
        buckets = head

    return buckets
# ===================== 4) Prompt 组装：附加“片段清单” =====================

def build_segment_prompt(
    base_prompt_tmpl: str,
    msg_type: str,
    emotion: str,
    emphasis_json: str,
    cleaned_transcript: str,
    segments_payload: List[Dict]
) -> str:
    """
    - 主模板填入 emotion/emphasis/cleaned_transcript
    - 在结尾追加“片段清单（时间戳+原文片段）”，要求模型按顺序总结，但不要修改 time_range
    """
    try:
        main = base_prompt_tmpl.format(
            emotion=emotion,
            emphasis_words=emphasis_json,
            cleaned_transcript=cleaned_transcript
        )
    except KeyError:
        main = base_prompt_tmpl

    lines = ["", "【片段清单（供参考；不要修改时间范围；按顺序总结）】"]
    for i, seg in enumerate(segments_payload, 1):
        tr = seg.get("time_range", "")
        frag = seg.get("original_fragment", "").strip()
        if len(frag) > 500:
            frag = frag[:500] + " …"
        lines.append(f"- 段{i} | 时间：{tr}\n  原文片段：{frag}")

    tail_rule = """
请严格按上述“片段清单”的顺序生成 `segments`，并保持 `time_range` 不变。
每个 `segment_summary` 用 1~2 句概括；`original_fragment` 请尽量贴近清单中的原文，不要虚构。
仅返回 JSON，不要额外解释。
"""
    prompt = main + "\n" + "\n".join(lines) + "\n" + tail_rule
    return prompt