# -*- coding: utf-8 -*-
"""
把《对比文本.txt》转换为 CSV（列名符合当前 pipeline）：
ID,msg_type,emotions,transcript_raw,transcript_clean,summary_ref,emphasized_words,reasons

使用：
python tools/convert_compare_text_to_csv.py \
  --input data/对比文本.txt \
  --output data/compare_text.csv \
  --default_emotion 中性

说明：
- 自动识别大类：任务类→task，闲聊类→chitchat，通知类→notice
- 自动识别子项（①②③…）及其正文（直到下一个子项或新大类/小节）
- 默认 emotions=中性（可通过 --default_emotion 修改）
- transcript_clean 先与 transcript_raw 相同（后续你可自行做去口语化）
- emphasized_words、summary_ref、reasons 先留空（"" / "[]")
"""
import argparse
import csv
import os
import re
import sys

CIRCLED_NUMS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
CIRCLED_PATTERN = re.compile(rf"^[{CIRCLED_NUMS}]（?.*?）?$")  # 例如：①（朋友邀约）
SECTION_MAP = {
    "任务类": "task",
    "闲聊类": "chitchat",
    "通知类": "notice",
}
# 可选：识别“短消息/长消息”的小节，仅做记录（不影响 msg_type）
SUBSECTION_HINTS = ["短消息", "长消息", "长语音"]

def read_text(path):
    # 尝试多种编码，容错
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    print("无法读取文本文件，请确认编码。", file=sys.stderr)
    sys.exit(1)

def normalize_line(s: str) -> str:
    return s.strip().replace("\u3000", " ").replace("\xa0", " ")

def is_section_header(line: str):
    return any(k in line for k in SECTION_MAP.keys())

def which_section(line: str):
    for k, v in SECTION_MAP.items():
        if k in line:
            return v
    return None

def is_subsection_header(line: str):
    return any(h in line for h in SUBSECTION_HINTS)

def is_circled_item_header(line: str):
    # 形如：①（朋友邀约）
    l = normalize_line(line)
    # 容宽匹配：以①②…开头即可
    return len(l) > 0 and l[0] in CIRCLED_NUMS

def split_items(lines):
    """
    根据“① ② …”拆分条目，每个条目为 (title_line, [paragraph_lines])
    """
    items = []
    current_title = None
    current_block = []
    for line in lines:
        l = normalize_line(line)
        if not l:
            # 空行直接入块（可能是段落换行）
            if current_title is not None:
                current_block.append("")
            continue

        if is_circled_item_header(l):
            # 开启新条目前，把上一个收尾
            if current_title is not None:
                items.append((current_title, current_block))
            current_title = l
            current_block = []
        else:
            # 非条目标题行
            if current_title is not None:
                current_block.append(l)
    # 收尾
    if current_title is not None:
        items.append((current_title, current_block))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="原始对比文本 .txt")
    ap.add_argument("--output", required=True, help="输出 CSV 路径")
    ap.add_argument("--default_emotion", default="中性", help="默认情感标签")
    args = ap.parse_args()

    raw = read_text(args.input)
    # 按行切
    all_lines = [normalize_line(x) for x in raw.splitlines()]

    rows = []
    current_section = None  # task/chitchat/notice
    current_subsection = None  # 短消息/长消息/长语音（可选记录）
    buffer_for_items = []

    def flush_buffer():
        # 将 buffer_for_items 解析为若干条目
        nonlocal rows, current_section, current_subsection
        if not buffer_for_items:
            return
        items = split_items(buffer_for_items)
        for title, paras in items:
            transcript = "\n".join([p for p in paras if p is not None]).strip()
            if not transcript:
                continue
            rows.append({
                "ID": None,  # 之后统一编号
                "msg_type": current_section or "",
                "emotions": args.default_emotion,
                "transcript_raw": transcript,
                "transcript_clean": transcript,  # 先与 raw 相同
                "summary_ref": "",
                "emphasized_words": "[]",
                "reasons": "",
                # 可选元数据（便于溯源；不在你现有 schema 中，可忽略）
                "_title": title,
                "_subsection": current_subsection or "",
            })
        buffer_for_items.clear()

    for line in all_lines:
        if not line:
            # 空行，照常进入条目缓存
            if buffer_for_items is not None:
                buffer_for_items.append(line)
            continue

        # 新的大类（任务类/闲聊类/通知类）
        if is_section_header(line):
            # 先把上一个小节里的条目写出
            flush_buffer()
            current_section = which_section(line)
            current_subsection = None
            continue

        # 新的小节（短消息/长消息/长语音）
        if is_subsection_header(line):
            flush_buffer()
            current_subsection = line
            continue

        # 普通内容，进入条目缓存
        buffer_for_items.append(line)

    # 末尾收尾
    flush_buffer()

    # 统一编号
    for idx, r in enumerate(rows, start=1):
        r["ID"] = idx

    # 写 CSV
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["ID","msg_type","emotions","transcript_raw","transcript_clean",
                  "summary_ref","emphasized_words","reasons"]
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"已写出 {len(rows)} 条到 {args.output}")
    print("提示：若有对应音频，可在 CSV 里新增 audio_path 列后续再对齐。")

if __name__ == "__main__":
    main()