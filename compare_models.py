# -*- coding: utf-8 -*-
"""
统一对比评估脚本：OpenAI vs Qwen
- 支持有参考摘要（summary_ref）与无参考两种数据源
- 评估总摘要（overall_summary）与分段摘要（segments 拼接）
- 兼容 ID 异常、BOM 列名、类型不一致、缺行等问题
- 额外在导出的 CSV 中同时保存分段摘要（JSON 串 + 拼接文本）
- 新增 --expand_segments：将分段摘要平铺为多列，便于 Excel 浏览

用法示例：

# 1) 有参考摘要（建议科研评估）
python compare_models.py \
  --openai_jsonl baseline_segments_openai.jsonl \
  --qwen_jsonl   baseline_segments_qwen.jsonl \
  --csv          data/CHI智能耳机_数据表.csv \
  --output_csv   model_comparison_results_detailed.csv \
  --fig          model_comparison_chart.png \
  --expand_segments

# 2) 无参考摘要（compare_text.csv），仅导出对照（含分段展开）
python compare_models.py \
  --openai_jsonl baseline_segments_openai.jsonl \
  --qwen_jsonl   baseline_segments_qwen.jsonl \
  --csv          data/compare_text.csv \
  --output_csv   model_comparison_results_no_ref.csv \
  --expand_segments
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def lazy_load_metrics():
    """仅在有参考摘要时加载评估工具。"""
    try:
        import evaluate
        from bert_score import score as bert_score
        rouge_metric = evaluate.load("rouge")
        return rouge_metric, bert_score
    except Exception as e:
        print(f"警告：无法加载评估工具（evaluate/bert_score）：{e}\n"
              f"将跳过自动指标计算，仅输出两模型结果对照。")
        return None, None

def load_jsonl_dict(path: str) -> Dict[str, dict]:
    """读取 JSONL → dict: id(str) -> payload(dict)；要求行内含 ID 字段。"""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"警告：{path} 第{ln}行 JSON 解析失败，已跳过")
                continue
            if "ID" not in obj:
                print(f"警告：{path} 第{ln}行缺少 ID 字段，已跳过")
                continue
            data[str(obj["ID"])] = obj
    return data

def normalize_id_series(s: pd.Series) -> pd.Series:
    """将 ID 列统一为字符串，并去除可能存在的小数点尾巴等。"""
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

def pick_id_column(df: pd.DataFrame) -> str:
    """自动选择 ID 列名，兼容 BOM：\ufeffID。"""
    if "ID" in df.columns:
        return "ID"
    for c in df.columns:
        if c.replace("\ufeff", "") == "ID":
            return c
    raise ValueError(f"CSV 中未找到 ID 列，当前列名：{df.columns.to_list()}")

def has_summary_ref(df: pd.DataFrame) -> bool:
    return "summary_ref" in df.columns and df["summary_ref"].notna().any()

def extract_overall_text(d: dict) -> str:
    """取 overall_summary（优先）或 summary 字段。"""
    if "overall_summary" in d and isinstance(d["overall_summary"], str):
        return d["overall_summary"]
    if "summary" in d and isinstance(d["summary"], str):
        return d["summary"]
    return ""

def concat_segment_summaries(d: dict) -> str:
    """将 segments 中的 segment_summary 依次拼成一段文本（用于评估/对照）。"""
    segs = d.get("segments", [])
    if not isinstance(segs, list):
        return ""
    parts = []
    for seg in segs:
        txt = seg.get("segment_summary", "")
        if isinstance(txt, str) and txt.strip():
            parts.append(txt.strip())
    return " ".join(parts)

def compute_rouge(rouge_metric, refs: List[str], hyps: List[str]) -> Tuple[float, float]:
    if not refs or not hyps:
        return np.nan, np.nan
    result = rouge_metric.compute(predictions=hyps, references=refs)
    r1 = float(result["rouge1"].mid.fmeasure) if "rouge1" in result else np.nan
    rL = float(result["rougeL"].mid.fmeasure) if "rougeL" in result else np.nan
    return r1, rL

def compute_bertscore(bert_score_fn, refs: List[str], hyps: List[str], lang="zh") -> float:
    if not refs or not hyps:
        return np.nan
    try:
        P, R, F1 = bert_score_fn(hyps, refs, lang=lang)
        return float(F1.mean())
    except Exception:
        return np.nan

def expand_segments_to_columns(payload: dict, prefix: str, max_segments: int = 5) -> dict:
    """
    将 segments 平铺为若干列：{prefix}_segment_1_summary / _fragment / _time_range ...
    不足的段落留空；多于 max_segments 的截断。
    """
    out = {}
    segs = payload.get("segments", [])
    if not isinstance(segs, list):
        segs = []
    for idx in range(max_segments):
        col_base = f"{prefix}_segment_{idx+1}"
        if idx < len(segs):
            seg = segs[idx] or {}
            out[f"{col_base}_summary"] = seg.get("segment_summary", "")
            out[f"{col_base}_fragment"] = seg.get("original_fragment", "")
            out[f"{col_base}_time_range"] = seg.get("time_range", "")
        else:
            out[f"{col_base}_summary"] = ""
            out[f"{col_base}_fragment"] = ""
            out[f"{col_base}_time_range"] = ""
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openai_jsonl", required=True, help="OpenAI 结果 JSONL")
    ap.add_argument("--qwen_jsonl", required=True, help="Qwen 结果 JSONL")
    ap.add_argument("--csv", required=True, help="原始 CSV（有无 summary_ref 均可）")
    ap.add_argument("--output_csv", required=True, help="导出的对照/评估明细 CSV")
    ap.add_argument("--fig", default="", help="若提供文件名，则在有参考时生成对比图（png）")
    ap.add_argument("--expand_segments", action="store_true",
                    help="将两模型的分段摘要平铺为多列（最多 5 段）")
    ap.add_argument("--max_segments", type=int, default=5, help="平铺的最大段数（默认 5）")
    args = ap.parse_args()

    # 载入两模型结果
    openai_map = load_jsonl_dict(args.openai_jsonl)
    qwen_map   = load_jsonl_dict(args.qwen_jsonl)

    # 载入 CSV
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    id_col = pick_id_column(df)
    df[id_col] = normalize_id_series(df[id_col])

    # （可选）参考摘要
    ref_available = has_summary_ref(df)
    if ref_available:
        df = df.copy()
        df["summary_ref"] = df["summary_ref"].fillna("").astype(str)
        df = df[df["summary_ref"].str.strip() != ""]
        if df.empty:
            ref_available = False

    # 汇总公共 ID（两模型都要有）
    common_ids = sorted(set(openai_map.keys()) & set(qwen_map.keys()))
    if not common_ids:
        print("错误：两模型结果无公共 ID，请确认 JSONL 是否来自同一批数据。")
        sys.exit(1)

    # 对齐到 CSV（仅用于读取参考摘要 & 额外元数据）
    csv_ids = set(df[id_col].tolist())
    if ref_available:
        ids_for_eval = [cid for cid in common_ids if cid in csv_ids]
        if not ids_for_eval:
            print("警告：虽有参考摘要，但与模型结果无公共 ID；将跳过自动评估，仅导出结果对照。")
            ref_available = False
    else:
        ids_for_eval = []

    print(f"公共 ID 数量：{len(common_ids)}；可评估 ID 数量：{len(ids_for_eval)}（有参考时）")

    # === 构建“对照明细表”数据（无论是否有参考，都会输出） ===
    rows_for_csv = []
    for cid in common_ids:
        # 取 CSV 匹配行（若存在）
        ref_text = ""
        extra = {}
        if cid in csv_ids:
            row_ref = df[df[id_col] == cid].iloc[0]
            if "summary_ref" in row_ref:
                ref_text = str(row_ref["summary_ref"])
            # 额外元数据（若存在）
            for meta_key in ["msg_type", "emotions", "transcript_clean", "transcript_raw"]:
                if meta_key in df.columns:
                    extra[meta_key] = row_ref.get(meta_key, "")

        o_payload = openai_map[cid]
        q_payload = qwen_map[cid]

        base_row = {
            "ID": cid,
            "summary_ref": ref_text,
            "overall_openai": extract_overall_text(o_payload),
            "overall_qwen":   extract_overall_text(q_payload),
            # 保存 segments JSON 串（原样）
            "segments_openai_json": json.dumps(o_payload.get("segments", []), ensure_ascii=False),
            "segments_qwen_json":   json.dumps(q_payload.get("segments", []), ensure_ascii=False),
            # 保存 segments 拼接文本（用于阅读/评估）
            "segments_openai_concat": concat_segment_summaries(o_payload),
            "segments_qwen_concat":   concat_segment_summaries(q_payload),
            **extra
        }

        # 可选：平铺 segments 为多列
        if args.expand_segments:
            base_row.update(expand_segments_to_columns(o_payload, "openai", max_segments=args.max_segments))
            base_row.update(expand_segments_to_columns(q_payload, "qwen",   max_segments=args.max_segments))

        rows_for_csv.append(base_row)

    df_out = pd.DataFrame(rows_for_csv)
    df_out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"已导出对照明细：{args.output_csv}")

    # === 若无参考摘要，至此结束 ===
    if not ref_available:
        print("未检测到参考摘要（summary_ref），跳过自动指标计算。\n"
              "建议：进行人工评估或补充参考摘要后再运行本脚本以获得量化对比。")
        return

    # === 存在参考摘要：进行自动评估 ===
    rouge_metric, bert_score_fn = lazy_load_metrics()
    if rouge_metric is None or bert_score_fn is None:
        print("评估工具不可用，跳过自动指标计算。")
        return

    # 仅对可评估 ID 计算指标
    refs_overall, hyp_openai_overall, hyp_qwen_overall = [], [], []
    refs_segments, hyp_openai_segments, hyp_qwen_segments = [], [], []

    for cid in ids_for_eval:
        ref_row = df[df[id_col] == cid].iloc[0]
        ref_text = str(ref_row["summary_ref"]).strip()

        o_payload = openai_map.get(cid, {})
        q_payload = qwen_map.get(cid, {})

        # 总摘要
        refs_overall.append(ref_text)
        hyp_openai_overall.append(extract_overall_text(o_payload))
        hyp_qwen_overall.append(extract_overall_text(q_payload))

        # 分段摘要（拼接）
        refs_segments.append(ref_text)
        hyp_openai_segments.append(concat_segment_summaries(o_payload))
        hyp_qwen_segments.append(concat_segment_summaries(q_payload))

    # 计算指标：总体
    o_r1, o_rL = compute_rouge(rouge_metric, refs_overall, hyp_openai_overall)
    q_r1, q_rL = compute_rouge(rouge_metric, refs_overall, hyp_qwen_overall)
    o_bs = compute_bertscore(bert_score_fn, refs_overall, hyp_openai_overall, lang="zh")
    q_bs = compute_bertscore(bert_score_fn, refs_overall, hyp_qwen_overall, lang="zh")

    print("\n=== 总摘要指标（overall_summary） ===")
    print(f"OpenAI: ROUGE-1={o_r1:.4f}, ROUGE-L={o_rL:.4f}, BERTScore={o_bs:.4f}")
    print(f"Qwen  : ROUGE-1={q_r1:.4f}, ROUGE-L={q_rL:.4f}, BERTScore={q_bs:.4f}")

    # 计算指标：分段拼接
    os_r1, os_rL = compute_rouge(rouge_metric, refs_segments, hyp_openai_segments)
    qs_r1, qs_rL = compute_rouge(rouge_metric, refs_segments, hyp_qwen_segments)
    os_bs = compute_bertscore(bert_score_fn, refs_segments, hyp_openai_segments, lang="zh")
    qs_bs = compute_bertscore(bert_score_fn, refs_segments, hyp_qwen_segments, lang="zh")

    print("\n=== 分段摘要指标（segments 拼接） ===")
    print(f"OpenAI: ROUGE-1={os_r1:.4f}, ROUGE-L={os_rL:.4f}, BERTScore={os_bs:.4f}")
    print(f"Qwen  : ROUGE-1={qs_r1:.4f}, ROUGE-L={qs_rL:.4f}, BERTScore={qs_bs:.4f}")

    # 绘图（可选）
    if args.fig:
        try:
            models = ["OpenAI", "Qwen"]
            w = 0.25

            # overall
            plt.figure(figsize=(9, 5))
            x = np.arange(len(models))
            plt.bar(x - w, [o_r1, q_r1], width=w, label="ROUGE-1 (overall)")
            plt.bar(x,     [o_rL, q_rL], width=w, label="ROUGE-L (overall)")
            plt.bar(x + w, [o_bs, q_bs], width=w, label="BERTScore (overall)")
            plt.xticks(x, models)
            plt.ylabel("Score")
            plt.title("OpenAI vs Qwen（Overall Summary）")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.fig.replace(".png", "_overall.png"), dpi=200)

            # segments concat
            plt.figure(figsize=(9, 5))
            x = np.arange(len(models))
            plt.bar(x - w, [os_r1, qs_r1], width=w, label="ROUGE-1 (segments)")
            plt.bar(x,     [os_rL, qs_rL], width=w, label="ROUGE-L (segments)")
            plt.bar(x + w, [os_bs, qs_bs], width=w, label="BERTScore (segments)")
            plt.xticks(x, models)
            plt.ylabel("Score")
            plt.title("OpenAI vs Qwen（Segments Concat）")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.fig.replace(".png", "_segments.png"), dpi=200)

            print(f"\n已保存图表：{args.fig.replace('.png', '_overall.png')} 与 {args.fig.replace('.png', '_segments.png')}")
        except Exception as e:
            print(f"绘图失败（已忽略）：{e}")

    # 路由建议（仅在有参考指标时）
    print("\n=== 路由建议（基于 BERTScore-Overall） ===")
    if np.isnan(o_bs) or np.isnan(q_bs):
        print("无法生成建议：BERTScore 计算失败或缺失。")
    else:
        gap = q_bs - o_bs
        if gap > 0.02:
            print("建议：优先使用 Qwen；若成本敏感可仅在长文本/任务类调用。")
        elif gap < -0.02:
            print("建议：优先使用 OpenAI；Qwen 可作为备选或特定场景（如通知类）。")
        else:
            print("建议：两模型表现接近，可按成本/延迟选择，或引入任务类型/长度路由。")

if __name__ == "__main__":
    main()