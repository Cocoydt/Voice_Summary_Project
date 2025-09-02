import json
import csv
import argparse
from rouge_score import rouge_scorer

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_reference(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="生成的 JSONL 文件")
    parser.add_argument("--reference", type=str, required=True, help="参考 CSV 文件")
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline)
    reference = {row["ID"]: row for row in load_reference(args.reference)}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge_scores = []
    emotion_match = 0
    count = 0

    for item in baseline:
        ref = reference.get(item["ID"])
        if not ref:
            continue
        count += 1
        r = scorer.score(ref["summary_ref"], item["summary"])
        rouge_scores.append(r)
        if item.get("emotion") == ref.get("emotions"):
            emotion_match += 1

    avg_rouge1 = (sum(s["rouge1"].fmeasure for s in rouge_scores) / len(rouge_scores)
                  if rouge_scores else 0)
    avg_rougeL = (sum(s["rougeL"].fmeasure for s in rouge_scores) / len(rouge_scores)
                  if rouge_scores else 0)
    emotion_acc = emotion_match / count if count else 0

    report = {
        "samples_evaluated": count,
        "avg_rouge1": avg_rouge1,
        "avg_rougeL": avg_rougeL,
        "emotion_accuracy": emotion_acc
    }

    with open("baseline_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== 质量评估完成 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()