import json
import csv
from rouge_score import rouge_scorer

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_reference(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def main():
    baseline = load_jsonl("baseline_results.jsonl")
    reference = {row["ID"]: row for row in load_reference("CHI智能耳机_数据表.csv")}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge_scores = []
    emotion_match = 0
    count = 0

    for item in baseline:
        ref = reference.get(item["ID"])
        if not ref:
            continue
        count += 1
        # ROUGE
        r = scorer.score(ref["summary_ref"], item["summary"])
        rouge_scores.append(r)
        # Emotion match
        if item.get("emotion") == ref.get("emotions"):
            emotion_match += 1

    avg_rouge1 = sum(s["rouge1"].fmeasure for s in rouge_scores) / len(rouge_scores)
    avg_rougeL = sum(s["rougeL"].fmeasure for s in rouge_scores) / len(rouge_scores)
    emotion_acc = emotion_match / count

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