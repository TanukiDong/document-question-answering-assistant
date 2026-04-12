import json
import re
import string
from collections import Counter


PREFIX = "nq_384_256_3"
GOLD_FILE = "data/input_nq.json"
PRED_FILE = f"output/{PREFIX}.json"
ERROR_FILE = f"output/error/{PREFIX}_error.json"
DETAIL_FILE = f"output/eval/{PREFIX}_eval.json"
SAVE_OUTPUT = False

def normalize_text(s: str) -> str:
    if s is None:
        return ""

    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list) -> float:
    if not ground_truths:
        return metric_fn(prediction, "")
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def best_matching_gold(prediction: str, ground_truths: list):
    """
    Return the gold answer that gives the highest F1.
    If there is a tie, prefer the one with higher EM.
    """
    if not ground_truths:
        return "", 0, 0.0

    best_gold = ground_truths[0]
    best_em = exact_match_score(prediction, best_gold)
    best_f1 = f1_score(prediction, best_gold)

    for gt in ground_truths[1:]:
        em = exact_match_score(prediction, gt)
        f1 = f1_score(prediction, gt)

        if (f1 > best_f1) or (f1 == best_f1 and em > best_em):
            best_gold = gt
            best_em = em
            best_f1 = f1

    return best_gold, best_em, best_f1


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def evaluate(gold_file: str, pred_file: str, error_file: str, detail_file: str):
    gold_data = load_json(gold_file)
    pred_data = load_json(pred_file)

    gold_dict = {}
    question_dict = {}

    for item in gold_data:
        qid = item["question_id"]
        answers = item.get("golden_answers", [])
        if isinstance(answers, str):
            answers = [answers]

        gold_dict[qid] = answers
        question_dict[qid] = item.get("question", "")

    pred_dict = {}
    for item in pred_data:
        qid = item["question_id"]
        pred_dict[qid] = item.get("answer", "")

    total = len(gold_dict)
    if total == 0:
        raise ValueError("Gold file is empty.")

    em_sum = 0.0
    f1_sum = 0.0
    missing_predictions = []

    detailed_results = []
    error_analysis = []

    for qid, gold_answers in gold_dict.items():
        pred_answer = pred_dict.get(qid, "")
        question = question_dict.get(qid, "")

        if qid not in pred_dict:
            missing_predictions.append(qid)

        best_gold, em, f1 = best_matching_gold(pred_answer, gold_answers)

        em_sum += em
        f1_sum += f1

        result = {
            "question_id": qid,
            "question": question,
            "prediction": pred_answer,
            "golden_answers": gold_answers,
            "best_matching_gold": best_gold,
            "exact_match": int(em),
            "f1": round(f1, 4),
            "prediction_normalized": normalize_text(pred_answer),
            "best_gold_normalized": normalize_text(best_gold),
            "is_missing_prediction": qid not in pred_dict,
        }
        detailed_results.append(result)

        if em == 0 or f1 < 0.5:
            error_analysis.append(result)

    em_percent = 100.0 * em_sum / total
    f1_percent = 100.0 * f1_sum / total

    summary = {
        "total_questions": total,
        "missing_predictions": len(missing_predictions),
        "exact_match": round(em_percent, 2),
        "token_f1": round(f1_percent, 2),
    }

    if SAVE_OUTPUT:
        save_json(
            {
                "summary": summary,
                "errors": error_analysis,
            },
            error_file,
        )
        print(f"Saved error analysis to: {error_file}")

        save_json(
            {
                "summary": summary,
                "results": detailed_results,
            },
            detail_file,
        )
        print(f"Saved detailed results to: {detail_file}")

    print(f"Total questions: {total}")
    print(f"Missing predictions: {len(missing_predictions)}")
    print(f"Exact Match (EM): {em_percent:.2f}")
    print(f"Token-level F1:  {f1_percent:.2f}")

    return summary


if __name__ == "__main__":
    evaluate(GOLD_FILE, PRED_FILE, ERROR_FILE, DETAIL_FILE)