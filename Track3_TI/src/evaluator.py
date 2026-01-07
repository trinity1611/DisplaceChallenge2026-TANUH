import numpy as np
from bert_score import score as bert_score


def semantic_prf_with_scores(
    gt_texts,
    pred_texts,
    threshold: float = 0.85,
    bert_model_type: str = "roberta-base"
):
    """
    Computes TP / FP / FN using BERTScore-F1
    Computes PARTIAL per-sample accuracy (recall-based)

    Returns:
        metrics (dict)
        matches (list)
        tp_score_sum (float)
        tp_count (int)
        accuracy (float in [0, 1])
    """

    
    if len(gt_texts) == 0 and len(pred_texts) == 0:
        return _empty_metrics(), [], 0.0, 0, 1.0

    if len(pred_texts) == 0:
        return {
            "TP": 0,
            "FP": 0,
            "FN": len(gt_texts),
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
        }, [], 0.0, 0, 0.0

    if len(gt_texts) == 0:
        return {
            "TP": 0,
            "FP": len(pred_texts),
            "FN": 0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
        }, [], 0.0, 0, 0.0

    
    bert_f1_matrix = np.zeros((len(pred_texts), len(gt_texts)))

    for i, pred in enumerate(pred_texts):
        _, _, f1 = bert_score(
            [pred] * len(gt_texts),
            gt_texts,
            model_type=bert_model_type,
            verbose=False
        )
        bert_f1_matrix[i] = f1.numpy()

    matched_gt = set()
    TP = 0
    matches = []

    tp_score_sum = 0.0
    tp_count = 0


    for pred_idx, pred_text in enumerate(pred_texts):
        best_gt_idx = int(np.argmax(bert_f1_matrix[pred_idx]))
        best_score = float(bert_f1_matrix[pred_idx][best_gt_idx])
        best_gt_text = gt_texts[best_gt_idx]

        if best_score >= threshold and best_gt_idx not in matched_gt:
            TP += 1
            matched_gt.add(best_gt_idx)
            match_type = "TP"
            tp_score_sum += best_score
            tp_count += 1
        else:
            match_type = "FP"

        matches.append({
            "prediction": pred_text,
            "matched_gt": best_gt_text,
            "bertscore_f1": round(best_score, 3),
            "type": match_type
        })

    FP = len(pred_texts) - TP
    FN = len(gt_texts) - TP

    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

   
    accuracy = TP / len(gt_texts) if len(gt_texts) > 0 else 0.0

    metrics = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
    }

    return metrics, matches, tp_score_sum, tp_count, round(accuracy, 3)


def compute_final_metrics(
    total_TP,
    total_FP,
    total_FN,
    total_tp_score,
    total_tp_count,
    total_accuracy,
    num_samples
):
    """
    Aggregates metrics across all samples
    """

    precision = (
        total_TP / (total_TP + total_FP)
        if (total_TP + total_FP) > 0 else 0.0
    )

    recall = (
        total_TP / (total_TP + total_FN)
        if (total_TP + total_FN) > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    avg_tp_score = (
        total_tp_score / total_tp_count
        if total_tp_count > 0 else 0.0
    )

    avg_accuracy = total_accuracy / num_samples if num_samples > 0 else 0.0

    return {
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
        "Avg_TP_BERTScore": round(avg_tp_score, 3),
        "Accuracy": round(avg_accuracy, 3),
    }


def _empty_metrics():
    return {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
    }
