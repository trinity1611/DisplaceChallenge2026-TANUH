import os
import yaml
import pandas as pd
from tqdm import tqdm

from model import load_qwen
from predictor import predict_topic
from evaluator import semantic_prf_with_scores, compute_final_metrics
from llama_translator import LlamaTranslator


def read_asr_text(asr_dir, pattern, rec_id):
    """
    Read ASR transcription file for a given recording ID.
    """
    file_path = os.path.join(asr_dir, pattern.format(rec_id=rec_id))
    if not os.path.exists(file_path):
        print(f"[WARNING] ASR file not found: {file_path}")
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_prediction(output_dir, pattern, rec_id, text):
    """
    Save predicted topics to a file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, pattern.format(rec_id=rec_id))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text.strip())


def main():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    tokenizer, model = load_qwen(
        base_model=config["model"]["base_model"],
        adapter_model=config["model"]["adapter_model"],
    )

    translator = None
    if config["translation"]["enabled"]:
        translator = LlamaTranslator(config["translation"])

    gt_path = config["paths"]["gt_path"]
    df_gt = pd.read_csv(gt_path)

    rec_id_col = config["columns"]["rec_id"]
    gt_col = config["columns"]["gt_summary"]

    df_gt[gt_col] = (
        df_gt[gt_col]
        .astype(str)
        .str.lower()
        .str.replace("&", ",", regex=False)
        .str.split(",")
    )
    df_gt[gt_col] = df_gt[gt_col].apply(
        lambda topics: [t.strip() for t in topics if t.strip()]
    )

    asr_dir = config["paths"]["input_path"]
    topic_pred_dir = config["paths"]["topic_pred_dir"]
    metrics_dir = config["paths"]["metrics_dir"]
    asr_pattern = config["paths"]["asr_pattern"]
    topic_pred_pattern = config["paths"]["topic_pred_pattern"]

    os.makedirs(metrics_dir, exist_ok=True)

    total_TP = total_FP = total_FN = 0
    total_tp_score = total_tp_count = 0
    total_accuracy = 0

    print("\nRunning Medical Topic Identification...\n")


    for idx, row in tqdm(df_gt.iterrows(), total=len(df_gt)):

        rec_id = str(row[rec_id_col])
        gt_topics = row[gt_col]

        conversation_text = read_asr_text(asr_dir, asr_pattern, rec_id)
        if conversation_text.strip() == "":
            print(f"[WARNING] Empty ASR text for {rec_id}")
            continue
        
        if translator is not None:
            conversation_text = translator.translate(conversation_text)

        pred_text = predict_topic(
            text=conversation_text,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=config["model"]["max_new_tokens"],
            temperature=config["translation"]["inference"]["temperature"],
        )

        pred_topics = [x.strip() for x in pred_text.split(",") if x.strip()]

        save_prediction(topic_pred_dir, topic_pred_pattern, rec_id, pred_text)

    
        metrics, matches, tp_score, tp_cnt, acc = semantic_prf_with_scores(
            gt_texts=gt_topics,
            pred_texts=pred_topics,
        )

        total_TP += metrics["TP"]
        total_FP += metrics["FP"]
        total_FN += metrics["FN"]
        total_tp_score += tp_score
        total_tp_count += tp_cnt
        total_accuracy += acc

        if idx < 3:
            print("\nREC ID:", rec_id)
            print("GT:", gt_topics)
            print("PR:", pred_topics)
            print("Metrics:", metrics)
            print("Matches:")
            for m in matches:
                print(m)
            print("-" * 60)

    final_metrics = compute_final_metrics(
        total_TP=total_TP,
        total_FP=total_FP,
        total_FN=total_FN,
        total_tp_score=total_tp_score,
        total_tp_count=total_tp_count,
        total_accuracy=total_accuracy,
        num_samples=len(df_gt),
    )

    metrics_path = os.path.join(metrics_dir, "final_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in final_metrics.items():
            f.write(f"{k}: {v}\n")

    
    print("\n================= FINAL EVALUATION =================")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")
    print("===================================================")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
