from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_loader import build_account_graph, load_transactions


def evaluate(csv_path: str, predictions_path: str, output_path: str):
    df = load_transactions(csv_path)

    if "Is Laundering" not in df.columns:
        raise ValueError("CSV không có cột Is Laundering nên không thể đánh giá.")

    graph = build_account_graph(
        df,
        directed_edges=True,
    )

    pred_df = pd.read_csv(predictions_path)

    if "node_id" not in pred_df.columns:
        raise ValueError("predictions file thiếu cột node_id.")

    pred_df["node_id"] = pred_df["node_id"].astype(int)

    y_true_df = pd.DataFrame(
        {
            "node_id": range(len(graph.node_accounts)),
            "true_label": graph.data.y.cpu().numpy().astype(int),
        }
    )

    merged = pred_df.merge(y_true_df, on="node_id", how="inner")

    if len(merged) == 0:
        raise ValueError("Không merge được node_id nào.")

    y_true = merged["true_label"].astype(int)
    y_pred = merged["predicted_label"].astype(int)
    y_score = merged["risk_score"].astype(float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result = {
        "accounts_evaluated": int(len(merged)),
        "positive_accounts": int(y_true.sum()),
        "negative_accounts": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_true.nunique() == 2 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "negative_predictive_value": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        csv_path=args.csv_path,
        predictions_path=args.predictions_path,
        output_path=args.output_path,
    )