from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def safe_roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return 0.0


def safe_pr_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, probs))
    except ValueError:
        return 0.0


def metric_dict(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    specificity = tn / max(1, tn + fp)
    npv = tn / max(1, tn + fn)

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": safe_pr_auc(y_true, probs),
        "roc_auc": safe_roc_auc(y_true, probs),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "negative_predictive_value": float(npv),
        "threshold": float(threshold),
        "support": {
            "negative": int((y_true == 0).sum()),
            "positive": int((y_true == 1).sum()),
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def best_threshold_by_f1(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_metrics = metric_dict(y_true, probs, best_threshold)
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = metric_dict(y_true, probs, float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def evaluation_schema() -> list[dict[str, str]]:
    return [
        {"name": "recall", "purpose": "Tỷ lệ tài khoản rửa tiền thật được phát hiện.", "why": "AML ưu tiên giảm bỏ sót."},
        {"name": "precision", "purpose": "Tỷ lệ cảnh báo đúng trong các cảnh báo.", "why": "Giảm báo động giả."},
        {"name": "f1", "purpose": "Cân bằng precision và recall.", "why": "Dùng chọn threshold."},
        {"name": "pr_auc", "purpose": "Diện tích dưới Precision-Recall curve.", "why": "Phù hợp dữ liệu mất cân bằng."},
        {"name": "roc_auc", "purpose": "Khả năng phân tách hai lớp.", "why": "So sánh tổng quan mô hình."},
        {"name": "confusion_matrix", "purpose": "TP/FP/FN/TN.", "why": "Phân tích lỗi model."},
        {"name": "specificity", "purpose": "Tỷ lệ tài khoản bình thường được nhận diện đúng.", "why": "Đo khả năng giảm FP."},
        {"name": "balanced_accuracy", "purpose": "Trung bình recall và specificity.", "why": "Ổn định hơn accuracy khi lệch lớp."},
    ]
