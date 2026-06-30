from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

from data_loader import build_account_graph, load_transactions
from metrics import best_threshold_by_f1, metric_dict


def tensor_to_numpy(tensor, dtype=None):
    arr = np.asarray(tensor.detach().cpu().tolist())
    return arr.astype(dtype) if dtype is not None else arr


def run_mlp(data_path: str, output_dir: str = "data/models", sample_rows: int | None = None):
    df = load_transactions(data_path, sample_rows=sample_rows)
    graph = build_account_graph(df)
    data = graph.data
    x = tensor_to_numpy(data.x, dtype=float)
    y = tensor_to_numpy(data.y, dtype=int)
    train_mask = tensor_to_numpy(data.train_mask, dtype=bool)
    val_mask = tensor_to_numpy(data.val_mask, dtype=bool)
    test_mask = tensor_to_numpy(data.test_mask, dtype=bool)

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, early_stopping=True, random_state=42)
    clf.fit(x[train_mask], y[train_mask])
    val_probs = clf.predict_proba(x[val_mask])[:, 1]
    threshold, val_metrics = best_threshold_by_f1(y[val_mask], val_probs)
    test_probs = clf.predict_proba(x[test_mask])[:, 1]
    test_metrics = metric_dict(y[test_mask], test_probs, threshold)

    result = {"best_threshold": threshold, "validation": val_metrics, "test": test_metrics}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "mlp_baseline_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/raw/HI-Small_Trans.csv")
    parser.add_argument("--output-dir", default="data/models")
    parser.add_argument("--sample-rows", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run_mlp(args.data_path, args.output_dir, args.sample_rows), ensure_ascii=False, indent=2))
