from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import NeighborLoader

from data_loader import build_account_graph, load_transactions
from losses import positive_probabilities
from models.gnn_models import build_model
from network_utils import export_web_graph


def tensor_to_numpy(tensor: torch.Tensor, dtype: type | None = None) -> np.ndarray:
    arr = np.asarray(tensor.detach().cpu().tolist())
    return arr.astype(dtype) if dtype is not None else arr


def load_bundle(bundle_path: str | Path) -> dict[str, Any]:
    bundle = torch.load(bundle_path, map_location="cpu")
    if not isinstance(bundle, dict) or "model_state_dict" not in bundle:
        raise ValueError(f"Invalid model bundle: {bundle_path}")
    return bundle


@torch.no_grad()
def predict_all_nodes(
    model,
    data,
    device,
    batch_size: int = 2048,
    num_neighbors: list[int] | None = None,
) -> np.ndarray:
    model.eval()
    num_neighbors = num_neighbors or [25, 15]

    input_nodes = torch.arange(data.num_nodes)
    loader = NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
    )

    scores = np.zeros(data.num_nodes, dtype=np.float32)

    for batch in loader:
        batch = batch.to(device)

        out = model(
            batch.x,
            batch.edge_index,
            getattr(batch, "edge_attr", None),
        )

        seed_count = int(batch.batch_size)
        probs = positive_probabilities(out[:seed_count])

        node_ids = tensor_to_numpy(batch.n_id[:seed_count], dtype=int)
        scores[node_ids] = tensor_to_numpy(probs, dtype=float)

    return scores


def predict_csv(
    bundle_path: str | Path,
    csv_path: str | Path,
    output_dir: str | Path = "data/outputs",
    batch_size: int = 2048,
) -> dict[str, Any]:
    start = time.time()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(bundle_path)
    cfg = bundle["model_config"]

    threshold = float(bundle.get("best_threshold", 0.5))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_transactions(csv_path)

    # Inference/demo mode:
    # Uploaded CSV files may NOT contain "Is Laundering".
    # The model does not use this column for prediction.
    # We add a dummy label only because build_account_graph may expect it.
    has_ground_truth = "Is Laundering" in df.columns
    if not has_ground_truth:
        df["Is Laundering"] = 0

    graph = build_account_graph(
        df,
        directed_edges=True,
        feature_names=list(bundle["feature_names"]),
        feature_mean=list(bundle["feature_mean"]),
        feature_std=list(bundle["feature_std"]),
    )

    data = graph.data

    edge_attr = getattr(data, "edge_attr", None)
    edge_dim = int(cfg.get("edge_dim") or (edge_attr.size(1) if edge_attr is not None else 0))

    model = build_model(
        model_type=cfg["model_type"],
        in_channels=int(cfg["in_channels"]),
        hidden_channels=int(cfg["hidden_channels"]),
        out_channels=int(cfg["out_channels"]),
        edge_dim=edge_dim,
        heads=int(cfg.get("heads", 4)),
        dropout=float(cfg.get("dropout", 0.2)),
    ).to(device)

    model.load_state_dict(bundle["model_state_dict"])

    risk_scores = predict_all_nodes(
        model=model,
        data=data,
        device=device,
        batch_size=batch_size,
    )

    labels = (risk_scores >= threshold).astype(int)

    predictions = pd.DataFrame(
        {
            "node_id": np.arange(len(graph.node_accounts)),
            "account": graph.node_accounts,
            "risk_score": risk_scores,
            "predicted_label": labels,
        }
    ).sort_values("risk_score", ascending=False)

    predictions_path = output_dir / "account_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    web_graph_path = output_dir / "web_graph.json"

    # export_web_graph should export suspicious subgraph for web visualization.
    # Do not visualize the whole graph when the CSV is very large.
    web_graph = export_web_graph(
        transactions=graph.transactions,
        node_accounts=graph.node_accounts,
        risk_scores=risk_scores,
        threshold=threshold,
        output_path=str(web_graph_path),
        include_neighbors=True,
    )

    summary = {
        "bundle_path": str(bundle_path),
        "csv_path": str(csv_path),
        "threshold": threshold,
        "has_ground_truth": bool(has_ground_truth),
        "total_accounts": int(len(graph.node_accounts)),
        "total_transactions": int(len(graph.transactions)),
        "suspicious_accounts_detected": int(labels.sum()),
        "laundering_groups_detected": int(len(web_graph.get("groups", []))),
        "inference_seconds": float(time.time() - start),
        "outputs": {
            "account_predictions_csv": str(predictions_path),
            "web_graph_json": str(web_graph_path),
            "inference_summary_json": str(output_dir / "inference_summary.json"),
        },
    }

    summary_path = output_dir / "inference_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AML GNN inference on uploaded CSV transactions."
    )
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", default="data/outputs")
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_csv(
        bundle_path=args.bundle_path,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))