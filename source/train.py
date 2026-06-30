from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

from data_loader import build_account_graph, load_transactions
from losses import make_loss, masked_or_seed_loss, positive_probabilities
from metrics import best_threshold_by_f1, evaluation_schema, metric_dict
from models.gnn_models import build_model


@dataclass
class TrainConfig:
    data_path: str = "data/raw/HI-Small_Trans.csv"
    output_dir: str = "data/models"
    model_type: str = "graphsage"
    sample_rows: int | None = None
    epochs: int = 30
    hidden_dim: int = 128
    heads: int = 4
    dropout: float = 0.2
    lr: float = 3e-4
    weight_decay: float = 5e-4
    batch_size: int = 1024
    num_neighbors: str = "15,10"
    eval_num_neighbors: str = "25,15"
    loss: str = "weighted_ce"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    patience: int = 8
    seed: int = 42
    directed_edges: bool = True


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_fanouts(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def tensor_to_numpy(tensor: torch.Tensor, dtype: type | None = None) -> np.ndarray:
    arr = np.asarray(tensor.detach().cpu().tolist())
    return arr.astype(dtype) if dtype is not None else arr


def make_loaders(data, config: TrainConfig):
    train_loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=parse_fanouts(config.num_neighbors),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=data.val_mask,
        num_neighbors=parse_fanouts(config.eval_num_neighbors),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=data.test_mask,
        num_neighbors=parse_fanouts(config.eval_num_neighbors),
        batch_size=config.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate_loader(model, loader, device, threshold: float | None = None):
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))
        seed_count = int(batch.batch_size)
        probs = positive_probabilities(out[:seed_count])
        labels = batch.y[:seed_count]
        all_probs.extend(tensor_to_numpy(probs, dtype=float).tolist())
        all_labels.extend(tensor_to_numpy(labels, dtype=int).tolist())

    y_true = np.asarray(all_labels, dtype=int)
    probs = np.asarray(all_probs, dtype=float)
    if threshold is None:
        return best_threshold_by_f1(y_true, probs)
    return threshold, metric_dict(y_true, probs, threshold)


def train_one_epoch(model, loader, criterion, optimizer, device, loss_name: str) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))
        seed_count = int(batch.batch_size)
        loss = masked_or_seed_loss(out[:seed_count], batch.y[:seed_count], criterion, loss_name)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * seed_count
        total_examples += seed_count
    return total_loss / max(1, total_examples)


def train(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {config.model_type}")
    print(f"Directed graph: {config.directed_edges}")

    df = load_transactions(config.data_path, sample_rows=config.sample_rows, seed=config.seed)
    graph = build_account_graph(df, seed=config.seed, directed_edges=config.directed_edges)
    data = graph.data
    out_channels = 2 if config.loss == "weighted_ce" else 1
    model = build_model(
        model_type=config.model_type,
        in_channels=data.num_features,
        hidden_channels=config.hidden_dim,
        out_channels=out_channels,
        edge_dim=data.edge_attr.size(1) if getattr(data, "edge_attr", None) is not None else None,
        heads=config.heads,
        dropout=config.dropout,
    ).to(device)

    train_loader, val_loader, test_loader = make_loaders(data, config)
    train_labels = data.y[data.train_mask]
    criterion = make_loss(config.loss, train_labels, config.focal_alpha, config.focal_gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state = None
    best_val_f1 = -1.0
    best_threshold = 0.5
    best_epoch = 0
    stale = 0
    history = []
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config.loss)
        val_threshold, val_metrics = evaluate_loader(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.5f} | "
            f"val_f1={val_metrics['f1']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f} "
            f"threshold={val_threshold:.2f}"
        )

        if float(val_metrics["f1"]) > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_threshold = float(val_threshold)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best checkpoint was created.")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    _, val_report = evaluate_loader(model, val_loader, device, threshold=best_threshold)
    _, test_report = evaluate_loader(model, test_loader, device, threshold=best_threshold)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"gnn_{config.model_type}"
    model_path = output_dir / f"{model_name}_model.pth"
    bundle_path = output_dir / f"{model_name}_bundle.pth"
    metrics_path = output_dir / f"{model_name}_metrics.json"

    model_config = {
        "model_type": config.model_type,
        "in_channels": data.num_features,
        "hidden_channels": config.hidden_dim,
        "out_channels": out_channels,
        "edge_dim": data.edge_attr.size(1),
        "heads": config.heads,
        "dropout": config.dropout,
        "loss": config.loss,
    }

    bundle = {
        "bundle_version": "2.0",
        "task_type": "node_classification_then_suspicious_subgraph_visualization",
        "model_state_dict": best_state,
        "model_config": model_config,
        "feature_names": graph.feature_names,
        "edge_feature_names": graph.edge_feature_names,
        "feature_mean": graph.feature_mean,
        "feature_std": graph.feature_std,
        "node_accounts": graph.node_accounts,
        "best_threshold": best_threshold,
        "label_map": {0: "normal_account", 1: "suspicious_intermediary_account"},
        "training_config": asdict(config),
        "validation_report": val_report,
        "test_report": test_report,
        "evaluation_schema": evaluation_schema(),
    }

    torch.save(best_state, model_path)
    torch.save(bundle, bundle_path)
    result = {
        "model_path": str(model_path),
        "bundle_path": str(bundle_path),
        "metrics_path": str(metrics_path),
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "training_seconds": time.time() - start_time,
        "validation_report": val_report,
        "test_report": test_report,
        "history": history,
        "note": "GNN predicts suspicious intermediary accounts. Laundering networks are derived by suspicious subgraph extraction.",
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model: {model_path}")
    print(f"Saved bundle: {bundle_path}")
    print(f"Saved metrics: {metrics_path}")
    return result


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train GraphSAGE/GAT for AML suspicious intermediary account detection.")
    parser.add_argument("--data-path", default=TrainConfig.data_path)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--model-type", choices=["graphsage", "gat"], default=TrainConfig.model_type)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--hidden-dim", type=int, default=TrainConfig.hidden_dim)
    parser.add_argument("--heads", type=int, default=TrainConfig.heads)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num-neighbors", default=TrainConfig.num_neighbors)
    parser.add_argument("--eval-num-neighbors", default=TrainConfig.eval_num_neighbors)
    parser.add_argument("--loss", choices=["weighted_ce", "focal", "bce"], default=TrainConfig.loss)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--undirected", action="store_true", help="Use symmetrized edges. Default keeps the graph directed.")
    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        sample_rows=args.sample_rows,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_neighbors=args.num_neighbors,
        eval_num_neighbors=args.eval_num_neighbors,
        loss=args.loss,
        patience=args.patience,
        seed=args.seed,
        directed_edges=not args.undirected,
    )


if __name__ == "__main__":
    train(parse_args())
