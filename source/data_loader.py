from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


REQUIRED_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


@dataclass
class GraphBuildResult:
    data: Data
    node_accounts: list[str]
    feature_names: list[str]
    edge_feature_names: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    transactions: pd.DataFrame


def require_columns(df: pd.DataFrame, columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_transactions(path: str | Path, sample_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample_rows is not None and sample_rows < len(df):
        df = df.sample(sample_rows, random_state=seed).reset_index(drop=True)
    return df


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df)
    df = df.copy()

    for col in ["Account", "Account.1", "From Bank", "To Bank", "Receiving Currency", "Payment Currency", "Payment Format"]:
        df[col] = df[col].fillna("UNK").astype(str)

    df["src_account"] = df["From Bank"].astype(str) + "_" + df["Account"].astype(str)
    df["dst_account"] = df["To Bank"].astype(str) + "_" + df["Account.1"].astype(str)

    df["Is Laundering"] = pd.to_numeric(df["Is Laundering"], errors="coerce").fillna(0).astype(int)
    df["Amount Received"] = pd.to_numeric(df["Amount Received"], errors="coerce").fillna(0.0)
    df["Amount Paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0.0)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["hour"] = df["Timestamp"].dt.hour.fillna(0).astype(int)
    df["dayofweek"] = df["Timestamp"].dt.dayofweek.fillna(0).astype(int)
    ts_numeric = df["Timestamp"].astype("int64", errors="ignore")
    if not np.issubdtype(ts_numeric.dtype, np.number):
        df["timestamp_norm"] = 0.0
    else:
        ts = pd.to_numeric(ts_numeric, errors="coerce").fillna(0.0)
        span = float(ts.max() - ts.min())
        df["timestamp_norm"] = 0.0 if span <= 0 else (ts - ts.min()) / span

    df["is_self_loop"] = (df["src_account"] == df["dst_account"]).astype(int)
    df["currency_mismatch"] = (df["Receiving Currency"] != df["Payment Currency"]).astype(int)
    df["bank_mismatch"] = (df["From Bank"] != df["To Bank"]).astype(int)

    for col in ["Payment Format", "Receiving Currency", "Payment Currency"]:
        codes, _ = pd.factorize(df[col].astype(str), sort=True)
        df[col + "_code"] = codes.astype(float)

    return df


def stratified_node_split(labels: torch.Tensor, train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    train_parts, val_parts, test_parts = [], [], []

    for class_value in [0, 1]:
        idx = torch.where(labels == class_value)[0]
        if idx.numel() == 0:
            continue
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        n = idx.numel()
        train_end = max(1, int(n * train_ratio))
        val_end = int(n * (train_ratio + val_ratio))
        if n >= 3:
            val_end = max(train_end + 1, val_end)
            val_end = min(n - 1, val_end)
        else:
            val_end = min(n, max(train_end, val_end))
        train_parts.append(idx[:train_end])
        val_parts.append(idx[train_end:val_end])
        test_parts.append(idx[val_end:])

    def cat_or_empty(parts):
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)

    train_idx = cat_or_empty(train_parts)
    val_idx = cat_or_empty(val_parts)
    test_idx = cat_or_empty(test_parts)
    return train_idx, val_idx, test_idx


def build_node_features(df: pd.DataFrame, num_nodes: int) -> tuple[pd.DataFrame, list[str]]:
    out_stats = df.groupby("src_idx").agg(
        out_tx_count=("src_idx", "size"),
        out_amount_sum=("Amount Paid", "sum"),
        out_amount_mean=("Amount Paid", "mean"),
        out_unique_targets=("dst_idx", "nunique"),
        out_hour_mean=("hour", "mean"),
        out_dayofweek_mean=("dayofweek", "mean"),
        out_self_loop_count=("is_self_loop", "sum"),
        out_currency_mismatch_count=("currency_mismatch", "sum"),
        out_bank_mismatch_count=("bank_mismatch", "sum"),
    )
    in_stats = df.groupby("dst_idx").agg(
        in_tx_count=("dst_idx", "size"),
        in_amount_sum=("Amount Received", "sum"),
        in_amount_mean=("Amount Received", "mean"),
        in_unique_sources=("src_idx", "nunique"),
        in_hour_mean=("hour", "mean"),
        in_dayofweek_mean=("dayofweek", "mean"),
        in_self_loop_count=("is_self_loop", "sum"),
        in_currency_mismatch_count=("currency_mismatch", "sum"),
        in_bank_mismatch_count=("bank_mismatch", "sum"),
    )

    node_df = pd.DataFrame(index=range(num_nodes))
    node_df = node_df.join(out_stats, how="left").join(in_stats, how="left").fillna(0)

    node_df["total_tx_count"] = node_df["out_tx_count"] + node_df["in_tx_count"]
    node_df["total_amount_sum"] = node_df["out_amount_sum"] + node_df["in_amount_sum"]
    node_df["amount_balance"] = node_df["in_amount_sum"] - node_df["out_amount_sum"]
    node_df["counterpart_count"] = node_df["out_unique_targets"] + node_df["in_unique_sources"]
    node_df["avg_amount_per_tx"] = node_df["total_amount_sum"] / node_df["total_tx_count"].clip(lower=1)
    node_df["amount_in_out_ratio"] = node_df["in_amount_sum"] / node_df["out_amount_sum"].clip(lower=1e-6)
    node_df["count_in_out_ratio"] = node_df["in_tx_count"] / node_df["out_tx_count"].clip(lower=1e-6)
    node_df["active_hour_span"] = (node_df["out_hour_mean"] - node_df["in_hour_mean"]).abs()
    node_df["active_day_span"] = (node_df["out_dayofweek_mean"] - node_df["in_dayofweek_mean"]).abs()
    node_df["self_loop_count"] = node_df["out_self_loop_count"] + node_df["in_self_loop_count"]
    node_df["currency_mismatch_count"] = node_df["out_currency_mismatch_count"] + node_df["in_currency_mismatch_count"]
    node_df["bank_mismatch_count"] = node_df["out_bank_mismatch_count"] + node_df["in_bank_mismatch_count"]

    node_df = node_df.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    return node_df, node_df.columns.tolist()


def align_features(feature_frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    aligned = feature_frame.copy()
    for name in feature_names:
        if name not in aligned.columns:
            aligned[name] = 0.0
    return aligned[feature_names].replace([np.inf, -np.inf], 0).fillna(0).astype("float32")


def normalize_features(x: torch.Tensor, train_mask: torch.Tensor | None = None, mean: list[float] | None = None, std: list[float] | None = None):
    if mean is None or std is None:
        base = x[train_mask] if train_mask is not None and train_mask.any() else x
        mu = base.mean(dim=0, keepdim=True)
        sigma = base.std(dim=0, keepdim=True).clamp(min=1e-6)
    else:
        mu = torch.tensor(mean, dtype=torch.float32).view(1, -1)
        sigma = torch.tensor(std, dtype=torch.float32).view(1, -1).clamp(min=1e-6)
    return (x - mu) / sigma, mu.view(-1).tolist(), sigma.view(-1).tolist()


def build_account_graph(
    df: pd.DataFrame,
    seed: int = 42,
    directed_edges: bool = True,
    feature_names: list[str] | None = None,
    feature_mean: list[float] | None = None,
    feature_std: list[float] | None = None,
) -> GraphBuildResult:
    df = prepare_transactions(df)
    accounts = pd.Index(pd.concat([df["src_account"], df["dst_account"]], ignore_index=True).unique())
    account_to_idx = {account: idx for idx, account in enumerate(accounts)}
    df["src_idx"] = df["src_account"].map(account_to_idx).astype(int)
    df["dst_idx"] = df["dst_account"].map(account_to_idx).astype(int)

    feature_frame, computed_feature_names = build_node_features(df, len(accounts))
    if feature_names is not None:
        feature_frame = align_features(feature_frame, feature_names)
        computed_feature_names = feature_names

    x = torch.tensor(feature_frame.to_numpy(), dtype=torch.float32)

    fraud_edges = df["Is Laundering"] == 1
    fraud_nodes = set(df.loc[fraud_edges, "src_idx"].tolist()) | set(df.loc[fraud_edges, "dst_idx"].tolist())
    y = torch.tensor([1 if idx in fraud_nodes else 0 for idx in range(len(accounts))], dtype=torch.long)

    src = df["src_idx"].to_numpy(dtype=np.int64)
    dst = df["dst_idx"].to_numpy(dtype=np.int64)
    edge_index_np = np.vstack([src, dst])

    edge_feature_names = [
        "Amount Paid",
        "Amount Received",
        "timestamp_norm",
        "Payment Format_code",
        "Payment Currency_code",
        "Receiving Currency_code",
        "currency_mismatch",
        "bank_mismatch",
        "is_self_loop",
    ]
    edge_attr_np = df[edge_feature_names].replace([np.inf, -np.inf], 0).fillna(0).to_numpy(dtype=np.float32)

    if not directed_edges:
        edge_index_np = np.hstack([edge_index_np, edge_index_np[::-1]])
        edge_attr_np = np.vstack([edge_attr_np, edge_attr_np])

    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

    train_idx, val_idx, test_idx = stratified_node_split(y, seed=seed)
    train_mask = torch.zeros(len(accounts), dtype=torch.bool)
    val_mask = torch.zeros(len(accounts), dtype=torch.bool)
    test_mask = torch.zeros(len(accounts), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    x, mu, sigma = normalize_features(x, train_mask=train_mask, mean=feature_mean, std=feature_std)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.num_nodes = len(accounts)

    return GraphBuildResult(
        data=data,
        node_accounts=accounts.tolist(),
        feature_names=computed_feature_names,
        edge_feature_names=edge_feature_names,
        feature_mean=mu,
        feature_std=sigma,
        transactions=df,
    )
