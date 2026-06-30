from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def build_nx_graph(transactions: pd.DataFrame, node_accounts: list[str], risk_scores: np.ndarray, labels: np.ndarray) -> nx.DiGraph:
    graph = nx.DiGraph()
    for idx, account in enumerate(node_accounts):
        graph.add_node(
            int(idx),
            account=str(account),
            risk_score=float(risk_scores[idx]),
            predicted_label=int(labels[idx]),
        )

    grouped = transactions.groupby(["src_idx", "dst_idx"]).agg(
        amount=("Amount Paid", "sum"),
        tx_count=("src_idx", "size"),
        laundering_edge=("Is Laundering", "max"),
    ).reset_index()

    for row in grouped.itertuples(index=False):
        graph.add_edge(
            int(row.src_idx),
            int(row.dst_idx),
            amount=float(row.amount),
            tx_count=int(row.tx_count),
            laundering_edge=int(row.laundering_edge),
        )
    return graph


def extract_suspicious_subgraph(graph: nx.DiGraph, include_neighbors: bool = True, max_neighbors_per_node: int = 20) -> nx.DiGraph:
    suspicious = [n for n, d in graph.nodes(data=True) if int(d.get("predicted_label", 0)) == 1]
    selected = set(suspicious)
    if include_neighbors:
        for node in suspicious:
            neighbors = list(graph.predecessors(node))[:max_neighbors_per_node] + list(graph.successors(node))[:max_neighbors_per_node]
            selected.update(neighbors)
    return graph.subgraph(selected).copy()


def detect_patterns(subgraph: nx.DiGraph, component_nodes: list[int]) -> list[str]:
    component = subgraph.subgraph(component_nodes).copy()
    patterns: list[str] = []
    if any(component.out_degree(n) >= 5 for n in component.nodes):
        patterns.append("fan_out")
    if any(component.in_degree(n) >= 5 for n in component.nodes):
        patterns.append("fan_in")
    try:
        if any(True for _ in nx.simple_cycles(component)):
            patterns.append("cycle")
    except nx.NetworkXNoCycle:
        pass
    if component.number_of_nodes() >= 4 and component.number_of_edges() >= component.number_of_nodes() - 1:
        patterns.append("layering")
    return patterns or ["suspicious_subgraph"]


def find_laundering_groups(subgraph: nx.DiGraph) -> list[dict[str, Any]]:
    groups = []
    weak_components = list(nx.weakly_connected_components(subgraph))
    group_id = 1
    for comp in weak_components:
        comp_nodes = list(comp)
        suspicious_nodes = [n for n in comp_nodes if subgraph.nodes[n].get("predicted_label", 0) == 1]
        if not suspicious_nodes:
            continue
        risks = [float(subgraph.nodes[n].get("risk_score", 0.0)) for n in suspicious_nodes]
        volume = sum(float(d.get("amount", 0.0)) for _, _, d in subgraph.subgraph(comp_nodes).edges(data=True))
        groups.append(
            {
                "group_id": group_id,
                "risk_score": float(np.mean(risks)) if risks else 0.0,
                "num_accounts": int(len(comp_nodes)),
                "num_suspicious_accounts": int(len(suspicious_nodes)),
                "transaction_volume": float(volume),
                "accounts": [subgraph.nodes[n].get("account", str(n)) for n in comp_nodes],
                "suspicious_accounts": [subgraph.nodes[n].get("account", str(n)) for n in suspicious_nodes],
                "detected_patterns": detect_patterns(subgraph, comp_nodes),
            }
        )
        group_id += 1
    groups.sort(key=lambda g: (g["risk_score"], g["transaction_volume"]), reverse=True)
    return groups


def export_web_graph(
    transactions: pd.DataFrame,
    node_accounts: list[str],
    risk_scores: np.ndarray,
    threshold: float,
    output_path: str,
    include_neighbors: bool = True,
) -> dict[str, Any]:
    labels = (risk_scores >= threshold).astype(int)
    graph = build_nx_graph(transactions, node_accounts, risk_scores, labels)
    subgraph = extract_suspicious_subgraph(graph, include_neighbors=include_neighbors)
    groups = find_laundering_groups(subgraph)

    max_amount = max([float(d.get("amount", 0.0)) for _, _, d in subgraph.edges(data=True)] or [1.0])
    nodes = []
    for n, d in subgraph.nodes(data=True):
        nodes.append(
            {
                "id": int(n),
                "account": d.get("account", str(n)),
                "risk_score": float(d.get("risk_score", 0.0)),
                "predicted_label": int(d.get("predicted_label", 0)),
                "color": "red" if int(d.get("predicted_label", 0)) == 1 else "blue",
            }
        )

    edges = []
    for u, v, d in subgraph.edges(data=True):
        amount = float(d.get("amount", 0.0))
        edges.append(
            {
                "source": int(u),
                "target": int(v),
                "amount": amount,
                "tx_count": int(d.get("tx_count", 1)),
                "width": 1.0 + 6.0 * amount / max_amount,
                "is_suspicious": bool(subgraph.nodes[u].get("predicted_label", 0) or subgraph.nodes[v].get("predicted_label", 0)),
            }
        )

    result = {
        "nodes": nodes,
        "edges": edges,
        "groups": groups,
        "summary": {
            "total_accounts": int(len(node_accounts)),
            "total_transactions": int(len(transactions)),
            "suspicious_accounts_detected": int(labels.sum()),
            "laundering_groups_detected": int(len(groups)),
            "threshold": float(threshold),
        },
    }

    import json
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
