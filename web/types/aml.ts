export interface Summary {
  total_accounts: number;
  total_transactions: number;
  suspicious_accounts_detected: number;
  laundering_groups_detected: number;
  threshold?: number;
  inference_seconds?: number;
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  pr_auc: number;
}

export interface GraphNode {
  id: number | string;
  account?: string;
  risk_score?: number;
  predicted_label?: number;
}

export interface GraphEdge {
  source: number | string;
  target: number | string;
  amount?: number;
}

export interface LaunderingGroup {
  group_id?: number;
  members?: string[];
  num_accounts?: number;
  risk_score?: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  groups: LaunderingGroup[];
}

export interface DashboardResponse {
  summary: Summary;
  metrics: Metrics | null;
  graph: GraphData;
}