"use client";

import dynamic from "next/dynamic";
import { GraphData } from "@/types/aml";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
});

interface Props {
  graph?: GraphData;
}

export default function NetworkGraph({ graph }: Props) {
  const graphData = graph
    ? {
        nodes: graph.nodes.map((node) => ({
          ...node,
          id: String(node.id),
        })),
        links: graph.edges.map((edge) => ({
          ...edge,
          source: String(edge.source),
          target: String(edge.target),
        })),
      }
    : { nodes: [], links: [] };

  return (
    <div className="rounded-2xl border border-amber-500/10 bg-slate-950/80 p-6">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-amber-300">
            Transaction Network
          </h2>
          <p className="text-sm text-slate-400">
            Mạng lưới giao dịch nghi ngờ sau khi GNN phát hiện.
          </p>
        </div>

        <div className="text-sm text-slate-400">
          {graphData.nodes.length.toLocaleString("vi-VN")} nodes •{" "}
          {graphData.links.length.toLocaleString("vi-VN")} edges
        </div>
      </div>

      <div className="h-[650px] overflow-hidden rounded-xl border border-slate-800 bg-black">
        {graphData.nodes.length > 0 ? (
          <ForceGraph2D
            graphData={graphData}
            backgroundColor="#000000"
            nodeLabel={(node) => {
              const item = node as {
                account?: string;
                id?: string;
                risk_score?: number;
                predicted_label?: number;
              };

              return `
                Account: ${item.account ?? item.id}
                Risk: ${item.risk_score?.toFixed(4) ?? "--"}
                Label: ${item.predicted_label === 1 ? "Suspicious" : "Normal"}
              `;
            }}
            nodeColor={(node) => {
              const item = node as { predicted_label?: number };
              return item.predicted_label === 1 ? "#ef4444" : "#38bdf8";
            }}
            nodeRelSize={5}
            linkColor={() => "rgba(251, 191, 36, 0.35)"}
            linkWidth={(link) => {
              const item = link as { amount?: number };
              return item.amount ? Math.max(1, Math.log10(item.amount + 1)) : 1;
            }}
            cooldownTicks={100}
            onEngineStop={() => {
              // graph tự ổn định layout
            }}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-slate-500">
            Không có dữ liệu graph.
          </div>
        )}
      </div>
    </div>
  );
}