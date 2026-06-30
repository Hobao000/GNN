import { Metrics, Summary } from "@/types/aml";

interface Props {
  summary?: Summary;
  metrics?: Metrics | null;
}

export default function SummaryCharts({ summary, metrics }: Props) {
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div className="rounded-2xl border border-amber-500/10 bg-slate-950/80 p-6">
        <h2 className="mb-4 text-xl font-semibold text-amber-300">Evaluation Metrics</h2>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <Metric label="Accuracy" value={metrics?.accuracy} />
          <Metric label="Precision" value={metrics?.precision} />
          <Metric label="Recall" value={metrics?.recall} />
          <Metric label="F1-score" value={metrics?.f1_score} />
          <Metric label="ROC-AUC" value={metrics?.roc_auc} />
          <Metric label="PR-AUC" value={metrics?.pr_auc} />
        </div>
      </div>

      <div className="rounded-2xl border border-amber-500/10 bg-slate-950/80 p-6">
        <h2 className="mb-4 text-xl font-semibold text-amber-300">Prediction Summary</h2>
        <p className="text-slate-400">Threshold: {summary?.threshold ?? "--"}</p>
        <p className="text-slate-400">
          Inference time: {summary?.inference_seconds?.toFixed(2) ?? "--"}s
        </p>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value?: number }) {
  return (
    <div className="rounded-xl bg-black/40 p-4">
      <p className="text-slate-400">{label}</p>
      <p className="mt-1 text-2xl font-bold text-white">
        {typeof value === "number" ? value.toFixed(4) : "--"}
      </p>
    </div>
  );
}