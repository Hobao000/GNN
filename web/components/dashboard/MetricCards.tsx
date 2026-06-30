import { Summary } from "@/types/aml";

interface Props {
  summary?: Summary;
}

export default function MetricCards({ summary }: Props) {
  const cards = [
    { title: "Total Accounts", value: summary?.total_accounts, color: "text-amber-300" },
    { title: "Transactions", value: summary?.total_transactions, color: "text-cyan-300" },
    { title: "Suspicious Accounts", value: summary?.suspicious_accounts_detected, color: "text-red-400" },
    { title: "Laundering Groups", value: summary?.laundering_groups_detected, color: "text-emerald-400" },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
      {cards.map((card) => (
        <div key={card.title} className="rounded-2xl border border-amber-500/10 bg-slate-950/80 p-6">
          <p className="text-sm text-slate-400">{card.title}</p>
          <h3 className={`mt-2 text-3xl font-bold ${card.color}`}>
            {card.value?.toLocaleString("vi-VN") ?? "--"}
          </h3>
        </div>
      ))}
    </div>
  );
}