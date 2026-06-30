import { LaunderingGroup } from "@/types/aml";

interface Props {
  groups: LaunderingGroup[];
}

export default function LaunderingGroupsTable({ groups }: Props) {
  return (
    <div className="rounded-2xl border border-amber-500/10 bg-slate-950/80 p-6">
      <h2 className="mb-4 text-xl font-semibold text-amber-300">Laundering Groups</h2>

      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700 text-left text-slate-300">
            <th className="py-2">Group</th>
            <th>Accounts</th>
            <th>Risk</th>
          </tr>
        </thead>

        <tbody>
          {groups.slice(0, 50).map((group, index) => (
            <tr key={group.group_id ?? index} className="border-b border-slate-800">
              <td className="py-3">{group.group_id ?? index + 1}</td>
              <td>{group.members?.length ?? group.num_accounts ?? "--"}</td>
              <td>{group.risk_score?.toFixed(4) ?? "--"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}