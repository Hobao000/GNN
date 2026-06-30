export default function UploadSection() {
  return (
    <div className="rounded-2xl border border-amber-500/20 bg-slate-950/80 p-6 shadow-xl shadow-amber-500/5">
      <h2 className="text-xl font-semibold text-amber-300">
        Upload CSV Transaction File
      </h2>

      <p className="text-slate-400 mt-1">
        Tải dữ liệu giao dịch để mô hình GraphSAGE phân tích tài khoản trung chuyển nghi ngờ.
      </p>

      <div className="mt-5 flex flex-col md:flex-row gap-3">
        <input
          type="file"
          accept=".csv"
          className="block w-full rounded-lg border border-slate-700 bg-black px-4 py-2 text-slate-300"
        />

        <button className="rounded-lg bg-amber-500 px-5 py-2 font-semibold text-black hover:bg-amber-400 shadow-lg shadow-amber-500/20">
          Run Prediction
        </button>
      </div>
    </div>
  );
}