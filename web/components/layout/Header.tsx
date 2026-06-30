export default function Header() {
  return (
    <header className="border-b border-amber-500/20 bg-black/60 backdrop-blur">
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex items-center gap-3">
          <div className="h-12 w-12 rounded-full bg-amber-500 text-black flex items-center justify-center text-2xl font-black shadow-lg shadow-amber-500/30">
            ₿
          </div>

          <div>
            <h1 className="text-3xl font-bold tracking-tight">
              AML Crypto Finance Dashboard
            </h1>

            <p className="text-amber-200/80 mt-1">
              GNN phát hiện đường dây rửa tiền trong mạng lưới giao dịch tài chính
            </p>
          </div>
        </div>
      </div>
    </header>
  );
}