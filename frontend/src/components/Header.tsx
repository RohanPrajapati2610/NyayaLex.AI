export default function Header() {
  return (
    <header className="border-b border-gray-800 px-6 py-4 flex items-center gap-3">
      <div className="w-8 h-8 rounded-lg bg-brand-500 flex items-center justify-center text-white font-bold text-sm">
        N
      </div>
      <div>
        <h1 className="text-white font-semibold text-lg leading-none">NyayaLex.AI</h1>
        <p className="text-gray-400 text-xs mt-0.5">Statute-Aware Legal Q&amp;A</p>
      </div>
    </header>
  );
}
