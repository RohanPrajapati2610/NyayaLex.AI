import type { Citation } from "./ChatWindow";

type Props = {
  citations: Citation[];
};

export default function CitationPanel({ citations }: Props) {
  return (
    <aside className="w-80 border-l border-gray-800 overflow-y-auto p-4 space-y-3">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-4">
        Sources
      </h2>
      {citations.map((c) => (
        <div key={c.id} className="bg-gray-800 rounded-xl p-3 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-brand-100 truncate">{c.source}</span>
            <span className="text-xs text-gray-400 ml-2 shrink-0">
              {(c.score * 100).toFixed(0)}%
            </span>
          </div>
          {c.court && (
            <p className="text-xs text-gray-400">{c.court}{c.date ? ` · ${c.date}` : ""}</p>
          )}
          <p className="text-xs text-gray-300 leading-relaxed line-clamp-4">{c.excerpt}</p>
        </div>
      ))}
    </aside>
  );
}
