"use client";

import { useState, useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";
import ChatInput from "./ChatInput";
import CitationPanel from "./CitationPanel";

export type Citation = {
  id: number;
  source: string;
  court?: string;
  date?: string;
  excerpt: string;
  score: number;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  conflictWarning?: string;
};

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeCitations, setActiveCitations] = useState<Citation[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage(text: string) {
    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
      const data = await res.json();

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer,
        citations: data.citations,
        conflictWarning: data.conflict_warning,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setActiveCitations(data.citations ?? []);
    } catch {
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "assistant", content: "Something went wrong. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-1 overflow-hidden">
      <div className="flex flex-col flex-1 overflow-hidden">
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center text-gray-500 space-y-2">
              <p className="text-lg font-medium text-gray-300">Ask a legal question</p>
              <p className="text-sm">Type or speak your question about US case law or federal statutes.</p>
            </div>
          )}
          {messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              onCitationClick={() => setActiveCitations(msg.citations ?? [])}
            />
          ))}
          {loading && (
            <div className="flex gap-2 items-center text-gray-400 text-sm px-2">
              <span className="animate-pulse">Retrieving sources...</span>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
        <ChatInput onSend={sendMessage} disabled={loading} />
      </div>
      {activeCitations.length > 0 && <CitationPanel citations={activeCitations} />}
    </div>
  );
}
