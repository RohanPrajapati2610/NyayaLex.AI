"use client";

import { useEffect } from "react";
import type { Message } from "./ChatWindow";

type Props = {
  message: Message;
  onCitationClick: () => void;
};

export default function MessageBubble({ message, onCitationClick }: Props) {
  const isUser = message.role === "user";

  useEffect(() => {
    if (!isUser && message.content && typeof window !== "undefined") {
      const synth = window.speechSynthesis;
      const utterance = new SpeechSynthesisUtterance(message.content);
      utterance.lang = "en-US";
      utterance.rate = 1;
      synth.cancel();
      synth.speak(utterance);
    }
  }, [message.content, isUser]);

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-2xl rounded-2xl px-4 py-3 text-sm ${
        isUser
          ? "bg-brand-500 text-white"
          : "bg-gray-800 text-gray-100"
      }`}>
        {message.conflictWarning && (
          <div className="mb-2 text-xs bg-yellow-500/20 text-yellow-300 border border-yellow-500/30 rounded-lg px-3 py-1.5">
            Conflict detected: {message.conflictWarning}
          </div>
        )}

        <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>

        {message.citations && message.citations.length > 0 && (
          <button
            onClick={onCitationClick}
            className="mt-2 text-xs text-brand-100 underline underline-offset-2 opacity-70 hover:opacity-100"
          >
            {message.citations.length} source{message.citations.length > 1 ? "s" : ""} cited
          </button>
        )}
      </div>
    </div>
  );
}
