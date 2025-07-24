"use client";
import React, { useState } from "react";

export default function Home() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<null | { label: string; prob: number }>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const [feedbackMsg, setFeedbackMsg] = useState("");

  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

  const checkToxic = async () => {
    setPending(true);
    setError("");
    setResult(null);
    setFeedbackMsg("");
    try {
      const res = await fetch(`${apiBase}/api/check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      });
      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      setResult(data);
    } catch {
      setError("Backend error: is Flask running?");
    }
    setPending(false);
  };

  const sendFeedback = async (isToxic: boolean) => {
    setFeedbackMsg("Sending feedback...");
    try {
      await fetch(`${apiBase}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input, label: isToxic ? 1 : 0 }),
      });
      setFeedbackMsg(isToxic ? "Marked as Toxic! Model will learn soon." : "Marked as Safe! Model will learn soon.");
    } catch {
      setFeedbackMsg("Could not send feedback.");
    }
  };

  return (
    <div className="min-h-screen bg-[#181D31] flex flex-col items-center">
      <main className="w-full max-w-xl bg-[#232946] mt-16 rounded-xl shadow-xl p-8 flex flex-col items-center">
        <h1 className="text-3xl md:text-4xl font-bold text-blue-400 mb-6 tracking-tight drop-shadow">Local AI Toxicity Filter</h1>
        <form
          className="flex w-full space-x-2 mb-8"
          onSubmit={e => {
            e.preventDefault();
            checkToxic();
          }}
        >
          <input
            type="text"
            className="flex-1 rounded-lg px-4 py-2 bg-[#181D31] text-blue-100 border border-[#3C4251] focus:border-blue-500 focus:outline-none"
            placeholder="Type your phrase or sentence..."
            value={input}
            onChange={e => setInput(e.target.value)}
            autoFocus
          />
          <button
            type="submit"
            className="bg-blue-600 hover:bg-blue-800 text-white px-6 py-2 rounded-lg font-semibold transition"
            disabled={!input || pending}
          >
            {pending ? "Checking..." : "Check"}
          </button>
        </form>
        {error && <div className="text-pink-400 mb-2">{error}</div>}
        {result && (
          <div className="w-full mb-2">
            <div className="text-lg font-semibold mb-2">
              Result:{" "}
              {result.label === "toxic" || result.label === "❌ Toxic" ? (
                <span className="text-pink-400 font-bold">❌ Toxic</span>
              ) : (
                <span className="text-cyan-300 font-bold">✅ Safe</span>
              )}{" "}
              <span className="text-blue-200">({result.prob.toFixed(3)})</span>
            </div>
            <div className="flex space-x-3">
              <button
                className="bg-pink-600 hover:bg-pink-800 text-white px-3 py-1 rounded transition"
                onClick={() => sendFeedback(true)}
              >
                Mark as Toxic
              </button>
              <button
                className="bg-cyan-700 hover:bg-cyan-900 text-white px-3 py-1 rounded transition"
                onClick={() => sendFeedback(false)}
              >
                Mark as Safe
              </button>
            </div>
          </div>
        )}
        {feedbackMsg && (
          <div className="text-green-400 mt-2">{feedbackMsg}</div>
        )}
      </main>
      <footer className="mt-8 text-blue-200 text-sm opacity-60 text-center">
        Created by <span className="font-bold">Ayush Kunjadia</span> &middot; Local AI Safety Filter ⚡️
      </footer>
    </div>
  );
}
