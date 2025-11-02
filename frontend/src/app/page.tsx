"use client";
import React, { useMemo, useState } from "react";

type ApiResult = {
  label: string;
  prob: number;
  scores?: Record<string, number>;
  source?: string;
};

const TOXICITY_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] as const;

const formatLabel = (label: string) =>
  label
    .split("_")
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

export default function Home() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<ApiResult | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const [feedbackMsg, setFeedbackMsg] = useState("");
  const [enabledLabels, setEnabledLabels] = useState<string[]>([...TOXICITY_LABELS]);

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

  const toggleLabel = (label: string) => {
    setEnabledLabels(current => {
      const isActive = current.includes(label);
      if (isActive) {
        if (current.length === 1) {
          return current; // keep at least one label active
        }
        return current.filter(item => item !== label);
      }
      return [...current, label];
    });
  };

  const orderedScores = useMemo(() => {
    if (!result?.scores) return [] as Array<[string, number]>;
    const entries = Object.entries(result.scores);
    return entries.sort((a, b) => {
      if (a[0] === "toxic") return -1;
      if (b[0] === "toxic") return 1;
      return a[0].localeCompare(b[0]);
    });
  }, [result]);

  const focusedScore = useMemo(() => {
    if (!result?.scores) return 0;
    return enabledLabels.reduce((max, label) => {
      const value = result.scores?.[label] ?? 0;
      return value > max ? value : max;
    }, 0);
  }, [enabledLabels, result]);

  const focusedVerdict = focusedScore > 0.5 ? "❌ Toxic" : "✅ Safe";

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
        <div className="w-full mb-6">
          <div className="text-sm text-blue-200 mb-2">Focus on categories:</div>
          <div className="flex flex-wrap gap-2">
            {TOXICITY_LABELS.map(label => {
              const active = enabledLabels.includes(label);
              return (
                <button
                  key={label}
                  type="button"
                  className={`${active ? "bg-pink-600 text-white" : "bg-[#181D31] text-blue-200"} px-3 py-1 rounded-md border border-[#3C4251] transition hover:border-blue-500 text-sm`}
                  onClick={() => toggleLabel(label)}
                >
                  {formatLabel(label)}
                </button>
              );
            })}
          </div>
        </div>
        {result && (
          <div className="w-full mb-2">
            <div className="text-lg font-semibold mb-1 text-blue-100">
              API Verdict:{" "}
              {result.label === "toxic" || result.label === "❌ Toxic" ? (
                <span className="text-pink-400 font-bold">❌ Toxic</span>
              ) : (
                <span className="text-cyan-300 font-bold">✅ Safe</span>
              )}{" "}
              <span className="text-blue-200">({result.prob.toFixed(3)})</span>
            </div>
            {result.scores && (
              <div className="text-sm text-blue-200 mb-3">
                Focus verdict ({enabledLabels.length ? enabledLabels.map(formatLabel).join(", ") : "None"}):{" "}
                <span className={focusedVerdict === "❌ Toxic" ? "text-pink-300 font-semibold" : "text-cyan-200 font-semibold"}>
                  {focusedVerdict}
                </span>{" "}
                <span className="text-blue-200">({focusedScore.toFixed(3)})</span>
              </div>
            )}
            {result.scores && orderedScores.length > 0 && (
              <div className="bg-[#181D31] border border-[#3C4251] rounded-lg p-3 mb-3">
                <div className="text-sm text-blue-200 mb-2">Category probabilities</div>
                <ul className="space-y-1">
                  {orderedScores.map(([label, value]) => {
                    const active = enabledLabels.includes(label);
                    return (
                      <li key={label} className={active ? "text-pink-200" : "text-blue-200"}>
                        <span className="font-semibold">{formatLabel(label)}:</span> {value.toFixed(3)}
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
            {result.source && (
              <div className="text-xs uppercase tracking-wide text-blue-300 mb-2">Source: {result.source}</div>
            )}
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
