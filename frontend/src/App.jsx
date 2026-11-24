import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { sendChat } from "./api.js";

const newSessionId = () =>
  "s-" + Math.random().toString(16).slice(2, 8) + Date.now().toString(16).slice(-4);

const FAILURE_PRESETS = [
  { label: "None", value: "" },
  { label: "key_1 → 429 (fail over to key_2)", value: "key_1=429" },
  { label: "key_1 → 500 (fail over to key_2)", value: "key_1=500" },
  { label: "key_1 → 429, key_2 → 500", value: "key_1=429,key_2=500" },
  { label: "All keys → 429 (expect 503)", value: "*=429" },
];

function Meta({ label, testid, value }) {
  return (
    <span className="meta-item">
      <span className="meta-label">{label}</span>{" "}
      <span className="meta-value" data-testid={testid}>{value}</span>
    </span>
  );
}

function AssistantMessage({ msg }) {
  const m = msg.metadata;
  const cached = m.resolution_source === "LLM_CACHE";
  return (
    <div className="row assistant">
      <div className="bubble" data-testid="assistant-message">
        <div className="text markdown">
          <ReactMarkdown>{msg.text}</ReactMarkdown>
        </div>
        <div className="meta">
          <span className={"badge " + (cached ? "cache" : "live")}>
            <span data-testid="meta-source">{m.resolution_source}</span>
          </span>
          <Meta label="Model" testid="meta-model" value={m.model} />
          <Meta label="Latency" testid="meta-latency" value={`${m.latency_ms} ms`} />
          <Meta label="Input tokens" testid="meta-input-tokens" value={m.input_tokens} />
          <Meta label="Output tokens" testid="meta-output-tokens" value={m.output_tokens} />
          <span className="meta-item dim">
            key: {m.key_id ?? "none"} · fallbacks: {m.fallback_attempts}
          </span>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [sessionId] = useState(newSessionId); // a new session on every page load
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulate, setSimulate] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  async function send() {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setMessages((prev) => [...prev, { kind: "user", text }]);
    setLoading(true);
    try {
      const data = await sendChat({
        sessionId,
        message: text,
        systemPrompt,
        simulateFailure: simulate,
      });
      setMessages((prev) => [
        ...prev,
        { kind: "assistant", text: data.reply, metadata: data.metadata },
      ]);
    } catch (err) {
      setMessages((prev) => [...prev, { kind: "error", text: err.message }]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="app">
      <header>
        <h1>LLM Chat</h1>
        <span className="session">session {sessionId}</span>
      </header>

      <details className="settings">
        <summary>Session settings</summary>
        <label>
          System prompt (optional)
          <textarea
            data-testid="system-prompt-input"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="e.g. You are a concise assistant."
            maxLength={4000}
            rows={2}
          />
        </label>
        <label>
          Simulate failure (demo of key failover)
          <select value={simulate} onChange={(e) => setSimulate(e.target.value)}>
            {FAILURE_PRESETS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
      </details>

      <main className="chat">
        {messages.length === 0 && (
          <p className="empty">
            Say something to start the conversation. Reload the page and send the same first
            message to see a cache hit.
          </p>
        )}
        {messages.map((msg, i) =>
          msg.kind === "assistant" ? (
            <AssistantMessage key={i} msg={msg} />
          ) : msg.kind === "user" ? (
            <div className="row user" key={i}>
              <div className="bubble">{msg.text}</div>
            </div>
          ) : (
            <div className="row error" key={i} role="alert">
              <div className="bubble">{msg.text}</div>
            </div>
          )
        )}
        {loading && (
          <div className="row assistant">
            <div className="bubble typing">Thinking…</div>
          </div>
        )}
        <div ref={endRef} />
      </main>

      <footer>
        <textarea
          data-testid="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type a message… (Enter to send, Shift+Enter for a new line)"
          rows={2}
          maxLength={4000}
        />
        <button data-testid="send-button" onClick={send} disabled={loading || !input.trim()}>
          Send
        </button>
      </footer>
    </div>
  );
}
