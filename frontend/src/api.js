const BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");

/** Turn any backend/network failure into a message a person can read. */
function describeError(status, body) {
  const detail = body && body.detail;
  if (status === 503) return "All AI providers are unavailable right now. Please try again in a moment.";
  if (status === 422) {
    if (Array.isArray(detail) && detail.length) {
      return "Invalid request: " + detail.map((d) => d.msg || String(d)).join("; ");
    }
    return "Invalid request: " + (typeof detail === "string" ? detail : "please check your input.");
  }
  if (typeof detail === "string") return detail;
  return `Something went wrong (HTTP ${status}).`;
}

export async function sendChat({ sessionId, message, systemPrompt, simulateFailure }) {
  const headers = { "Content-Type": "application/json" };
  if (simulateFailure) headers["X-Simulate-Failure"] = simulateFailure;

  const payload = { session_id: sessionId, message };
  if (systemPrompt && systemPrompt.trim()) payload.system_prompt = systemPrompt.trim();

  let res;
  try {
    res = await fetch(`${BASE}/chat`, { method: "POST", headers, body: JSON.stringify(payload) });
  } catch {
    throw new Error("Cannot reach the server. Check your connection and try again.");
  }

  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON error page */
  }
  if (!res.ok) throw new Error(describeError(res.status, body));
  return body;
}
