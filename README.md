# LLM Chatbot with Key Failover and Response Caching

A small full-stack chat app: **React** frontend, **FastAPI** backend, **SQLite** storage and a
**LiteLLM** gateway in front of **Gemini 3.5 Flash** (`gemini/gemini-3.5-flash`).

- Multi-turn conversations stored per session in SQLite
- A pool of API keys with automatic failover on 429 / quota / 5xx errors
- LiteLLM response caching, reported as `LLM_CACHE` vs `LIVE_INFERENCE`
- Per-request token usage, latency, model and key shown next to every reply

## 1. Live URLs

| | URL |
|---|---|
| Frontend | https://lyftar-ai.vercel.app |
| Backend | https://llm-chat-backend-cp5q.onrender.com (`/health`, `/chat`, `/sessions/{id}/messages`) |
| Repository | https://github.com/muskan-424/lyftar-ai |
| Demo video | _add_ |

The backend runs on Render's free plan: it sleeps after about 15 minutes idle (the first request
then takes around a minute) and its SQLite history resets on redeploy and after a sleep.

Deploying: backend as a Docker web service (see `render.yaml`; set `LLM_API_KEYS`,
`ALLOWED_ORIGINS=<frontend url>`, `ENABLE_FAILURE_SIMULATION=true`), frontend on
Vercel/Netlify/Cloudflare Pages (root `frontend`, build `npm run build`, output `dist`,
env `VITE_API_BASE_URL=<backend url>`).

## 2. Run locally

### With Docker

```bash
cp .env.example .env      # then put 2+ real Gemini keys in LLM_API_KEYS
make up                   # frontend http://localhost:3000, backend http://localhost:8000
make down
```

`make up` has no external dependencies besides Docker. To try the app without keys, set
`MOCK_LLM=true` in `.env` (the provider call is faked; caching and failover still run through
LiteLLM). SQLite lives in the named volume `app-data`, mounted at `/data`, so history survives
`make down` / `make up`.

### Without Docker

```bash
# backend (Python 3.12)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export LLM_API_KEYS=key1,key2 DATABASE_URL=sqlite:///./data/app.db \
       ALLOWED_ORIGINS=http://localhost:3000 ENABLE_FAILURE_SIMULATION=true
uvicorn app.main:app --port 8000

# frontend (Node 20+), in another terminal
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev      # http://localhost:3000
```

Tests: `cd backend && pytest` (or `make test`, which runs them in a container). They use
`MOCK_LLM`, so no keys or network are needed.

### Configuration

All configuration is environment variables; see `.env.example` for the full list
(`LLM_API_KEYS`, `DATABASE_URL`, `ALLOWED_ORIGINS`, `ENABLE_FAILURE_SIMULATION`, plus optional
`LLM_MODEL`, `CACHE_ENABLED`, `CACHE_TYPE`, `LLM_REASONING_EFFORT`, ...).

## 3. Design decisions

### Key fallback
`LLM_API_KEYS` is parsed into `key_1, key_2, ...`. For each request the gateway
(`backend/app/llm.py`) tries the keys in order, calling `litellm.completion(api_key=...)` with
`num_retries=0`; any failure (429, quota, 5xx, timeout, empty reply) moves on to the next key and
increments `fallback_attempts`. If every key fails the API returns
`503 {"detail": "all providers unavailable"}`. Keys are never logged or returned; only the id
(`key_1`) is exposed. I fail over per call rather than using a router's cooldown state so the
reported `key_id` / `fallback_attempts` are exact and deterministic for each request.

### Caching
`litellm.cache = Cache(type="local")` (in-memory; `CACHE_TYPE=disk` switches to a disk cache).
LiteLLM builds the cache key from the model and the full message list (system prompt + history +
new message), so the same first message in a new session is a hit while any different message,
or the same message after a different history or system prompt, is a miss. A hit is detected
through LiteLLM's `cache_hit` flag, returns the stored reply and stored token usage, and reports
`key_id: null`, `fallback_attempts: 0`. Requests carrying `X-Simulate-Failure` bypass cache
reads, so failover can always be demonstrated even for previously seen messages.

### SQLite session storage
Stdlib `sqlite3`, file at `DATABASE_URL` (default `sqlite:////data/app.db` on a Docker volume).
The schema is created on startup. Tables: `messages` (as specified in the brief, plus an index on
`session_id`) and `sessions` (stores the per-session system prompt). The client sends only the new
message; the server rebuilds the context from SQLite. A user+assistant exchange is written in a
single transaction only after a successful reply, so 422/503 requests never touch history.
Per-request token counts come from the provider's usage for that request, never a running total.

## 4. Verification

Replace `$API` with the backend URL.

```bash
curl $API/health
# {"status":"ok","keys_configured":2,"cache_enabled":true}

# Live call, then the same first message in a NEW session -> cache hit
curl -s $API/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-1","message":"What is the capital of France?"}'
curl -s $API/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-2","message":"What is the capital of France?"}'
# 2nd: resolution_source LLM_CACHE, identical reply, key_id null, lower latency_ms

# Failover
curl -s $API/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=429' \
  -d '{"session_id":"s-3","message":"hello"}'     # 200, key_id key_2, fallback_attempts 1
curl -s -i $API/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: *=429' \
  -d '{"session_id":"s-4","message":"hello"}'     # 503 {"detail":"all providers unavailable"}
curl -i $API/sessions/s-4/messages               # 404: failed request was not stored

# History and validation
curl $API/sessions/s-1/messages
curl -i $API/chat -H 'Content-Type: application/json' -d '{"session_id":"s","message":""}'   # 422
```

In the UI each reply shows source, model, latency and input/output tokens (for that request only).
Open **Session settings** to set a system prompt or pick a simulated failure.

## 5. Setup Used

- Editor: VS Code with the Claude Code extension
- AI tools: Claude Code (Claude Sonnet 5.5) for scaffolding, implementation and tests
- Runtime: Python 3.12, FastAPI, LiteLLM, SQLite, React 18 + Vite, Docker Compose

## 6. Bonus features

- Failure-simulation picker in the UI (no curl needed to demo failover)
- Optional per-session system prompt that persists for the session
- `MOCK_LLM` offline mode so the whole stack and the test-suite run without keys
- Disk cache option (`CACHE_TYPE=disk`), non-root backend image with a healthcheck
- Backend test suite (cache, failover, validation, persistence, CORS)
