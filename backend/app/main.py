"""FastAPI app: /health, /chat, /sessions/{session_id}/messages."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .db import Database
from .llm import AllProvidersUnavailable, LLMGateway, parse_simulation_header
from .schemas import ChatMetadata, ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = Database(settings.database_url)
    gateway: LLMGateway | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal gateway
        db.init()  # schema is created automatically on startup
        gateway = LLMGateway(settings)
        if gateway.keys_configured < 2:
            log.warning("only %d API key(s) configured; failover needs at least 2", gateway.keys_configured)
        log.info(
            "ready: model=%s keys=%d cache=%s simulation=%s",
            settings.model, gateway.keys_configured, gateway.cache_enabled,
            settings.enable_failure_simulation,
        )
        yield

    app = FastAPI(title="LLM Chatbot with Key Failover and Caching", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],  # includes X-Simulate-Failure
    )

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.get("/health")
    def health():
        ready = db.ping() and gateway is not None and gateway.keys_configured > 0
        body = {
            "status": "ok" if ready else "unavailable",
            "keys_configured": gateway.keys_configured if gateway else 0,
            "cache_enabled": bool(gateway and gateway.cache_enabled),
        }
        return JSONResponse(status_code=200 if ready else 503, content=body)

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        body: ChatRequest,
        x_simulate_failure: str | None = Header(default=None),
    ):
        started = time.perf_counter()
        simulate = (
            parse_simulation_header(x_simulate_failure)
            if settings.enable_failure_simulation
            else {}
        )

        history = db.get_history(body.session_id)
        system_prompt = (
            body.system_prompt
            or db.get_system_prompt(body.session_id)
            or settings.default_system_prompt
        )
        messages = (
            [{"role": "system", "content": system_prompt}]
            + [{"role": m["role"], "content": m["content"]} for m in history]
            + [{"role": "user", "content": body.message}]
        )

        try:
            result = gateway.complete(messages, simulate)
        except AllProvidersUnavailable:
            raise HTTPException(status_code=503, detail="all providers unavailable")

        metadata = ChatMetadata(
            resolution_source=result.resolution_source,
            model=result.model,
            latency_ms=int(round((time.perf_counter() - started) * 1000)),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )
        # Only successful exchanges are stored; 422/503 never reach this line.
        db.save_exchange(
            body.session_id, body.system_prompt, body.message, result.reply, metadata.model_dump()
        )
        return ChatResponse(session_id=body.session_id, reply=result.reply, metadata=metadata)

    @app.get("/sessions/{session_id}/messages")
    def session_messages(session_id: str):
        if not db.session_exists(session_id):
            raise HTTPException(status_code=404, detail="session not found")
        return {"session_id": session_id, "messages": db.get_history(session_id)}

    return app


app = create_app()
