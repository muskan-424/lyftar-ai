"""Runtime configuration, read entirely from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


REASONING_EFFORTS = {"low", "medium", "high", "disable"}


def _reasoning_effort(value: str | None) -> str:
    """Only a known level is forwarded to the provider; anything else means 'model default'.

    (A stray `# comment` pasted into an empty env value must not reach Gemini as a parameter.)
    """
    level = (value or "").strip().lower()
    return level if level in REASONING_EFFORTS else ""


@dataclass
class Settings:
    llm_api_keys: list[str] = field(default_factory=list)
    model: str = "gemini/gemini-3.5-flash-lite"
    database_url: str = "sqlite:///./data/app.db"
    allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    enable_failure_simulation: bool = False
    cache_enabled: bool = True
    cache_type: str = "local"  # "local" (in-memory) or "disk"
    cache_dir: str = "/data/llm_cache"
    default_system_prompt: str = "You are a helpful assistant."
    request_timeout: float = 30.0
    reasoning_effort: str = ""  # "", "low", "medium", "high", "disable"
    # Offline mode for tests / demos without real keys. Goes through the same
    # LiteLLM code path (cache + failover); only the provider call is faked.
    mock_llm: bool = False
    mock_llm_delay_ms: int = 150

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            llm_api_keys=_csv(env.get("LLM_API_KEYS")),
            model=env.get("LLM_MODEL", cls.model),
            database_url=env.get("DATABASE_URL", cls.database_url),
            allowed_origins=_csv(env.get("ALLOWED_ORIGINS")) or ["http://localhost:3000"],
            enable_failure_simulation=_bool(env.get("ENABLE_FAILURE_SIMULATION")),
            cache_enabled=_bool(env.get("CACHE_ENABLED"), True),
            cache_type=env.get("CACHE_TYPE", cls.cache_type).strip().lower(),
            cache_dir=env.get("CACHE_DIR", cls.cache_dir),
            default_system_prompt=env.get("DEFAULT_SYSTEM_PROMPT", cls.default_system_prompt),
            request_timeout=float(env.get("LLM_TIMEOUT_SECONDS", cls.request_timeout)),
            reasoning_effort=_reasoning_effort(env.get("LLM_REASONING_EFFORT")),
            mock_llm=_bool(env.get("MOCK_LLM")),
            mock_llm_delay_ms=int(env.get("MOCK_LLM_DELAY_MS", cls.mock_llm_delay_ms)),
        )
