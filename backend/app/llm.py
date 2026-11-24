"""LiteLLM gateway: key pool with automatic failover and built-in response caching."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import litellm
from litellm import ModelResponse
from litellm.caching.caching import Cache

from .config import Settings

log = logging.getLogger("app.llm")

litellm.telemetry = False
litellm.suppress_debug_info = True
litellm.drop_params = True

SIMULATABLE_STATUSES = {429, 500, 502, 503}


class AllProvidersUnavailable(Exception):
    """Every key in the pool failed (rate limit, quota, 5xx, timeout, ...)."""


class SimulatedProviderError(Exception):
    """Stand-in for a provider error, raised by the X-Simulate-Failure hook."""

    def __init__(self, status_code: int):
        super().__init__(f"simulated provider error {status_code}")
        self.status_code = status_code


@dataclass
class LLMResult:
    reply: str
    model: str
    input_tokens: int
    output_tokens: int
    resolution_source: str  # LIVE_INFERENCE | LLM_CACHE
    key_id: str | None
    fallback_attempts: int


def parse_simulation_header(value: str | None) -> dict[str, int]:
    """'key_1=429,key_2=500' / '*=429' -> {'key_1': 429, 'key_2': 500}. Bad pairs are ignored."""
    rules: dict[str, int] = {}
    for part in (value or "").split(","):
        key, sep, status = part.partition("=")
        key, status = key.strip(), status.strip()
        if sep and key and status.isdigit() and int(status) in SIMULATABLE_STATUSES:
            rules[key] = int(status)
    return rules


class LLMGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.model
        self.keys = {f"key_{i}": key for i, key in enumerate(settings.llm_api_keys, start=1)}
        self.cache_enabled = settings.cache_enabled
        if self.cache_enabled:
            if settings.cache_type == "disk":
                litellm.cache = Cache(type="disk", disk_cache_dir=settings.cache_dir)
            else:
                litellm.cache = Cache(type="local")
        else:
            litellm.cache = None
        if self.cache_enabled:
            self._warm_up_cache()

    @property
    def keys_configured(self) -> int:
        return len(self.keys)

    def complete(self, messages: list[dict], simulate: dict[str, int] | None = None) -> LLMResult:
        """Answer `messages`, trying keys key_1, key_2, ... until one succeeds.

        The cache sits in front of the key pool (LiteLLM looks up the cache before
        the provider call), so a cache hit never consumes a key.
        """
        simulate = simulate or {}
        failures = 0
        for key_id, api_key in self.keys.items():
            status = simulate.get(key_id, simulate.get("*"))
            try:
                if status is not None:
                    raise SimulatedProviderError(status)
                response = self._call(messages, api_key, bypass_cache=bool(simulate))
            except Exception as exc:  # any provider failure -> try the next key
                failures += 1
                log.warning(
                    "%s failed (%s: %s); %s",
                    key_id,
                    type(exc).__name__,
                    getattr(exc, "status_code", "n/a"),
                    "trying next key" if failures < len(self.keys) else "no keys left",
                )
                continue

            reply = (response.choices[0].message.content or "").strip()
            if not reply:  # an empty answer is as useless as an error: fail over
                failures += 1
                log.warning("%s returned an empty reply; trying next key", key_id)
                continue

            cache_hit = bool((getattr(response, "_hidden_params", None) or {}).get("cache_hit"))
            usage = getattr(response, "usage", None)
            return LLMResult(
                reply=reply,
                model=self.model,
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                resolution_source="LLM_CACHE" if cache_hit else "LIVE_INFERENCE",
                key_id=None if cache_hit else key_id,
                fallback_attempts=0 if cache_hit else failures,
            )
        raise AllProvidersUnavailable()

    def _call(self, messages: list[dict], api_key: str, bypass_cache: bool):
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "api_key": api_key,
            "timeout": self.settings.request_timeout,
            "num_retries": 0,  # failover is handled across keys, not by retrying one key
        }
        if self.settings.reasoning_effort:
            kwargs["reasoning_effort"] = self.settings.reasoning_effort
        if bypass_cache:
            # Failure-simulation requests must be answered live, never from cache.
            kwargs["cache"] = {"no-cache": True}
        if self.settings.mock_llm:
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            kwargs["mock_response"] = self._mock_response(messages, f"[mock] You said: {last_user}")
            kwargs["mock_delay"] = self.settings.mock_llm_delay_ms / 1000
        return litellm.completion(**kwargs)

    def _mock_response(self, messages: list[dict], text: str) -> ModelResponse:
        """A canned reply whose token usage scales with the conversation, like a real model.

        (LiteLLM's plain string mock always reports 10 input / 20 output tokens.)
        """
        prompt = litellm.token_counter(model=self.model, messages=messages)
        completion = litellm.token_counter(model=self.model, text=text)
        return ModelResponse(
            model=self.model,
            choices=[{"message": {"role": "assistant", "content": text}, "finish_reason": "stop", "index": 0}],
            usage={
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        )

    def _warm_up_cache(self) -> None:
        """Run one cached round trip at startup.

        LiteLLM lazily imports a large chunk of its proxy package the first time it
        serves a cache hit (seconds on a small machine). Doing it here keeps that cost
        off the first real user request. Uses a mock response, so no API key is spent.
        """
        messages = [{"role": "user", "content": "__cache_warm_up__"}]
        try:
            for _ in range(2):
                litellm.completion(
                    model=self.model,
                    messages=messages,
                    api_key="warm-up",
                    mock_response=self._mock_response(messages, "ok"),
                )
        except Exception as exc:  # warm-up is best effort
            log.warning("cache warm-up skipped (%s: %s)", type(exc).__name__, exc)

