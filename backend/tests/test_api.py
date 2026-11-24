import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_client(tmp_path, **overrides):
    settings = Settings(
        llm_api_keys=["fake-key-1", "fake-key-2"],
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        enable_failure_simulation=True,
        mock_llm=True,
        mock_llm_delay_ms=250,
        **overrides,
    )
    return TestClient(create_app(settings))


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def chat(client, session_id, message, headers=None, **extra):
    return client.post(
        "/chat", json={"session_id": session_id, "message": message, **extra}, headers=headers or {}
    )


# ---------- health ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


# ---------- chat contract ----------

def test_chat_response_shape(client):
    r = chat(client, "s-shape", "What is the capital of France?")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "s-shape"
    assert body["reply"]
    meta = body["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["model"] == "gemini/gemini-3.5-flash-lite"
    assert isinstance(meta["latency_ms"], int)
    assert meta["input_tokens"] > 0 and meta["output_tokens"] > 0
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0


# ---------- caching ----------

def test_same_first_message_in_new_session_hits_cache(client):
    live = chat(client, "s-a", "Tell me a fact about cats").json()
    hit = chat(client, "s-b", "Tell me a fact about cats").json()
    assert live["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert hit["metadata"]["resolution_source"] == "LLM_CACHE"
    assert hit["reply"] == live["reply"]
    assert hit["metadata"]["key_id"] is None
    assert hit["metadata"]["fallback_attempts"] == 0
    assert hit["metadata"]["input_tokens"] == live["metadata"]["input_tokens"]
    assert hit["metadata"]["output_tokens"] == live["metadata"]["output_tokens"]
    assert hit["metadata"]["latency_ms"] < live["metadata"]["latency_ms"]


def test_different_message_never_comes_from_cache(client):
    chat(client, "s-c", "alpha question")
    other = chat(client, "s-d", "beta question").json()
    assert other["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_different_system_prompt_is_a_cache_miss(client):
    chat(client, "s-e", "hello there")
    other = chat(client, "s-f", "hello there", system_prompt="Answer like a pirate.").json()
    assert other["metadata"]["resolution_source"] == "LIVE_INFERENCE"


# ---------- tokens ----------

def test_tokens_are_per_request_not_cumulative(client):
    first = chat(client, "s-t1", "one two three").json()["metadata"]
    second = chat(client, "s-t1", "four five six").json()["metadata"]
    other = chat(client, "s-t2", "one two three").json()["metadata"]
    # second turn includes history, so it is bigger than a fresh first turn, but
    # a new session with the same-length message reports the same tokens as the first.
    assert second["input_tokens"] > first["input_tokens"]
    assert other["input_tokens"] == first["input_tokens"]


# ---------- multi-turn + history ----------

def test_multi_turn_history_and_isolation(client):
    chat(client, "s-h1", "first question")
    chat(client, "s-h1", "second question")
    chat(client, "s-h2", "unrelated question")

    h1 = client.get("/sessions/s-h1/messages").json()
    assert h1["session_id"] == "s-h1"
    assert [m["role"] for m in h1["messages"]] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in h1["messages"] if m["role"] == "user"] == [
        "first question",
        "second question",
    ]
    assert all("system" != m["role"] for m in h1["messages"])
    assistant_meta = h1["messages"][1]["metadata"]
    assert {"resolution_source", "input_tokens", "output_tokens"} <= set(assistant_meta)
    assert "metadata" not in h1["messages"][0]

    h2 = client.get("/sessions/s-h2/messages").json()
    assert len(h2["messages"]) == 2


def test_history_is_sent_to_the_model(client, monkeypatch):
    seen = []
    from app import llm

    original = llm.LLMGateway._call

    def spy(self, messages, api_key, bypass_cache):
        seen.append([m["content"] for m in messages])
        return original(self, messages, api_key, bypass_cache)

    monkeypatch.setattr(llm.LLMGateway, "_call", spy)
    chat(client, "s-mem", "my name is Zed", system_prompt="Be brief.")
    chat(client, "s-mem", "what is my name?")
    assert seen[1][0] == "Be brief."  # session system prompt is remembered
    assert "my name is Zed" in seen[1]  # earlier turn is in the context


def test_unknown_session_is_404(client):
    r = client.get("/sessions/does-not-exist/messages")
    assert r.status_code == 404
    assert r.json() == {"detail": "session not found"}


def test_history_survives_app_restart(tmp_path):
    with make_client(tmp_path) as c1:
        chat(c1, "s-persist", "remember me")
    with make_client(tmp_path) as c2:
        assert len(c2.get("/sessions/s-persist/messages").json()["messages"]) == 2


# ---------- failover ----------

def test_simulated_429_on_key_1_falls_over_to_key_2(client):
    r = chat(client, "s-f1", "failover please", headers={"X-Simulate-Failure": "key_1=429"})
    assert r.status_code == 200
    meta = r.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["resolution_source"] == "LIVE_INFERENCE"


def test_simulated_500_on_key_1_falls_over_to_key_2(client):
    r = chat(client, "s-f2", "failover 500", headers={"X-Simulate-Failure": "key_1=500"})
    assert r.status_code == 200 and r.json()["metadata"]["key_id"] == "key_2"


def test_simulated_failure_is_answered_live_even_if_cached(client):
    chat(client, "s-f3a", "cached then failover")
    r = chat(client, "s-f3b", "cached then failover", headers={"X-Simulate-Failure": "key_1=429"})
    assert r.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert r.json()["metadata"]["key_id"] == "key_2"


def test_all_keys_failing_returns_503_and_stores_nothing(client):
    r = chat(client, "s-f4", "everything fails", headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}
    assert client.get("/sessions/s-f4/messages").status_code == 404


def test_both_keys_listed_returns_503(client):
    r = chat(client, "s-f5", "both fail", headers={"X-Simulate-Failure": "key_1=429,key_2=500"})
    assert r.status_code == 503


def test_failure_hook_is_ignored_when_switch_is_off(tmp_path):
    settings = Settings(
        llm_api_keys=["a", "b"],
        database_url=f"sqlite:///{tmp_path / 'off.db'}",
        enable_failure_simulation=False,
        mock_llm=True,
    )
    with TestClient(create_app(settings)) as c:
        r = chat(c, "s-off", "hook disabled", headers={"X-Simulate-Failure": "*=429"})
        assert r.status_code == 200 and r.json()["metadata"]["key_id"] == "key_1"


def test_real_provider_error_triggers_failover(client, monkeypatch):
    from app import llm

    original = llm.LLMGateway._call

    def flaky(self, messages, api_key, bypass_cache):
        if api_key == "fake-key-1":
            raise RuntimeError("429 quota exceeded")
        return original(self, messages, api_key, bypass_cache)

    monkeypatch.setattr(llm.LLMGateway, "_call", flaky)
    r = chat(client, "s-real", "provider is down for key one")
    assert r.status_code == 200
    assert r.json()["metadata"]["key_id"] == "key_2"
    assert r.json()["metadata"]["fallback_attempts"] == 1


# ---------- validation ----------

@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "s", "message": ""},
        {"session_id": "s", "message": "   "},
        {"session_id": "", "message": "hi"},
        {"message": "hi"},
        {"session_id": "s"},
        {"session_id": "s", "message": "x" * 4001},
        {"session_id": "s" * 129, "message": "hi"},
        {"session_id": "s", "message": "hi", "system_prompt": "x" * 4001},
    ],
)
def test_validation_errors_are_422_json(client, payload):
    r = client.post("/chat", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_malformed_json_is_422(client):
    r = client.post("/chat", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_message_is_trimmed(client):
    chat(client, "s-trim", "   padded   ")
    msgs = client.get("/sessions/s-trim/messages").json()["messages"]
    assert msgs[0]["content"] == "padded"


def test_cors_allows_configured_origin(client):
    r = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-simulate-failure",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
