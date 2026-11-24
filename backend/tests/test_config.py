import pytest

from app.config import Settings


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, ""),
        ("", ""),
        ("   ", ""),
        ("low", "low"),
        (" High ", "high"),
        ("disable", "disable"),
        # An inline comment after an empty value (Docker Compose passes it through as the value)
        ("# empty = model default; low | medium | high | disable", ""),
        ("turbo", ""),
    ],
)
def test_reasoning_effort_only_accepts_known_levels(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("LLM_REASONING_EFFORT", raw)
    assert Settings.from_env().reasoning_effort == expected
