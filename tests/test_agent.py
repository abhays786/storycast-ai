"""Tests for agent.py."""

import types

import pytest

import agent


class _ResponseError(Exception):
    pass


def _install_fake_ollama(monkeypatch, *, raise_response=False, raise_other=None, message=None, response=None):
    fake_ollama = types.SimpleNamespace()
    fake_ollama.ResponseError = _ResponseError

    class _Client:
        def __init__(self, host):
            self.host = host
        def chat(self, model, messages, options):
            if raise_response is not False:
                raise _ResponseError(raise_response)
            if raise_other is not None:
                raise raise_other
            return response or {"message": {"content": message}}

    fake_ollama.Client = _Client
    monkeypatch.setattr(agent, "ollama", fake_ollama)
    monkeypatch.setattr(agent, "_OLLAMA_RESPONSE_ERROR", _ResponseError)


# ── Input guardrail short-circuits ───────────────────────────────────────────

def test_safety_rejects_empty():
    title, body, err = agent.generate_story("", "both")
    assert title == "" and body == ""
    assert "enter a story topic" in err


def test_safety_rejects_blocked():
    title, body, err = agent.generate_story("a story about killing", "both")
    assert err and title == ""


# ── No ollama installed ──────────────────────────────────────────────────────

def test_no_ollama_returns_friendly_message(monkeypatch):
    monkeypatch.setattr(agent, "ollama", None)
    title, body, err = agent.generate_story("A friendly dragon and a robot", "both")
    assert title == ""
    assert "not installed" in err


# ── Success paths ────────────────────────────────────────────────────────────

def test_success_with_title(monkeypatch):
    body_text = "Once upon a time " * 30
    msg = f"Title: The Brave Dragon\n\n{body_text}"
    _install_fake_ollama(monkeypatch, raise_response=False, message=msg)
    title, body, err = agent.generate_story("A friendly dragon", "both")
    assert err == ""
    assert title == "The Brave Dragon"
    assert "Once upon a time" in body


def test_success_without_title_uses_default(monkeypatch):
    body_text = "Big text " * 30
    _install_fake_ollama(monkeypatch, raise_response=False, message=body_text)
    title, body, err = agent.generate_story("Big tale", "boy")
    assert err == ""
    assert title == "A Wonderful Story"
    assert body.strip().startswith("Big text")


def test_success_with_quoted_title(monkeypatch):
    msg = 'Title: "A Quoted Title"\n\n' + ("blah " * 60)
    _install_fake_ollama(monkeypatch, raise_response=False, message=msg)
    title, _, err = agent.generate_story("A friendly dragon", "girl")
    assert err == ""
    assert title == "A Quoted Title"


def test_gender_focus_unknown_value(monkeypatch):
    body_text = "story text " * 30
    _install_fake_ollama(monkeypatch, raise_response=False, message=body_text)
    title, _, err = agent.generate_story("A bird", "alien")
    assert err == ""


# ── Failure paths ────────────────────────────────────────────────────────────

def test_response_error_model_not_found(monkeypatch):
    _install_fake_ollama(monkeypatch, raise_response="model 'foo' not found")
    title, _, err = agent.generate_story("A friendly dragon and a robot", "both")
    assert title == ""
    assert "is not pulled yet" in err


def test_response_error_other(monkeypatch):
    _install_fake_ollama(monkeypatch, raise_response="some other ollama error")
    _, _, err = agent.generate_story("A friendly dragon and a robot", "both")
    assert "Ollama error" in err


def test_connection_error(monkeypatch):
    _install_fake_ollama(monkeypatch, raise_other=ConnectionError("connection refused"))
    _, _, err = agent.generate_story("A friendly dragon and a robot", "both")
    assert "Cannot reach Ollama" in err


def test_unexpected_exception(monkeypatch):
    _install_fake_ollama(monkeypatch, raise_other=ValueError("weird"))
    _, _, err = agent.generate_story("A friendly dragon and a robot", "both")
    assert "Unexpected error" in err


def test_output_guardrail_rejects_short_story(monkeypatch):
    _install_fake_ollama(monkeypatch, raise_response=False, message="Title: X\n\nshort body")
    _, _, err = agent.generate_story("A friendly dragon", "both")
    assert "too short" in err.lower()


def test_output_guardrail_rejects_blocked_term(monkeypatch):
    body = "Title: X\n\n" + ("explicit " * 50)
    _install_fake_ollama(monkeypatch, raise_response=False, message=body)
    _, _, err = agent.generate_story("A friendly dragon", "both")
    assert "inappropriate" in err.lower()


def test_hindi_uses_hindi_model(monkeypatch):
    captured = {}
    fake_ollama = types.SimpleNamespace()
    fake_ollama.ResponseError = _ResponseError
    class _Client:
        def __init__(self, host): pass
        def chat(self, model, messages, options):
            captured["model"]   = model
            captured["system"]  = messages[0]["content"]
            return {"message": {"content": "Title: कहानी\n\n" + ("कहानी " * 60)}}
    fake_ollama.Client = _Client
    monkeypatch.setattr(agent, "ollama", fake_ollama)
    monkeypatch.setattr(agent, "_OLLAMA_RESPONSE_ERROR", _ResponseError)

    # Override hindi model
    from settings import settings
    monkeypatch.setattr(agent, "settings", type(settings)(
        story_model="llama3", hindi_story_model="llama3-hindi",
        translation_model="llama3", ollama_host="http://x",
        gemini_api_key="", gemini_tts_model="g", ui_theme="classic",
        enabled_backends=("piper",),
    ))

    title, body, err = agent.generate_story("A friendly dragon", "both", language="hi")
    assert err == ""
    assert title == "कहानी"
    assert captured["model"] == "llama3-hindi"
    # Hindi instruction made it into the system prompt
    assert "हिंदी" in captured["system"]


def test_long_title_is_truncated(monkeypatch):
    long_title = "X" * 80
    msg = f"Title: {long_title}\n\n" + ("body " * 60)
    _install_fake_ollama(monkeypatch, raise_response=False, message=msg)
    title, _, err = agent.generate_story("A friendly dragon", "both")
    assert err == ""
    assert len(title) <= 50
