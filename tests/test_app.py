"""Tests for app.py."""

import sys
import types

import pytest

import app


# ── check_ollama ────────────────────────────────────────────────────────────

def _install_requests(monkeypatch, *, raise_=None, models=None):
    mod = types.ModuleType("requests")

    class _Resp:
        def __init__(self, data):
            self._data = data
        def json(self):
            return {"models": [{"name": m} for m in (self._data or [])]}

    def get(url, timeout):
        if raise_ is not None:
            raise raise_
        return _Resp(models)

    mod.get = get
    monkeypatch.setitem(sys.modules, "requests", mod)
    return mod


def test_check_ollama_model_present(monkeypatch):
    _install_requests(monkeypatch, models=["llama3:latest", "mistral"])
    assert app.check_ollama() is True


def test_check_ollama_model_missing(monkeypatch):
    _install_requests(monkeypatch, models=["mistral"])
    assert app.check_ollama() is False


def test_check_ollama_connection_failure(monkeypatch):
    _install_requests(monkeypatch, raise_=RuntimeError("connection refused"))
    assert app.check_ollama() is False


def test_check_ollama_model_name_with_tag_matches(monkeypatch):
    # If settings.story_model is "llama3" and pulled is "llama3:latest",
    # the prefix-match branch must succeed.
    _install_requests(monkeypatch, models=["llama3:latest"])
    assert app.check_ollama() is True


# ── main ────────────────────────────────────────────────────────────────────

def test_main_wires_components(monkeypatch):
    calls = {"bootstrap": 0, "ensure": 0, "build": 0, "check": 0, "launch": 0}

    monkeypatch.setattr(app, "register_default_backends",
                        lambda: calls.__setitem__("bootstrap", calls["bootstrap"] + 1))
    monkeypatch.setattr(app, "ensure_all_models",
                        lambda: calls.__setitem__("ensure", calls["ensure"] + 1))
    monkeypatch.setattr(app, "check_ollama",
                        lambda: (calls.__setitem__("check", calls["check"] + 1), True)[1])

    class _Blocks:
        def queue(self):
            class _Q:
                def launch(self_, **kw):
                    calls["launch"] += 1
                    calls["launch_kwargs"] = kw
            return _Q()

    class _Built:
        blocks = _Blocks()
        css = "/* css */"

    monkeypatch.setattr(app, "build_app", lambda theme: (
        calls.__setitem__("build", calls["build"] + 1), _Built()
    )[1])

    app.main()

    assert calls["bootstrap"] == 1
    assert calls["ensure"] == 1
    assert calls["check"] == 1
    assert calls["build"] == 1
    assert calls["launch"] == 1
    assert calls["launch_kwargs"]["css"] == "/* css */"
    assert calls["launch_kwargs"]["server_port"] == 7860
