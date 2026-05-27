"""Tests for tts/gemini_backend.py."""

import sys
import types
from pathlib import Path

import pytest

from tts import gemini_backend
from tts.gemini_backend import GeminiBackend


def _install_genai(monkeypatch, *, data=b"x" * 5000, mime="audio/pcm", raise_=None):
    google_mod = types.ModuleType("google")
    genai_mod  = types.ModuleType("google.genai")
    types_mod  = types.ModuleType("google.genai.types")

    types_mod.GenerateContentConfig = lambda **kw: kw
    types_mod.SpeechConfig = lambda **kw: kw
    types_mod.VoiceConfig = lambda **kw: kw
    types_mod.PrebuiltVoiceConfig = lambda **kw: kw

    class _Models:
        def generate_content(self, model, contents, config):
            if raise_ is not None:
                raise raise_
            part = types.SimpleNamespace(
                inline_data=types.SimpleNamespace(mime_type=mime, data=data)
            )
            cand = types.SimpleNamespace(
                content=types.SimpleNamespace(parts=[part])
            )
            return types.SimpleNamespace(candidates=[cand])

    class _Client:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = _Models()

    genai_mod.Client = _Client
    genai_mod.types  = types_mod

    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)


@pytest.fixture
def with_key(monkeypatch):
    fake_settings = types.SimpleNamespace(
        gemini_api_key="sk-test",
        gemini_tts_model="gemini-flash-tts",
    )
    monkeypatch.setattr(gemini_backend, "settings", fake_settings)
    return fake_settings


def test_display_name():
    assert "Gemini" in GeminiBackend().display_name


def test_is_available_with_key(monkeypatch):
    monkeypatch.setattr(gemini_backend, "settings",
                        types.SimpleNamespace(gemini_api_key="x", gemini_tts_model="m"))
    assert GeminiBackend().is_available() is True


def test_is_available_no_key(monkeypatch):
    monkeypatch.setattr(gemini_backend, "settings",
                        types.SimpleNamespace(gemini_api_key="", gemini_tts_model="m"))
    assert GeminiBackend().is_available() is False


def test_generate_missing_genai(monkeypatch, with_key):
    monkeypatch.delitem(sys.modules, "google.genai", raising=False)
    monkeypatch.delitem(sys.modules, "google.genai.types", raising=False)
    import builtins
    real_import = builtins.__import__
    def fail(name, *a, **k):
        if "google" in name:
            raise ImportError("no genai")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fail)
    assert GeminiBackend().generate("hi", "boy", 1.0) is None


def test_generate_no_api_key(monkeypatch):
    monkeypatch.setattr(gemini_backend, "settings",
                        types.SimpleNamespace(gemini_api_key="", gemini_tts_model="m"))
    _install_genai(monkeypatch)
    assert GeminiBackend().generate("hi", "boy", 1.0) is None


def test_generate_pcm_path(monkeypatch, with_key):
    _install_genai(monkeypatch, mime="audio/pcm", data=b"\x00\x00" * 2000)
    out = GeminiBackend().generate("hello", "girl", 1.0)
    assert out is not None
    assert Path(out).exists()
    assert Path(out).stat().st_size > 1024


def test_generate_wav_path(monkeypatch, with_key):
    _install_genai(monkeypatch, mime="audio/wav", data=b"RIFF" + b"x" * 5000)
    out = GeminiBackend().generate("hello", "boy", 1.0)
    assert out is not None
    assert Path(out).read_bytes().startswith(b"RIFF")


def test_generate_data_already_bytes(monkeypatch, with_key):
    _install_genai(monkeypatch, mime="audio/pcm", data=bytearray(b"\x00" * 4000))
    out = GeminiBackend().generate("hello", "both", 1.0)
    assert out is not None


def test_generate_unknown_gender_falls_back(monkeypatch, with_key):
    _install_genai(monkeypatch)
    out = GeminiBackend().generate("hi", "alien", 1.0)
    assert out is not None


def test_generate_tiny_file_returns_none(monkeypatch, with_key):
    _install_genai(monkeypatch, mime="audio/pcm", data=b"\x00" * 100)
    out = GeminiBackend().generate("hi", "boy", 1.0)
    assert out is None


def test_generate_api_exception_returns_none(monkeypatch, with_key):
    _install_genai(monkeypatch, raise_=RuntimeError("API failed"))
    assert GeminiBackend().generate("hi", "boy", 1.0) is None


def test_generate_returns_none_when_no_voice(monkeypatch, with_key):
    _install_genai(monkeypatch)
    monkeypatch.setattr(gemini_backend, "VOICES", [])
    assert GeminiBackend().generate("hi", "boy", 1.0) is None


def test_voices_returns_catalog():
    voices = GeminiBackend().voices()
    # Three base voices × 2 languages = 6 entries
    assert len(voices) == 6


def test_generate_applies_speed(monkeypatch, with_key):
    _install_genai(monkeypatch, mime="audio/pcm", data=b"\x00\x00" * 4000)
    captured = {}
    real_speed = gemini_backend.apply_speed_pydub
    def trace(path, speed):
        captured["speed"] = speed
        return real_speed(path, speed)
    monkeypatch.setattr(gemini_backend, "apply_speed_pydub", trace)
    GeminiBackend().generate("hi", "girl", 1.2)
    assert captured["speed"] == pytest.approx(1.2)
