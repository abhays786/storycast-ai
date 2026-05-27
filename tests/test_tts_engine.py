"""Tests for tts_engine.py (the dispatcher)."""

import pytest

import tts_engine
from tts import registry
from tts.base import TTSBackend


def _make_backend(id_, *, result="ok.wav"):
    """Build a fresh TTSBackend subclass + instance bound to `id_`."""
    Calls = {"value": []}
    Models = {"value": False}

    class _B(TTSBackend):
        id = id_

        @property
        def display_name(self):
            return id_.upper()

        def voices(self):
            return []

        def generate(self, text, gender="both", speed=1.0, *,
                     language="en", voice_id=None, output_basename=None):
            Calls["value"].append((text, gender, speed))
            return result

        def ensure_models(self):
            Models["value"] = True

    inst = _B()
    inst._calls = Calls
    inst._models = Models
    return inst


def test_generate_audio_known_engine_success():
    piper = _make_backend("piper")
    registry.register(piper)
    out = tts_engine.generate_audio("hi", "boy", "piper", 1.0)
    assert out == "ok.wav"
    assert piper._calls["value"] == [("hi", "boy", 1.0)]


def test_generate_audio_speed_clamped():
    piper = _make_backend("piper")
    registry.register(piper)
    tts_engine.generate_audio("hi", "boy", "piper", 99.0)
    _, _, speed = piper._calls["value"][0]
    assert speed == 2.0
    tts_engine.generate_audio("hi", "boy", "piper", -1.0)
    _, _, speed = piper._calls["value"][1]
    assert speed == 0.5


def test_generate_audio_unknown_engine_uses_fallback():
    piper = _make_backend("piper")
    registry.register(piper)
    out = tts_engine.generate_audio("hi", "boy", "made-up-engine", 1.0)
    assert out == "ok.wav"
    assert piper._calls["value"]


def test_generate_audio_unknown_engine_no_fallback_returns_none():
    # No piper registered → fallback lookup returns None.
    out = tts_engine.generate_audio("hi", "boy", "anything", 1.0)
    assert out is None


def test_generate_audio_backend_fails_uses_fallback():
    bad    = _make_backend("gemini", result=None)
    fallback = _make_backend("piper", result="fallback.wav")
    registry.register(bad)
    registry.register(fallback)
    out = tts_engine.generate_audio("hi", "boy", "gemini", 1.0)
    assert out == "fallback.wav"
    assert fallback._calls["value"]


def test_generate_audio_fallback_itself_fails_returns_none():
    only_piper = _make_backend("piper", result=None)
    registry.register(only_piper)
    out = tts_engine.generate_audio("hi", "boy", "piper", 1.0)
    assert out is None


def test_generate_audio_no_fallback_registered_returns_none():
    # An engine that isn't the fallback id, and no fallback id registered.
    other = _make_backend("gemini", result=None)
    registry.register(other)
    out = tts_engine.generate_audio("hi", "boy", "gemini", 1.0)
    assert out is None


def test_generate_audio_empty_engine_string_uses_fallback():
    piper = _make_backend("piper")
    registry.register(piper)
    out = tts_engine.generate_audio("hi", "boy", "", 1.0)
    assert out == "ok.wav"


def test_ensure_all_models_invokes_each_backend():
    a = _make_backend("piper")
    b = _make_backend("gemini")
    registry.register(a)
    registry.register(b)
    tts_engine.ensure_all_models()
    assert a._models["value"] is True
    assert b._models["value"] is True
