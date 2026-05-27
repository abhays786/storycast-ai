"""Tests for settings.py."""

import importlib
import os
import sys

import settings as settings_module


def test_settings_object_present():
    s = settings_module.settings
    assert isinstance(s.story_model, str) and s.story_model
    assert s.ollama_host.startswith("http")
    assert isinstance(s.gemini_api_key, str)
    assert isinstance(s.enabled_backends, tuple)


def test_parse_enabled_backends_normalizes_whitespace_and_lowercase():
    assert settings_module._parse_enabled_backends("Piper, Gemini ") == ("piper", "gemini")


def test_parse_enabled_backends_skips_empty():
    assert settings_module._parse_enabled_backends("piper,,gemini, ") == ("piper", "gemini")


def test_parse_enabled_backends_empty_string():
    assert settings_module._parse_enabled_backends("") == ()


def test_load_reads_env(monkeypatch):
    monkeypatch.setenv("STORY_MODEL", "llama3:custom")
    monkeypatch.setenv("OLLAMA_HOST", "http://foo:9999")
    monkeypatch.setenv("GEMINI_API_KEY", "abc123 ")
    monkeypatch.setenv("GEMINI_TTS_MODEL", "gemini-xyz")
    monkeypatch.setenv("UI_THEME", "modern")
    monkeypatch.setenv("ENABLED_BACKENDS", "piper,bark")

    s = settings_module._load()

    assert s.story_model == "llama3:custom"
    assert s.ollama_host == "http://foo:9999"
    assert s.gemini_api_key == "abc123"
    assert s.gemini_tts_model == "gemini-xyz"
    assert s.ui_theme == "modern"
    assert s.enabled_backends == ("piper", "bark")


def test_load_uses_defaults_when_env_missing(monkeypatch):
    for v in ("STORY_MODEL", "OLLAMA_HOST", "GEMINI_API_KEY",
              "GEMINI_TTS_MODEL", "UI_THEME", "ENABLED_BACKENDS"):
        monkeypatch.delenv(v, raising=False)
    s = settings_module._load()
    from config import LLM_MODEL_DEFAULT, OLLAMA_HOST_DEFAULT, GEMINI_TTS_MODEL_DEFAULT
    assert s.story_model == LLM_MODEL_DEFAULT
    assert s.ollama_host == OLLAMA_HOST_DEFAULT
    assert s.gemini_tts_model == GEMINI_TTS_MODEL_DEFAULT
    assert s.ui_theme == "classic"
    assert s.enabled_backends == ("piper", "gemini")
