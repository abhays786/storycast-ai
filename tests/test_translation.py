"""Tests for translation.py."""

import types

import pytest

import translation


class _ResponseError(Exception):
    pass


def _install_fake_ollama(monkeypatch, *, response=None, raise_response=None, raise_other=None):
    fake = types.SimpleNamespace()
    fake.ResponseError = _ResponseError

    class _Client:
        def __init__(self, host):
            self.host = host
        def chat(self, model, messages, options):
            if raise_response is not None:
                raise _ResponseError(raise_response)
            if raise_other is not None:
                raise raise_other
            return {"message": {"content": response or ""}}

    fake.Client = _Client
    monkeypatch.setattr(translation, "ollama", fake)
    monkeypatch.setattr(translation, "_OLLAMA_RESPONSE_ERROR", _ResponseError)


def test_normalize_lang_codes():
    assert translation._normalize_lang(None) == "en"
    assert translation._normalize_lang("") == "en"
    assert translation._normalize_lang("en-IN") == "en"
    assert translation._normalize_lang("hi-IN") == "hi"
    assert translation._normalize_lang("HI") == "hi"
    assert translation._normalize_lang("fr") == "en"   # unknown → fallback


def test_translate_same_language_is_noop():
    text, err = translation.translate("hello", "en", "en")
    assert text == "hello"
    assert err == ""


def test_translate_empty_text_is_noop():
    text, err = translation.translate("", "en", "hi")
    assert text == ""
    assert err == ""


def test_translate_no_ollama(monkeypatch):
    monkeypatch.setattr(translation, "ollama", None)
    text, err = translation.translate("hello", "en", "hi")
    assert text == "hello"
    assert "Translation unavailable" in err


def test_translate_success(monkeypatch):
    _install_fake_ollama(monkeypatch, response="नमस्ते")
    text, err = translation.translate("hello", "en", "hi")
    assert text == "नमस्ते"
    assert err == ""


def test_translate_response_error(monkeypatch):
    _install_fake_ollama(monkeypatch, raise_response="model not found")
    text, err = translation.translate("hello", "en", "hi")
    assert text == "hello"
    assert "Translation failed" in err


def test_translate_generic_error(monkeypatch):
    _install_fake_ollama(monkeypatch, raise_other=ConnectionError("refused"))
    text, err = translation.translate("hello", "en", "hi")
    assert text == "hello"
    assert "Translation failed" in err


def test_translate_empty_response_keeps_original(monkeypatch):
    _install_fake_ollama(monkeypatch, response="")
    text, err = translation.translate("hello", "en", "hi")
    # When Ollama returns nothing, we fall back to the original text
    assert text == "hello"
    assert err == ""
