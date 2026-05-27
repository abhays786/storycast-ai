"""Tests for stt.py."""

import sys
import types

import pytest

import stt


class _UnknownValueError(Exception): pass
class _RequestError(Exception): pass


def _make_sr(*, results=None, raise_unknown=False, raise_request=False, raise_generic=False):
    """
    results: dict mapping language code -> transcript string (or "" for empty).
             Default returns "" for every language.
    """
    results = results or {}
    mod = types.ModuleType("speech_recognition")
    mod.UnknownValueError = _UnknownValueError
    mod.RequestError = _RequestError

    class _Recognizer:
        def adjust_for_ambient_noise(self, source, duration):
            pass
        def record(self, source):
            return "audio-data"
        def recognize_google(self, audio_data, language=None):
            if raise_unknown:
                raise _UnknownValueError("nope")
            if raise_request:
                raise _RequestError("network")
            if raise_generic:
                raise RuntimeError("boom")
            return results.get(language, "")

    class _AudioFile:
        def __init__(self, path):
            self.path = path
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    mod.Recognizer = _Recognizer
    mod.AudioFile = _AudioFile
    return mod


def _make_pydub(success=True):
    mod = types.ModuleType("pydub")

    class _AudioSegment:
        @staticmethod
        def from_file(path):
            if not success:
                raise RuntimeError("pydub failed")
            seg = _AudioSegment()
            seg._path = path
            return seg
        def export(self, target, format):
            with open(target, "wb") as f:
                f.write(b"RIFF....WAVE")

    mod.AudioSegment = _AudioSegment
    return mod


# ── transcribe_audio tests ──────────────────────────────────────────────────

def test_transcribe_missing_file_returns_empty(tmp_path):
    assert stt.transcribe_audio("") == ("", "")
    assert stt.transcribe_audio(str(tmp_path / "nope.wav")) == ("", "")


def test_transcribe_wav_explicit_en(tmp_path, monkeypatch):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFF....WAVE")
    monkeypatch.setitem(sys.modules, "speech_recognition",
                        _make_sr(results={"en-IN": "hello"}))
    text, lang = stt.transcribe_audio(str(wav), language="en")
    assert text == "hello"
    assert lang == "en-IN"


def test_transcribe_wav_explicit_hi(tmp_path, monkeypatch):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFF....WAVE")
    monkeypatch.setitem(sys.modules, "speech_recognition",
                        _make_sr(results={"hi-IN": "नमस्ते"}))
    text, lang = stt.transcribe_audio(str(wav), language="hi")
    assert text == "नमस्ते"
    assert lang == "hi-IN"


def test_transcribe_auto_picks_longest_result(tmp_path, monkeypatch):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFF....WAVE")
    # Hindi result is longer
    monkeypatch.setitem(sys.modules, "speech_recognition",
                        _make_sr(results={"en-IN": "hi", "hi-IN": "नमस्ते दोस्त"}))
    text, lang = stt.transcribe_audio(str(wav), language="auto")
    assert text == "नमस्ते दोस्त"
    assert lang == "hi-IN"


def test_transcribe_auto_default_is_none_language(tmp_path, monkeypatch):
    wav = tmp_path / "in.wav"
    wav.write_bytes(b"RIFF....WAVE")
    monkeypatch.setitem(sys.modules, "speech_recognition",
                        _make_sr(results={"en-IN": "hi there"}))
    text, lang = stt.transcribe_audio(str(wav), language=None)
    assert text == "hi there"
    assert lang == "en-IN"


def test_transcribe_non_wav_converts_via_pydub(tmp_path, monkeypatch):
    src = tmp_path / "in.mp3"
    src.write_bytes(b"fake-mp3")
    monkeypatch.setitem(sys.modules, "speech_recognition",
                        _make_sr(results={"en-IN": "ok"}))
    monkeypatch.setitem(sys.modules, "pydub", _make_pydub(success=True))
    text, _ = stt.transcribe_audio(str(src), language="en")
    assert text == "ok"


def test_transcribe_non_wav_pydub_failure_falls_back(tmp_path, monkeypatch):
    src = tmp_path / "in.webm"
    src.write_bytes(b"fake-webm")
    monkeypatch.setitem(sys.modules, "speech_recognition",
                        _make_sr(results={"en-IN": "still ok"}))
    monkeypatch.setitem(sys.modules, "pydub", _make_pydub(success=False))
    text, _ = stt.transcribe_audio(str(src), language="en")
    assert text == "still ok"


def test_transcribe_unknown_value_returns_empty(tmp_path, monkeypatch):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFF....WAVE")
    monkeypatch.setitem(sys.modules, "speech_recognition",
                        _make_sr(raise_unknown=True))
    text, _ = stt.transcribe_audio(str(wav), language="en")
    assert text == ""


def test_transcribe_request_error_returns_empty(tmp_path, monkeypatch):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFF....WAVE")
    monkeypatch.setitem(sys.modules, "speech_recognition",
                        _make_sr(raise_request=True))
    text, _ = stt.transcribe_audio(str(wav), language="en")
    assert text == ""


def test_transcribe_generic_exception_returns_empty(tmp_path, monkeypatch):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFF....WAVE")
    monkeypatch.setitem(sys.modules, "speech_recognition",
                        _make_sr(raise_generic=True))
    text, _ = stt.transcribe_audio(str(wav), language="en")
    assert text == ""


def test_transcribe_cleanup_handles_unlink_error(tmp_path, monkeypatch):
    src = tmp_path / "in.mp3"
    src.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "speech_recognition",
                        _make_sr(results={"en-IN": "hi"}))
    monkeypatch.setitem(sys.modules, "pydub", _make_pydub(success=True))
    monkeypatch.setattr(stt.os, "unlink",
                        lambda p: (_ for _ in ()).throw(OSError("locked")))
    text, _ = stt.transcribe_audio(str(src), language="en")
    assert text == "hi"
