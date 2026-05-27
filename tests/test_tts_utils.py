"""Tests for tts/_utils.py."""

import wave
from pathlib import Path

import pytest

from tts import _utils


# ── apply_speed_pydub ────────────────────────────────────────────────────────

def _write_wav(path: Path, frames: bytes = b"\x00\x00" * 100, rate: int = 22050) -> None:
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


def test_apply_speed_close_to_one_returns_path_unchanged(tmp_path):
    wav = tmp_path / "a.wav"
    _write_wav(wav)
    result = _utils.apply_speed_pydub(str(wav), 1.0)
    assert result == str(wav)


def test_apply_speed_real_change(tmp_path):
    wav = tmp_path / "b.wav"
    _write_wav(wav)
    result = _utils.apply_speed_pydub(str(wav), 1.5)
    assert result == str(wav)
    # File still exists & is non-empty after re-export
    assert Path(result).stat().st_size > 0


def test_apply_speed_swallows_pydub_error(tmp_path):
    missing = str(tmp_path / "does-not-exist.wav")
    # Pydub will fail to open this file; function returns the path anyway.
    out = _utils.apply_speed_pydub(missing, 1.5)
    assert out == missing


# ── chunk_text ───────────────────────────────────────────────────────────────

def test_chunk_text_single_short_chunk():
    assert _utils.chunk_text("Hello world.", 100) == ["Hello world."]


def test_chunk_text_paragraph_split():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = _utils.chunk_text(text, 15)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 30  # some slack for whitespace joining


def test_chunk_text_sentence_split_for_long_paragraph():
    long_para = "First sentence here. Second sentence here. Third sentence here. Fourth."
    chunks = _utils.chunk_text(long_para, 25)
    assert len(chunks) >= 2


def test_chunk_text_falls_back_when_text_unchunkable():
    # No newlines, no sentence punctuation, and shorter than max_chars yields one chunk.
    assert _utils.chunk_text("nopunct", 100) == ["nopunct"]


def test_chunk_text_empty_returns_single_chunk():
    assert _utils.chunk_text("", 10) == [""]


def test_chunk_text_very_long_single_word():
    text = "x" * 50
    chunks = _utils.chunk_text(text, 10)
    # Single paragraph longer than max_chars triggers sentence-split fallback
    assert chunks


def test_chunk_text_sentence_split_grows_current():
    # Each sentence is short — first goes into current, next extends current.
    text = "S1. S2. S3. " + ("x" * 30) + "."
    chunks = _utils.chunk_text(text, 20)
    assert chunks


# ── new_output_path ──────────────────────────────────────────────────────────

def test_new_output_path_creates_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(_utils, "ASSETS_DIR", tmp_path / "assets")
    out = _utils.new_output_path()
    assert (tmp_path / "assets").exists()
    assert str(out).endswith(".wav")


def test_new_output_path_uniqueness(tmp_path, monkeypatch):
    monkeypatch.setattr(_utils, "ASSETS_DIR", tmp_path / "assets")
    p1 = _utils.new_output_path()
    p2 = _utils.new_output_path()
    assert p1 != p2


def test_new_output_path_prefix_and_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(_utils, "ASSETS_DIR", tmp_path / "assets")
    out = _utils.new_output_path(prefix="speech", suffix=".raw")
    assert "speech_" in out.name
    assert out.name.endswith(".raw")


def test_new_output_path_with_basename(tmp_path, monkeypatch):
    monkeypatch.setattr(_utils, "ASSETS_DIR", tmp_path / "assets")
    out = _utils.new_output_path(basename="The Brave Dragon!")
    assert "The_Brave_Dragon" in out.name


def test_safe_filename_token_strips_unsafe():
    assert _utils.safe_filename_token("Hello/World?") == "HelloWorld"


def test_safe_filename_token_empty_falls_back_to_audio():
    assert _utils.safe_filename_token("???") == "audio"
    assert _utils.safe_filename_token("") == "audio"
    assert _utils.safe_filename_token(None) == "audio"
