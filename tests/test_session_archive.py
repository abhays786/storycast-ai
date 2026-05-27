"""Tests for session_archive.py."""

from pathlib import Path

import session_archive as sa


def test_safe_name_strips_unsafe_chars():
    assert sa._safe_name("Hello / World!?") == "Hello_World"


def test_safe_name_truncates():
    assert len(sa._safe_name("a" * 200, maxlen=10)) == 10


def test_safe_name_empty_falls_back_to_session():
    assert sa._safe_name("###") == "session"


def test_ts_format():
    ts = sa._ts()
    assert len(ts) == 15  # YYYYMMDD_HHMMSS
    assert "_" in ts


def test_copy_missing_source(tmp_path):
    res = sa._copy(tmp_path / "missing.wav", tmp_path / "dest", "x.wav")
    assert res is None


def test_copy_success(tmp_path):
    src = tmp_path / "src.wav"
    src.write_bytes(b"hello")
    dest_dir = tmp_path / "dest"
    res = sa._copy(src, dest_dir, "renamed.wav")
    assert res is not None
    assert res.read_bytes() == b"hello"


def test_log_story_session_writes_inputs_and_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "LOG_DIR", tmp_path)
    mic = tmp_path / "mic.wav"
    mic.write_bytes(b"mic-audio")
    out = tmp_path / "out.wav"
    out.write_bytes(b"out-audio")

    sa.log_story_session(
        topic="A dragon", gender="both", engine="piper", speed="1.0x",
        language="en", title="Brave Dragon", story_text="Once upon a time...",
        input_audio_path=str(mic), output_audio_path=str(out),
    )

    in_txt_files = list((tmp_path / "inputs" / "text").iterdir())
    assert any(f.name.endswith("_topic.txt") for f in in_txt_files)

    out_txt_files = list((tmp_path / "outputs" / "text").iterdir())
    assert any("Brave_Dragon" in f.name for f in out_txt_files)

    in_audio_files = list((tmp_path / "inputs" / "audio").iterdir())
    assert any(f.name.endswith("_mic.wav") for f in in_audio_files)

    out_audio_files = list((tmp_path / "outputs" / "audio").iterdir())
    assert any("Brave_Dragon" in f.name for f in out_audio_files)


def test_log_story_session_optional_audio_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "LOG_DIR", tmp_path)
    sa.log_story_session(
        topic="X", gender="boy", engine="gemini", speed="1.2x",
        language="hi", title="X", story_text="X" * 10,
        input_audio_path=None, output_audio_path=None,
    )
    assert not (tmp_path / "inputs" / "audio").exists()
    assert not (tmp_path / "outputs" / "audio").exists()


def test_log_tts_session_writes_text_and_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "LOG_DIR", tmp_path)
    out = tmp_path / "tts.wav"
    out.write_bytes(b"audio")

    sa.log_tts_session(
        text="Hello world", engine="piper", speed="1.0x",
        output_audio_path=str(out),
    )

    in_files = list((tmp_path / "inputs" / "text").iterdir())
    assert any(f.name.endswith("_tts_input.txt") for f in in_files)

    out_files = list((tmp_path / "outputs" / "audio").iterdir())
    assert any("_tts" in f.name for f in out_files)


def test_log_tts_session_without_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "LOG_DIR", tmp_path)
    sa.log_tts_session(text="hi", engine="piper", speed="1.0x")
    assert not (tmp_path / "outputs" / "audio").exists()


def test_log_a2t_session(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "LOG_DIR", tmp_path)
    mic = tmp_path / "src.wav"
    mic.write_bytes(b"audio-input")
    sa.log_a2t_session(
        input_audio_path=str(mic),
        detected_language="hi",
        output_language="en",
        transcript_raw="नमस्ते",
        transcript_out="hello",
    )
    assert any(f.name.endswith("_a2t.wav")
               for f in (tmp_path / "inputs" / "audio").iterdir())
    assert any(f.name.endswith("_a2t.txt")
               for f in (tmp_path / "outputs" / "text").iterdir())


def test_log_a2a_session(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "LOG_DIR", tmp_path)
    mic = tmp_path / "in.wav"
    mic.write_bytes(b"audio-input")
    out = tmp_path / "out.wav"
    out.write_bytes(b"audio-output")
    sa.log_a2a_session(
        input_audio_path=str(mic),
        detected_language="en", output_language="hi",
        transcript_raw="hello", transcript_out="नमस्ते",
        engine="piper", voice_id="hi_IN-rohan-medium", speed="1.0x",
        output_audio_path=str(out),
    )
    in_audio = list((tmp_path / "inputs" / "audio").iterdir())
    out_audio = list((tmp_path / "outputs" / "audio").iterdir())
    assert any("_a2a" in f.name for f in in_audio)
    assert any("_a2a" in f.name for f in out_audio)


def test_log_a2t_session_no_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "LOG_DIR", tmp_path)
    sa.log_a2t_session(
        input_audio_path="", detected_language="en", output_language="en",
        transcript_raw="hi", transcript_out="hi",
    )
    # No audio dir should be created when input path is empty
    assert not (tmp_path / "inputs" / "audio").exists()


def test_log_a2a_session_no_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "LOG_DIR", tmp_path)
    sa.log_a2a_session(
        input_audio_path="", detected_language="en", output_language="en",
        transcript_raw="hi", transcript_out="hi",
        engine="piper", voice_id="", speed="1.0x",
    )
    assert not (tmp_path / "inputs" / "audio").exists()
    assert not (tmp_path / "outputs" / "audio").exists()
