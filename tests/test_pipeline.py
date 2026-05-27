"""Tests for pipeline.py."""

import pytest

import pipeline


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture
def patched_pipeline(monkeypatch):
    """Replace every external call with simple stubs we can configure."""
    calls = {"story": [], "audio": [], "stt": [], "log_story": [], "log_tts": []}

    def fake_generate_story(topic, gender, language="en"):
        calls["story"].append((topic, gender, language))
        return ("Tale", "Once upon a time " * 30, "")

    def fake_generate_audio(text, gender="both", engine="piper", speed=1.0, **kw):
        calls["audio"].append((text, gender, engine, speed, kw))
        return "/tmp/out.wav"

    def fake_transcribe(path, language=None):
        calls["stt"].append((path, language))
        return ("transcribed topic", "en-IN")

    def fake_log_story(**kw):
        calls["log_story"].append(kw)

    def fake_log_tts(**kw):
        calls["log_tts"].append(kw)

    monkeypatch.setattr(pipeline, "generate_story", fake_generate_story)
    monkeypatch.setattr(pipeline, "generate_audio", fake_generate_audio)
    monkeypatch.setattr(pipeline, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(pipeline, "log_story_session", fake_log_story)
    monkeypatch.setattr(pipeline, "log_tts_session", fake_log_tts)
    return calls


def _events(gen):
    return list(gen)


# ── clamp ────────────────────────────────────────────────────────────────────

def test_clamp_speed_extremes():
    assert pipeline._clamp_speed(0.0) == 0.5
    assert pipeline._clamp_speed(99.0) == 2.0
    assert pipeline._clamp_speed(1.0) == 1.0


# ── run_story ────────────────────────────────────────────────────────────────

def test_run_story_empty_no_mic(patched_pipeline):
    out = _events(pipeline.run_story(
        topic="", audio_input=None, gender="both", engine_id="piper", speed=1.0,
    ))
    assert isinstance(out[0], pipeline.Failed)
    assert "enter a story topic" in out[0].reason


def test_run_story_mic_only_stt_returns_empty(monkeypatch, patched_pipeline):
    monkeypatch.setattr(pipeline, "transcribe_audio", lambda p, language=None: ("", ""))
    out = _events(pipeline.run_story(
        topic="", audio_input="/m.wav", gender="both", engine_id="piper", speed=1.0,
    ))
    assert isinstance(out[0], pipeline.Transcribing)
    assert isinstance(out[1], pipeline.Failed)


def test_run_story_mic_only_stt_success(patched_pipeline):
    out = _events(pipeline.run_story(
        topic="", audio_input="/mic.wav", gender="both", engine_id="piper", speed=1.0,
    ))
    kinds = [type(e).__name__ for e in out]
    assert kinds[0] == "Transcribing"
    assert kinds[1] == "Transcribed"
    assert "AudioReady" in kinds


def test_run_story_blocked_topic(patched_pipeline):
    out = _events(pipeline.run_story(
        topic="a story about killing", audio_input=None, gender="both",
        engine_id="piper", speed=1.0,
    ))
    assert isinstance(out[0], pipeline.Failed)
    assert "isn't quite right" in out[0].reason


def test_run_story_llm_error(monkeypatch, patched_pipeline):
    monkeypatch.setattr(pipeline, "generate_story",
                        lambda t, g, language="en": ("", "", "Ollama not running"))
    out = _events(pipeline.run_story(
        topic="A friendly dragon and a curious robot",
        audio_input=None, gender="both", engine_id="piper", speed=1.0,
    ))
    kinds = [type(e).__name__ for e in out]
    assert kinds == ["GeneratingStory", "Failed"]
    assert out[-1].reason == "Ollama not running"


def test_run_story_full_success(patched_pipeline):
    out = _events(pipeline.run_story(
        topic="A friendly dragon",
        audio_input=None, gender="both", engine_id="piper", speed=1.2,
    ))
    kinds = [type(e).__name__ for e in out]
    assert kinds == ["GeneratingStory", "StoryGenerated", "Synthesizing", "AudioReady"]
    ready = out[-1]
    assert ready.title == "Tale"
    assert ready.audio_path == "/tmp/out.wav"
    assert patched_pipeline["log_story"]


def test_run_story_passes_language(patched_pipeline):
    out = _events(pipeline.run_story(
        topic="A friendly dragon",
        audio_input=None, gender="both", engine_id="piper", speed=1.0,
        language="hi",
    ))
    assert any(isinstance(e, pipeline.AudioReady) for e in out)
    # generate_story was called with language="hi"
    assert patched_pipeline["story"][0][2] == "hi"
    # generate_audio received language="hi" kwarg
    _, _, _, _, kw = patched_pipeline["audio"][0]
    assert kw.get("language") == "hi"


def test_run_story_audio_fails(monkeypatch, patched_pipeline):
    monkeypatch.setattr(pipeline, "generate_audio", lambda *a, **k: None)
    out = _events(pipeline.run_story(
        topic="A friendly dragon",
        audio_input=None, gender="both", engine_id="piper", speed=1.0,
    ))
    last = out[-1]
    assert isinstance(last, pipeline.Failed)
    assert last.after_story is True
    assert last.title == "Tale"


# ── run_tts ──────────────────────────────────────────────────────────────────

def test_run_tts_empty_text(patched_pipeline):
    out = _events(pipeline.run_tts(text="", engine_id="piper", speed=1.0))
    assert isinstance(out[0], pipeline.Failed)


def test_run_tts_too_long(patched_pipeline):
    out = _events(pipeline.run_tts(
        text="x" * 9000, engine_id="piper", speed=1.0,
    ))
    assert isinstance(out[0], pipeline.Failed)
    assert "too long" in out[0].reason.lower()


def test_run_tts_success(patched_pipeline):
    out = _events(pipeline.run_tts(text="Hello", engine_id="piper", speed=1.0))
    kinds = [type(e).__name__ for e in out]
    assert kinds == ["Synthesizing", "AudioReady"]


def test_run_tts_audio_fails(monkeypatch, patched_pipeline):
    monkeypatch.setattr(pipeline, "generate_audio", lambda *a, **k: None)
    out = _events(pipeline.run_tts(text="Hello", engine_id="piper", speed=1.0))
    kinds = [type(e).__name__ for e in out]
    assert kinds == ["Synthesizing", "Failed"]


# ── run_a2t ──────────────────────────────────────────────────────────────────

def test_run_a2t_no_audio(patched_pipeline):
    out = _events(pipeline.run_a2t(audio_input=None, output_language="auto"))
    assert isinstance(out[0], pipeline.Failed)


def test_run_a2t_stt_empty(monkeypatch, patched_pipeline):
    monkeypatch.setattr(pipeline, "transcribe_audio", lambda p, language=None: ("", ""))
    out = _events(pipeline.run_a2t(audio_input="/mic.wav", output_language="auto"))
    kinds = [type(e).__name__ for e in out]
    assert kinds == ["Transcribing", "Failed"]


def test_run_a2t_auto_no_translation(patched_pipeline):
    out = _events(pipeline.run_a2t(audio_input="/mic.wav", output_language="auto"))
    kinds = [type(e).__name__ for e in out]
    assert "TextReady" in kinds
    ready = out[-1]
    assert ready.text == "transcribed topic"
    assert ready.source_language == "en"
    assert ready.output_language == "en"


def test_run_a2t_with_translation(monkeypatch, patched_pipeline):
    # Spoken Hindi → output English
    monkeypatch.setattr(pipeline, "transcribe_audio",
                        lambda p, language=None: ("नमस्ते दोस्त", "hi-IN"))
    monkeypatch.setattr(pipeline, "translate",
                        lambda text, src, tgt: ("hello friend", ""))
    out = _events(pipeline.run_a2t(audio_input="/mic.wav", output_language="en"))
    kinds = [type(e).__name__ for e in out]
    assert "Translating" in kinds
    assert "Translated" in kinds
    assert "TextReady" in kinds
    assert out[-1].text == "hello friend"
    assert out[-1].output_language == "en"


def test_run_a2t_translation_error(monkeypatch, patched_pipeline):
    monkeypatch.setattr(pipeline, "transcribe_audio",
                        lambda p, language=None: ("नमस्ते", "hi-IN"))
    monkeypatch.setattr(pipeline, "translate",
                        lambda text, src, tgt: (text, "Translation failed: boom"))
    out = _events(pipeline.run_a2t(audio_input="/mic.wav", output_language="en"))
    assert isinstance(out[-1], pipeline.Failed)


# ── run_a2a ──────────────────────────────────────────────────────────────────

def test_run_a2a_no_audio(patched_pipeline):
    out = _events(pipeline.run_a2a(
        audio_input=None, output_language="en", engine_id="piper",
        voice_id=None, speed=1.0,
    ))
    assert isinstance(out[0], pipeline.Failed)


def test_run_a2a_full_flow(monkeypatch, patched_pipeline):
    out = _events(pipeline.run_a2a(
        audio_input="/mic.wav", output_language="auto",
        engine_id="piper", voice_id=None, speed=1.0,
    ))
    kinds = [type(e).__name__ for e in out]
    assert "AudioReady" in kinds


def test_run_a2a_cross_language(monkeypatch, patched_pipeline):
    monkeypatch.setattr(pipeline, "transcribe_audio",
                        lambda p, language=None: ("hello", "en-IN"))
    monkeypatch.setattr(pipeline, "translate",
                        lambda text, src, tgt: ("नमस्ते", ""))
    out = _events(pipeline.run_a2a(
        audio_input="/mic.wav", output_language="hi",
        engine_id="piper", voice_id="hi_IN-rohan-medium", speed=1.0,
    ))
    kinds = [type(e).__name__ for e in out]
    assert "Translating" in kinds and "Translated" in kinds
    assert "AudioReady" in kinds


def test_run_a2a_translation_failure(monkeypatch, patched_pipeline):
    monkeypatch.setattr(pipeline, "transcribe_audio",
                        lambda p, language=None: ("hello", "en-IN"))
    monkeypatch.setattr(pipeline, "translate",
                        lambda t, s, g: (t, "boom"))
    out = _events(pipeline.run_a2a(
        audio_input="/mic.wav", output_language="hi",
        engine_id="piper", voice_id=None, speed=1.0,
    ))
    assert isinstance(out[-1], pipeline.Failed)


def test_run_a2a_stt_empty(monkeypatch, patched_pipeline):
    monkeypatch.setattr(pipeline, "transcribe_audio",
                        lambda p, language=None: ("", ""))
    out = _events(pipeline.run_a2a(
        audio_input="/mic.wav", output_language="en",
        engine_id="piper", voice_id=None, speed=1.0,
    ))
    assert isinstance(out[-1], pipeline.Failed)


def test_run_a2a_audio_fails(monkeypatch, patched_pipeline):
    monkeypatch.setattr(pipeline, "generate_audio", lambda *a, **k: None)
    out = _events(pipeline.run_a2a(
        audio_input="/mic.wav", output_language="auto",
        engine_id="piper", voice_id=None, speed=1.0,
    ))
    assert isinstance(out[-1], pipeline.Failed)


# ── preview_voice ───────────────────────────────────────────────────────────

def test_preview_voice_calls_generate_audio(monkeypatch, patched_pipeline):
    monkeypatch.setattr(pipeline.shutil, "copy2", lambda src, dst: None)
    out = pipeline.preview_voice(engine_id="piper", voice_id="en_US-arctic-medium",
                                  language="en")
    import config
    assert out == str(config.PREVIEW_CACHE_DIR / "piper_en_US-arctic-medium_en.wav")
    _, _, _, _, kw = patched_pipeline["audio"][0]
    assert kw.get("voice_id") == "en_US-arctic-medium"


def test_preview_voice_hindi(monkeypatch, patched_pipeline):
    monkeypatch.setattr(pipeline.shutil, "copy2", lambda src, dst: None)
    pipeline.preview_voice(engine_id="piper", voice_id="hi_IN-rohan-medium",
                            language="hi")
    _, _, _, _, kw = patched_pipeline["audio"][0]
    assert kw.get("language") == "hi"


def test_preview_voice_cache_hit(patched_pipeline):
    import config
    cache_dir = config.PREVIEW_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "piper_en_US-arctic-medium_en.wav"
    cache_file.write_bytes(b"X" * 2000)

    out = pipeline.preview_voice(engine_id="piper", voice_id="en_US-arctic-medium",
                                  language="en")
    assert out == str(cache_file)
    assert len(patched_pipeline["audio"]) == 0   # no synthesis on cache hit


def test_preview_voice_generate_fails_returns_none(monkeypatch):
    monkeypatch.setattr(pipeline, "generate_audio", lambda *a, **kw: None)
    out = pipeline.preview_voice(engine_id="piper", voice_id="en_US-arctic-medium",
                                  language="en")
    assert out is None


# ── Event dataclass behaviour ────────────────────────────────────────────────

def test_events_are_frozen():
    ev = pipeline.AudioReady(audio_path="/x.wav")
    with pytest.raises(Exception):
        ev.audio_path = "/y.wav"   # type: ignore[misc]
