"""Tests for ui/builder.py."""

import gradio as gr
import pytest

import pipeline
from tts.base import TTSBackend
from tts import registry
from ui import builder
from ui.themes import CLASSIC, MODERN


from tts.voices import VoiceInfo


class _Backend(TTSBackend):
    id = "piper"
    @property
    def display_name(self):
        return "Piper — Test"
    def voices(self):
        return [
            VoiceInfo(id="en_test", display_name="EN Test", language="en", gender="both"),
            VoiceInfo(id="hi_test", display_name="HI Test", language="hi", gender="both"),
        ]
    def generate(self, text, gender="both", speed=1.0, *,
                 language="en", voice_id=None, output_basename=None):
        return None


@pytest.fixture
def with_backend():
    registry.register(_Backend())


# ── BuiltApp ────────────────────────────────────────────────────────────────

def test_built_app_dataclass(with_backend):
    built = builder.build_app(CLASSIC)
    assert isinstance(built, builder.BuiltApp)
    assert built.css == CLASSIC.css
    assert isinstance(built.blocks, gr.Blocks)


# ── pure helpers ────────────────────────────────────────────────────────────

def test_engine_label_known(with_backend):
    assert builder._engine_label("piper") == "Piper — Test"


def test_engine_label_unknown_falls_back_to_id():
    assert builder._engine_label("nope") == "nope"


@pytest.mark.parametrize("speed, label", [
    (1.0, "1x"),
    (2.0, "2x"),
    (1.5, "1.5x"),
    (0.7, "0.7x"),
])
def test_format_speed(speed, label):
    assert builder._format_speed(speed) == label


def test_story_md_with_content():
    md = builder._story_md("T", "B")
    assert md.startswith("## T")


def test_story_md_without_content():
    assert builder._story_md(None, None).startswith("*Your story")
    assert builder._story_md("T", None).startswith("*Your story")
    assert builder._story_md(None, "B").startswith("*Your story")


# ── _format_story_event ─────────────────────────────────────────────────────

def test_format_story_event_each_type(with_backend):
    ev_types = [
        pipeline.Transcribing(),
        pipeline.Transcribed(text="hello"),
        pipeline.GeneratingStory(),
        pipeline.StoryGenerated(title="T", body="B"),
        pipeline.Synthesizing(engine_id="piper", speed=1.0, title="T", body="B"),
        pipeline.AudioReady(audio_path="/x.wav", title="T", body="B"),
        pipeline.Failed(reason="oops"),
        pipeline.Failed(reason="audio failed", title="T", body="B", after_story=True),
    ]
    for ev in ev_types:
        status, md, audio = builder._format_story_event(ev)
        assert isinstance(status, str)
        assert isinstance(md, str)


def test_format_story_event_after_story_uses_working_kind():
    status, _, _ = builder._format_story_event(
        pipeline.Failed(reason="x", title="T", body="B", after_story=True)
    )
    assert "status-working" in status


def test_format_story_event_failed_pre_story_uses_error_kind():
    status, _, _ = builder._format_story_event(pipeline.Failed(reason="x"))
    assert "status-error" in status


# ── _format_tts_event ───────────────────────────────────────────────────────

def test_format_tts_event_each_type(with_backend):
    syn = builder._format_tts_event(
        pipeline.Synthesizing(engine_id="piper", speed=1.0)
    )
    assert syn[1] is None
    ready = builder._format_tts_event(pipeline.AudioReady(audio_path="/x.wav"))
    assert ready[1] == "/x.wav"
    fail = builder._format_tts_event(pipeline.Failed(reason="x"))
    assert fail[1] is None
    assert "status-error" in fail[0]


# ── Adapters ────────────────────────────────────────────────────────────────

def test_story_adapter_yields_tuples(monkeypatch, with_backend):
    def fake_run_story(**kw):
        yield pipeline.Transcribing()
        yield pipeline.Failed(reason="x")
    monkeypatch.setattr(builder, "run_story", fake_run_story)
    out = list(builder._story_adapter("topic", None, "both", "piper", "1.0x", "en", None))
    assert len(out) == 2
    for t in out:
        assert len(t) == 3


def test_tts_adapter_yields_tuples(monkeypatch, with_backend):
    def fake_run_tts(**kw):
        yield pipeline.Synthesizing(engine_id="piper", speed=1.0)
        yield pipeline.AudioReady(audio_path="/x.wav")
    monkeypatch.setattr(builder, "run_tts", fake_run_tts)
    out = list(builder._tts_adapter("text", "piper", "1.0x", "en", None))
    assert len(out) == 2


def test_a2t_adapter_yields_tuples(monkeypatch, with_backend):
    def fake_run_a2t(**kw):
        yield pipeline.Transcribing()
        yield pipeline.TextReady(text="x", source_language="en",
                                  output_language="en",
                                  raw_transcript="x", output_transcript="x")
    monkeypatch.setattr(builder, "run_a2t", fake_run_a2t)
    out = list(builder._a2t_adapter("/m.wav", None, "auto"))
    assert len(out) == 2


def test_a2a_adapter_yields_tuples(monkeypatch, with_backend):
    def fake_run_a2a(**kw):
        yield pipeline.Transcribing()
        yield pipeline.AudioReady(audio_path="/x.wav")
    monkeypatch.setattr(builder, "run_a2a", fake_run_a2a)
    out = list(builder._a2a_adapter("/m.wav", None, "en", "piper", "en_test", "1.0x"))
    assert len(out) == 2


def test_format_a2t_events(with_backend):
    events = [
        pipeline.Transcribing(),
        pipeline.Transcribed(text="hi", language="en"),
        pipeline.Translating(source="en", target="hi"),
        pipeline.Translated(text="नमस्ते", source="en", target="hi"),
        pipeline.TextReady(text="hi", source_language="en", output_language="en",
                            raw_transcript="hi", output_transcript="hi"),
        pipeline.Failed(reason="oops"),
    ]
    for ev in events:
        s, t = builder._format_a2t_event(ev)
        assert isinstance(s, str) and isinstance(t, str)


def test_format_a2a_events(with_backend):
    events = [
        pipeline.Transcribing(),
        pipeline.Transcribed(text="hi", language="en"),
        pipeline.Translating(source="en", target="hi"),
        pipeline.Translated(text="ok", source="en", target="hi"),
        pipeline.Synthesizing(engine_id="piper", speed=1.0),
        pipeline.AudioReady(audio_path="/x.wav"),
        pipeline.Failed(reason="oops"),
    ]
    for ev in events:
        s, a = builder._format_a2a_event(ev)
        assert isinstance(s, str)


def test_preview_adapter_no_voice():
    assert builder._preview_adapter("piper", None, "en") is None


def test_preview_adapter_calls_preview(monkeypatch, with_backend):
    monkeypatch.setattr(builder, "preview_voice", lambda **kw: "/p.wav")
    assert builder._preview_adapter("piper", "en_test", "en") == "/p.wav"


def test_clear_a2t():
    res = builder._clear_a2t()
    assert len(res) == 4
    assert res[0] is None


def test_clear_a2a():
    res = builder._clear_a2a()
    assert len(res) == 4
    assert res[0] is None


# ── clear handlers ──────────────────────────────────────────────────────────

def test_clear_story_shape():
    res = builder._clear_story()
    assert len(res) == 5
    assert res[0] == ""
    assert res[1] is None


def test_clear_tts_shape():
    res = builder._clear_tts()
    assert len(res) == 3
    assert res[0] == ""


# ── build_app ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("theme", [CLASSIC, MODERN])
def test_build_app_runs_for_each_theme(theme, with_backend):
    built = builder.build_app(theme)
    assert isinstance(built.blocks, gr.Blocks)
    assert built.css == theme.css


def test_refresh_voices_a2a_resolves_auto(monkeypatch, with_backend):
    captured = {}
    monkeypatch.setattr(builder, "update_voices",
                        lambda eid, lang: captured.setdefault("lang", lang))
    builder._refresh_voices_a2a("piper", "auto")
    assert captured["lang"] == "en"
    captured.clear()
    builder._refresh_voices_a2a("piper", "hi")
    assert captured["lang"] == "hi"
