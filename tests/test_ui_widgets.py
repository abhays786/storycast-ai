"""Tests for ui/widgets.py."""

import types

import gradio as gr
import pytest

from tts.base import TTSBackend
from tts import registry
from ui import widgets


class _Stub(TTSBackend):
    id = "piper"
    @property
    def display_name(self):
        return "Piper — Test"
    def voices(self):
        return []
    def generate(self, text, gender="both", speed=1.0, *,
                 language="en", voice_id=None, output_basename=None):
        return None


def test_banner_each_kind():
    assert "status-idle"    in widgets.banner("x", "idle")
    assert "status-working" in widgets.banner("x", "working")
    assert "status-ready"   in widgets.banner("x", "ready")
    assert "status-error"   in widgets.banner("x", "error")


def test_banner_unknown_kind_falls_back_to_idle():
    assert "status-idle" in widgets.banner("x", "unrecognized")


def test_placeholder_story_returns_md():
    assert widgets.placeholder_story().startswith("*Your story")


@pytest.mark.parametrize("label, expected", [
    ("1.0x", 1.0),
    ("0.7x", 0.7),
    ("1.0x (Normal)", 1.0),
    ("1.3x", 1.3),
])
def test_parse_speed_label(label, expected):
    assert widgets.parse_speed_label(label) == pytest.approx(expected)


def test_audio_settings_with_registered_backend(monkeypatch):
    registry.register(_Stub())
    fake_settings = types.SimpleNamespace(enabled_backends=("piper",))
    monkeypatch.setattr(widgets, "settings", fake_settings)
    with gr.Blocks():
        engine_dd, speed_dd = widgets.audio_settings()
    assert engine_dd.value == "piper"
    assert "1.0" in speed_dd.value


def test_audio_settings_with_no_backends(monkeypatch):
    fake_settings = types.SimpleNamespace(enabled_backends=())
    monkeypatch.setattr(widgets, "settings", fake_settings)
    with gr.Blocks():
        engine_dd, speed_dd = widgets.audio_settings()
    # default None when there are no choices
    assert engine_dd.value is None


def test_voices_for_unknown_engine_returns_empty():
    assert widgets.voices_for("does-not-exist", "en") == []


def test_voices_for_known_engine_returns_choices(monkeypatch):
    from tts.voices import VoiceInfo
    class _B(_Stub):
        id = "piper"
        def voices(self):
            return [VoiceInfo(id="v1", display_name="V1", language="en", gender="both")]
    registry.register(_B())
    assert widgets.voices_for("piper", "en") == [("V1", "v1")]


def test_update_voices_returns_dropdown(monkeypatch):
    from tts.voices import VoiceInfo
    class _B(_Stub):
        id = "piper"
        def voices(self):
            return [VoiceInfo(id="x", display_name="X", language="en", gender="both")]
    registry.register(_B())
    with gr.Blocks():
        dd = widgets.update_voices("piper", "en")
    assert dd is not None


def test_voice_dropdown_for_unknown_engine():
    with gr.Blocks():
        dd = widgets.voice_dropdown("missing", "en")
    assert dd.value is None


def test_language_dropdown_with_auto():
    with gr.Blocks():
        dd = widgets.language_dropdown(include_auto=True)
    assert dd.value == "auto"


def test_playback_rate_dropdown_default():
    with gr.Blocks():
        dd = widgets.playback_rate_dropdown()
    assert "1.0" in dd.value
