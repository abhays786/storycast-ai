"""Reusable widget factories and small UI helpers."""

from typing import Iterable

import gradio as gr

from config import (
    LANGUAGE_DEFAULT, LANGUAGE_LABELS, LANGUAGE_OPTIONS,
    PLAYBACK_RATE_DEFAULT, PLAYBACK_RATE_OPTIONS,
    SPEED_DEFAULT, SPEED_OPTIONS,
)
from settings import settings
from tts.registry import get as get_backend, ui_choices


# ── Status helpers ──────────────────────────────────────────────────────────

def banner(msg: str, kind: str) -> str:
    """Render the status banner HTML for a given kind."""
    cls = {
        "idle":    "status-idle",
        "working": "status-working",
        "ready":   "status-ready",
        "error":   "status-error",
    }.get(kind, "status-idle")
    return f'<div id="status-banner" class="{cls}">{msg}</div>'


def placeholder_story() -> str:
    return "*Your story will appear here once it has been created...*"


def placeholder_text() -> str:
    return "*Transcription will appear here once the audio is processed...*"


def parse_speed_label(speed_str: str) -> float:
    """Convert UI speed label like '1.2x' / '1.0x (Normal)' to a float."""
    return float(speed_str.replace("x", "").split()[0])


# ── Dropdown factories ──────────────────────────────────────────────────────

def language_dropdown(*, label: str = "Language", include_auto: bool = False,
                      default: str = LANGUAGE_DEFAULT) -> gr.Dropdown:
    choices = [(label_, code) for label_, code in LANGUAGE_OPTIONS]
    if include_auto:
        choices = [("Auto-detect", "auto"), *choices]
        default = "auto"
    return gr.Dropdown(choices=choices, value=default, label=label, show_label=True)


def engine_dropdown(label: str = "TTS Engine") -> gr.Dropdown:
    choices = ui_choices(enabled_ids=settings.enabled_backends)
    default = choices[0][1] if choices else None
    return gr.Dropdown(choices=choices, value=default, label=label, show_label=True)


def voices_for(engine_id: str, language: str) -> list[tuple[str, str]]:
    """Return (display_name, voice_id) pairs for a given engine + language."""
    backend = get_backend(engine_id)
    if backend is None:
        return []
    return [(v.display_name, v.id) for v in backend.voices_for(language)]


def voice_dropdown(engine_id: str, language: str, label: str = "Voice") -> gr.Dropdown:
    choices = voices_for(engine_id, language)
    default = choices[0][1] if choices else None
    return gr.Dropdown(choices=choices, value=default, label=label, show_label=True)


def speed_dropdown(label: str = "Speed") -> gr.Dropdown:
    return gr.Dropdown(
        choices=SPEED_OPTIONS, value=SPEED_DEFAULT,
        label=label, show_label=True,
    )


def playback_rate_dropdown(label: str = "Playback speed") -> gr.Dropdown:
    return gr.Dropdown(
        choices=PLAYBACK_RATE_OPTIONS, value=PLAYBACK_RATE_DEFAULT,
        label=label, show_label=True,
    )


def update_voices(engine_id: str, language: str) -> gr.Dropdown:
    """Event handler used to refresh the voice dropdown when engine/language changes."""
    choices = voices_for(engine_id, language)
    return gr.Dropdown(
        choices=choices,
        value=(choices[0][1] if choices else None),
    )


def audio_settings() -> tuple[gr.Dropdown, gr.Dropdown]:
    """Compact engine + speed dropdowns side by side (legacy two-column layout)."""
    with gr.Row():
        with gr.Column(elem_classes=["compact-drop"]):
            engine_dd = engine_dropdown()
        with gr.Column(elem_classes=["compact-drop"]):
            speed_dd = speed_dropdown()
    return engine_dd, speed_dd


# ── Browser-side JS for the playback-rate dropdown ──────────────────────────

PLAYBACK_RATE_JS = """
(rate_str, audio_elem_id) => {
    const wrap = document.getElementById(audio_elem_id);
    if (!wrap) return rate_str;
    const audio = wrap.querySelector("audio");
    if (audio) audio.playbackRate = parseFloat(String(rate_str).replace('x',''));
    return rate_str;
}
"""
