"""
Single themed Gradio app builder.

`build_app(theme)` returns a `BuiltApp` (Blocks + CSS). Adapters in this file
translate semantic `pipeline.Event`s into Gradio's output tuples.
"""

from dataclasses import dataclass
from typing import Iterator, Optional

import gradio as gr

from config import (
    AGE_MAX, AGE_MIN, LANGUAGE_LABELS,
    MAX_TTS_TEXT_LENGTH, STORY_EXAMPLES,
)
from pipeline import (
    AudioReady, Failed, GeneratingStory, StoryGenerated,
    Synthesizing, TextReady, Transcribed, Transcribing,
    Translated, Translating,
    preview_voice, run_a2a, run_a2t, run_story, run_tts,
)
from tts.registry import get as get_backend
from ui.themes import Theme
from ui.widgets import (
    PLAYBACK_RATE_JS,
    audio_settings, banner, engine_dropdown, language_dropdown,
    parse_speed_label, placeholder_story, placeholder_text,
    playback_rate_dropdown, speed_dropdown, update_voices,
    voice_dropdown, voices_for,
)


# ── Return type ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BuiltApp:
    """Bundle returned by build_app() — caller passes both into launch()."""
    blocks: gr.Blocks
    css:    str


# ── Event → presentation helpers ─────────────────────────────────────────────

def _engine_label(engine_id: str) -> str:
    backend = get_backend(engine_id)
    return backend.display_name if backend else engine_id


def _format_speed(speed: float) -> str:
    if speed == int(speed):
        return f"{int(speed)}x"
    return f"{speed}x"


def _story_md(title: Optional[str], body: Optional[str]) -> str:
    if title and body:
        return f"## {title}\n\n{body}"
    return placeholder_story()


def _lang_label(code: str) -> str:
    return LANGUAGE_LABELS.get(code, code or "?")


# ── Story-tab event formatter ────────────────────────────────────────────────

def _format_story_event(ev) -> tuple[str, str, Optional[str]]:
    if isinstance(ev, Transcribing):
        return banner("Transcribing your voice...", "working"), placeholder_story(), None
    if isinstance(ev, Transcribed):
        return (
            banner(f"Heard: '{ev.text}' — writing your story...", "working"),
            placeholder_story(),
            None,
        )
    if isinstance(ev, GeneratingStory):
        return (
            banner("Crafting your magical story... please wait!", "working"),
            placeholder_story(),
            None,
        )
    if isinstance(ev, StoryGenerated):
        return banner("Story written!", "working"), _story_md(ev.title, ev.body), None
    if isinstance(ev, Synthesizing):
        msg = f"Recording with {_engine_label(ev.engine_id)} at {_format_speed(ev.speed)}..."
        return banner(msg, "working"), _story_md(ev.title, ev.body), None
    if isinstance(ev, AudioReady):
        return (
            banner("Your story is ready — press play to listen!", "ready"),
            _story_md(ev.title, ev.body),
            ev.audio_path,
        )
    if isinstance(ev, Failed):
        kind = "working" if ev.after_story else "error"
        return banner(ev.reason, kind), _story_md(ev.title, ev.body), None
    raise TypeError(f"Unknown story event: {type(ev).__name__}")  # pragma: no cover


def _format_tts_event(ev) -> tuple[str, Optional[str]]:
    if isinstance(ev, Synthesizing):
        msg = f"Converting with {_engine_label(ev.engine_id)} at {_format_speed(ev.speed)}..."
        return banner(msg, "working"), None
    if isinstance(ev, AudioReady):
        return banner("Done! Press play to listen.", "ready"), ev.audio_path
    if isinstance(ev, Failed):
        return banner(ev.reason, "error"), None
    raise TypeError(f"Unknown TTS event: {type(ev).__name__}")  # pragma: no cover


def _format_a2t_event(ev) -> tuple[str, str]:
    if isinstance(ev, Transcribing):
        return banner("Transcribing audio...", "working"), placeholder_text()
    if isinstance(ev, Transcribed):
        return (
            banner(f"Detected language: {_lang_label(ev.language)}", "working"),
            placeholder_text(),
        )
    if isinstance(ev, Translating):
        return (
            banner(f"Translating {_lang_label(ev.source)} → {_lang_label(ev.target)}...", "working"),
            placeholder_text(),
        )
    if isinstance(ev, Translated):
        return (
            banner(f"Translated to {_lang_label(ev.target)}.", "working"),
            placeholder_text(),
        )
    if isinstance(ev, TextReady):
        return banner("Done!", "ready"), ev.text
    if isinstance(ev, Failed):
        return banner(ev.reason, "error"), placeholder_text()
    raise TypeError(f"Unknown A2T event: {type(ev).__name__}")  # pragma: no cover


def _format_a2a_event(ev) -> tuple[str, Optional[str]]:
    if isinstance(ev, Transcribing):
        return banner("Transcribing audio...", "working"), None
    if isinstance(ev, Transcribed):
        return banner(f"Detected language: {_lang_label(ev.language)}", "working"), None
    if isinstance(ev, Translating):
        return (
            banner(f"Translating {_lang_label(ev.source)} → {_lang_label(ev.target)}...",
                   "working"),
            None,
        )
    if isinstance(ev, Translated):
        return banner(f"Translated to {_lang_label(ev.target)}.", "working"), None
    if isinstance(ev, Synthesizing):
        return (
            banner(f"Recording with {_engine_label(ev.engine_id)} "
                   f"at {_format_speed(ev.speed)}...", "working"),
            None,
        )
    if isinstance(ev, AudioReady):
        return banner("Done! Press play to listen.", "ready"), ev.audio_path
    if isinstance(ev, Failed):
        return banner(ev.reason, "error"), None
    raise TypeError(f"Unknown A2A event: {type(ev).__name__}")  # pragma: no cover


# ── Gradio adapters ──────────────────────────────────────────────────────────

def _story_adapter(topic, mic, gender, engine, speed_str, language, voice_id) -> Iterator[tuple]:
    speed = parse_speed_label(speed_str)
    for ev in run_story(
        topic=topic, audio_input=mic, gender=gender,
        engine_id=engine, speed=speed, language=language, voice_id=voice_id,
    ):
        yield _format_story_event(ev)


def _tts_adapter(text, engine, speed_str, language, voice_id) -> Iterator[tuple]:
    speed = parse_speed_label(speed_str)
    for ev in run_tts(
        text=text, engine_id=engine, speed=speed,
        language=language, voice_id=voice_id,
    ):
        yield _format_tts_event(ev)


def _a2t_adapter(mic_audio, upload_audio, output_language) -> Iterator[tuple]:
    audio_input = mic_audio or upload_audio
    for ev in run_a2t(audio_input=audio_input, output_language=output_language):
        yield _format_a2t_event(ev)


def _a2a_adapter(mic_audio, upload_audio, output_language,
                 engine, voice_id, speed_str) -> Iterator[tuple]:
    audio_input = mic_audio or upload_audio
    speed = parse_speed_label(speed_str)
    for ev in run_a2a(
        audio_input=audio_input, output_language=output_language,
        engine_id=engine, voice_id=voice_id, speed=speed,
    ):
        yield _format_a2a_event(ev)


def _preview_adapter(engine_id, voice_id, language):
    if not voice_id:
        return None
    return preview_voice(engine_id=engine_id, voice_id=voice_id, language=language)


def _clear_story():
    return "", None, banner("", "idle"), placeholder_story(), None


def _clear_tts():
    return "", banner("", "idle"), None


def _clear_a2t():
    return None, None, banner("", "idle"), placeholder_text()


def _clear_a2a():
    return None, None, banner("", "idle"), None


def _refresh_voices_a2a(engine_id, lang_code):
    """A2A treats 'auto' as English when picking the voice list."""
    effective = "en" if lang_code == "auto" else lang_code
    return update_voices(engine_id, effective)


# ── Theme-driven column helpers ──────────────────────────────────────────────

def _left_column(theme: Theme):
    return gr.Column(scale=1, elem_classes=list(theme.left_panel_classes))


def _right_column(theme: Theme):
    return gr.Column(scale=1, elem_classes=list(theme.right_panel_classes))


def _audio_player(theme: Theme, label: str, elem_id: str | None = None):
    kwargs = {"label": label, "show_label": False, "autoplay": False}
    if elem_id:
        kwargs["elem_id"] = elem_id
    if theme.use_audio_wrap:
        with gr.Group(elem_classes=["audio-wrap"]):
            return gr.Audio(**kwargs)
    return gr.Audio(**kwargs)


def _maybe_card_title(text: Optional[str]) -> None:
    if text:
        gr.HTML(f'<div class="card-title">{text}</div>')


# ── Tab builders ─────────────────────────────────────────────────────────────

def _build_story_tab(theme: Theme) -> None:
    with gr.Row(equal_height=False):

        with _left_column(theme):
            _maybe_card_title(theme.story_card_title)

            gr.HTML('<p class="section-label">What should the story be about?</p>')
            topic_box = gr.Textbox(
                placeholder="e.g. A tiny wizard who accidentally turns all the cats purple...",
                lines=2, max_lines=3, show_label=False,
            )

            gr.HTML('<p class="section-label">Or record your idea:</p>')
            mic_input = gr.Audio(sources=["microphone"], type="filepath",
                                 label="Speak your idea", show_label=False)

            with gr.Row(elem_classes=["gender-row"]):
                with gr.Column(scale=2, min_width=0):
                    gr.HTML('<p class="section-label">Story focus</p>')
                    gender_radio = gr.Radio(choices=["boy", "girl", "both"], value="both",
                                            show_label=False)
                with gr.Column(scale=2, min_width=0):
                    gr.HTML('<p class="section-label">Story language</p>')
                    lang_dd = language_dropdown(label="Story language")
                with gr.Column(scale=1, min_width=0):
                    gr.HTML('<p class="section-label">Age</p>')
                    gr.HTML(
                        f'<div style="margin-top:4px"><span class="age-badge">'
                        f'{AGE_MIN}-{AGE_MAX} yrs</span></div>'
                    )

            with gr.Accordion("Audio Settings", open=True, elem_classes=["gr-accordion"]):
                story_engine, story_speed = audio_settings()
                # Voice is auto-resolved from (language, gender) when None.
                voice_state = gr.State(value=None)

            with gr.Row():
                story_gen_btn   = gr.Button("Create My Story!", variant="primary",
                                            elem_id="gen-btn", scale=3)
                story_clear_btn = gr.Button("Clear", variant="secondary", scale=1)

            with gr.Accordion("Need Inspiration?", open=False, elem_classes=["gr-accordion"]):
                with gr.Column(elem_classes=["example-btn"]):
                    for ex in STORY_EXAMPLES:
                        gr.Button(ex, size="sm").click(fn=lambda e=ex: e, outputs=[topic_box])

        with _right_column(theme):
            _maybe_card_title(theme.story_output_title)

            story_status = gr.HTML(value=banner("", "idle"))
            if theme.section_label_html:
                gr.HTML(theme.section_label_html)

            story_md = gr.Markdown(value=placeholder_story(), elem_id="story-output")
            gr.HTML(theme.listen_label_html)
            story_audio = _audio_player(theme, label="Story audio", elem_id="story-audio")

    story_gen_btn.click(
        fn=_story_adapter,
        inputs=[topic_box, mic_input, gender_radio, story_engine, story_speed,
                lang_dd, voice_state],
        outputs=[story_status, story_md, story_audio],
    )
    story_clear_btn.click(
        fn=_clear_story,
        outputs=[topic_box, mic_input, story_status, story_md, story_audio],
    )


def _build_tts_tab(theme: Theme) -> None:
    with gr.Row(equal_height=False):

        with _left_column(theme):
            _maybe_card_title(theme.tts_card_title)

            gr.HTML('<p class="section-label">Type or paste your text</p>')
            tts_text = gr.Textbox(
                placeholder=f"Paste any text here — up to {MAX_TTS_TEXT_LENGTH:,} characters...",
                lines=theme.tts_text_lines, max_lines=14, show_label=False,
            )

            with gr.Accordion("Audio Settings", open=True, elem_classes=["gr-accordion"]):
                with gr.Row():
                    tts_engine = engine_dropdown()
                    tts_lang   = language_dropdown(label="Language")
                with gr.Row():
                    tts_voice  = voice_dropdown(tts_engine.value, tts_lang.value)
                    tts_speed  = speed_dropdown()

                with gr.Row():
                    preview_btn = gr.Button("Preview voice", variant="secondary",
                                            elem_id="preview-btn", scale=1)
                preview_audio = gr.Audio(label="Voice preview", show_label=True,
                                         autoplay=True, elem_id="tts-preview-audio")

            with gr.Row():
                tts_btn       = gr.Button("Convert to Audio", variant="primary",
                                          elem_id="tts-btn", scale=3)
                tts_clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        with _right_column(theme):
            _maybe_card_title(theme.tts_output_title)

            tts_status = gr.HTML(value=banner("", "idle"))
            gr.HTML(theme.tts_listen_label_html)
            tts_audio  = _audio_player(theme, label="Output audio", elem_id="tts-audio")

            with gr.Row():
                tts_rate = playback_rate_dropdown(label="Playback speed")
                tts_audio_id = gr.Textbox(value="tts-audio", visible=False)

            gr.HTML(
                f'<p class="{theme.note_class}">'
                f'All outputs are saved to the <code>logs/</code> folder.</p>'
            )

    # ── Wiring ────────────────────────────────────────────────────────────
    # Refresh voice list when engine or language changes.
    tts_engine.change(fn=update_voices, inputs=[tts_engine, tts_lang], outputs=tts_voice)
    tts_lang.change(  fn=update_voices, inputs=[tts_engine, tts_lang], outputs=tts_voice)

    preview_btn.click(
        fn=_preview_adapter,
        inputs=[tts_engine, tts_voice, tts_lang],
        outputs=preview_audio,
    )

    tts_btn.click(
        fn=_tts_adapter,
        inputs=[tts_text, tts_engine, tts_speed, tts_lang, tts_voice],
        outputs=[tts_status, tts_audio],
    )
    tts_clear_btn.click(fn=_clear_tts, outputs=[tts_text, tts_status, tts_audio])

    # Browser-side: changing the playback-rate dropdown sets the <audio>'s
    # playbackRate via JS (no server round-trip / no re-render).
    tts_rate.change(fn=None, inputs=[tts_rate, tts_audio_id], js=PLAYBACK_RATE_JS)


def _build_a2t_tab(theme: Theme) -> None:
    with gr.Row(equal_height=False):

        with _left_column(theme):
            _maybe_card_title(theme.a2t_card_title)

            gr.HTML('<p class="section-label">Record audio</p>')
            mic_input = gr.Audio(sources=["microphone"], type="filepath",
                                 label="Speak", show_label=False)

            gr.HTML('<p class="section-label with-top-gap">…or upload a file</p>')
            upload_input = gr.Audio(sources=["upload"], type="filepath",
                                    label="Upload", show_label=False)

            with gr.Accordion("Output Settings", open=True, elem_classes=["gr-accordion"]):
                out_lang = language_dropdown(label="Output language", include_auto=True)

            with gr.Row():
                a2t_btn = gr.Button("Transcribe", variant="primary",
                                    elem_id="a2t-btn", scale=3)
                a2t_clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        with _right_column(theme):
            _maybe_card_title(theme.a2t_output_title)

            a2t_status = gr.HTML(value=banner("", "idle"))
            a2t_text   = gr.Markdown(value=placeholder_text(), elem_id="a2t-output")
            gr.HTML(
                f'<p class="{theme.note_class}">'
                f'All transcripts are saved to the <code>logs/</code> folder.</p>'
            )

    a2t_btn.click(
        fn=_a2t_adapter,
        inputs=[mic_input, upload_input, out_lang],
        outputs=[a2t_status, a2t_text],
    )
    a2t_clear_btn.click(
        fn=_clear_a2t,
        outputs=[mic_input, upload_input, a2t_status, a2t_text],
    )


def _build_a2a_tab(theme: Theme) -> None:
    with gr.Row(equal_height=False):

        with _left_column(theme):
            _maybe_card_title(theme.a2a_card_title)

            gr.HTML('<p class="section-label">Record audio</p>')
            mic_input = gr.Audio(sources=["microphone"], type="filepath",
                                 label="Speak", show_label=False)

            gr.HTML('<p class="section-label with-top-gap">…or upload a file</p>')
            upload_input = gr.Audio(sources=["upload"], type="filepath",
                                    label="Upload", show_label=False)

            with gr.Accordion("Output Settings", open=True, elem_classes=["gr-accordion"]):
                out_lang = language_dropdown(label="Output language", include_auto=True)
                with gr.Row():
                    a2a_engine = engine_dropdown()
                    a2a_voice  = voice_dropdown(a2a_engine.value,
                                                "en" if out_lang.value == "auto" else out_lang.value)
                a2a_speed = speed_dropdown()

            with gr.Row():
                a2a_btn = gr.Button("Re-render", variant="primary",
                                    elem_id="a2a-btn", scale=3)
                a2a_clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        with _right_column(theme):
            _maybe_card_title(theme.a2a_output_title)

            a2a_status = gr.HTML(value=banner("", "idle"))
            gr.HTML(theme.tts_listen_label_html)
            a2a_audio  = _audio_player(theme, label="Output audio", elem_id="a2a-audio")
            gr.HTML(
                f'<p class="{theme.note_class}">'
                f'Inputs and outputs are saved to <code>logs/</code>.</p>'
            )

    a2a_engine.change(fn=_refresh_voices_a2a, inputs=[a2a_engine, out_lang], outputs=a2a_voice)
    out_lang.change(  fn=_refresh_voices_a2a, inputs=[a2a_engine, out_lang], outputs=a2a_voice)

    a2a_btn.click(
        fn=_a2a_adapter,
        inputs=[mic_input, upload_input, out_lang, a2a_engine, a2a_voice, a2a_speed],
        outputs=[a2a_status, a2a_audio],
    )
    a2a_clear_btn.click(
        fn=_clear_a2a,
        outputs=[mic_input, upload_input, a2a_status, a2a_audio],
    )


# ── Top-level builder ────────────────────────────────────────────────────────

def build_app(theme: Theme) -> BuiltApp:
    """Construct the Gradio app for the given theme."""
    with gr.Blocks(title="KidsStory Magic") as blocks:
        gr.HTML(f'<div id="app-title"><h1>{theme.title_text}</h1></div>')
        gr.HTML(
            f'<p id="app-subtitle">Your personal story-adventure creator — '
            f'ages {AGE_MIN}-{AGE_MAX}{theme.subtitle_suffix}</p>'
        )

        with gr.Tabs(elem_classes=list(theme.tab_elem_classes)):
            with gr.Tab("Story Magic"):
                _build_story_tab(theme)
            with gr.Tab("Text to Audio"):
                _build_tts_tab(theme)
            with gr.Tab("Audio to Text"):
                _build_a2t_tab(theme)
            with gr.Tab("Audio to Audio"):
                _build_a2a_tab(theme)

        gr.HTML(f'<div id="footer">{theme.footer_text}</div>')

    return BuiltApp(blocks=blocks, css=theme.css)
