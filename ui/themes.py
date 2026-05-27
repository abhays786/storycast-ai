"""
UI themes — CSS strings + structural toggles for the single themed builder.
"""

from dataclasses import dataclass


# ── Classic CSS ──────────────────────────────────────────────────────────────
CLASSIC_CSS = """
.gradio-container {
    background: linear-gradient(160deg, #f0f4ff 0%, #fdf0ff 100%) !important;
    padding: 0 8px 0 !important;
}
#app-title { text-align: center; padding: 4px 0 0; }
#app-title h1 {
    font-size: 2.0em; font-weight: 900;
    background: linear-gradient(90deg, #ff6b6b, #f7b731, #20bf6b, #45aaf2, #a55eea);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.5px; line-height: 1.1;
}
#app-subtitle { text-align: center; color: #666; font-size: 0.9em; margin: 0 0 2px; }
.input-card { background: white; border-radius: 14px; padding: 8px 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.07); }
.section-label { font-size: 0.85em; font-weight: 700; color: #444; margin: 3px 0 1px; }
.section-label.with-top-gap { margin-top: 10px; }
.footer-note { font-size: 0.85em; margin-top: 6px; color: #6b7280; }
#status-banner { text-align: center; font-size: 1.0em; font-weight: 700; padding: 7px 12px; border-radius: 10px; min-height: 36px; }
.status-idle    { background: #f3f4f6; color: #9ca3af; }
.status-working { background: #fef9c3; color: #92400e; }
.status-ready   { background: linear-gradient(90deg,#d1fae5,#cffafe); color: #065f46; font-size: 1.1em; }
.status-error   { background: #fee2e2; color: #991b1b; }
#story-output, #a2t-output {
    background: #fffdf5; border: 2px solid #fde68a; border-radius: 12px;
    padding: 12px 16px; font-size: 0.98em; line-height: 1.75;
    height: 220px; min-height: unset !important; overflow-y: auto !important; color: #1f2937;
}
#story-output h2 { color: #7c3aed; font-size: 1.2em; margin-bottom: 8px; }
#gen-btn, #tts-btn, #preview-btn, #a2t-btn, #a2a-btn {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    color: white !important; font-size: 1.0em !important; font-weight: 700 !important;
    border-radius: 10px !important; padding: 10px !important;
}
.example-btn button {
    background: #ede9fe !important; color: #5b21b6 !important;
    border-radius: 18px !important; font-size: 0.82em !important;
    padding: 3px 10px !important; margin: 2px !important; border: none !important;
}
.age-badge { display: inline-block; background: #dbeafe; color: #1e40af; border-radius: 20px; padding: 2px 10px; font-weight: 600; font-size: 0.85em; }
.compact-drop select { padding: 4px 6px !important; font-size: 0.9em !important; }
#footer { text-align: center; color: #9ca3af; font-size: 0.78em; padding: 3px 0 8px; }
.gr-accordion > .label-wrap { padding: 4px 10px !important; }
"""

# ── Modern CSS ───────────────────────────────────────────────────────────────
MODERN_CSS = """
.gradio-container {
    background: linear-gradient(140deg, #ede9fe 0%, #dbeafe 100%) !important;
    padding: 0 20px 20px !important;
}
#app-title { text-align: center; padding: 14px 0 2px; }
#app-title h1 { font-size: 2.1em; font-weight: 900; color: #3b0764; letter-spacing: -0.5px; }
#app-subtitle { text-align: center; color: #6d28d9; font-size: 0.92em; margin: 0 0 10px; }

.left-panel {
    background: #ffffff !important;
    border-radius: 22px !important;
    padding: 18px 20px 14px !important;
    box-shadow: 0 6px 28px rgba(109,40,217,0.12) !important;
}
.right-panel {
    background: #d1fae5 !important;
    border-radius: 22px !important;
    padding: 18px 20px 14px !important;
    box-shadow: 0 6px 28px rgba(16,185,129,0.10) !important;
}

.card-title {
    font-size: 1.15em; font-weight: 800; color: #3b0764;
    margin: 0 0 12px 0; padding-bottom: 8px;
    border-bottom: 2px solid #ede9fe;
}
.section-label { font-size: 0.86em; font-weight: 700; color: #374151; margin: 6px 0 2px; }
.section-label.with-top-gap { margin-top: 10px; }
.footer-note { font-size: 0.84em; margin-top: 8px; color: #059669; }

#status-banner { text-align: center; font-size: 0.95em; font-weight: 700; padding: 7px 12px; border-radius: 10px; min-height: 34px; }
.status-idle    { background: #f3f4f6; color: #9ca3af; }
.status-working { background: #fef9c3; color: #92400e; }
.status-ready   { background: linear-gradient(90deg,#d1fae5,#cffafe); color: #065f46; }
.status-error   { background: #fee2e2; color: #991b1b; }

#story-output, #a2t-output {
    background: #ffffff; border: none; border-radius: 14px;
    padding: 14px 18px; font-size: 0.97em; line-height: 1.78;
    height: 220px; min-height: unset !important; overflow-y: auto !important;
    color: #1f2937; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
#story-output h2 { color: #6d28d9; font-size: 1.15em; margin-bottom: 8px; }

.audio-wrap {
    background: #3b0764 !important;
    border-radius: 14px !important;
    padding: 10px 14px !important;
    margin-top: 6px !important;
}
.audio-wrap audio { width: 100% !important; }
.audio-wrap label, .audio-wrap .label-wrap { color: #e9d5ff !important; font-weight: 600 !important; }

#gen-btn, #tts-btn, #preview-btn, #a2t-btn, #a2a-btn {
    background: #3b0764 !important;
    color: white !important; font-size: 1.0em !important; font-weight: 700 !important;
    border-radius: 12px !important; padding: 11px !important;
}
#gen-btn:hover, #tts-btn:hover, #preview-btn:hover, #a2t-btn:hover, #a2a-btn:hover {
    background: #6d28d9 !important;
}

.gender-row label { font-size: 0.9em !important; }

.age-badge {
    display: inline-block; background: #ede9fe; color: #5b21b6;
    border-radius: 20px; padding: 2px 12px; font-weight: 600; font-size: 0.85em;
}

.example-btn button {
    background: #ede9fe !important; color: #5b21b6 !important;
    border-radius: 18px !important; font-size: 0.82em !important;
    padding: 3px 10px !important; margin: 2px !important; border: none !important;
}

.compact-drop select { padding: 4px 6px !important; font-size: 0.9em !important; }

.gr-accordion > .label-wrap {
    padding: 6px 12px !important;
    background: #f5f3ff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: #5b21b6 !important;
}

.tab-nav button { font-weight: 600 !important; border-radius: 10px 10px 0 0 !important; }
.tab-nav button.selected { background: #3b0764 !important; color: white !important; }

#footer { text-align: center; color: #8b5cf6; font-size: 0.8em; padding: 8px 0 12px; }
"""


@dataclass(frozen=True)
class Theme:
    """Every cosmetic + structural choice a builder reads."""
    name:                 str
    css:                  str
    title_text:           str
    subtitle_suffix:      str
    footer_text:          str
    tts_text_lines:       int
    left_panel_classes:   tuple[str, ...]
    right_panel_classes:  tuple[str, ...]
    tab_elem_classes:     tuple[str, ...]
    # Card titles per tab — None ⇒ section not rendered
    story_card_title:     str | None
    story_output_title:   str | None
    tts_card_title:       str | None
    tts_output_title:     str | None
    a2t_card_title:       str | None
    a2t_output_title:     str | None
    a2a_card_title:       str | None
    a2a_output_title:     str | None
    # Listen-label HTML / section-label HTML
    listen_label_html:    str
    tts_listen_label_html: str
    section_label_html:   str
    use_audio_wrap:       bool
    note_class:           str


CLASSIC = Theme(
    name="classic",
    css=CLASSIC_CSS,
    title_text="KidsStory Magic",
    subtitle_suffix="!",
    footer_text="Made with love for curious young minds &nbsp;|&nbsp; Powered by Llama 3 + Gemini/Piper TTS",
    tts_text_lines=7,
    left_panel_classes=("input-card",),
    right_panel_classes=(),
    tab_elem_classes=(),
    story_card_title=None,
    story_output_title="Your Story",
    tts_card_title=None,
    tts_output_title=None,
    a2t_card_title=None,
    a2t_output_title="Transcription",
    a2a_card_title=None,
    a2a_output_title="Rendered Audio",
    listen_label_html='<p class="section-label">Listen</p>',
    tts_listen_label_html='<p class="section-label">Listen</p>',
    section_label_html='<p class="section-label">Your Story</p>',
    use_audio_wrap=False,
    note_class="footer-note",
)

MODERN = Theme(
    name="modern",
    css=MODERN_CSS,
    title_text="📚 KidsStory Magic",
    subtitle_suffix="",
    footer_text="Made with love for curious young minds &nbsp;|&nbsp; Powered by Llama 3 + Piper TTS",
    tts_text_lines=8,
    left_panel_classes=("left-panel",),
    right_panel_classes=("right-panel",),
    tab_elem_classes=("tab-nav",),
    story_card_title="✏️ Story Settings",
    story_output_title="📖 Your Story",
    tts_card_title="📝 Text to Audio",
    tts_output_title="📖 Output",
    a2t_card_title="🎙️ Audio to Text",
    a2t_output_title="📝 Transcription",
    a2a_card_title="🔁 Audio to Audio",
    a2a_output_title="🔊 Rendered Audio",
    listen_label_html='<p class="section-label with-top-gap">🔊 Listen</p>',
    tts_listen_label_html='<p class="section-label with-top-gap">🔊 Listen to your story!</p>',
    section_label_html="",
    use_audio_wrap=True,
    note_class="footer-note",
)


def get_theme(name: str) -> Theme:
    """Resolve a theme by name; unknown names fall back to classic."""
    return MODERN if (name or "").lower() == "modern" else CLASSIC
