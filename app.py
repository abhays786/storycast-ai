"""
KidsStory Magic — Gradio app entry point.

Run with:  py app.py
Then open: http://localhost:7860

UI theme is controlled by UI_THEME in .env:
  UI_THEME=classic  (default) — compact card layout
  UI_THEME=modern             — purple/green two-panel layout
"""

import gradio as gr

from app_logging import get_logger
from settings import settings
from tts.bootstrap import register_default_backends
from tts_engine import ensure_all_models
from ui import build_app, get_theme

log = get_logger(__name__)


def check_ollama() -> bool:
    """Probe the configured Ollama server; log a friendly hint on failure.

    Returns True if the configured model is pulled, False otherwise.
    """
    import requests

    try:
        resp   = requests.get(f"{settings.ollama_host}/api/tags", timeout=3)
        pulled = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        if (settings.story_model in pulled
                or settings.story_model.split(":")[0] in pulled):
            log.info("Ollama OK — using model '%s'", settings.story_model)
            return True
        log.warning(
            "Model '%s' is not pulled yet. Run:  ollama pull %s",
            settings.story_model, settings.story_model,
        )
        return False
    except Exception:
        log.warning(
            "Ollama does not appear to be running at %s. "
            "Download from https://ollama.com/download, start it, "
            "then run: ollama pull %s",
            settings.ollama_host, settings.story_model,
        )
        return False


def main() -> None:
    register_default_backends()

    log.info("UI theme: %s", settings.ui_theme)
    theme = get_theme(settings.ui_theme)
    built = build_app(theme)

    check_ollama()
    ensure_all_models()

    built.blocks.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="violet"),
        css=built.css,
    )


if __name__ == "__main__":   # pragma: no cover
    main()
