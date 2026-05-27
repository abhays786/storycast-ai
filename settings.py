"""
Runtime settings — single composition root for environment variables.

All env / .env reads happen here so the rest of the codebase has one
typed source of truth. `load_dotenv()` is intentionally called only
in this module.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from config import LLM_MODEL_DEFAULT, OLLAMA_HOST_DEFAULT, GEMINI_TTS_MODEL_DEFAULT

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # LLM (Ollama)
    story_model:       str    # model used for English stories + default
    hindi_story_model: str    # model used for Hindi stories (defaults to story_model)
    translation_model: str    # model used for Hindi <-> English translation
    ollama_host:       str
    # Gemini TTS
    gemini_api_key:   str
    gemini_tts_model: str
    # UI
    ui_theme:         str                  # "classic" | "modern"
    enabled_backends: tuple[str, ...]      # backends shown in the UI dropdown


def _parse_enabled_backends(raw: str) -> tuple[str, ...]:
    items = [item.strip().lower() for item in raw.split(",")]
    return tuple(item for item in items if item)


def _load() -> Settings:
    story_model = os.getenv("STORY_MODEL", LLM_MODEL_DEFAULT)
    return Settings(
        story_model=story_model,
        hindi_story_model=os.getenv("HINDI_STORY_MODEL", story_model),
        translation_model=os.getenv("TRANSLATION_MODEL", story_model),
        ollama_host=os.getenv("OLLAMA_HOST", OLLAMA_HOST_DEFAULT),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_tts_model=os.getenv("GEMINI_TTS_MODEL", GEMINI_TTS_MODEL_DEFAULT),
        ui_theme=os.getenv("UI_THEME", "classic"),
        enabled_backends=_parse_enabled_backends(
            os.getenv("ENABLED_BACKENDS", "piper,gemini")
        ),
    )


settings: Settings = _load()
