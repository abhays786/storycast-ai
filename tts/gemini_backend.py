"""
Gemini TTS backend — AI voice synthesis via Google Gemini Flash TTS.

Gemini voices are language-agnostic: the same voice can speak Hindi or
English depending on the script of the text passed in. We expose duplicate
catalog entries (one per language) so the UI shows the voice under the
correct language section.
"""

import wave
from pathlib import Path

from app_logging import get_logger
from settings import settings
from tts._utils import apply_speed_pydub, new_output_path
from tts.base import TTSBackend
from tts.voices import VoiceInfo

log = get_logger(__name__)


# ── Voice catalog ────────────────────────────────────────────────────────────
# Voice id format: "<gemini-voice>" (e.g., "Charon"). Display name carries the
# friendly description. Each underlying voice is registered once per language
# so the UI dropdown shows it for both English and Hindi.
_BASE_VOICES = [
    ("Charon", "Charon (deep, calm male)",      "boy"),
    ("Kore",   "Kore (bright friendly female)", "girl"),
    ("Puck",   "Puck (warm expressive)",        "both"),
]
VOICES: list[VoiceInfo] = [
    VoiceInfo(id=vid, display_name=name, language=lang, gender=gen)
    for vid, name, gen in _BASE_VOICES
    for lang in ("en", "hi")
]

PCM_RATE = 24000   # Hz — Gemini returns 24 kHz 16-bit mono PCM


class GeminiBackend(TTSBackend):
    id = "gemini"

    @property
    def display_name(self) -> str:
        return "Gemini Flash TTS — AI voice"

    def voices(self) -> list[VoiceInfo]:
        return list(VOICES)

    def is_available(self) -> bool:
        return bool(settings.gemini_api_key)

    def generate(
        self,
        text: str,
        gender: str = "both",
        speed: float = 1.0,
        *,
        language: str = "en",
        voice_id: str | None = None,
        output_basename: str | None = None,
    ) -> str | None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            log.error("google-genai not installed. Run: pip install google-genai")
            return None

        if not settings.gemini_api_key:
            log.error("GEMINI_API_KEY not set in .env")
            return None

        voice = self.resolve_voice(language, gender, voice_id)
        if voice is None:
            log.error("No Gemini voice available for language=%s gender=%s", language, gender)
            return None

        model_name = settings.gemini_tts_model
        out_path   = str(new_output_path(basename=output_basename))

        log.info("Gemini TTS: voice=%s model=%s lang=%s", voice.id, model_name, language)

        try:
            client   = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice.id
                            )
                        )
                    ),
                ),
            )

            part      = response.candidates[0].content.parts[0]
            mime      = getattr(part.inline_data, "mime_type", "audio/pcm")
            data      = part.inline_data.data
            raw_bytes = data if isinstance(data, (bytes, bytearray)) else bytes(data)

            if "wav" in mime.lower():
                with open(out_path, "wb") as fh:
                    fh.write(raw_bytes)
            else:
                with wave.open(out_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(PCM_RATE)
                    wf.writeframes(raw_bytes)

            apply_speed_pydub(out_path, speed)

            if Path(out_path).exists() and Path(out_path).stat().st_size > 1024:
                log.info("Gemini TTS complete: %s (%.1f kB)",
                         Path(out_path).name, Path(out_path).stat().st_size / 1024)
                return out_path

            log.warning("Gemini TTS produced an empty file.")
            return None

        except Exception as exc:
            log.error("Gemini TTS error: %s", exc)
            return None
