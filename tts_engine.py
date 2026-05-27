"""
TTS dispatcher — thin facade over the registry.

Owns one piece of policy: if the chosen backend fails (returns None), retry
once with the registry's `DEFAULT_FALLBACK_ID`. Adding / removing backends
is done by editing `tts/bootstrap.py`, not this file.
"""

from app_logging import get_logger
from tts.registry import DEFAULT_FALLBACK_ID, all_backends, get, get_fallback

log = get_logger(__name__)


def generate_audio(
    text: str,
    gender: str = "both",
    engine: str = "piper",
    speed: float = 1.0,
    *,
    language: str = "en",
    voice_id: str | None = None,
    output_basename: str | None = None,
) -> str | None:
    """
    Synthesise text to a WAV file and return its path, or None on failure.

    If the chosen engine returns None and is not the fallback, the dispatcher
    retries once with the default fallback backend.
    """
    engine_id = (engine or "").lower().strip()
    speed     = max(0.5, min(2.0, float(speed)))

    backend = get(engine_id)
    if backend is None:
        log.warning("Unknown engine '%s' — using fallback.", engine_id)
        backend = get_fallback()
        if backend is None:
            log.error("No fallback backend registered.")
            return None

    log.info("generate_audio: engine=%s gender=%s lang=%s voice=%s speed=%.2f chars=%d",
             backend.id, gender, language, voice_id or "<default>", speed, len(text))

    result = backend.generate(
        text, gender, speed,
        language=language, voice_id=voice_id, output_basename=output_basename,
    )
    if result is not None:
        return result

    if backend.id == DEFAULT_FALLBACK_ID:
        return None

    fallback = get_fallback()
    if fallback is None or fallback is backend:
        return None

    log.warning("%s failed — falling back to %s.", backend.id, fallback.id)
    return fallback.generate(
        text, gender, speed,
        language=language, voice_id=voice_id, output_basename=output_basename,
    )


def ensure_all_models() -> None:
    """Pre-download / warm models for every registered backend."""
    for backend in all_backends():
        backend.ensure_models()
