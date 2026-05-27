"""
Orchestration layer — emits semantic events that any front-end can render.

Four flows live here:
  • run_story  (text → story → audio)
  • run_tts    (text → audio)
  • run_a2t    (audio → text)
  • run_a2a    (audio → text → audio)

Events are a sum type; the UI adapter (`ui/builder.py`) matches on type to
produce user-facing strings and Gradio output tuples.
"""

import shutil
from dataclasses import dataclass
from typing import Iterator, Optional, Union

from agent import generate_story
from config import MAX_TTS_TEXT_LENGTH
from safety import check_input
from session_archive import (
    log_a2a_session, log_a2t_session,
    log_story_session, log_tts_session,
)
from stt import transcribe_audio
from translation import translate, _normalize_lang
from tts_engine import generate_audio


# ── Event sum type ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Transcribing:
    """Pipeline is asking STT to convert audio to text."""


@dataclass(frozen=True)
class Transcribed:
    """STT returned text."""
    text:     str
    language: str = ""


@dataclass(frozen=True)
class Translating:
    """Cross-language: translating transcript to target output language."""
    source: str
    target: str


@dataclass(frozen=True)
class Translated:
    """Translation finished."""
    text:   str
    source: str
    target: str


@dataclass(frozen=True)
class GeneratingStory:
    """Talking to the LLM."""


@dataclass(frozen=True)
class StoryGenerated:
    title: str
    body:  str


@dataclass(frozen=True)
class Synthesizing:
    engine_id: str
    speed:     float
    title:     Optional[str] = None
    body:      Optional[str] = None


@dataclass(frozen=True)
class AudioReady:
    audio_path: str
    title:      Optional[str] = None
    body:       Optional[str] = None


@dataclass(frozen=True)
class TextReady:
    """Output of an audio-to-text flow."""
    text:              str
    source_language:   str
    output_language:   str
    raw_transcript:    str
    output_transcript: str


@dataclass(frozen=True)
class Failed:
    reason:      str
    title:       Optional[str] = None
    body:        Optional[str] = None
    after_story: bool = False


Event = Union[
    Transcribing, Transcribed,
    Translating, Translated,
    GeneratingStory, StoryGenerated,
    Synthesizing, AudioReady,
    TextReady, Failed,
]


# ── Constants ────────────────────────────────────────────────────────────────

MIN_SPEED = 0.5
MAX_SPEED = 2.0


def _clamp_speed(speed: float) -> float:
    return max(MIN_SPEED, min(MAX_SPEED, float(speed)))


# ── Story pipeline ────────────────────────────────────────────────────────────

def run_story(
    *,
    topic: str,
    audio_input: Optional[str],
    gender: str,
    engine_id: str,
    speed: float,
    language: str = "en",
    voice_id: Optional[str] = None,
) -> Iterator[Event]:
    topic = (topic or "").strip()
    speed = _clamp_speed(speed)

    if not topic and audio_input:
        yield Transcribing()
        topic, detected = transcribe_audio(audio_input, language=language)
        if not topic:
            yield Failed(reason="Could not understand the audio. Please type your idea instead.")
            return
        yield Transcribed(text=topic, language=detected)
    elif not topic:
        yield Failed(reason="Please enter a story topic or record your voice!")
        return

    safe, err = check_input(topic)
    if not safe:
        yield Failed(reason=err)
        return

    yield GeneratingStory()
    title, body, error = generate_story(topic, gender, language=language)
    if error:
        yield Failed(reason=error)
        return
    yield StoryGenerated(title=title, body=body)

    yield Synthesizing(engine_id=engine_id, speed=speed, title=title, body=body)
    audio_path = generate_audio(
        body, gender, engine_id, speed,
        language=language, voice_id=voice_id, output_basename=title,
    )

    log_story_session(
        topic=topic, gender=gender, engine=engine_id, speed=f"{speed}x",
        language=language, title=title, story_text=body,
        input_audio_path=audio_input, output_audio_path=audio_path,
    )

    if audio_path:
        yield AudioReady(audio_path=audio_path, title=title, body=body)
    else:
        yield Failed(
            reason="Audio generation failed — check the console.",
            title=title, body=body, after_story=True,
        )


# ── Text-to-audio pipeline ────────────────────────────────────────────────────

def run_tts(
    *,
    text: str,
    engine_id: str,
    speed: float,
    language: str = "en",
    voice_id: Optional[str] = None,
) -> Iterator[Event]:
    text  = (text or "").strip()
    speed = _clamp_speed(speed)

    if not text:
        yield Failed(reason="Please enter some text to convert.")
        return
    if len(text) > MAX_TTS_TEXT_LENGTH:
        yield Failed(reason=f"Text is too long (max {MAX_TTS_TEXT_LENGTH:,} characters).")
        return

    yield Synthesizing(engine_id=engine_id, speed=speed)
    audio_path = generate_audio(
        text, "both", engine_id, speed,
        language=language, voice_id=voice_id,
    )
    log_tts_session(
        text=text, engine=engine_id, speed=f"{speed}x",
        language=language, voice_id=voice_id or "", output_audio_path=audio_path,
    )

    if audio_path:
        yield AudioReady(audio_path=audio_path)
    else:
        yield Failed(reason="Audio generation failed — check the console.")


# ── Audio-to-text pipeline ────────────────────────────────────────────────────

def run_a2t(
    *,
    audio_input: Optional[str],
    output_language: str = "auto",
) -> Iterator[Event]:
    """
    Transcribe an audio file. If `output_language` is "auto" or matches the
    detected language, the spoken text is returned directly. Otherwise the
    transcript is translated to the chosen output language via the LLM.
    """
    if not audio_input:
        yield Failed(reason="Please record or upload an audio clip first.")
        return

    yield Transcribing()
    raw_text, detected = transcribe_audio(audio_input, language="auto")
    if not raw_text:
        yield Failed(reason="Could not transcribe the audio. Try a clearer clip.")
        return
    detected_iso = _normalize_lang(detected) or "en"
    yield Transcribed(text=raw_text, language=detected_iso)

    if output_language == "auto" or _normalize_lang(output_language) == detected_iso:
        out_text = raw_text
        out_lang = detected_iso
    else:
        out_lang = _normalize_lang(output_language)
        yield Translating(source=detected_iso, target=out_lang)
        translated, err = translate(raw_text, detected_iso, out_lang)
        if err:
            yield Failed(reason=err)
            return
        out_text = translated
        yield Translated(text=translated, source=detected_iso, target=out_lang)

    log_a2t_session(
        input_audio_path=audio_input,
        detected_language=detected_iso,
        output_language=out_lang,
        transcript_raw=raw_text,
        transcript_out=out_text,
    )

    yield TextReady(
        text=out_text,
        source_language=detected_iso,
        output_language=out_lang,
        raw_transcript=raw_text,
        output_transcript=out_text,
    )


# ── Audio-to-audio pipeline ───────────────────────────────────────────────────

def run_a2a(
    *,
    audio_input: Optional[str],
    output_language: str,
    engine_id: str,
    voice_id: Optional[str],
    speed: float,
    gender: str = "both",
) -> Iterator[Event]:
    """STT → optional translation → TTS."""
    if not audio_input:
        yield Failed(reason="Please record or upload an audio clip first.")
        return

    speed = _clamp_speed(speed)

    yield Transcribing()
    raw_text, detected = transcribe_audio(audio_input, language="auto")
    if not raw_text:
        yield Failed(reason="Could not transcribe the audio. Try a clearer clip.")
        return
    detected_iso = _normalize_lang(detected) or "en"
    yield Transcribed(text=raw_text, language=detected_iso)

    if output_language == "auto" or _normalize_lang(output_language) == detected_iso:
        out_text = raw_text
        out_lang = detected_iso
    else:
        out_lang = _normalize_lang(output_language)
        yield Translating(source=detected_iso, target=out_lang)
        translated, err = translate(raw_text, detected_iso, out_lang)
        if err:
            yield Failed(reason=err)
            return
        out_text = translated
        yield Translated(text=translated, source=detected_iso, target=out_lang)

    yield Synthesizing(engine_id=engine_id, speed=speed)
    audio_path = generate_audio(
        out_text, gender, engine_id, speed,
        language=out_lang, voice_id=voice_id,
    )

    log_a2a_session(
        input_audio_path=audio_input,
        detected_language=detected_iso,
        output_language=out_lang,
        transcript_raw=raw_text,
        transcript_out=out_text,
        engine=engine_id,
        voice_id=voice_id or "",
        speed=f"{speed}x",
        output_audio_path=audio_path,
    )

    if audio_path:
        yield AudioReady(audio_path=audio_path)
    else:
        yield Failed(reason="Audio generation failed — check the console.")


# ── Voice preview helper ─────────────────────────────────────────────────────

def preview_voice(
    *,
    engine_id: str,
    voice_id: str,
    language: str = "en",
) -> Optional[str]:
    """Render a short preview clip; result is cached per (engine, voice, language)."""
    from config import PREVIEW_CACHE_DIR, PREVIEW_TEXT

    cache_path = PREVIEW_CACHE_DIR / f"{engine_id}_{voice_id}_{language}.wav"
    if cache_path.exists() and cache_path.stat().st_size > 1024:
        return str(cache_path)

    text = PREVIEW_TEXT.get(language, PREVIEW_TEXT["en"])
    result = generate_audio(
        text, "both", engine_id, 1.0,
        language=language, voice_id=voice_id,
        output_basename=f"preview_{engine_id}_{voice_id}",
    )
    if result is None:
        return None

    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result, cache_path)
    return str(cache_path)
