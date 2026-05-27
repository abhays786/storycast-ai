"""Shared audio post-processing, text-chunking, and output-path helpers."""

import re
import time
from itertools import count
from pathlib import Path

from app_logging import get_logger
from config import ASSETS_DIR

log = get_logger(__name__)

_output_counter = count()   # monotonic — disambiguates within the same second


def apply_speed_pydub(wav_path: str, speed: float) -> str:
    """
    Adjust playback speed by resampling the WAV frame rate.
    Speed > 1 = faster, speed < 1 = slower. Pitch shifts with speed.
    Returns wav_path unchanged on error or when speed is ~1.0.
    """
    if abs(speed - 1.0) < 0.02:
        return wav_path
    try:
        from pydub import AudioSegment
        sound = AudioSegment.from_wav(wav_path)
        altered = sound._spawn(
            sound.raw_data,
            overrides={"frame_rate": int(sound.frame_rate * speed)},
        ).set_frame_rate(sound.frame_rate)
        altered.export(wav_path, format="wav")
    except Exception as exc:
        log.warning("Speed adjustment failed: %s", exc)
    return wav_path


def chunk_text(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks bounded by `max_chars`, preferring paragraph then
    sentence boundaries. Returns at least one element.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current = (current + " " + para).strip()
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(para) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 <= max_chars:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sent
        else:
            current = para

    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


_FILENAME_UNSAFE = re.compile(r"[^\w\s\-]")


def safe_filename_token(text: str, maxlen: int = 40) -> str:
    """Sanitise an arbitrary string for use inside a filename."""
    cleaned = _FILENAME_UNSAFE.sub("", text or "").strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:maxlen] or "audio"


def new_output_path(
    prefix: str = "story",
    suffix: str = ".wav",
    *,
    basename: str | None = None,
) -> Path:
    """
    Allocate a unique output path under ASSETS_DIR.

    Pass `basename` (already sanitized or not — we sanitise either way) to
    embed the story title or another label in the filename. Falls back to
    `prefix` when basename is empty.
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    seq = next(_output_counter)
    label = safe_filename_token(basename) if basename else prefix
    return ASSETS_DIR / f"{label}_{int(time.time())}_{seq}{suffix}"
