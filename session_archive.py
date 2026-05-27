"""
Session archiver — persists each user interaction to logs/ for later review.

Directory layout (rooted at config.LOG_DIR):
    logs/
    ├── inputs/
    │   ├── text/   ← typed topics & TTS / A2T / A2A input text (.txt)
    │   └── audio/  ← mic recordings / uploaded audio files
    └── outputs/
        ├── text/   ← generated story text, transcriptions, translations
        └── audio/  ← synthesised WAV files
"""

import re
import shutil
import time
from pathlib import Path

from config import LOG_DIR


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_name(text: str, maxlen: int = 40) -> str:
    """Strip characters unsafe for filenames and truncate."""
    cleaned = re.sub(r"[^\w\s-]", "", text).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:maxlen] or "session"


def _copy(src: str | Path, dest_dir: Path, filename: str) -> Path | None:
    src = Path(src)
    if not src.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    shutil.copy2(src, dest)
    return dest


# ── Public API ────────────────────────────────────────────────────────────────

def log_story_session(
    *,
    topic: str,
    gender: str,
    engine: str,
    speed: str,
    language: str,
    title: str,
    story_text: str,
    input_audio_path: str | None = None,
    output_audio_path: str | None = None,
) -> None:
    """Log a story-generation session (inputs + outputs)."""
    ts = _ts()
    safe_title = _safe_name(title)

    in_txt = LOG_DIR / "inputs" / "text"
    in_txt.mkdir(parents=True, exist_ok=True)
    (in_txt / f"{ts}_{safe_title}_topic.txt").write_text(
        f"[Story Session — {ts}]\n"
        f"Topic    : {topic}\n"
        f"Gender   : {gender}\n"
        f"Engine   : {engine}\n"
        f"Speed    : {speed}\n"
        f"Language : {language}\n",
        encoding="utf-8",
    )

    if input_audio_path:
        src = Path(input_audio_path)
        _copy(src, LOG_DIR / "inputs" / "audio", f"{ts}_{safe_title}_mic{src.suffix}")

    out_txt = LOG_DIR / "outputs" / "text"
    out_txt.mkdir(parents=True, exist_ok=True)
    (out_txt / f"{ts}_{safe_title}.txt").write_text(
        f"[Story — {ts}]\n"
        f"Title    : {title}\n"
        f"Topic    : {topic}\n"
        f"Language : {language}\n\n"
        f"{story_text}\n",
        encoding="utf-8",
    )

    if output_audio_path:
        src = Path(output_audio_path)
        _copy(src, LOG_DIR / "outputs" / "audio", f"{ts}_{safe_title}{src.suffix}")


def log_tts_session(
    *,
    text: str,
    engine: str,
    speed: str,
    language: str = "en",
    voice_id: str = "",
    output_audio_path: str | None = None,
) -> None:
    """Log a plain text-to-audio conversion session."""
    ts = _ts()

    in_txt = LOG_DIR / "inputs" / "text"
    in_txt.mkdir(parents=True, exist_ok=True)
    (in_txt / f"{ts}_tts_input.txt").write_text(
        f"[TTS Session — {ts}]\n"
        f"Engine   : {engine}\n"
        f"Voice    : {voice_id}\n"
        f"Speed    : {speed}\n"
        f"Language : {language}\n\n"
        f"{text}\n",
        encoding="utf-8",
    )

    if output_audio_path:
        src = Path(output_audio_path)
        _copy(src, LOG_DIR / "outputs" / "audio", f"{ts}_tts{src.suffix}")


def log_a2t_session(
    *,
    input_audio_path: str,
    detected_language: str,
    output_language: str,
    transcript_raw: str,
    transcript_out: str,
) -> None:
    """Log an audio-to-text session."""
    ts = _ts()

    if input_audio_path:
        src = Path(input_audio_path)
        _copy(src, LOG_DIR / "inputs" / "audio", f"{ts}_a2t{src.suffix}")

    in_txt = LOG_DIR / "inputs" / "text"
    in_txt.mkdir(parents=True, exist_ok=True)
    (in_txt / f"{ts}_a2t_meta.txt").write_text(
        f"[Audio-to-Text — {ts}]\n"
        f"Detected language : {detected_language}\n"
        f"Output language   : {output_language}\n",
        encoding="utf-8",
    )

    out_txt = LOG_DIR / "outputs" / "text"
    out_txt.mkdir(parents=True, exist_ok=True)
    (out_txt / f"{ts}_a2t.txt").write_text(
        f"[Audio-to-Text — {ts}]\n"
        f"Detected : {detected_language}\n"
        f"Output   : {output_language}\n\n"
        f"--- transcript (as spoken) ---\n{transcript_raw}\n\n"
        f"--- transcript (output language) ---\n{transcript_out}\n",
        encoding="utf-8",
    )


def log_a2a_session(
    *,
    input_audio_path: str,
    detected_language: str,
    output_language: str,
    transcript_raw: str,
    transcript_out: str,
    engine: str,
    voice_id: str,
    speed: str,
    output_audio_path: str | None = None,
) -> None:
    """Log an audio-to-audio session."""
    ts = _ts()

    if input_audio_path:
        src = Path(input_audio_path)
        _copy(src, LOG_DIR / "inputs" / "audio", f"{ts}_a2a{src.suffix}")

    in_txt = LOG_DIR / "inputs" / "text"
    in_txt.mkdir(parents=True, exist_ok=True)
    (in_txt / f"{ts}_a2a_meta.txt").write_text(
        f"[Audio-to-Audio — {ts}]\n"
        f"Detected language : {detected_language}\n"
        f"Output language   : {output_language}\n"
        f"Engine            : {engine}\n"
        f"Voice             : {voice_id}\n"
        f"Speed             : {speed}\n",
        encoding="utf-8",
    )

    out_txt = LOG_DIR / "outputs" / "text"
    out_txt.mkdir(parents=True, exist_ok=True)
    (out_txt / f"{ts}_a2a.txt").write_text(
        f"[Audio-to-Audio — {ts}]\n"
        f"Detected : {detected_language}\n"
        f"Output   : {output_language}\n\n"
        f"--- transcript (as spoken) ---\n{transcript_raw}\n\n"
        f"--- transcript (output language) ---\n{transcript_out}\n",
        encoding="utf-8",
    )

    if output_audio_path:
        src = Path(output_audio_path)
        _copy(src, LOG_DIR / "outputs" / "audio", f"{ts}_a2a{src.suffix}")
