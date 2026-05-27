"""
Speech-to-text — separate from TTS to honour SRP.

Uses Google's STT via `speech_recognition`. Supports English and Hindi.
"""

import os
import tempfile
from pathlib import Path


# Language code mapping for Google STT.
_GOOGLE_LANG = {
    "en":    "en-IN",
    "en-IN": "en-IN",
    "en-US": "en-US",
    "hi":    "hi-IN",
    "hi-IN": "hi-IN",
}

# Order tried when language="auto" (longest non-empty result wins).
_AUTO_TRY_LANGS = ["en-IN", "hi-IN"]


def _normalize(lang: str | None) -> str:
    if not lang:
        return "auto"
    lang = lang.lower()
    if lang in ("auto", ""):
        return "auto"
    return _GOOGLE_LANG.get(lang, lang)


def _ensure_wav(audio_path: str) -> tuple[str, bool]:
    """Return a path to a .wav file (converting via pydub if needed) plus a
    bool flag indicating whether we created a temp file (and so must clean it
    up afterwards)."""
    if audio_path.lower().endswith(".wav"):
        return audio_path, False
    try:
        from pydub import AudioSegment
        sound = AudioSegment.from_file(audio_path)
        tmp   = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sound.export(tmp.name, format="wav")
        return tmp.name, True
    except Exception:
        return audio_path, False


def transcribe_audio(audio_path: str, language: str | None = None) -> tuple[str, str]:
    """
    Transcribe an audio file. Returns (recognised_text, detected_language).

    `language` may be:
      - "en" / "en-IN" / "en-US" → force English
      - "hi" / "hi-IN"           → force Hindi
      - None / "auto"            → try Hindi and English, pick the better
                                   transcription (longest non-empty result)

    On failure both return values are empty strings.
    """
    if not audio_path or not Path(audio_path).exists():
        return "", ""

    import speech_recognition as sr

    target = _normalize(language)
    recognizer = sr.Recognizer()
    wav_path, created_tmp = _ensure_wav(audio_path)

    try:
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)

        candidates = _AUTO_TRY_LANGS if target == "auto" else [target]
        best_text, best_lang = "", ""
        for lang_code in candidates:
            try:
                text = recognizer.recognize_google(audio_data, language=lang_code)
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                continue
            if len(text) > len(best_text):
                best_text, best_lang = text, lang_code
        return best_text, best_lang

    except Exception:
        return "", ""
    finally:
        if created_tmp and Path(wav_path).exists():
            try:
                os.unlink(wav_path)
            except Exception:
                pass
