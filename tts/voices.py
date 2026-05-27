"""
Voice catalog primitives.

`VoiceInfo` is the per-voice record every backend exposes. Backends keep their
own list of voices (with language + gender hints), and the registry / UI ask
for "voices available for this language" rather than baking voice ids
anywhere outside the backend.
"""

from dataclasses import dataclass
from typing import Iterable


LANGUAGE_CODES = ("en", "hi")


@dataclass(frozen=True)
class VoiceInfo:
    """Metadata for one synthesis voice."""
    id:           str           # backend-specific identifier (file id / preset name)
    display_name: str           # human-friendly name shown in dropdowns
    language:     str           # "en" | "hi"
    gender:       str           # "boy" | "girl" | "both"


def filter_voices(voices: Iterable[VoiceInfo], language: str) -> list[VoiceInfo]:
    """Return only voices that match the given language."""
    return [v for v in voices if v.language == language]


def default_voice_for(
    voices: Iterable[VoiceInfo],
    language: str,
    gender: str,
) -> VoiceInfo | None:
    """Pick the best default voice for (language, gender).

    Preference order:
      1. exact (language, gender) match
      2. (language, "both")
      3. first voice in the language
      4. first voice overall
    """
    voices = list(voices)
    matches = [v for v in voices if v.language == language]
    if not matches:
        return voices[0] if voices else None

    exact = [v for v in matches if v.gender == gender]
    if exact:
        return exact[0]

    both = [v for v in matches if v.gender == "both"]
    if both:
        return both[0]

    return matches[0]
