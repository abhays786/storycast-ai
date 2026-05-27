"""
TTS backend interface.

`speed` is best-effort: backends apply it differently (length-scale in Piper,
native time-stretch in Coqui, post-resampling for Gemini/Bark).

Backends now publish a voice catalog via `voices()`. Callers may pass an
explicit `voice_id` to `generate()`; if omitted, the backend picks a default
using (language, gender). The legacy `gender` parameter is still honoured
for backward compatibility.
"""

from abc import ABC, abstractmethod

from tts.voices import VoiceInfo, default_voice_for, filter_voices


class TTSBackend(ABC):
    """Abstract base class for a TTS engine."""

    # Stable string identifier (used by registry, .env, logs). Subclasses MUST
    # override; the registration check below enforces this at class-creation
    # time so typos are caught early.
    id: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("__abstract_backend__", False):
            return
        if not getattr(cls, "id", ""):
            raise TypeError(
                f"{cls.__name__} must set a non-empty class attribute `id`"
            )

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in the UI. May include hardware hints."""

    @abstractmethod
    def voices(self) -> list[VoiceInfo]:
        """Return every voice this backend can synthesise with."""

    @abstractmethod
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
        """
        Synthesise text to a WAV file and return its path, or None on failure.

        Args:
            text:            Text to speak.
            gender:          Gender hint for default voice selection.
            speed:           Playback speed multiplier in [0.5, 2.0]. Best-effort.
            language:        ISO-639-1 code; influences voice selection.
            voice_id:        Explicit voice override; if None, backend picks default.
            output_basename: Optional filename prefix for the output WAV.
        """

    def ensure_models(self) -> None:
        """Optional hook to pre-download / warm models. Default: no-op."""
        return None

    def is_available(self) -> bool:
        """Return True if this backend can actually run right now."""
        return True

    # ── Helpers shared by every concrete backend ────────────────────────────

    def voices_for(self, language: str) -> list[VoiceInfo]:
        """Voices restricted to the given language."""
        return filter_voices(self.voices(), language)

    def resolve_voice(self, language: str, gender: str, voice_id: str | None) -> VoiceInfo | None:
        """
        Return the VoiceInfo to use for this call.

        Explicit `voice_id` wins; otherwise fall back to the language/gender default.
        """
        catalog = self.voices()
        if voice_id:
            for v in catalog:
                if v.id == voice_id:
                    return v
        return default_voice_for(catalog, language, gender)
