"""
Coqui TTS backend — natural neural voices via the jenny single-speaker model.

Jenny is English-only. Speed is passed natively to `tts_to_file`.
Currently disabled in the default UI dropdown — enable via `ENABLED_BACKENDS`.
"""

from pathlib import Path

from app_logging import get_logger
from tts._utils import new_output_path
from tts.base import TTSBackend
from tts.devices import torch_device
from tts.voices import VoiceInfo

log = get_logger(__name__)


VOICES: list[VoiceInfo] = [
    VoiceInfo(id="jenny", display_name="Jenny (US English)", language="en", gender="girl"),
]

_JENNY_MODEL = "tts_models/en/jenny/jenny"


def _patch_transformers_compat() -> None:
    """Shim for `isin_mps_friendly` removed in transformers 4.40+."""
    try:
        import transformers.pytorch_utils as _pu
        if not hasattr(_pu, "isin_mps_friendly"):
            import torch as _torch
            def isin_mps_friendly(elements, test_elements):
                return _torch.isin(elements, test_elements)
            _pu.isin_mps_friendly = isin_mps_friendly
    except Exception:
        pass


class CoquiBackend(TTSBackend):
    id = "coqui"

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    @property
    def display_name(self) -> str:
        return "Coqui TTS — Natural neural voices"

    def voices(self) -> list[VoiceInfo]:
        return list(VOICES)

    def is_available(self) -> bool:
        try:
            import TTS.api  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_model(self, model_name: str):
        if model_name not in self._cache:
            _patch_transformers_compat()
            from TTS.api import TTS
            dev = torch_device()
            log.info("Loading Coqui TTS model '%s' on %s (first load may take a minute)...",
                     model_name, dev.upper())
            tts = TTS(model_name, gpu=(dev == "cuda"))
            if dev == "xpu":
                try:
                    import torch
                    tts.synthesizer.tts_model.to(torch.device("xpu"))
                    log.info("Coqui model moved to XPU.")
                except Exception as e:
                    log.warning("XPU move skipped: %s", e)
            self._cache[model_name] = tts
        return self._cache[model_name]

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
            _patch_transformers_compat()
            from TTS.api import TTS  # noqa: F401 — availability check
        except ImportError:
            log.error("Coqui TTS not installed. Run: pip install coqui-tts")
            return None

        if language != "en":
            log.warning("Coqui (jenny) supports English only; falling back to English voice.")

        out_path = str(new_output_path(basename=output_basename))

        try:
            tts = self._get_model(_JENNY_MODEL)
            tts.tts_to_file(text=text, file_path=out_path, speed=speed)

            if Path(out_path).exists() and Path(out_path).stat().st_size > 1024:
                log.info("Coqui synthesis complete: %s", Path(out_path).name)
                return out_path

            log.warning("Coqui produced an empty file.")
            return None

        except Exception as exc:
            log.error("Coqui TTS error: %s", exc)
            return None
