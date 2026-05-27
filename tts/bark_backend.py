"""
BARK TTS backend — expressive, GPU-optional voice synthesis via Suno BARK.

BARK is English-leaning by default; we keep the same three preset voices and
expose them as English-only here. Currently disabled in the default UI.
"""

from pathlib import Path

from app_logging import get_logger
from tts._utils import apply_speed_pydub, chunk_text, new_output_path
from tts.base import TTSBackend
from tts.devices import torch_device
from tts.voices import VoiceInfo

log = get_logger(__name__)


VOICES: list[VoiceInfo] = [
    VoiceInfo(id="v2/en_speaker_6", display_name="Speaker 6 (male)",    language="en", gender="boy"),
    VoiceInfo(id="v2/en_speaker_9", display_name="Speaker 9 (female)",  language="en", gender="girl"),
    VoiceInfo(id="v2/en_speaker_1", display_name="Speaker 1 (neutral)", language="en", gender="both"),
]
MAX_CHUNK_CHARS = 300


class BarkBackend(TTSBackend):
    id = "bark"

    def __init__(self) -> None:
        self._loaded = False

    @property
    def display_name(self) -> str:
        tag = "GPU" if torch_device() != "cpu" else "slow on CPU"
        return f"BARK — Expressive [{tag}]"

    def voices(self) -> list[VoiceInfo]:
        return list(VOICES)

    def is_available(self) -> bool:
        try:
            import bark  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        try:
            from bark import preload_models
            import functools
            import torch as _torch
        except ImportError:
            log.error("BARK not installed. Run: pip install suno-bark")
            return False

        dev       = torch_device()
        use_gpu   = dev in ("cuda", "xpu")
        use_small = dev == "cpu"
        log.info(
            "Loading BARK %smodels on %s (first load downloads %s)...",
            "small " if use_small else "",
            dev.upper(),
            "~500 MB" if use_small else "~1-2 GB",
        )

        _orig_load = _torch.load
        _torch.load = functools.partial(_orig_load, weights_only=False)
        try:
            use_cuda = dev == "cuda"
            preload_models(
                text_use_gpu=use_cuda,   text_use_small=use_small,
                coarse_use_gpu=use_cuda, coarse_use_small=use_small,
                fine_use_gpu=use_cuda,   fine_use_small=use_small,
            )
            if dev == "xpu":
                try:
                    import bark.generation as _bg
                    for key in list(getattr(_bg, "models", {}).keys()):
                        m = _bg.models[key]
                        if hasattr(m, "to"):
                            _bg.models[key] = m.to("xpu")
                    log.info("BARK models moved to XPU.")
                except Exception as e:
                    log.warning("BARK XPU move skipped: %s", e)
        finally:
            _torch.load = _orig_load

        self._loaded = True
        return True

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
            from bark import generate_audio as bark_gen, SAMPLE_RATE
            import numpy as np
            from scipy.io.wavfile import write as write_wav
        except ImportError:
            log.error("BARK not installed. Run: pip install suno-bark")
            return None

        if not self._ensure_loaded():
            return None

        voice  = self.resolve_voice(language, gender, voice_id)
        preset = voice.id if voice else VOICES[-1].id
        chunks    = chunk_text(text, MAX_CHUNK_CHARS)
        n         = len(chunks)
        dev       = torch_device()
        secs_each = 18 if dev == "cpu" else 5
        log.info("BARK: %d chunk(s) — estimated %dm %ds on %s",
                 n, n * secs_each // 60, n * secs_each % 60, dev.upper())

        out_path = str(new_output_path(basename=output_basename))

        try:
            silence_arr = np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)
            audio_parts = []
            for i, chunk in enumerate(chunks, 1):
                log.info("BARK: generating chunk %d/%d ...", i, n)
                audio_parts.append(bark_gen(chunk, history_prompt=preset))
                if i < n:
                    audio_parts.append(silence_arr)

            full_audio = np.concatenate(audio_parts).astype(np.float32)
            write_wav(out_path, SAMPLE_RATE, full_audio)
            apply_speed_pydub(out_path, speed)

            if Path(out_path).exists() and Path(out_path).stat().st_size > 1024:
                log.info("BARK synthesis complete: %s", Path(out_path).name)
                return out_path

            log.warning("BARK produced an empty file.")
            return None

        except Exception as exc:
            log.error("BARK error: %s", exc)
            return None
