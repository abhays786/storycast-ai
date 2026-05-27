"""
Piper TTS backend — fast, fully offline synthesis.

Optionally accelerated via OpenVINO when an Intel GPU / NPU is present.
Speed is applied by adjusting Piper's `length_scale` (no pitch shift).

Voice catalog:
    en — Alba (UK female), Arctic (US neutral)
    hi — Rohan (Hindi neutral)
"""

import shutil
import wave
from pathlib import Path

from app_logging import get_logger
from config import MODEL_DIR
from tts._utils import new_output_path
from tts.base import TTSBackend
from tts.devices import ov_device
from tts.runtime import get_ov_session
from tts.voices import VoiceInfo

log = get_logger(__name__)


# ── Voice catalog ────────────────────────────────────────────────────────────
VOICES: list[VoiceInfo] = [
    VoiceInfo(id="en_GB-alba-medium",   display_name="Alba (UK English)",     language="en", gender="boy"),
    VoiceInfo(id="en_US-arctic-medium", display_name="Arctic (US English)",   language="en", gender="both"),
    VoiceInfo(id="hi_IN-rohan-medium",  display_name="Rohan (Hindi)",         language="hi", gender="both"),
]

# Piper exposes a per-voice length-scale tuning; we use a single base value here.
_BASE_LENGTH_SCALE = 1.05


# ── Model management helpers ──────────────────────────────────────────────────

def _model_paths(voice_id: str) -> tuple[Path, Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / f"{voice_id}.onnx", MODEL_DIR / f"{voice_id}.onnx.json"


def _hf_filename(voice_id: str, ext: str) -> str:
    parts       = voice_id.split("-")
    lang_region = parts[0]
    name        = parts[1]
    quality     = parts[2]
    lang        = lang_region.split("_")[0]
    return f"{lang}/{lang_region}/{name}/{quality}/{voice_id}{ext}"


def _download_file(hf_filename: str, dest: Path) -> bool:
    from huggingface_hub import hf_hub_download
    log.info("Downloading %s ...", dest.name)
    try:
        cached = hf_hub_download(repo_id="rhasspy/piper-voices", filename=hf_filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dest)
        log.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1_048_576)
        return True
    except Exception as exc:
        log.error("Download failed: %s", exc)
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def _download_voice(voice_id: str) -> bool:
    onnx, cfg = _model_paths(voice_id)
    ok = True
    for path, ext, min_sz in [(onnx, ".onnx", 4096), (cfg, ".onnx.json", 64)]:
        if path.exists() and path.stat().st_size > min_sz:
            continue
        ok = _download_file(_hf_filename(voice_id, ext), path) and ok
    return ok


def _voice_is_complete(voice_id: str) -> bool:
    onnx, cfg = _model_paths(voice_id)
    return (onnx.exists() and onnx.stat().st_size > 4096
            and cfg.exists() and cfg.stat().st_size > 64)


# ── Backend class ─────────────────────────────────────────────────────────────

class PiperBackend(TTSBackend):
    id = "piper"

    @property
    def display_name(self) -> str:
        dev = ov_device()
        tag = ("Intel Arc GPU" if dev == "GPU"
               else "Intel NPU" if dev == "NPU"
               else "Fast & Offline")
        return f"Piper — {tag}"

    def voices(self) -> list[VoiceInfo]:
        return list(VOICES)

    def ensure_models(self) -> None:
        dev  = ov_device()
        note = "(Intel Arc GPU)" if dev == "GPU" else "(NPU)" if dev == "NPU" else ""
        log.info("Piper device: OpenVINO %s %s", dev, note)
        log.info("Checking Piper voice models...")
        seen: set[str] = set()
        for v in VOICES:
            if v.id in seen:
                continue
            seen.add(v.id)
            if _voice_is_complete(v.id):
                log.info("  %s — already cached", v.id)
            else:
                log.info("  %s — downloading ...", v.id)
                _download_voice(v.id)
        log.info("Voice models ready.")

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
            from piper import PiperVoice
            from piper.config import SynthesisConfig
        except ImportError:
            log.error("piper-tts not installed. Run: pip install piper-tts")
            return None

        voice = self.resolve_voice(language, gender, voice_id)
        if voice is None:
            log.error("No Piper voice available for language=%s gender=%s", language, gender)
            return None

        length_scale = _BASE_LENGTH_SCALE / speed

        if not _voice_is_complete(voice.id):
            log.info("Voice model '%s' incomplete — downloading now...", voice.id)
            if not _download_voice(voice.id):
                return None

        onnx_path, _ = _model_paths(voice.id)
        out_path = new_output_path(basename=output_basename)

        try:
            import numpy as np
            piper_voice = PiperVoice.load(str(onnx_path), use_cuda=False)

            if ov_device() != "CPU":
                dummy = {
                    "input":         np.zeros((1, 10), dtype=np.int64),
                    "input_lengths": np.array([10],    dtype=np.int64),
                    "scales":        np.array([0.667, 1.0, 0.8], dtype=np.float32),
                }
                ov_session = get_ov_session(str(onnx_path), dummy_feed=dummy)
                if ov_session:
                    piper_voice.session = ov_session

            syn_cfg = SynthesisConfig(length_scale=length_scale)
            chunks  = list(piper_voice.synthesize(text, syn_cfg))

            if not chunks:
                log.warning("Piper returned no audio chunks.")
                return None

            first = chunks[0]
            with wave.open(str(out_path), "w") as wav_fh:
                wav_fh.setnchannels(first.sample_channels)
                wav_fh.setsampwidth(first.sample_width)
                wav_fh.setframerate(first.sample_rate)
                for chunk in chunks:
                    pcm = (chunk.audio_float_array * 32767).astype(np.int16)
                    wav_fh.writeframes(pcm.tobytes())

            if out_path.exists() and out_path.stat().st_size > 1024:
                log.info("Piper synthesis complete: %s (voice=%s)", out_path.name, voice.id)
                return str(out_path)

            log.warning("Piper produced an empty file.")
            return None

        except Exception as exc:
            log.error("Piper synthesis error: %s", exc)
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            return None
