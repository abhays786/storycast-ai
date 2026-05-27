"""Tests for tts/piper_backend.py."""

import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from tts import piper_backend
from tts.piper_backend import PiperBackend


# ── Test helpers / fakes ─────────────────────────────────────────────────────

class _FakeChunk:
    def __init__(self, samples=1024):
        self.sample_channels = 1
        self.sample_width = 2
        self.sample_rate = 22050
        self.audio_float_array = np.zeros(samples, dtype=np.float32)


class _FakeVoice:
    """Configurable PiperVoice replacement."""
    chunks_per_call = 2
    samples_per_chunk = 1024

    def __init__(self):
        self.session = None
        self.synth_calls = []

    @classmethod
    def load(cls, path, use_cuda):
        v = cls()
        v.path = path
        return v

    def synthesize(self, text, cfg):
        self.synth_calls.append((text, cfg))
        for _ in range(self.chunks_per_call):
            yield _FakeChunk(self.samples_per_chunk)


def _install_piper(monkeypatch, voice_cls=_FakeVoice):
    piper_mod = types.ModuleType("piper")
    piper_mod.PiperVoice = voice_cls
    piper_cfg = types.ModuleType("piper.config")
    piper_cfg.SynthesisConfig = lambda length_scale: types.SimpleNamespace(length_scale=length_scale)
    monkeypatch.setitem(sys.modules, "piper", piper_mod)
    monkeypatch.setitem(sys.modules, "piper.config", piper_cfg)
    return piper_mod


def _stamp_voice_files(model_dir: Path, voice_id: str, onnx_size=8192, json_size=128):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"{voice_id}.onnx").write_bytes(b"x" * onnx_size)
    (model_dir / f"{voice_id}.onnx.json").write_bytes(b"{}" + b"x" * (json_size - 2))


# ── display_name + is_available ─────────────────────────────────────────────

def test_display_name_cpu(monkeypatch):
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "CPU")
    assert "Fast & Offline" in PiperBackend().display_name


def test_display_name_gpu(monkeypatch):
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "GPU")
    assert "Intel Arc GPU" in PiperBackend().display_name


def test_display_name_npu(monkeypatch):
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "NPU")
    assert "Intel NPU" in PiperBackend().display_name


def test_is_available_default():
    assert PiperBackend().is_available() is True


def test_voices_returns_full_catalog():
    voices = PiperBackend().voices()
    ids = {v.id for v in voices}
    assert ids == {"en_GB-alba-medium", "en_US-arctic-medium", "hi_IN-rohan-medium"}


def test_voices_for_language():
    p = PiperBackend()
    en_ids = {v.id for v in p.voices_for("en")}
    hi_ids = {v.id for v in p.voices_for("hi")}
    assert en_ids == {"en_GB-alba-medium", "en_US-arctic-medium"}
    assert hi_ids == {"hi_IN-rohan-medium"}


def test_default_voice_mapping_matches_spec():
    p = PiperBackend()
    assert p.resolve_voice("en", "boy",  None).id == "en_GB-alba-medium"
    assert p.resolve_voice("en", "girl", None).id == "en_US-arctic-medium"
    assert p.resolve_voice("en", "both", None).id == "en_US-arctic-medium"
    assert p.resolve_voice("hi", "boy",  None).id == "hi_IN-rohan-medium"
    assert p.resolve_voice("hi", "girl", None).id == "hi_IN-rohan-medium"


# ── Helpers ─────────────────────────────────────────────────────────────────

def test_hf_filename_shape():
    assert piper_backend._hf_filename("en_US-arctic-medium", ".onnx") == \
        "en/en_US/arctic/medium/en_US-arctic-medium.onnx"
    assert piper_backend._hf_filename("hi_IN-rohan-medium", ".onnx.json") == \
        "hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json"


def test_voice_is_complete_true(monkeypatch, tmp_path):
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _stamp_voice_files(tmp_path, "en_US-arctic-medium")
    assert piper_backend._voice_is_complete("en_US-arctic-medium") is True


def test_voice_is_complete_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    assert piper_backend._voice_is_complete("en_US-arctic-medium") is False


def test_voice_is_complete_too_small(monkeypatch, tmp_path):
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    (tmp_path / "x.onnx").write_bytes(b"x" * 10)
    (tmp_path / "x.onnx.json").write_bytes(b"x" * 10)
    assert piper_backend._voice_is_complete("x") is False


# ── Download paths ───────────────────────────────────────────────────────────

def _install_hub(monkeypatch, succeed=True):
    mod = types.ModuleType("huggingface_hub")
    def hf_hub_download(repo_id, filename):
        if not succeed:
            raise RuntimeError("hub failed")
        p = Path(filename).name + ".cached"
        # Return a path that exists with non-zero size
        out = Path(monkeypatch._tmp / "hub" / p)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"X" * 5000)
        return str(out)
    mod.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return mod


def test_download_file_success(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    _install_hub(monkeypatch, succeed=True)
    dest = tmp_path / "out.onnx"
    assert piper_backend._download_file("en/foo/bar/medium/x.onnx", dest) is True
    assert dest.exists() and dest.stat().st_size > 0


def test_download_file_failure_removes_partial(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    _install_hub(monkeypatch, succeed=False)
    dest = tmp_path / "out.onnx"
    dest.write_bytes(b"old")
    assert piper_backend._download_file("x", dest) is False
    assert not dest.exists()


def test_download_voice_skips_when_present(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _stamp_voice_files(tmp_path, "en_US-arctic-medium")
    calls = []
    monkeypatch.setattr(
        piper_backend, "_download_file",
        lambda fn, dest: calls.append((fn, dest)) or True,
    )
    assert piper_backend._download_voice("en_US-arctic-medium") is True
    assert calls == []


def test_download_voice_downloads_missing(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    calls = []
    def fake_download(fn, dest):
        calls.append(dest.name)
        dest.write_bytes(b"X" * 5000)
        return True
    monkeypatch.setattr(piper_backend, "_download_file", fake_download)
    assert piper_backend._download_voice("en_US-arctic-medium") is True
    assert any(n.endswith(".onnx") for n in calls)


# ── ensure_models ────────────────────────────────────────────────────────────

def test_ensure_models_mixed_cached_and_downloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "GPU")
    # Arctic cached; Alba + Rohan not
    _stamp_voice_files(tmp_path, "en_US-arctic-medium")
    calls = []
    monkeypatch.setattr(piper_backend, "_download_voice",
                        lambda vid: calls.append(vid) or True)
    PiperBackend().ensure_models()
    assert "en_GB-alba-medium" in calls
    assert "hi_IN-rohan-medium" in calls
    assert "en_US-arctic-medium" not in calls


def test_ensure_models_npu_branch(monkeypatch, tmp_path):
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "NPU")
    monkeypatch.setattr(piper_backend, "_download_voice", lambda vid: True)
    PiperBackend().ensure_models()  # smoke


# ── generate ────────────────────────────────────────────────────────────────

def test_generate_missing_piper(monkeypatch, tmp_path):
    # Force `from piper import PiperVoice` to raise
    monkeypatch.delitem(sys.modules, "piper", raising=False)
    monkeypatch.delitem(sys.modules, "piper.config", raising=False)
    import builtins
    real_import = builtins.__import__
    def fail_piper(name, *args, **kwargs):
        if name.startswith("piper"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fail_piper)
    out = PiperBackend().generate("hi", "boy", 1.0)
    assert out is None


def test_generate_download_fails(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _install_piper(monkeypatch)
    monkeypatch.setattr(piper_backend, "_voice_is_complete", lambda vid: False)
    monkeypatch.setattr(piper_backend, "_download_voice", lambda vid: False)
    assert PiperBackend().generate("hi", "boy", 1.0) is None


def test_generate_cpu_success(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _stamp_voice_files(tmp_path, "en_US-arctic-medium")
    _install_piper(monkeypatch)
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "CPU")
    # ASSETS_DIR is already patched by conftest to tmp_path/assets
    out = PiperBackend().generate("hello world", "girl", 1.0)
    assert out is not None
    assert Path(out).exists()


def test_generate_uses_ov_when_accelerator_present(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _stamp_voice_files(tmp_path, "en_US-arctic-medium")

    captured = {}
    class _OVVoice(_FakeVoice):
        def synthesize(self, text, cfg):
            captured["session"] = self.session
            yield _FakeChunk()

    _install_piper(monkeypatch, voice_cls=_OVVoice)
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "GPU")

    fake_session = object()
    monkeypatch.setattr(piper_backend, "get_ov_session",
                        lambda path, dummy_feed=None: fake_session)
    out = PiperBackend().generate("hi", "both", 1.0)
    assert out is not None
    assert captured["session"] is fake_session


def test_generate_ov_session_returns_none_keeps_cpu(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _stamp_voice_files(tmp_path, "en_US-arctic-medium")
    _install_piper(monkeypatch)
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "NPU")
    monkeypatch.setattr(piper_backend, "get_ov_session",
                        lambda path, dummy_feed=None: None)
    assert PiperBackend().generate("hi", "both", 1.0) is not None


def test_generate_no_chunks_returns_none(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _stamp_voice_files(tmp_path, "en_US-arctic-medium")

    class _EmptyVoice(_FakeVoice):
        chunks_per_call = 0

    _install_piper(monkeypatch, voice_cls=_EmptyVoice)
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "CPU")
    assert PiperBackend().generate("hi", "boy", 1.0) is None


def test_generate_tiny_file_returns_none(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _stamp_voice_files(tmp_path, "en_US-arctic-medium")

    class _TinyVoice(_FakeVoice):
        chunks_per_call = 1
        samples_per_chunk = 4   # tiny — final wav stays under 1024 bytes
    _install_piper(monkeypatch, voice_cls=_TinyVoice)
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "CPU")
    out = PiperBackend().generate("hi", "boy", 1.0)
    assert out is None


def test_generate_unknown_gender_falls_back_to_both(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _stamp_voice_files(tmp_path, "en_US-arctic-medium")
    _install_piper(monkeypatch)
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "CPU")
    out = PiperBackend().generate("hi", "alien", 1.0)
    assert out is not None


def test_generate_exception_cleans_up(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _stamp_voice_files(tmp_path, "en_US-arctic-medium")

    class _BoomVoice(_FakeVoice):
        @classmethod
        def load(cls, path, use_cuda):
            raise RuntimeError("boom")

    _install_piper(monkeypatch, voice_cls=_BoomVoice)
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "CPU")
    out = PiperBackend().generate("hi", "boy", 1.0)
    assert out is None


def test_generate_late_exception_cleans_up_partial_file(monkeypatch, tmp_path):
    """Exception fired AFTER the wave file is written must unlink the partial."""
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _stamp_voice_files(tmp_path, "en_US-arctic-medium")

    class _LateBoomChunk:
        sample_channels = 1
        sample_width = 2
        sample_rate = 22050
        @property
        def audio_float_array(self):
            raise RuntimeError("late boom")

    class _LateBoomVoice(_FakeVoice):
        def synthesize(self, text, cfg):
            yield _LateBoomChunk()

    _install_piper(monkeypatch, voice_cls=_LateBoomVoice)
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "CPU")
    out = PiperBackend().generate("hi", "boy", 1.0)
    assert out is None
    # No leftover .wav files should remain in ASSETS_DIR
    from config import ASSETS_DIR
    if ASSETS_DIR.exists():
        assert not list(ASSETS_DIR.glob("story_*.wav"))


def test_generate_returns_none_when_no_voice_for_language(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _install_piper(monkeypatch)
    # Empty catalog → resolve_voice returns None
    monkeypatch.setattr(piper_backend, "VOICES", [])
    assert PiperBackend().generate("hi", "boy", 1.0) is None


def test_ensure_models_dedupes_repeated_ids(monkeypatch, tmp_path):
    from tts.voices import VoiceInfo
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "CPU")
    # Two entries with the SAME id — the second must be skipped via `seen`.
    monkeypatch.setattr(piper_backend, "VOICES", [
        VoiceInfo(id="en_GB-alba-medium", display_name="A1", language="en", gender="boy"),
        VoiceInfo(id="en_GB-alba-medium", display_name="A2", language="en", gender="girl"),
    ])
    calls: list[str] = []
    monkeypatch.setattr(piper_backend, "_download_voice",
                        lambda vid: calls.append(vid) or True)
    PiperBackend().ensure_models()
    assert calls == ["en_GB-alba-medium"]   # downloaded once


def test_generate_speed_scales_length(monkeypatch, tmp_path):
    monkeypatch._tmp = tmp_path
    monkeypatch.setattr(piper_backend, "MODEL_DIR", tmp_path)
    _stamp_voice_files(tmp_path, "en_US-arctic-medium")
    captured = {}
    class _Tracer(_FakeVoice):
        def synthesize(self, text, cfg):
            captured["length_scale"] = cfg.length_scale
            yield _FakeChunk()
    _install_piper(monkeypatch, voice_cls=_Tracer)
    monkeypatch.setattr(piper_backend, "ov_device", lambda: "CPU")
    PiperBackend().generate("hi", "girl", 2.0)
    assert captured["length_scale"] == pytest.approx(1.05 / 2.0)
