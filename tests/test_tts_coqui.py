"""Tests for tts/coqui_backend.py."""

import sys
import types
from pathlib import Path

import pytest

from tts import coqui_backend
from tts.coqui_backend import CoquiBackend


# ── Fake TTS.api ─────────────────────────────────────────────────────────────

def _install_tts(monkeypatch, *, raise_=None, file_size=5000):
    pkg = types.ModuleType("TTS")
    api = types.ModuleType("TTS.api")

    class _Synth:
        class _Model:
            def to(self, device):
                pass
        tts_model = _Model()

    class _TTS:
        def __init__(self, model_name, gpu=False):
            self.model_name = model_name
            self.gpu = gpu
            self.synthesizer = _Synth()
        def tts_to_file(self, text, file_path, speed, **kw):
            if raise_ is not None:
                raise raise_
            Path(file_path).write_bytes(b"X" * file_size)

    api.TTS = _TTS
    pkg.api = api
    monkeypatch.setitem(sys.modules, "TTS", pkg)
    monkeypatch.setitem(sys.modules, "TTS.api", api)
    return api


def test_display_name():
    assert "Coqui" in CoquiBackend().display_name


def test_is_available_yes(monkeypatch):
    _install_tts(monkeypatch)
    assert CoquiBackend().is_available() is True


def test_is_available_no(monkeypatch):
    monkeypatch.delitem(sys.modules, "TTS", raising=False)
    monkeypatch.delitem(sys.modules, "TTS.api", raising=False)
    import builtins
    real_import = builtins.__import__
    def fail(name, *a, **k):
        if name == "TTS.api" or name == "TTS":
            raise ImportError("no TTS")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fail)
    assert CoquiBackend().is_available() is False


def test_patch_transformers_compat_swallows_errors(monkeypatch):
    # Just confirms no exception propagates even when transformers is absent.
    import builtins
    real_import = builtins.__import__
    def fail(name, *a, **k):
        if name.startswith("transformers"):
            raise ImportError("no transformers")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fail)
    coqui_backend._patch_transformers_compat()


def test_patch_transformers_installs_shim(monkeypatch):
    # Build a fake transformers.pytorch_utils with no isin_mps_friendly attribute.
    transformers = types.ModuleType("transformers")
    pytorch_utils = types.ModuleType("transformers.pytorch_utils")
    transformers.pytorch_utils = pytorch_utils
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.pytorch_utils", pytorch_utils)
    # Provide a fake torch so the shim can call torch.isin
    fake_torch = types.SimpleNamespace(isin=lambda a, b: ("isin", a, b))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    coqui_backend._patch_transformers_compat()
    assert hasattr(pytorch_utils, "isin_mps_friendly")
    assert pytorch_utils.isin_mps_friendly(1, 2) == ("isin", 1, 2)


def test_patch_transformers_noop_when_already_patched(monkeypatch):
    transformers = types.ModuleType("transformers")
    pytorch_utils = types.ModuleType("transformers.pytorch_utils")
    pytorch_utils.isin_mps_friendly = lambda a, b: "existing"
    transformers.pytorch_utils = pytorch_utils
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.pytorch_utils", pytorch_utils)
    coqui_backend._patch_transformers_compat()
    assert pytorch_utils.isin_mps_friendly(1, 2) == "existing"


def test_generate_missing_library(monkeypatch):
    monkeypatch.delitem(sys.modules, "TTS", raising=False)
    monkeypatch.delitem(sys.modules, "TTS.api", raising=False)
    import builtins
    real_import = builtins.__import__
    def fail(name, *a, **k):
        if name == "TTS.api" or name == "TTS":
            raise ImportError("no TTS")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fail)
    assert CoquiBackend().generate("hi", "boy", 1.0) is None


def test_generate_success_cpu(monkeypatch):
    _install_tts(monkeypatch)
    monkeypatch.setattr(coqui_backend, "torch_device", lambda: "cpu")
    out = CoquiBackend().generate("hi there", "boy", 1.0)
    assert out is not None
    assert Path(out).exists()


def test_generate_success_cuda(monkeypatch):
    _install_tts(monkeypatch)
    monkeypatch.setattr(coqui_backend, "torch_device", lambda: "cuda")
    out = CoquiBackend().generate("hi", "boy", 1.0)
    assert out is not None


def test_generate_success_xpu(monkeypatch):
    _install_tts(monkeypatch)
    monkeypatch.setattr(coqui_backend, "torch_device", lambda: "xpu")
    # Stub torch so the .to(xpu) call works
    fake_torch = types.SimpleNamespace(device=lambda d: f"dev({d})")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    out = CoquiBackend().generate("hi", "boy", 1.0)
    assert out is not None


def test_generate_xpu_move_failure_logs_warning(monkeypatch):
    _install_tts(monkeypatch)
    monkeypatch.setattr(coqui_backend, "torch_device", lambda: "xpu")
    # Make `import torch` raise so the `try:` inside _get_model logs a warning
    import builtins
    real_import = builtins.__import__
    def fail(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fail)
    out = CoquiBackend().generate("hi", "boy", 1.0)
    assert out is not None


def test_generate_tiny_file_returns_none(monkeypatch):
    _install_tts(monkeypatch, file_size=100)
    monkeypatch.setattr(coqui_backend, "torch_device", lambda: "cpu")
    out = CoquiBackend().generate("hi", "boy", 1.0)
    assert out is None


def test_generate_exception_returns_none(monkeypatch):
    _install_tts(monkeypatch, raise_=RuntimeError("synth failed"))
    monkeypatch.setattr(coqui_backend, "torch_device", lambda: "cpu")
    assert CoquiBackend().generate("hi", "boy", 1.0) is None


def test_generate_unknown_gender_falls_back(monkeypatch):
    _install_tts(monkeypatch)
    monkeypatch.setattr(coqui_backend, "torch_device", lambda: "cpu")
    assert CoquiBackend().generate("hi", "alien", 1.0) is not None


def test_get_model_caches(monkeypatch):
    _install_tts(monkeypatch)
    monkeypatch.setattr(coqui_backend, "torch_device", lambda: "cpu")
    b = CoquiBackend()
    m1 = b._get_model("tts_models/en/jenny/jenny")
    m2 = b._get_model("tts_models/en/jenny/jenny")
    assert m1 is m2


def test_voices_returns_jenny():
    voices = CoquiBackend().voices()
    assert [v.id for v in voices] == ["jenny"]


def test_generate_hindi_language_logs_fallback(monkeypatch):
    _install_tts(monkeypatch)
    monkeypatch.setattr(coqui_backend, "torch_device", lambda: "cpu")
    # Coqui is English-only; passing language="hi" must still succeed
    # (falls back to the jenny English voice with a warning).
    out = CoquiBackend().generate("hi there", "girl", 1.0, language="hi")
    assert out is not None
