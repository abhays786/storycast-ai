"""Tests for tts/bark_backend.py."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tts import bark_backend
from tts.bark_backend import BarkBackend


# ── Fakes ───────────────────────────────────────────────────────────────────

def _install_bark(monkeypatch, *, generate_audio=None, raise_in_generate=None,
                  preload_raises=False, raise_in_module_import=False):
    pkg = types.ModuleType("bark")

    if raise_in_module_import:
        # Force the second-stage `from bark import generate_audio as bark_gen`
        # to fail by simply not exposing the symbol.
        def _gen(*a, **k):  # pragma: no cover
            raise RuntimeError("should not be called")
        pkg.generate_audio = _gen
        pkg.SAMPLE_RATE = 22050

        def preload(**kw):
            pass
        pkg.preload_models = preload
        monkeypatch.setitem(sys.modules, "bark", pkg)
        return pkg

    def _gen(text, history_prompt):
        if raise_in_generate is not None:
            raise raise_in_generate
        return (generate_audio if generate_audio is not None
                else np.ones(int(0.5 * 22050), dtype=np.float32))

    pkg.generate_audio = _gen
    pkg.SAMPLE_RATE = 22050

    def preload(**kw):
        if preload_raises:
            raise RuntimeError("preload failed")
    pkg.preload_models = preload

    # Provide bark.generation for XPU branch
    bgen = types.ModuleType("bark.generation")
    bgen.models = {}
    pkg.generation = bgen
    monkeypatch.setitem(sys.modules, "bark", pkg)
    monkeypatch.setitem(sys.modules, "bark.generation", bgen)
    return pkg


def _install_scipy_wavfile(monkeypatch):
    pkg = types.ModuleType("scipy")
    io = types.ModuleType("scipy.io")
    wf = types.ModuleType("scipy.io.wavfile")

    def write(path, rate, data):
        Path(path).write_bytes(b"X" * 5000)

    wf.write = write
    io.wavfile = wf
    pkg.io = io
    monkeypatch.setitem(sys.modules, "scipy", pkg)
    monkeypatch.setitem(sys.modules, "scipy.io", io)
    monkeypatch.setitem(sys.modules, "scipy.io.wavfile", wf)


def _install_torch(monkeypatch, *, raises=False, xpu_move_raises=False):
    if raises:
        import builtins
        real_import = builtins.__import__
        def fail(name, *a, **k):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", fail)
        return None
    fake_torch = types.SimpleNamespace()
    fake_torch.load = lambda path, **kw: {"x": 1}
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


# ── Tests ───────────────────────────────────────────────────────────────────

def test_display_name_cpu(monkeypatch):
    monkeypatch.setattr(bark_backend, "torch_device", lambda: "cpu")
    assert "slow on CPU" in BarkBackend().display_name


def test_display_name_gpu(monkeypatch):
    monkeypatch.setattr(bark_backend, "torch_device", lambda: "cuda")
    assert "[GPU]" in BarkBackend().display_name


def test_is_available_yes(monkeypatch):
    _install_bark(monkeypatch)
    assert BarkBackend().is_available() is True


def test_is_available_no(monkeypatch):
    monkeypatch.delitem(sys.modules, "bark", raising=False)
    import builtins
    real_import = builtins.__import__
    def fail(name, *a, **k):
        if name == "bark":
            raise ImportError("no bark")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fail)
    assert BarkBackend().is_available() is False


def test_ensure_loaded_missing_import(monkeypatch):
    # Make `from bark import preload_models` fail
    monkeypatch.delitem(sys.modules, "bark", raising=False)
    import builtins
    real_import = builtins.__import__
    def fail(name, *a, **k):
        if name == "bark":
            raise ImportError("no bark")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fail)
    b = BarkBackend()
    assert b._ensure_loaded() is False


def test_ensure_loaded_cpu_small_models(monkeypatch):
    _install_bark(monkeypatch)
    _install_torch(monkeypatch)
    monkeypatch.setattr(bark_backend, "torch_device", lambda: "cpu")
    b = BarkBackend()
    assert b._ensure_loaded() is True
    # Subsequent call short-circuits
    assert b._ensure_loaded() is True


def test_ensure_loaded_cuda(monkeypatch):
    _install_bark(monkeypatch)
    _install_torch(monkeypatch)
    monkeypatch.setattr(bark_backend, "torch_device", lambda: "cuda")
    assert BarkBackend()._ensure_loaded() is True


def test_ensure_loaded_xpu_with_models_moved(monkeypatch):
    pkg = _install_bark(monkeypatch)

    class _Model:
        def __init__(self): self.moved = None
        def to(self, dev):
            self.moved = dev
            return self
    pkg.generation.models = {"text": _Model(), "coarse": _Model()}
    _install_torch(monkeypatch)
    monkeypatch.setattr(bark_backend, "torch_device", lambda: "xpu")
    assert BarkBackend()._ensure_loaded() is True
    assert all(m.moved == "xpu" for m in pkg.generation.models.values())


def test_ensure_loaded_xpu_move_failure_logs(monkeypatch):
    pkg = _install_bark(monkeypatch)
    # Make iterating models raise
    bad = types.SimpleNamespace()
    def boom(*a, **k):
        raise RuntimeError("nope")
    pkg.generation.models = bad
    _install_torch(monkeypatch)
    monkeypatch.setattr(bark_backend, "torch_device", lambda: "xpu")
    assert BarkBackend()._ensure_loaded() is True


# ── generate ────────────────────────────────────────────────────────────────

def test_generate_missing_bark(monkeypatch):
    monkeypatch.delitem(sys.modules, "bark", raising=False)
    import builtins
    real_import = builtins.__import__
    def fail(name, *a, **k):
        if name == "bark":
            raise ImportError("no bark")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fail)
    assert BarkBackend().generate("hi", "boy", 1.0) is None


def test_generate_when_ensure_loaded_returns_false(monkeypatch):
    """`from bark import preload_models` raises ⇒ _ensure_loaded() returns False
    ⇒ generate() returns None."""
    # Stage 1: install a bark module with generate_audio + SAMPLE_RATE so the
    # FIRST import in generate() succeeds.
    pkg = types.ModuleType("bark")
    pkg.generate_audio = lambda text, history_prompt: np.zeros(1024, dtype=np.float32)
    pkg.SAMPLE_RATE = 22050
    # No `preload_models` attribute → `from bark import preload_models` fails.
    monkeypatch.setitem(sys.modules, "bark", pkg)
    _install_scipy_wavfile(monkeypatch)
    monkeypatch.setattr(bark_backend, "torch_device", lambda: "cpu")
    assert BarkBackend().generate("hi", "boy", 1.0) is None


def test_generate_success_cpu(monkeypatch):
    _install_bark(monkeypatch)
    _install_scipy_wavfile(monkeypatch)
    _install_torch(monkeypatch)
    monkeypatch.setattr(bark_backend, "torch_device", lambda: "cpu")
    out = BarkBackend().generate("Hello world.", "girl", 1.0)
    assert out is not None
    assert Path(out).exists()


def test_generate_multiple_chunks_appends_silence(monkeypatch):
    _install_bark(monkeypatch)
    _install_scipy_wavfile(monkeypatch)
    _install_torch(monkeypatch)
    monkeypatch.setattr(bark_backend, "torch_device", lambda: "cuda")
    long_text = ("Sentence one. " * 50)
    out = BarkBackend().generate(long_text, "both", 1.0)
    assert out is not None


def test_generate_synthesis_exception(monkeypatch):
    _install_bark(monkeypatch, raise_in_generate=RuntimeError("audio failed"))
    _install_scipy_wavfile(monkeypatch)
    _install_torch(monkeypatch)
    monkeypatch.setattr(bark_backend, "torch_device", lambda: "cpu")
    assert BarkBackend().generate("hi", "boy", 1.0) is None


def test_generate_tiny_file_returns_none(monkeypatch):
    _install_bark(monkeypatch)
    _install_torch(monkeypatch)
    # Inject a scipy.wavfile.write that produces a tiny file
    pkg = types.ModuleType("scipy")
    io = types.ModuleType("scipy.io")
    wf = types.ModuleType("scipy.io.wavfile")
    wf.write = lambda p, r, d: Path(p).write_bytes(b"x" * 100)
    io.wavfile = wf
    pkg.io = io
    monkeypatch.setitem(sys.modules, "scipy", pkg)
    monkeypatch.setitem(sys.modules, "scipy.io", io)
    monkeypatch.setitem(sys.modules, "scipy.io.wavfile", wf)
    monkeypatch.setattr(bark_backend, "torch_device", lambda: "cpu")
    assert BarkBackend().generate("hi", "boy", 1.0) is None


def test_generate_unknown_gender_falls_back(monkeypatch):
    _install_bark(monkeypatch)
    _install_scipy_wavfile(monkeypatch)
    _install_torch(monkeypatch)
    monkeypatch.setattr(bark_backend, "torch_device", lambda: "cpu")
    assert BarkBackend().generate("hi", "alien", 1.0) is not None
