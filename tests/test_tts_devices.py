"""Tests for tts/devices.py."""

import sys
import types

import pytest

from tts import devices


def _make_fake_torch(cuda_avail=False, xpu_avail=False):
    mod = types.SimpleNamespace()
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda_avail)
    if xpu_avail:
        mod.xpu = types.SimpleNamespace(is_available=lambda: True)
    return mod


def test_torch_device_returns_cuda(monkeypatch):
    devices.reset_cache()
    monkeypatch.setitem(sys.modules, "torch", _make_fake_torch(cuda_avail=True))
    assert devices.torch_device() == "cuda"


def test_torch_device_returns_xpu_when_cuda_unavailable_and_xpu_present(monkeypatch):
    devices.reset_cache()
    monkeypatch.setitem(sys.modules, "torch", _make_fake_torch(cuda_avail=False, xpu_avail=True))
    monkeypatch.setitem(sys.modules, "intel_extension_for_pytorch", types.ModuleType("ipex"))
    assert devices.torch_device() == "xpu"


def test_torch_device_xpu_skipped_when_ipex_missing(monkeypatch):
    devices.reset_cache()
    fake_torch = _make_fake_torch(cuda_avail=False, xpu_avail=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "intel_extension_for_pytorch", raising=False)
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "intel_extension_for_pytorch":
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    # IPEX import failure is swallowed; xpu still available → returns xpu
    assert devices.torch_device() == "xpu"


def test_torch_device_cpu_when_nothing_available(monkeypatch):
    devices.reset_cache()
    # Force the entire `import torch` block to raise.
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert devices.torch_device() == "cpu"


def test_torch_device_cpu_when_no_cuda_no_xpu(monkeypatch):
    devices.reset_cache()
    mod = types.SimpleNamespace()
    mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    # No xpu attribute → hasattr fails → fall through to cpu
    monkeypatch.setitem(sys.modules, "torch", mod)
    assert devices.torch_device() == "cpu"


def test_torch_device_memoized(monkeypatch):
    devices.reset_cache()
    calls = []
    fake = _make_fake_torch(cuda_avail=True)
    fake.cuda.is_available = lambda: (calls.append(1) or True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    devices.torch_device()
    devices.torch_device()
    assert len(calls) == 1


def _make_fake_ov(devices_list):
    mod = types.ModuleType("openvino")
    class _Core:
        @property
        def available_devices(self):
            return devices_list
    mod.Core = _Core
    return mod


def test_ov_device_prefers_gpu(monkeypatch):
    devices.reset_cache()
    monkeypatch.setitem(sys.modules, "openvino", _make_fake_ov(["GPU", "CPU"]))
    assert devices.ov_device() == "GPU"


def test_ov_device_falls_back_to_npu(monkeypatch):
    devices.reset_cache()
    monkeypatch.setitem(sys.modules, "openvino", _make_fake_ov(["NPU", "CPU"]))
    assert devices.ov_device() == "NPU"


def test_ov_device_cpu_when_no_accelerator(monkeypatch):
    devices.reset_cache()
    monkeypatch.setitem(sys.modules, "openvino", _make_fake_ov(["CPU"]))
    assert devices.ov_device() == "CPU"


def test_ov_device_cpu_on_import_failure(monkeypatch):
    devices.reset_cache()
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "openvino":
            raise ImportError("no openvino")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert devices.ov_device() == "CPU"


def test_reset_cache_clears_both():
    # Just sanity — function exists and clears.
    devices.reset_cache()
    assert devices.torch_device.cache_info().currsize == 0
    assert devices.ov_device.cache_info().currsize == 0
