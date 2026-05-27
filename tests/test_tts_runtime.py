"""Tests for tts/runtime.py."""

import sys
import types

import pytest

from tts import runtime


class _CompiledModel:
    def __init__(self, outputs=None, result=None):
        self.outputs = outputs or ["out"]
        self._result = result or {"out": "fake-output"}
        self.calls = []

    def __call__(self, feed):
        self.calls.append(feed)
        return self._result


def test_openvino_session_run_returns_outputs_in_order():
    cm = _CompiledModel(outputs=["a", "b"], result={"a": 1, "b": 2})
    sess = runtime.OpenVINOSession(cm)
    res = sess.run(None, {"input": 0})
    assert res == [1, 2]
    assert cm.calls == [{"input": 0}]


def _install_fake_openvino(monkeypatch, compiled_model):
    mod = types.ModuleType("openvino")
    class _Core:
        def read_model(self, path):
            return f"model({path})"
        def compile_model(self, model, device):
            assert isinstance(model, str)
            return compiled_model
    mod.Core = _Core
    monkeypatch.setitem(sys.modules, "openvino", mod)


def test_get_ov_session_compiles_and_caches(monkeypatch):
    runtime.reset_cache()
    cm = _CompiledModel()
    _install_fake_openvino(monkeypatch, cm)
    s1 = runtime.get_ov_session("/path/to/model.onnx")
    s2 = runtime.get_ov_session("/path/to/model.onnx")
    assert s1 is not None
    assert s1 is s2


def test_get_ov_session_runs_dummy_feed(monkeypatch):
    runtime.reset_cache()
    cm = _CompiledModel()
    _install_fake_openvino(monkeypatch, cm)
    runtime.get_ov_session("/m.onnx", dummy_feed={"x": 1})
    assert cm.calls == [{"x": 1}]


def test_get_ov_session_returns_none_on_failure(monkeypatch):
    runtime.reset_cache()
    mod = types.ModuleType("openvino")
    class _BrokenCore:
        def read_model(self, path):
            raise RuntimeError("boom")
    mod.Core = _BrokenCore
    monkeypatch.setitem(sys.modules, "openvino", mod)
    assert runtime.get_ov_session("/x.onnx") is None
