"""Tests for tts/registry.py."""

import pytest

from tts import registry
from tts.base import TTSBackend


class _Stub(TTSBackend):
    id = "stub"

    def __init__(self, *, available=True, name="Stub"):
        self._available = available
        self._name = name

    @property
    def display_name(self):
        return self._name

    def voices(self):
        return []

    def generate(self, text, gender="both", speed=1.0, *,
                 language="en", voice_id=None, output_basename=None):
        return None

    def is_available(self):
        return self._available


class _OtherStub(_Stub):
    id = "other"


def test_register_and_get():
    s = _Stub()
    registry.register(s)
    assert registry.get("stub") is s
    assert registry.get("missing") is None


def test_register_replaces_existing():
    a = _Stub(name="A")
    b = _Stub(name="B")
    registry.register(a)
    registry.register(b)
    assert registry.get("stub") is b
    # Order preserved — only one entry
    assert len(registry.all_backends()) == 1


def test_register_empty_id_raises():
    class _EmptyId(TTSBackend):
        __abstract_backend__ = True
        id = ""

        @property
        def display_name(self):
            return "x"
        def voices(self):
            return []
        def generate(self, text, gender="both", speed=1.0, *,
                     language="en", voice_id=None, output_basename=None):
            return None
    with pytest.raises(ValueError):
        registry.register(_EmptyId())


def test_clear_empties_the_registry():
    registry.register(_Stub())
    assert registry.all_backends() != []
    registry.clear()
    assert registry.all_backends() == []
    assert registry.get("stub") is None


def test_get_fallback_uses_default_id():
    class _Piper(_Stub):
        id = "piper"
    p = _Piper()
    registry.register(p)
    assert registry.get_fallback() is p


def test_get_fallback_none_when_not_registered():
    assert registry.get_fallback() is None


def test_all_backends_preserves_order():
    a = _Stub(name="A")
    b = _OtherStub(name="B")
    registry.register(a)
    registry.register(b)
    assert [x.display_name for x in registry.all_backends()] == ["A", "B"]


def test_ui_choices_no_filter_returns_available():
    a = _Stub(name="A", available=True)
    b = _OtherStub(name="B", available=False)
    registry.register(a)
    registry.register(b)
    out = registry.ui_choices()
    assert out == [("A", "stub")]


def test_ui_choices_with_filter():
    a = _Stub(name="A")
    b = _OtherStub(name="B")
    registry.register(a)
    registry.register(b)
    assert registry.ui_choices(enabled_ids=["other"]) == [("B", "other")]
    assert registry.ui_choices(enabled_ids=[]) == []


def test_ui_choices_skips_unavailable():
    a = _Stub(name="A", available=True)
    b = _OtherStub(name="B", available=False)
    registry.register(a)
    registry.register(b)
    assert registry.ui_choices(enabled_ids=["stub", "other"]) == [("A", "stub")]


def test_default_fallback_id():
    assert registry.DEFAULT_FALLBACK_ID == "piper"
