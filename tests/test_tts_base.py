"""Tests for tts/base.py."""

import pytest

from tts.base import TTSBackend


class _Concrete(TTSBackend):
    id = "concrete"

    @property
    def display_name(self) -> str:
        return "C"

    def voices(self):
        return []

    def generate(self, text, gender="both", speed=1.0, *,
                 language="en", voice_id=None, output_basename=None):
        return "/tmp/x.wav"


def test_subclass_with_id_works():
    b = _Concrete()
    assert b.id == "concrete"
    assert b.display_name == "C"
    assert b.generate("t", "boy", 1.0) == "/tmp/x.wav"
    assert b.is_available() is True
    # default ensure_models is a no-op
    assert b.ensure_models() is None


def test_resolve_voice_finds_explicit_id():
    from tts.voices import VoiceInfo

    class _WithVoices(TTSBackend):
        id = "v"
        @property
        def display_name(self): return "V"
        def voices(self):
            return [
                VoiceInfo(id="alpha", display_name="A", language="en", gender="boy"),
                VoiceInfo(id="beta",  display_name="B", language="en", gender="girl"),
            ]
        def generate(self, t, g="both", s=1.0, *, language="en",
                     voice_id=None, output_basename=None):
            return None

    b = _WithVoices()
    assert b.resolve_voice("en", "boy",  "beta").id == "beta"
    assert b.resolve_voice("en", "boy",  None).id  == "alpha"
    assert b.resolve_voice("en", "boy",  "missing").id == "alpha"   # falls back
    assert b.voices_for("en") and b.voices_for("hi") == []


def test_subclass_without_id_raises():
    with pytest.raises(TypeError, match="non-empty class attribute"):
        class _Broken(TTSBackend):
            @property
            def display_name(self):
                return "x"
            def voices(self):
                return []
            def generate(self, text, gender="both", speed=1.0, *,
                         language="en", voice_id=None, output_basename=None):
                return None


def test_intermediate_abstract_base_allowed():
    class _Inter(TTSBackend):
        __abstract_backend__ = True
        # No id required at this level.

    # Concrete subclass still must set id
    with pytest.raises(TypeError):
        class _Broken2(_Inter):
            @property
            def display_name(self):
                return "y"
            def voices(self):
                return []
            def generate(self, text, gender="both", speed=1.0, *,
                         language="en", voice_id=None, output_basename=None):
                return None
