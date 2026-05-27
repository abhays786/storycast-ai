"""Tests for tts/bootstrap.py."""

from tts import bootstrap, registry


def test_register_default_backends_populates_registry():
    bootstrap.register_default_backends()
    ids = [b.id for b in registry.all_backends()]
    assert ids == ["piper", "coqui", "bark", "gemini"]
