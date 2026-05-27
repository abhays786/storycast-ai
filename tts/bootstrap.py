"""
Explicit backend registration — no import side-effects.

`app.py` calls `register_default_backends()` once at startup. Tests can
call `tts.registry.clear()` and re-bootstrap a subset to control which
backends exist in their world.
"""

from tts.bark_backend import BarkBackend
from tts.coqui_backend import CoquiBackend
from tts.gemini_backend import GeminiBackend
from tts.piper_backend import PiperBackend
from tts.registry import register


def register_default_backends() -> None:
    """Register every shipped backend in the canonical order."""
    for backend in (PiperBackend(), CoquiBackend(), BarkBackend(), GeminiBackend()):
        register(backend)
