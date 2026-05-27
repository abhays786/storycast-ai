"""
TTS backend registry — lookup, listing, and fallback resolution.

The registry has no implicit population. Callers must explicitly register
backends (see `tts/bootstrap.py`). This avoids the "import-side-effect"
ordering hazard.
"""

from typing import Iterable

from tts.base import TTSBackend

# Engine id used as the default when an unknown / failing backend is requested.
DEFAULT_FALLBACK_ID = "piper"

_backends: dict[str, TTSBackend] = {}
_order: list[str] = []   # preserves registration order for UI listing


def register(backend: TTSBackend) -> None:
    """Add (or replace) a backend in the registry."""
    if not backend.id:
        raise ValueError(f"Backend {backend.__class__.__name__} has no id")
    if backend.id not in _backends:
        _order.append(backend.id)
    _backends[backend.id] = backend


def clear() -> None:
    """Remove every registered backend (used by tests)."""
    _backends.clear()
    _order.clear()


def get(engine_id: str) -> TTSBackend | None:
    """Return the backend for the given id, or None if unknown."""
    return _backends.get(engine_id)


def get_fallback() -> TTSBackend | None:
    """Return the configured fallback backend, or None if not registered."""
    return _backends.get(DEFAULT_FALLBACK_ID)


def all_backends() -> list[TTSBackend]:
    """All registered backends in registration order."""
    return [_backends[i] for i in _order]


def ui_choices(enabled_ids: Iterable[str] | None = None) -> list[tuple[str, str]]:
    """
    (display_name, id) tuples for the UI dropdown.

    Pass `enabled_ids` to restrict the list (e.g., from settings); pass None
    to include every registered, currently-available backend.
    """
    if enabled_ids is None:
        ids: set[str] | None = None
    else:
        ids = set(enabled_ids)
    out: list[tuple[str, str]] = []
    for b in all_backends():
        if ids is not None and b.id not in ids:
            continue
        if not b.is_available():
            continue
        out.append((b.display_name, b.id))
    return out
