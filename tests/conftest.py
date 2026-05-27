"""
Shared test fixtures.

* `isolate_filesystem` — every test runs with ASSETS_DIR / LOG_DIR / MODEL_DIR
  redirected into pytest's `tmp_path`, so test artifacts never pollute the
  repo's `logs/`, `assets/`, or `models/` directories.
* `silent_logging` (session) — replace every project logger's handlers with
  a NullHandler so tests don't open rotating log files.
* `clean_registry` — clear the TTS backend registry between tests so each test
  starts from a known empty state.
"""

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                # pragma: no cover
    sys.path.insert(0, str(ROOT))


# ── Silence file-writing loggers for the whole session ──────────────────────

@pytest.fixture(autouse=True, scope="session")
def silent_logging():
    """Swap every project logger's handler chain for a NullHandler."""
    import app_logging

    null = logging.NullHandler()

    def _silence(logger: logging.Logger) -> None:
        for h in logger.handlers[:]:
            try:
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)
        logger.addHandler(null)

    # Loggers already created at import time
    for logger in list(app_logging._loggers.values()):
        _silence(logger)

    # Future get_logger() calls — wrap the factory to skip file handlers.
    original = app_logging.get_logger

    def _silent_get_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
            _silence(logger)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
        return logger

    app_logging.get_logger = _silent_get_logger  # type: ignore[assignment]
    yield
    app_logging.get_logger = original           # type: ignore[assignment]


# ── Per-test filesystem isolation ───────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolate_filesystem(tmp_path, monkeypatch):
    """Redirect ASSETS_DIR / LOG_DIR / MODEL_DIR into a per-test tmp dir."""
    import config

    assets = tmp_path / "assets"
    logs   = tmp_path / "logs"
    models = tmp_path / "models"
    applog = logs / "app"

    monkeypatch.setattr(config, "ASSETS_DIR",        assets)
    monkeypatch.setattr(config, "PREVIEW_CACHE_DIR", assets / "previews")
    monkeypatch.setattr(config, "LOG_DIR",           logs)
    monkeypatch.setattr(config, "MODEL_DIR",         models)
    monkeypatch.setattr(config, "APP_LOG_DIR",       applog)

    # Modules that did `from config import X` captured their own binding —
    # patch those too.
    import tts._utils as _utils
    monkeypatch.setattr(_utils, "ASSETS_DIR", assets)

    import tts.piper_backend as piper_backend
    monkeypatch.setattr(piper_backend, "MODEL_DIR", models)

    import session_archive
    monkeypatch.setattr(session_archive, "LOG_DIR", logs)

    import app_logging
    monkeypatch.setattr(app_logging, "APP_LOG_DIR", applog)

    yield


# ── Registry isolation ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_registry():
    """Start every test with an empty TTS backend registry."""
    from tts import registry
    registry.clear()
    yield
    registry.clear()


# ── Cache isolation ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_caches():
    """Clear OV-session + device-detection caches between tests."""
    from tts import devices, runtime
    runtime.reset_cache()
    devices.reset_cache()
    yield
    runtime.reset_cache()
    devices.reset_cache()
