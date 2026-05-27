"""Tests for app_logging.py."""

import importlib
import logging

import pytest


@pytest.fixture
def fresh_app_logging(monkeypatch, tmp_path):
    """Re-import app_logging with patched APP_LOG_DIR and a cleared logger cache."""
    import app_logging
    # Restore the original (un-silenced) factory + handler for this test
    original_get_logger = app_logging._loggers.copy()
    app_logging._loggers.clear()

    # Pull the original (pre-silencing) functions back into scope
    real_get_logger = importlib.reload(app_logging).get_logger
    monkeypatch.setattr(app_logging, "APP_LOG_DIR", tmp_path / "logs" / "app")
    yield app_logging, real_get_logger
    # Restore cache so silent_logging session fixture's invariants hold
    app_logging._loggers.clear()
    app_logging._loggers.update(original_get_logger)


def test_get_logger_returns_cached_instance(fresh_app_logging, tmp_path):
    mod, get_logger = fresh_app_logging
    log1 = get_logger("test_a")
    log2 = get_logger("test_a")
    assert log1 is log2


def test_get_logger_creates_handlers(fresh_app_logging, tmp_path):
    mod, get_logger = fresh_app_logging
    log = get_logger("test_b")
    handler_types = {type(h).__name__ for h in log.handlers}
    assert "_SizedDailyRotatingHandler" in handler_types
    assert "StreamHandler" in handler_types
    assert (tmp_path / "logs" / "app").exists()


def test_sized_rotating_handler_size_rollover(fresh_app_logging, tmp_path):
    mod, _ = fresh_app_logging
    h = mod._SizedDailyRotatingHandler(
        str(tmp_path / "out.log"), max_bytes=10, backup_count=1
    )
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "hi", None, None)
    # First call opens the stream (delay=True). With max_bytes=10 and empty
    # stream, shouldRollover returns False initially.
    assert h.shouldRollover(rec) is False
    # Write more than max_bytes
    h.stream.write("x" * 50)
    h.stream.flush()
    assert h.shouldRollover(rec) is True


def test_sized_rotating_handler_zero_max_bytes(fresh_app_logging, tmp_path):
    mod, _ = fresh_app_logging
    h = mod._SizedDailyRotatingHandler(
        str(tmp_path / "out.log"), max_bytes=0, backup_count=1
    )
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "hi", None, None)
    # max_bytes=0 disables the size check; should fall back to time-based only.
    assert h.shouldRollover(rec) is False


def test_sized_rotating_handler_time_rollover(fresh_app_logging, tmp_path, monkeypatch):
    mod, _ = fresh_app_logging
    h = mod._SizedDailyRotatingHandler(
        str(tmp_path / "out.log"), max_bytes=1024, backup_count=1
    )
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "hi", None, None)
    # Force the time-based check to claim rollover is needed.
    monkeypatch.setattr(
        mod.TimedRotatingFileHandler, "shouldRollover", lambda self, r: True
    )
    assert h.shouldRollover(rec) is True
