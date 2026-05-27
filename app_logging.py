"""
Application logging — daily rotating file + size cap.

Separate from `session_archive.py` (which persists user-facing artifacts
like topics and generated audio). This module owns stdlib `logging`
configuration only.

Usage:
    from app_logging import get_logger
    log = get_logger(__name__)
    log.info("story generated for topic=%s", topic)

Log files land in logs/app/kidsstory.log (active) and
logs/app/kidsstory.log.YYYY-MM-DD (rotated).
"""

import logging
from logging.handlers import TimedRotatingFileHandler
from config import APP_LOG_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_loggers: dict[str, logging.Logger] = {}


class _SizedDailyRotatingHandler(TimedRotatingFileHandler):
    """Rotate at midnight OR when the file exceeds max_bytes, whichever comes first."""

    def __init__(self, filename: str, max_bytes: int, backup_count: int) -> None:
        super().__init__(
            filename,
            when="midnight",
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        self.max_bytes = max_bytes

    def shouldRollover(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        if super().shouldRollover(record):
            return True
        if self.max_bytes > 0:
            if self.stream is None:
                self.stream = self._open()
            self.stream.seek(0, 2)
            if self.stream.tell() >= self.max_bytes:
                return True
        return False


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with a daily-rotating file handler and console output.
    Safe to call multiple times with the same name — returns the cached instance.
    """
    if name in _loggers:
        return _loggers[name]

    APP_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        fh = _SizedDailyRotatingHandler(
            str(APP_LOG_DIR / "kidsstory.log"),
            max_bytes=LOG_MAX_BYTES,
            backup_count=LOG_BACKUP_COUNT,
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)-8s | %(name)s | %(message)s"))
        logger.addHandler(ch)

    _loggers[name] = logger
    return logger
