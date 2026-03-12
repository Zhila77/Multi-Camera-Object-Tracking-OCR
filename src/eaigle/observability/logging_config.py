"""
Structured logging configuration using structlog.
Produces JSON-formatted logs in production; pretty-printed in development.
"""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """
    Configure standard library logging.
    structlog is used for structured output; this function configures the
    stdlib root logger as the backend.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if json_logs:
        import json

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_obj = {
                    "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    log_obj["exc"] = self.formatException(record.exc_info)
                return json.dumps(log_obj)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
    else:
        fmt = "%(asctime)s %(levelname)-8s %(name)-30s %(message)s"
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "asyncio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
