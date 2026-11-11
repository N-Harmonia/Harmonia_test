from __future__ import annotations

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_CONFIGURED = False


def configure_logging(level: LogLevel = "INFO", fmt: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt or "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
