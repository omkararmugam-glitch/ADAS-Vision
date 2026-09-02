"""Structured logging and stage timing.

Every pipeline stage logs machine-parseable key=value records so that runs can be
diffed and timed without re-reading source. No coordinate frames are involved here.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from types import TracebackType

_CONFIGURED = False


class _KeyValueFormatter(logging.Formatter):
    """Render records as ``ts level logger msg key=value ...``."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name} {record.getMessage()}"
        extras = getattr(record, "kv", None)
        if extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: int = logging.INFO) -> None:
    """Install the key=value handler on the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_KeyValueFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring the root handler on first use."""
    configure_logging()
    return logging.getLogger(name)


def log_kv(logger: logging.Logger, level: int, msg: str, **kv: object) -> None:
    """Emit ``msg`` with structured ``key=value`` fields appended."""
    logger.log(level, msg, extra={"kv": kv})


@dataclass
class StageTimer:
    """Accumulate wall-clock time and item counts for one pipeline stage.

    Use as a context manager; on exit it logs ``items``, ``seconds`` and ``fps``.
    """

    stage: str
    logger: logging.Logger
    items: int = 0
    _t0: float = field(default=0.0, init=False)
    elapsed_s: float = field(default=0.0, init=False)

    def __enter__(self) -> "StageTimer":
        self._t0 = time.perf_counter()
        log_kv(self.logger, logging.INFO, "stage.start", stage=self.stage)
        return self

    def tick(self, n: int = 1) -> None:
        """Record that ``n`` more items (usually frames) were processed."""
        self.items += n

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_s = time.perf_counter() - self._t0
        rate = self.items / self.elapsed_s if self.elapsed_s > 0 else 0.0
        log_kv(
            self.logger,
            logging.ERROR if exc_type else logging.INFO,
            "stage.end",
            stage=self.stage,
            items=self.items,
            seconds=f"{self.elapsed_s:.3f}",
            items_per_s=f"{rate:.2f}",
            status="error" if exc_type else "ok",
        )
