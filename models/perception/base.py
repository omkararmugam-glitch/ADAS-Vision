"""The :class:`Detector` interface every perception backbone implements.

One required method, :meth:`Detector.infer`, so the stub, the YOLOv8 baseline and
the YOLOPv2 multi-task network are interchangeable everywhere in the pipeline.
Everything a detector returns is in original image pixels -- the letterbox frame
never escapes the model wrapper.

Failures are loud: missing weights raise, an unsupported device raises, and a CPU
fallback is reported at WARNING rather than being discovered later from the frame
rate.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from core.logging_utils import get_logger, log_kv
from core.types import CLASS_IDS, FrameResult

_log = get_logger(__name__)


class Detector(ABC):
    """A per-frame perception model.

    The single required method is :meth:`infer`, so the stub and the real network
    are interchangeable everywhere in the pipeline. Everything returned is in
    original image pixels.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier recorded in the cache manifest."""

    @property
    @abstractmethod
    def weights_hash(self) -> str:
        """Digest identifying the exact weights, for cache invalidation."""

    @property
    def class_names(self) -> tuple[str, ...]:
        """Classes this detector is capable of emitting.

        Recorded in the manifest so a cached run is never ambiguous about what it
        was *able* to see, as opposed to what it happened to find.
        """
        return tuple(CLASS_IDS)

    @abstractmethod
    def infer(self, frame: np.ndarray) -> FrameResult:
        """Run the model on one BGR frame and return results in original pixels."""

    def infer_batch(self, frames: list[np.ndarray]) -> list[FrameResult]:
        """Run on several frames. The default loops; subclasses may batch."""
        return [self.infer(f) for f in frames]

    def close(self) -> None:
        """Release any GPU resources."""


def resolve_device(requested: str) -> torch.device:
    """Pick a torch device, warning loudly when CUDA was wanted but is absent.

    A silent CPU fallback is the difference between a 20-minute run and an
    overnight one, so it is never silent.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log_kv(
            _log,
            logging.WARNING,
            "detect.cpu_fallback",
            requested="auto",
            reason="torch.cuda.is_available() is False",
            note="inference will be roughly an order of magnitude slower",
        )
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device '{requested}' was requested explicitly but CUDA is not "
            f"available. Set detector.device to 'cpu' or 'auto' to proceed."
        )
    if device.type == "cpu":
        log_kv(
            _log, logging.WARNING, "detect.cpu_selected",
            note="CPU inference was requested explicitly; this will be slow",
        )
    return device


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, so 150 MB of weights is not read into RAM."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
