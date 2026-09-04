"""MiDaS / DPT monocular depth, wrapped behind a small interface.

Used only where *relative* distance matters -- specifically, to supply a distance
for objects the homography cannot handle: an occluded ground-contact point, a
vehicle on a crest or in a dip, the back of a truck whose wheels are hidden.

**Accuracy caveat, stated once and meant.** MiDaS predicts relative inverse depth
with no metric scale. Everything metric in this module comes from fitting that
output to the calibrated ground plane (see :mod:`depth.align`), so MiDaS is never
the source of scale -- only of *shape*. Where the homography is available and the
object stands on flat road, the homography is the better estimate and is
preferred. MiDaS earns its place on the cases the homography cannot see at all.

The model is optional. Nothing in the pipeline requires it, the import is lazy,
and a missing model degrades to homography-only distances rather than failing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from core.config import DepthConfig
from core.logging_utils import get_logger, log_kv

_log = get_logger(__name__)


class DepthEstimator(ABC):
    """Predicts a relative depth map for one frame."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier recorded in the cache manifest."""

    @abstractmethod
    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Relative inverse depth, ``(H, W)`` float32, larger meaning nearer.

        The scale is arbitrary and differs between frames, which is why
        :mod:`depth.align` exists.
        """

    def infer_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return [self.infer(f) for f in frames]

    def close(self) -> None:
        """Release GPU resources."""


class MiDaSDepth(DepthEstimator):
    """MiDaS via torch.hub.

    ``MiDaS_small`` by default. On an 8 GB laptop GPU already hosting the
    detector and the tracker, the large DPT variants are the difference between
    a pipeline that runs and one that does not, and depth here feeds a *fallback*
    distance rather than the primary one -- so the small model is the right
    trade, not a compromise forced by hardware.
    """

    def __init__(self, cfg: DepthConfig) -> None:
        import torch

        self.cfg = cfg
        self._torch = torch
        self.device = self._resolve_device(cfg.device)
        log_kv(_log, logging.INFO, "depth.loading", model=cfg.model, device=str(self.device))

        self.model = torch.hub.load("intel-isl/MiDaS", cfg.model, trust_repo=True)
        self.model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = (
            transforms.small_transform if "small" in cfg.model.lower()
            else transforms.dpt_transform
        )
        self.half = bool(cfg.half and self.device.type == "cuda")
        if self.half:
            self.model.half()
        log_kv(_log, logging.INFO, "depth.loaded", model=cfg.model,
               device=str(self.device), half=self.half)

    def _resolve_device(self, requested: str):
        torch = self._torch
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            log_kv(_log, logging.WARNING, "depth.cpu_fallback",
                   note="MiDaS on CPU is roughly 20x slower")
            return torch.device("cpu")
        return torch.device(requested)

    @property
    def name(self) -> str:
        return f"midas:{self.cfg.model}"

    def infer(self, frame: np.ndarray) -> np.ndarray:
        torch = self._torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch = self.transform(rgb).to(self.device)
        if self.half:
            batch = batch.half()
        with torch.no_grad():
            pred = self.model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=frame.shape[:2],
                mode="bicubic", align_corners=False,
            ).squeeze()
        return pred.float().cpu().numpy().astype(np.float32)

    def close(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class PlanarStubDepth(DepthEstimator):
    """A deterministic stand-in that needs no weights and no network.

    Emits the inverse depth an ideal pinhole camera would see over a flat road,
    so a pipeline run and every alignment test can exercise the depth path
    without downloading a model. It is not a depth estimator -- it encodes the
    flat-ground assumption exactly, which makes it useful for testing the
    plumbing and useless for testing whether MiDaS helps.
    """

    def __init__(self, horizon_frac: float = 0.5, scale: float = 1.0) -> None:
        self.horizon_frac = horizon_frac
        self.scale = scale

    @property
    def name(self) -> str:
        return "depth-stub"

    def infer(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        rows = np.arange(h, dtype=np.float32)
        horizon = self.horizon_frac * h
        # Below the horizon, distance falls as 1/(y - horizon); inverse depth is
        # therefore linear in the row index, which is exactly the pinhole result.
        inv = np.clip((rows - horizon) / max(1.0, h - horizon), 0.0, None) * self.scale
        return np.repeat(inv[:, None], w, axis=1).astype(np.float32)


def build_depth_estimator(cfg: DepthConfig) -> DepthEstimator | None:
    """Construct the configured estimator, or None when depth is disabled.

    A failure to load is logged and downgraded to None rather than raised: depth
    is a fallback signal, and losing it must not take the collision pipeline with
    it.
    """
    if not cfg.enabled:
        return None
    if cfg.kind == "stub":
        return PlanarStubDepth()
    if cfg.kind == "midas":
        try:
            return MiDaSDepth(cfg)
        except Exception as exc:                      # noqa: BLE001 - deliberate
            log_kv(_log, logging.ERROR, "depth.load_failed",
                   model=cfg.model, error=str(exc)[:200],
                   note="continuing without depth; distances fall back to the homography")
            return None
    raise ValueError(f"unknown depth.kind '{cfg.kind}'")
