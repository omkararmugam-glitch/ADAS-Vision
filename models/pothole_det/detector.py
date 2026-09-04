"""Pothole and road-damage detection.

``ARCHITECTURE.md`` left this as an open question: a YOLOv8 detection class, or a
dedicated binary classifier over road patches? Both are implemented here behind
one interface so the choice is settled by measurement on the supplied dataset
rather than in advance -- which is the answer the brief asked for.

The trade-off, so the numbers can be read properly when they arrive:

**Detection** localises the damage, which the fusion engine wants: a pothole in
the ego corridor matters, one on the far verge does not. It needs box
annotations, and potholes have famously ill-defined boundaries -- annotator
disagreement on the *extent* of a patch of broken tarmac is large, which caps
achievable IoU and therefore mAP.

**Patch classification** needs only image-level labels, which are far cheaper and
far more consistent, and it sidesteps the boundary problem entirely. But it gives
no position, so the corridor test cannot be applied and every hazard reads as
equally relevant.

**Precision is weighted above recall either way**, and the config reflects that
with a deliberately high default threshold. A pothole warning that fires on every
tar seam, manhole and shadow gets ignored, and an ignored warning is worse than
no warning -- it also trains the driver to dismiss the alerts that matter.

Both paths are unified into ``RoadHazard`` records so the fusion engine does not
care which produced them; the classifier reports the patch it fired on as the
box, with ``localised=False`` recorded so a downstream reader is never misled
into treating a grid cell as a measured position.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import PotholeConfig
from core.logging_utils import get_logger, log_kv

_log = get_logger(__name__)


@dataclass(frozen=True)
class RoadHazard:
    """One detected patch of road damage.

    Attributes:
        box_px: ``(x0, y0, x1, y1)``. For the classifier this is the patch that
            fired, not a fitted boundary.
        score: Confidence.
        localised: True when the box is a real detection. False when it is a
            grid cell, so a reader cannot mistake it for a measured position.
        distance_m: Ground-plane distance, when calibration is available.
    """

    box_px: np.ndarray
    score: float
    localised: bool = True
    distance_m: float | None = None


class HazardDetector(ABC):
    """Finds road damage in one frame."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier recorded in the cache manifest."""

    @abstractmethod
    def infer(self, frame: np.ndarray) -> list[RoadHazard]:
        """Hazards in one BGR frame."""

    def close(self) -> None:
        """Release GPU resources."""


class PotholeYOLO(HazardDetector):
    """A YOLOv8 detector trained on pothole boxes.

    Single-class on purpose. Splitting "pothole" from "crack" from "patch" fails
    the same way every fine-grained damage taxonomy does: annotators disagree on
    the boundaries between them far more than on damage-versus-not, so the extra
    classes add label noise without adding anything the fusion engine can act on
    differently.
    """

    def __init__(self, cfg: PotholeConfig) -> None:
        from ultralytics import YOLO

        self.cfg = cfg
        weights = Path(cfg.weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"pothole weights not found: {weights}. Train with "
                f"`python models/pothole_det/train.py --data <dataset.yaml>`, "
                f"or set pothole.enabled=false."
            )
        self.model = YOLO(str(weights))
        log_kv(_log, logging.INFO, "pothole.loaded",
               weights=str(weights), conf=cfg.conf_threshold)

    @property
    def name(self) -> str:
        return f"pothole-yolo:{Path(self.cfg.weights).stem}"

    def infer(self, frame: np.ndarray) -> list[RoadHazard]:
        out = self.model.predict(
            source=frame, conf=self.cfg.conf_threshold,
            imgsz=self.cfg.input_size, verbose=False,
        )[0]
        boxes = out.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = boxes.conf.cpu().numpy().astype(np.float32)
        return [RoadHazard(box_px=b, score=float(s), localised=True)
                for b, s in zip(xyxy, scores)]

    def close(self) -> None:
        self.model = None


class PotholePatchClassifier(HazardDetector):
    """A binary classifier swept over road-surface patches.

    Only the lower part of the frame is examined, and only within the drivable
    area when one is supplied. That is not an optimisation: a classifier run over
    the whole image will happily call a dark window or a tree shadow a pothole,
    because it was only ever trained to separate damaged tarmac from sound
    tarmac and has no notion of "this is not road at all".
    """

    def __init__(self, cfg: PotholeConfig) -> None:
        import torch

        from models.weather_cls.classifier import build_model

        self.cfg = cfg
        self._torch = torch
        weights = Path(cfg.weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"pothole classifier weights not found: {weights}. Train with "
                f"`python models/pothole_det/train.py`, or set pothole.enabled=false."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(cfg.architecture, 2, pretrained=False)
        state = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(state.get("model", state))
        self.model.to(self.device).eval()
        log_kv(_log, logging.INFO, "pothole.loaded",
               mode="patch-classifier", weights=str(weights))

    @property
    def name(self) -> str:
        return f"pothole-patch:{Path(self.cfg.weights).stem}"

    def infer(
        self, frame: np.ndarray, drivable_mask: np.ndarray | None = None
    ) -> list[RoadHazard]:
        import cv2
        torch = self._torch

        h, w = frame.shape[:2]
        top = int(self.cfg.roi_top * h)
        size, stride = self.cfg.patch_px, self.cfg.patch_stride_px
        patches, origins = [], []
        for y in range(top, h - size + 1, stride):
            for x in range(0, w - size + 1, stride):
                if drivable_mask is not None:
                    cell = drivable_mask[y:y + size, x:x + size]
                    if cell.size and float((cell > 0).mean()) < self.cfg.min_drivable_fraction:
                        continue
                patch = cv2.resize(frame[y:y + size, x:x + size], (64, 64))
                patches.append(patch)
                origins.append((x, y))
        if not patches:
            return []

        arr = np.stack(patches).astype(np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        batch = torch.from_numpy(arr.transpose(0, 3, 1, 2)).float().to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)[:, 1].cpu().numpy()

        return [
            RoadHazard(
                box_px=np.array([x, y, x + size, y + size], dtype=np.float32),
                score=float(p), localised=False,
            )
            for (x, y), p in zip(origins, probs) if p >= self.cfg.conf_threshold
        ]

    def close(self) -> None:
        self.model = None


def build_hazard_detector(cfg: PotholeConfig) -> HazardDetector | None:
    """Construct the configured detector, or None when disabled or unavailable.

    A missing checkpoint degrades to None rather than raising: road hazards are
    an advisory signal, and losing them must not take the collision pipeline
    down with them.
    """
    if not cfg.enabled:
        return None
    try:
        if cfg.mode == "detector":
            return PotholeYOLO(cfg)
        if cfg.mode == "classifier":
            return PotholePatchClassifier(cfg)
    except Exception as exc:                          # noqa: BLE001 - deliberate
        log_kv(_log, logging.WARNING, "pothole.unavailable",
               mode=cfg.mode, error=str(exc)[:160],
               note="continuing without road-hazard detection")
        return None
    raise ValueError(f"unknown pothole.mode '{cfg.mode}'")
