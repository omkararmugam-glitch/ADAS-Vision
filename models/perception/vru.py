"""Vulnerable road users: pedestrians, cyclists, motorcyclists.

A second network exists here only because the multi-task backbone cannot cover
these classes (see :mod:`models.perception.yolopv2`). It emits ONLY
person/bicycle/motorcycle -- vehicles come from the primary detector, and taking
them from both would double-count every car in the frame.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from core.config import VRUConfig
from core.letterbox import LetterboxTransform
from core.logging_utils import get_logger, log_kv
from core.types import CLASS_IDS, FrameResult
from models.perception.base import Detector

_log = get_logger(__name__)


class VulnerableRoadUserDetector(Detector):
    """COCO-pretrained detector covering people, bicycles and motorcycles.

    YOLOPv2 has no pedestrian class, and no released multi-task network in that
    family does -- HybridNets ships ``obj_list: ['car']`` and YOLOP is the same.
    Getting vulnerable road users therefore requires a second model.

    This uses torchvision's COCO detector rather than adding an ultralytics
    dependency: torchvision is already a hard requirement, so this costs no new
    package. It deliberately emits *only* vulnerable road users -- vehicles come
    from YOLOPv2, and taking them from both would double-count every car.
    """

    #: COCO category id -> this project's class name.
    COCO_TO_CLASS = {1: "person", 2: "bicycle", 4: "motorcycle"}

    def __init__(self, cfg: VRUConfig, device: torch.device) -> None:
        from torchvision.models import detection as tvdet

        self.cfg = cfg
        self.device = device
        builder = getattr(tvdet, cfg.architecture, None)
        if builder is None:
            raise ValueError(
                f"unknown torchvision detection architecture '{cfg.architecture}'"
            )
        self.model = builder(weights="DEFAULT", box_score_thresh=cfg.conf_threshold)
        self.model.eval().to(device)
        self._weights_hash = self._resolve_weights_hash(cfg.architecture)
        log_kv(
            _log, logging.INFO, "detect.loaded",
            model="vru", architecture=cfg.architecture,
            sha256=self._weights_hash[:12], device=str(device),
            classes=",".join(self.class_names),
        )

    @staticmethod
    def _resolve_weights_hash(architecture: str) -> str:
        """Hash the cached checkpoint so swapping weights invalidates the cache."""
        hub = Path(torch.hub.get_dir()) / "checkpoints"
        candidates = sorted(hub.glob("*.pth")) if hub.is_dir() else []
        stem = architecture.replace("_", "")
        for path in candidates:
            if path.name.replace("_", "").lower().startswith(stem[:12].lower()):
                return file_sha256(path)
        # Not fatal: fall back to an identity string so the manifest still records
        # *something* that changes when the architecture does.
        return hashlib.sha256(architecture.encode()).hexdigest()

    @property
    def name(self) -> str:
        return f"vru:{self.cfg.architecture}"

    @property
    def weights_hash(self) -> str:
        return self._weights_hash

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self.COCO_TO_CLASS.values())

    def infer(self, frame: np.ndarray) -> FrameResult:
        return self.infer_batch([frame])[0]

    def infer_batch(self, frames: list[np.ndarray]) -> list[FrameResult]:
        if not frames:
            return []
        tensors = [
            torch.from_numpy(np.ascontiguousarray(f[..., ::-1].transpose(2, 0, 1)))
            .to(self.device)
            .float()
            / 255.0
            for f in frames
        ]
        with torch.no_grad():
            outputs = self.model(tensors)

        results = []
        for frame, out in zip(frames, outputs):
            h, w = frame.shape[:2]
            labels = out["labels"].cpu().numpy()
            wanted = np.isin(labels, list(self.COCO_TO_CLASS))
            boxes = out["boxes"].cpu().numpy()[wanted]
            scores = out["scores"].cpu().numpy()[wanted]
            ids = np.array(
                [CLASS_IDS[self.COCO_TO_CLASS[int(c)]] for c in labels[wanted]], dtype=np.int32
            )
            blank = np.zeros((h, w), dtype=np.uint8)
            results.append(
                FrameResult(
                    boxes_px=boxes.astype(np.float32),
                    class_ids=ids,
                    scores=scores.astype(np.float32),
                    drivable_mask=blank,
                    lane_mask=blank.copy(),
                )
            )
        return results
