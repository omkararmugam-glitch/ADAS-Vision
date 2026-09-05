"""A learned lane segmenter, distilled from the classical one.

BDD100K's lane annotations are a separate download and were not supplied, so
there are no ground-truth masks. What exists is :mod:`models.lane_seg.classical`,
which is right on roughly nine frames in ten. Training a network to reproduce it
is therefore weak supervision, and the interesting question is whether the
student can be *better than its teacher* -- which, unusually, it can be here.

The reason is that the classical detector's errors are **inconsistent** while its
successes are **consistent**. Painted markings look the same in every frame;
bonnet reflections, sunlit walls and foliage that happen to pass a threshold do
not. A network trained across thousands of frames sees the marking pattern
repeatedly and the spurious ones only once each, so gradient descent fits the
former and averages away the latter. It also has spatial context the classical
pipeline lacks: it can learn that lane markings do not appear on car bodies,
which no per-pixel threshold can express.

What it cannot do is invent a marking the teacher never found. Recall against
真 ground truth is still bounded by the classical detector; what improves is
precision.

LRASPP-MobileNetV3 as the backbone: designed for exactly this -- real-time
semantic segmentation on a constrained device -- and it leaves the GPU budget
free for the detector, which is the thing that must not be slowed down.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from core.logging_utils import get_logger, log_kv

_log = get_logger(__name__)

#: Input resolution the network sees. Lane markings are thin, so downsampling
#: too far erases them entirely -- but full resolution costs more than the
#: detector, which would be the wrong trade for a supporting signal.
INPUT_SIZE = (512, 288)


def build_segmentation_model(num_classes: int = 2, pretrained_backbone: bool = True):
    """LRASPP-MobileNetV3 with a lane/not-lane head."""
    import torchvision.models.segmentation as seg

    weights_backbone = "DEFAULT" if pretrained_backbone else None
    return seg.lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=weights_backbone, num_classes=num_classes
    )


class LearnedLaneSegmenter:
    """Runs a trained lane segmentation network.

    Interface-compatible with :class:`~models.lane_seg.classical.ClassicalLaneSegmenter`
    so the two are swappable wherever a lane mask is consumed.
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, weights: str | Path, device: str = "auto",
                 threshold: float = 0.5) -> None:
        import torch

        self._torch = torch
        self.threshold = threshold
        p = Path(weights)
        if not p.is_file():
            raise FileNotFoundError(
                f"lane segmenter weights not found: {p}. Train one with "
                f"`python models/lane_seg/train.py`, or use the classical "
                f"segmenter, which needs no weights."
            )
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        state = torch.load(p, map_location="cpu")
        self.model = build_segmentation_model(2, pretrained_backbone=False)
        self.model.load_state_dict(state.get("model", state))
        self.model.to(self.device).eval()
        self.input_size = tuple(state.get("input_size", INPUT_SIZE))
        log_kv(_log, logging.INFO, "lane_seg.loaded",
               weights=str(p), device=str(self.device), input_size=self.input_size)

    @property
    def name(self) -> str:
        return "lane-learned"

    def _preprocess(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_AREA)
        arr = (small.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        return self._torch.from_numpy(arr.transpose(2, 0, 1)[None]).to(self.device)

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Binary lane mask at the frame's own resolution."""
        torch = self._torch
        h, w = frame.shape[:2]
        with torch.no_grad():
            logits = self.model(self._preprocess(frame))["out"]
            prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
        # Nearest-neighbour upsampling of a *thresholded* mask would alias the
        # thin markings badly, so the probability map is resized and thresholded
        # afterwards.
        full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return ((full >= self.threshold) * 255).astype(np.uint8)

    def infer_detailed(self, frame: np.ndarray):
        """Match the classical segmenter's richer return type."""
        from models.lane_seg.classical import LaneMaskResult

        mask = self.infer(frame)
        coverage = float((mask > 0).sum()) / max(1, mask.size)
        return LaneMaskResult(mask=mask, coverage=coverage,
                              from_colour=0, from_gradient=0)

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
