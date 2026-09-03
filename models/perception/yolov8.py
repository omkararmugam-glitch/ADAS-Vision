"""YOLOv8 single-task baseline -- PLACEHOLDER, implemented in build step 3.

Why a baseline at all, when YOLOPv2 already works: the multi-task network shares a
backbone across three heads, which is cheaper but couples the tasks -- retraining
for potholes risks the lane head. The baseline keeps detection and lane
segmentation in separate models that can be trained and swapped independently, and
gives an honest speed/accuracy comparison for ``docs/EVAL_RESULTS.md``.

Unlike YOLOPv2 this model covers pedestrians directly (COCO class 0), so the
factory attaches no separate VRU detector to it. It produces no segmentation
masks; under ``detector.kind: yolov8`` the lane mask comes from the separate
model in ``models/lane_seg/``.

Deliberately raises rather than returning empty results: a backbone that silently
emits nothing looks exactly like a road with no cars in it.
"""

from __future__ import annotations

from core.config import DetectorConfig
from models.perception.base import Detector


class YOLOv8Detector(Detector):
    """Ultralytics YOLOv8 wrapper. Not yet implemented -- see module docstring."""

    def __init__(self, cfg: DetectorConfig) -> None:
        raise NotImplementedError(
            "the YOLOv8 baseline is build step 3 and is not implemented yet. "
            "Use detector.kind 'yolopv2' for the multi-task backbone, or 'stub' "
            "to exercise the pipeline without a model."
        )

    @property
    def name(self) -> str:
        return "yolov8"

    @property
    def weights_hash(self) -> str:
        raise NotImplementedError

    def infer(self, frame):  # type: ignore[no-untyped-def]
        raise NotImplementedError
