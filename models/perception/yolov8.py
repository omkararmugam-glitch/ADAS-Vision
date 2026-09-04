"""YOLOv8 single-task baseline.

Why a baseline at all, when YOLOPv2 already works: the multi-task network shares a
backbone across three heads, which is cheaper but couples the tasks -- retraining
for potholes risks the lane head. The baseline keeps detection and lane
segmentation in separate models that can be trained and swapped independently,
and gives an honest speed/accuracy comparison for ``docs/EVAL_RESULTS.md``.

Two differences from the YOLOPv2 path, both deliberate:

* **No VRU model is attached.** COCO already contains person, bicycle and
  motorcycle, so YOLOv8 covers vulnerable road users itself. Stacking the second
  detector on top would double-count every pedestrian.
* **No segmentation.** This is a detection model; it returns empty drivable and
  lane masks. Under ``detector.kind: yolov8`` the lane mask must come from the
  separate model in ``models/lane_seg/``, which is not built yet -- until it is,
  lane fitting on this path correctly reports ``NOT_DETECTED`` rather than
  inventing a lane.

Coordinates: ultralytics applies and inverts its own letterbox internally, so
``Results.boxes.xyxy`` is already in original image pixels. This wrapper therefore
does not use :class:`~core.letterbox.LetterboxTransform` -- applying it a second
time would scale the boxes twice.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from core.config import DetectorConfig, YOLOv8Config
from core.logging_utils import get_logger, log_kv
from core.types import CLASS_IDS, FrameResult
from models.perception.base import Detector, file_sha256, resolve_device

_log = get_logger(__name__)

#: Standard released checkpoints that may be downloaded on demand. Anything else
#: must already exist on disk -- quietly fetching an unrecognised name is how you
#: end up benchmarking a model you did not intend to run.
_PRETRAINED = re.compile(r"^yolo(v8|11|12)[nsmlx]\.pt$")


def _supports_quantize() -> bool:
    """Whether this ultralytics spells fp16 as ``quantize`` rather than ``half``."""
    try:
        from ultralytics.cfg import get_cfg

        return hasattr(get_cfg(), "quantize")
    except Exception:  # pragma: no cover - any failure means assume the old name
        return False


#: Resolved once: the answer cannot change within a process.
_SUPPORTS_QUANTIZE = _supports_quantize()


class YOLOv8Detector(Detector):
    """Ultralytics YOLOv8, mapped onto this project's class vocabulary.

    COCO's three vehicle categories collapse to one ``vehicle`` class, matching
    what YOLOPv2 emits, so the two backbones stay directly comparable.

    COCO's ``traffic light`` and ``stop sign`` map onto the project's traffic
    control classes. Note the asymmetry that creates: COCO knows exactly one
    kind of sign, so ``traffic_sign`` from this detector means "a stop sign",
    while the same class from a BDD100K-trained model means any sign. The
    manifest records which detector produced a run, so the two are never
    silently mixed.
    """

    #: COCO class index -> this project's class name. car/bus/truck all become
    #: ``vehicle``; a bus is a vehicle for every downstream purpose here.
    COCO_TO_CLASS = {
        0: "person",
        1: "bicycle",
        2: "vehicle",   # car
        3: "motorcycle",
        5: "vehicle",   # bus
        7: "vehicle",   # truck
        9: "traffic_light",
        11: "traffic_sign",   # COCO's only sign class is the stop sign
    }

    @classmethod
    def _build_index_map(cls, model) -> dict[int, str]:
        """Map this checkpoint's class indices onto the project vocabulary.

        Two cases are recognised. A stock COCO checkpoint is matched by its class
        names, not by its class count, so a differently-ordered 80-class model
        cannot slip through. A checkpoint already trained on this project's
        vocabulary maps identically.

        Raises:
            ValueError: when the checkpoint's classes are neither, rather than
                guessing an alignment.
        """
        names = getattr(model, "names", None) or {}
        by_index = {int(k): str(v) for k, v in names.items()}
        lowered = {i: n.lower() for i, n in by_index.items()}

        # Already speaking our vocabulary?
        if set(lowered.values()) <= set(CLASS_IDS):
            return {i: n for i, n in lowered.items()}

        # Stock COCO, verified by name rather than by index.
        coco_expected = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                         5: "bus", 7: "truck", 9: "traffic light", 11: "stop sign"}
        if all(lowered.get(i) == n for i, n in coco_expected.items()):
            return dict(cls.COCO_TO_CLASS)

        raise ValueError(
            f"cannot map this checkpoint's classes onto the project vocabulary. "
            f"Expected either COCO, or a subset of {sorted(CLASS_IDS)}; got "
            f"{sorted(by_index.values())[:10]}. Retrain with the documented class "
            f"order, or extend core/types.py:CLASS_NAMES."
        )

    def __init__(self, cfg: DetectorConfig) -> None:
        from ultralytics import YOLO

        self.cfg = cfg
        self.sub: YOLOv8Config = cfg.yolov8
        self.device = resolve_device(cfg.device)
        self.half = bool(cfg.half and self.device.type == "cuda")

        weights = self._resolve_weights(self.sub)
        self.model = YOLO(str(weights))
        self.model.to(self.device)
        self._weights_path = weights
        self._weights_hash = file_sha256(weights)
        # Built from the checkpoint's OWN class names, never assumed. A model
        # fine-tuned on this project's vocabulary emits index 0 for `vehicle`,
        # which under the COCO table would be read as `person` -- silently
        # wrong, and invisible until a pedestrian alert fires at a parked car.
        self._index_map = self._build_index_map(self.model)

        log_kv(
            _log, logging.INFO, "detect.loaded",
            model="yolov8", weights=str(weights), sha256=self._weights_hash[:12],
            device=str(self.device), half=self.half,
            classes=",".join(self.class_names),
            note="detection only; lane mask must come from models/lane_seg",
        )

    # ------------------------------------------------------------------ #
    # weights
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_weights(sub: YOLOv8Config) -> Path:
        """Return an existing checkpoint path, downloading a standard one if allowed.

        A missing checkpoint raises unless it is both a recognised pretrained name
        and ``allow_download`` is set. The download is logged at WARNING: which
        weights produced a number is exactly the thing that must not be silent.
        """
        path = Path(sub.weights)
        if path.is_file():
            return path
        if not (sub.allow_download and _PRETRAINED.match(path.name)):
            raise FileNotFoundError(
                f"YOLOv8 weights not found at {path}. Either place a checkpoint "
                f"there, or set detector.yolov8.weights to a standard pretrained "
                f"name (e.g. yolov8n.pt) with allow_download: true."
            )

        from ultralytics.utils.downloads import attempt_download_asset

        path.parent.mkdir(parents=True, exist_ok=True)
        log_kv(
            _log, logging.WARNING, "detect.weights_download",
            name=path.name, dest=str(path),
            note="COCO-pretrained baseline; not fine-tuned on this project's data",
        )
        fetched = Path(attempt_download_asset(path.name))
        if fetched.resolve() != path.resolve():
            fetched.replace(path)
        if not path.is_file():
            raise FileNotFoundError(f"download of {path.name} did not produce {path}")
        return path

    # ------------------------------------------------------------------ #
    # Detector interface
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return f"yolov8:{self._weights_path.stem}"

    @property
    def weights_hash(self) -> str:
        return self._weights_hash

    @classmethod
    def vocabulary(cls) -> tuple[str, ...]:
        """Distinct class names this detector can emit, in first-seen order.

        ``dict.fromkeys`` removes the duplicate ``vehicle`` that car, bus and
        truck all map to, without sorting -- the manifest should list a class
        once, not three times.
        """
        return tuple(dict.fromkeys(cls.COCO_TO_CLASS.values()))

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.vocabulary()

    def _precision_kwargs(self) -> dict[str, object]:
        """fp16 selection, spelled the way the installed ultralytics expects.

        8.4 replaced the boolean ``half`` with ``quantize`` (16 or 32) and warns on
        the old name. Probing the config schema keeps this working across both
        without pinning the dependency to a single generation.
        """
        if _SUPPORTS_QUANTIZE:
            return {"quantize": 16 if self.half else 32}
        return {"half": self.half}

    def infer(self, frame: np.ndarray) -> FrameResult:
        return self.infer_batch([frame])[0]

    def infer_batch(self, frames: list[np.ndarray]) -> list[FrameResult]:
        if not frames:
            return []
        outputs = self.model.predict(
            frames,
            imgsz=self.cfg.input_size,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            max_det=self.cfg.max_detections,
            device=self.device,
            verbose=False,
            **self._precision_kwargs(),
        )

        results: list[FrameResult] = []
        for frame, out in zip(frames, outputs):
            h, w = frame.shape[:2]
            boxes = out.boxes
            if boxes is None or len(boxes) == 0:
                results.append(FrameResult.empty((w, h)))
                continue

            raw = boxes.cls.cpu().numpy().astype(int)
            keep = np.isin(raw, list(self._index_map))
            xyxy = boxes.xyxy.cpu().numpy()[keep].astype(np.float32)
            scores = boxes.conf.cpu().numpy()[keep].astype(np.float32)
            ids = np.array(
                [CLASS_IDS[self._index_map[int(c)]] for c in raw[keep]], dtype=np.int32
            )

            # Degenerate boxes would trip FrameResult's own validation; drop them
            # here so one bad box cannot fail an entire frame.
            wide = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
            xyxy, scores, ids = xyxy[wide], scores[wide], ids[wide]

            blank = np.zeros((h, w), dtype=np.uint8)
            results.append(
                FrameResult(
                    boxes_px=xyxy,
                    class_ids=ids,
                    scores=scores,
                    drivable_mask=blank,
                    lane_mask=blank.copy(),
                )
            )
        return results

    def close(self) -> None:
        import torch

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
