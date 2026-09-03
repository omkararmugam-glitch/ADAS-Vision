"""Construction of the detector named by ``detector.kind`` in the config.

The three kinds are deliberately interchangeable behind one config flag, which is
what makes the baseline-vs-multi-task benchmark in ``docs/EVAL_RESULTS.md`` a
one-line change rather than a branch:

  ``stub``     no model at all -- exercises the whole pipeline offline.
  ``yolov8``   the single-task baseline: YOLOv8 boxes + a separate lane model.
  ``yolopv2``  the multi-task backbone: boxes and both masks in one pass.

Imports of the heavy wrappers are deferred into each branch so that selecting one
backbone never pays the import cost -- or the dependency -- of the others.
"""

from __future__ import annotations

from core.config import DetectorConfig
from models.perception.base import Detector, resolve_device


def build_detector(cfg: DetectorConfig, camera: object | None = None) -> Detector:
    """Construct the detector named by the config.

    Args:
        cfg: Detector settings.
        camera: A :class:`~core.calibrate.CameraModel`, required only by the stub,
            which generates its fake output through the calibration.

    Raises:
        ValueError: for an unknown ``detector.kind``, or a stub without a camera.
        FileNotFoundError: if the weights are missing.
    """
    if cfg.kind == "stub":
        from models.perception.stub import StubDetector

        if camera is None:
            raise ValueError("the stub detector requires a CameraModel")
        return StubDetector(camera, cfg)  # type: ignore[arg-type]

    if cfg.kind == "yolopv2":
        from models.perception.composite import CompositeDetector
        from models.perception.yolopv2 import YOLOPv2Detector

        primary: Detector = YOLOPv2Detector(cfg)
        if cfg.vru.enabled:
            return CompositeDetector(primary, [build_vru_detector(cfg)])
        return primary

    if cfg.kind == "yolov8":
        from models.perception.yolov8 import YOLOv8Detector

        # The YOLOv8 baseline detects people itself, so unlike YOLOPv2 it needs no
        # separate VRU model -- and stacking one on would double-count pedestrians.
        return YOLOv8Detector(cfg)

    raise ValueError(f"unknown detector kind '{cfg.kind}'")


def build_vru_detector(cfg: DetectorConfig) -> Detector:
    """Build the vulnerable-road-user detector on the same device as the primary."""
    from models.perception.vru import VulnerableRoadUserDetector

    return VulnerableRoadUserDetector(cfg.vru, resolve_device(cfg.device))


# Retained under the original name so existing callers and tests keep working.
VRUDetectorFactory = build_vru_detector
