"""Object detection and multi-task perception backbones.

``build_detector`` is the only entry point the pipeline uses; everything else here
is an implementation of the :class:`~models.perception.base.Detector` interface.
"""

from models.perception.base import Detector, file_sha256, resolve_device
from models.perception.factory import VRUDetectorFactory, build_detector, build_vru_detector

__all__ = [
    "Detector",
    "build_detector",
    "build_vru_detector",
    "VRUDetectorFactory",
    "resolve_device",
    "file_sha256",
]
