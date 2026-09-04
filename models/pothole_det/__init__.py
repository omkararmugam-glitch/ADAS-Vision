"""Road-hazard (pothole) detection.

Two approaches behind one interface, because the brief asks for whichever gives
better precision on the supplied dataset -- a measurement, not a prior belief.
Detection localises the damage so the ego-corridor test can be applied but needs
box labels on objects with famously ill-defined boundaries; patch classification
needs only cheap image-level labels but yields no position.

Precision is weighted above recall in both: a hazard warning that fires on every
tar seam gets ignored, and an ignored warning is worse than none.
"""

from models.pothole_det.detector import (
    HazardDetector,
    PotholePatchClassifier,
    PotholeYOLO,
    RoadHazard,
    build_hazard_detector,
)

__all__ = [
    "HazardDetector",
    "RoadHazard",
    "PotholeYOLO",
    "PotholePatchClassifier",
    "build_hazard_detector",
]
