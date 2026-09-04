"""Monocular depth, used only where the calibrated ground plane cannot help.

The homography gives accurate metric distance for objects standing on flat road,
and says nothing at all about anything else -- an occluded contact point, a
vehicle on a crest, a truck whose wheels are hidden. MiDaS covers every pixel but
carries no scale.

:mod:`depth.align` combines them: the homography supplies metric anchors, an
affine fit transfers that scale to the relative depth map, and the result extends
metric distance to the pixels the homography could not reach. MiDaS is therefore
never the source of scale, only of shape -- and where both are available, the
homography wins.
"""

from depth.align import MIN_ANCHORS, DepthScale, fit_scale, sample_depth
from depth.midas import (
    DepthEstimator,
    MiDaSDepth,
    PlanarStubDepth,
    build_depth_estimator,
)

__all__ = [
    "DepthEstimator",
    "MiDaSDepth",
    "PlanarStubDepth",
    "build_depth_estimator",
    "DepthScale",
    "fit_scale",
    "sample_depth",
    "MIN_ANCHORS",
]
