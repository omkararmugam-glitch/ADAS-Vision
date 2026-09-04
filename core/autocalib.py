"""Estimate the ground-plane homography from road geometry, with no target.

A chessboard calibration is strictly better and should be used whenever the
camera is to hand. This exists for the case that actually occurs with supplied
footage: dashcam frames with no calibration rig, no intrinsics, and no way to go
back and measure anything.

The method is standard and rests on two facts about roads:

1. **Lane markings are parallel in the world**, so in the image they converge on
   a vanishing point. That point fixes the camera's pitch and yaw relative to the
   road, which is what the horizon position encodes.
2. **Lane width is roughly known.** Motorway lanes are near 3.5 m almost
   everywhere. That supplies the one absolute distance the geometry otherwise
   lacks, and turns a shape into metres.

**What this costs, stated plainly.** Three quantities are assumed rather than
measured -- focal length, camera height, lane width -- and every distance the
pipeline reports scales with them:

* ``camera_height_m`` scales distance **linearly**. A dashcam mounted 1.5 m up
  but assumed at 1.2 m makes every distance read 20% short.
* ``lane_width_m`` sets the lateral scale the same way.
* ``focal_px`` affects how quickly distance grows toward the horizon, so its
  error grows with range.

So the output is a *usable* homography, not an accurate one. Every field it
writes is recorded in the YAML, including ``method: autocalib`` and the
assumptions used, so no downstream number can later be mistaken for the product
of a real calibration. :func:`estimate` refuses rather than guesses when the
evidence is too weak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

#: Typical dashcam horizontal field of view. Most consumer dashcams sit between
#: 100 and 170 degrees diagonal; 90 horizontal is a middling, deliberately
#: conservative default when nothing better is known.
DEFAULT_HFOV_DEG = 90.0

#: Standard lane widths. 3.5 m is the common motorway figure; urban lanes are
#: narrower, which is why this is configurable and recorded.
DEFAULT_LANE_WIDTH_M = 3.5

#: Typical dashcam mounting height, windscreen behind the mirror.
DEFAULT_CAMERA_HEIGHT_M = 1.35

#: Plausibility gates on the recovered pose. A forward-facing dashcam is mounted
#: roughly square to the vehicle, so a large yaw means the vanishing point was
#: fitted to something that is not the road -- in urban scenes, building edges
#: converge on their own vanishing point and will happily out-vote lane markings.
#: Rejecting is the right response: a wrong pose yields confident wrong metres,
#: and the pipeline already degrades gracefully when no calibration is present.
MAX_PLAUSIBLE_YAW_DEG = 12.0
MAX_PLAUSIBLE_PITCH_DEG = 20.0

#: The horizon must land within this fractional band of frame height. Outside it
#: the "road" being fitted is almost certainly a facade or a railing.
HORIZON_BAND = (0.25, 0.75)


@dataclass(frozen=True)
class AutoCalibration:
    """The estimate, with the assumptions that produced it.

    Attributes:
        H_image_to_ground: ``(3, 3)`` homography, image px -> ego metres.
        vanishing_point: ``(u, v)`` in pixels.
        horizon_v: Image row of the horizon.
        pitch_rad: Downward tilt implied by the vanishing point.
        yaw_rad: Sideways rotation implied by it.
        focal_px: Focal length used, in pixels.
        camera_height_m: Assumed mounting height.
        lane_width_m: Assumed lane width.
        n_lines: Lane lines that supported the vanishing point.
        inlier_ratio: Share of candidate lines agreeing with it.
        assumptions: Human-readable list, copied into the YAML.
    """

    H_image_to_ground: np.ndarray
    vanishing_point: tuple[float, float]
    horizon_v: float
    pitch_rad: float
    yaw_rad: float
    focal_px: float
    camera_height_m: float
    lane_width_m: float
    n_lines: int
    inlier_ratio: float
    assumptions: list[str] = field(default_factory=list)


def focal_from_fov(width_px: int, hfov_deg: float = DEFAULT_HFOV_DEG) -> float:
    """Focal length in pixels implied by a horizontal field of view."""
    return float(width_px / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0)))


def _candidate_lines(
    frame: np.ndarray, roi_top: float = 0.40, min_slope: float = 0.15
) -> list[tuple[float, float, float, float]]:
    """Find near-straight, road-like line segments in the lower image.

    Restricted to the lower frame because that is where road markings are, and
    filtered by slope: near-horizontal segments are almost always buildings,
    wires or the horizon itself, and they corrupt a vanishing point badly
    because they are nearly parallel to the direction being solved for.

    Both cut-offs are deliberately generous. Tightening them looks appealing on
    one scene and silently rejects another -- the horizon sits anywhere between
    a third and two-thirds of frame height depending on mounting and pitch, so a
    tight ROI encodes one camera's geometry as if it were universal.
    """
    h, w = frame.shape[:2]
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(grey, 60, 160)
    edges[: int(roi_top * h), :] = 0

    segments = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=55,
        minLineLength=int(0.08 * h), maxLineGap=int(0.02 * h),
    )
    if segments is None or len(segments) == 0:
        return []
    # OpenCV has returned (N, 1, 4) historically and (N, 4) since 5.0. Reshape
    # rather than index a fixed layout, so this works across versions.
    segments = np.asarray(segments).reshape(-1, 4)
    out = []
    for x1, y1, x2, y2 in segments:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dx) < 1e-6:
            out.append((float(x1), float(y1), float(x2), float(y2)))
            continue
        if abs(dy / dx) < min_slope:
            continue
        out.append((float(x1), float(y1), float(x2), float(y2)))
    return out


def vanishing_point(
    lines: list[tuple[float, float, float, float]], size: tuple[int, int],
    threshold_px: float = 18.0, iterations: int = 400, seed: int = 0,
) -> tuple[tuple[float, float], int, float] | None:
    """RANSAC the intersection of road lines.

    RANSAC rather than least squares because a single kerb, railing or building
    edge that survives the slope filter would drag a least-squares vanishing
    point far off, and the horizon it implies with it.
    """
    if len(lines) < 4:
        return None
    w, h = size
    rng = np.random.default_rng(seed)
    homog = []
    for x1, y1, x2, y2 in lines:
        homog.append(np.cross([x1, y1, 1.0], [x2, y2, 1.0]))
    homog = np.asarray(homog, dtype=np.float64)

    best_point, best_inliers = None, 0
    for _ in range(iterations):
        i, j = rng.choice(len(homog), size=2, replace=False)
        p = np.cross(homog[i], homog[j])
        if abs(p[2]) < 1e-9:
            continue
        u, v = p[0] / p[2], p[1] / p[2]
        # A vanishing point far outside the frame means near-parallel lines and
        # a numerically hopeless intersection.
        if not (-2 * w < u < 3 * w and -2 * h < v < 2 * h):
            continue
        d = np.abs(homog @ np.array([u, v, 1.0])) / (
            np.linalg.norm(homog[:, :2], axis=1) + 1e-9)
        n = int((d < threshold_px).sum())
        if n > best_inliers:
            best_point, best_inliers = (float(u), float(v)), n
    if best_point is None or best_inliers < 4:
        return None
    return best_point, best_inliers, best_inliers / len(lines)


def homography_from_pose(
    focal_px: float, principal: tuple[float, float],
    pitch_rad: float, yaw_rad: float, height_m: float,
) -> np.ndarray:
    """Ground-plane homography for a camera at a known pose.

    Maps image pixels to ego-frame metres (X right, Y forward), with the origin
    at the camera's projection onto the road.
    """
    cx, cy = principal
    K = np.array([[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]])
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy_, sy_ = np.cos(yaw_rad), np.sin(yaw_rad)
    # Camera looks down the +Y (forward) axis, tilted down by pitch.
    R_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    R_yaw = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]], dtype=np.float64)
    R = R_pitch @ R_yaw
    # Ground plane Z=0 in world, so only the X and Y columns and the
    # translation survive. In camera coordinates Y points DOWN and Z forward, and
    # the road is height_m *below* the camera -- hence +height_m, not -. The
    # negative sign here produced correct magnitudes with a negated forward
    # distance, which `CameraModel.is_on_ground` rejects outright (it requires
    # forward > 0), silently making every calibration useless.
    world_to_cam = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.0, height_m],
                             [0.0, 1.0, 0.0]])
    H_ground_to_image = K @ R @ world_to_cam
    return np.linalg.inv(H_ground_to_image)


def estimate(
    frames: list[np.ndarray],
    hfov_deg: float = DEFAULT_HFOV_DEG,
    camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M,
    lane_width_m: float = DEFAULT_LANE_WIDTH_M,
    min_inlier_ratio: float = 0.25,
    seed: int = 0,
) -> AutoCalibration | None:
    """Estimate a ground homography from road frames.

    Pools line evidence across several frames: markings are dashed, a single
    frame often carries too few segments, and pooling averages out the ones that
    happen to catch a kerb.

    Returns:
        The estimate, or None when the evidence does not support one. Refusing
        happens on three grounds -- too few lines, too few agreeing on a
        vanishing point, and a recovered pose outside what a windscreen mount can
        physically be. That last gate matters most in cities, where building
        edges converge on their own vanishing point and will out-vote the lane
        markings; the resulting pose looks numerically fine and is nonsense.
    """
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    lines: list[tuple[float, float, float, float]] = []
    for f in frames:
        lines.extend(_candidate_lines(f))
    if len(lines) < 6:
        return None

    found = vanishing_point(lines, (w, h), seed=seed)
    if found is None:
        return None
    (u, v), n_lines, ratio = found
    if ratio < min_inlier_ratio:
        return None

    focal = focal_from_fov(w, hfov_deg)
    cx, cy = w / 2.0, h / 2.0
    if not (HORIZON_BAND[0] * h <= v <= HORIZON_BAND[1] * h):
        return None
    # The vanishing point of the road direction gives the camera's rotation: its
    # offset from the principal point, in focal lengths, is the tangent of the
    # angle. The sign convention must match homography_from_pose, where the
    # horizon of the resulting homography sits at cy - focal*tan(pitch) -- so a
    # VP BELOW the principal point is a camera tilted up, i.e. negative pitch.
    #
    # Getting this backwards is not a small error. It puts the homography's
    # horizon on the wrong side of the image centre, and the recovered distance
    # then never diverges near the true horizon: 5 m was reported where the
    # geometry says 39 m, with no obvious symptom because the numbers still
    # increase monotonically with range.
    pitch = float(np.arctan2(cy - v, focal))
    yaw = float(np.arctan2(u - cx, focal))
    if abs(np.rad2deg(yaw)) > MAX_PLAUSIBLE_YAW_DEG:
        return None
    if abs(np.rad2deg(pitch)) > MAX_PLAUSIBLE_PITCH_DEG:
        return None
    H = homography_from_pose(focal, (cx, cy), pitch, yaw, camera_height_m)

    return AutoCalibration(
        H_image_to_ground=H,
        vanishing_point=(u, v),
        horizon_v=float(v),
        pitch_rad=pitch,
        yaw_rad=yaw,
        focal_px=focal,
        camera_height_m=camera_height_m,
        lane_width_m=lane_width_m,
        n_lines=n_lines,
        inlier_ratio=float(ratio),
        assumptions=[
            f"horizontal field of view assumed {hfov_deg:.0f} deg "
            f"-> focal {focal:.0f} px (affects how distance grows with range)",
            f"camera height assumed {camera_height_m:.2f} m "
            f"(scales EVERY distance linearly)",
            f"lane width assumed {lane_width_m:.2f} m (sets lateral scale)",
            "principal point assumed at the image centre",
            "road assumed flat and straight where the vanishing point was fitted",
            "NOT a measured calibration: use `run.py calibrate` with a "
            "chessboard whenever the camera is available",
        ],
    )
