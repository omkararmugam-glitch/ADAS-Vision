"""Lane-marking segmentation without a network.

Exists because the only model in this project that produces lane masks is
YOLOPv2, whose weights were lost, and because a lane detector that needs no
weights is worth having regardless: it runs anywhere, adds no GPU memory beside
the detector, and cannot silently degrade on a domain it was never trained for.

It is not as good as a learned segmenter and does not pretend to be. What it
gets right is the common case -- painted markings on tarmac, reasonable light --
and it fails in ways that are legible rather than mysterious: worn paint, snow
cover, and strong shadow edges parallel to the lane are the known weaknesses,
listed here so a bad frame is recognisable rather than surprising.

Two cues are combined, because each covers the other's blind spot:

**Colour.** Markings are white or yellow. Working in HLS rather than RGB matters:
lightness separates from hue there, so a white line stays white through a shadow
that would move it a long way in RGB. Yellow is thresholded on saturation, which
is what distinguishes yellow paint from pale tarmac.

**Gradient.** A marking is a bright stripe with an edge on each side. A Sobel in
x responds to exactly that and, unlike colour, keeps working when paint is grey
with age. It also responds to every other vertical edge, which is why it is
intersected with a road-region mask rather than used alone.

**Orientation.** Lane markings run along the direction of travel. Zebra
crossings, stop lines and give-way markings run across it, and are otherwise
indistinguishable -- white, thin, on tarmac, inside the road region. Without an
orientation test the fitter strings a crossing into a "lane" that cuts
diagonally over the junction, which is what it did before this filter existed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: HLS lightness above which a pixel is a candidate white marking.
WHITE_LIGHTNESS = 165

#: HLS hue/saturation window for yellow paint. Saturation is the discriminating
#: axis: pale tarmac shares yellow's hue but has almost none of its saturation.
YELLOW_HUE = (15, 40)
YELLOW_SATURATION = 70

#: Sobel magnitude, normalised to 0-255, above which an edge counts.
GRADIENT_THRESHOLD = 45

#: Fraction of frame height above which nothing is considered road. Generous --
#: the horizon moves with pitch, and cutting too tightly removes the far lane
#: exactly where the fit most needs support.
ROI_TOP = 0.45

#: Fraction of frame height below which nothing is considered road either. In a
#: dashcam frame the bottom band is the ego vehicle's own bonnet, and at night it
#: is a mirror: shop signs, headlights and tail lights all reflect off it as
#: bright saturated streaks that every threshold here would otherwise accept.
#: Those reflections sit at the very bottom of the image, which is where the
#: sliding-window fit starts, so they do not merely add noise -- they anchor the
#: whole lane fit to the bonnet.
ROI_BOTTOM = 0.88

#: A marking is a thin stripe. Anything wider than this fraction of the frame is
#: a wall, a vehicle flank or a sunlit kerb, not paint.
MAX_STRIPE_FRAC = 0.10

#: Minimum angle from horizontal, in degrees, for a stripe to count as a lane
#: marking. Lane markings run roughly along the direction of travel, so in the
#: image they converge on the vanishing point and are far from horizontal.
#:
#: This exists because of ZEBRA CROSSINGS. Their stripes are white, thin, and
#: pass every other test here -- but they run ACROSS the road, and the fitter
#: happily strings them into a "lane" that cuts diagonally over the junction.
#: Stop lines and give-way markings fail the same way. Orientation is the only
#: cue that separates them from real markings.
MIN_STRIPE_ANGLE_DEG = 25.0

#: Maximum angle, in degrees, between a stripe's own direction and the direction
#: from that stripe to the vanishing point.
#:
#: This is the cue that actually separates lane markings from zebra crossings,
#: and the flat-angle test above is only a fallback for when no vanishing point
#: is known. Under perspective a crossing's bars are NOT horizontal in the image
#: -- they run diagonally, and steeply enough to pass any angle threshold loose
#: enough to keep real markings near the horizon. What remains true is that a
#: lane marking lies along the direction of travel and therefore points AT the
#: vanishing point, while a crossing bar lies across it and points well away.
MAX_VP_MISALIGNMENT_DEG = 22.0


@dataclass(frozen=True)
class LaneMaskResult:
    """A lane mask and the evidence behind it.

    Attributes:
        mask: ``(H, W)`` uint8, 255 where a lane marking is believed to be.
        coverage: Fraction of the region of interest marked. A useful sanity
            signal -- a plausible road has a small percentage, and a large value
            means the thresholds have latched onto something that is not paint.
        from_colour: Pixels contributed by the colour cue.
        from_gradient: Pixels contributed by the gradient cue.
    """

    mask: np.ndarray
    coverage: float
    from_colour: int
    from_gradient: int

    @property
    def is_plausible(self) -> bool:
        """Whether the mask looks like lane markings rather than noise.

        An empty mask and a mask covering a third of the road are both wrong, and
        both are better reported than fitted: ``LaneFitter`` will refuse a sparse
        mask on its own, but a *dense* one would happily produce a confident and
        entirely wrong lane.
        """
        return 0.0005 <= self.coverage <= 0.15


def _roi(shape: tuple[int, int], roi_top: float,
         roi_bottom: float = ROI_BOTTOM) -> np.ndarray:
    """Trapezoid covering the road ahead, excluding sky, edges and the bonnet."""
    h, w = shape
    top = int(roi_top * h)
    bottom = int(roi_bottom * h)
    poly = np.array([[
        (int(0.02 * w), bottom), (int(0.42 * w), top),
        (int(0.58 * w), top), (int(0.98 * w), bottom),
    ]], dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, poly, 255)
    return mask


def lane_mask(
    frame: np.ndarray,
    roi_top: float = ROI_TOP,
    roi_bottom: float = ROI_BOTTOM,
    white_lightness: int = WHITE_LIGHTNESS,
    yellow_saturation: int = YELLOW_SATURATION,
    gradient_threshold: int = GRADIENT_THRESHOLD,
    vanishing_point: tuple[float, float] | None = None,
) -> LaneMaskResult:
    """Segment lane markings in one BGR frame.

    Args:
        vanishing_point: ``(u, v)`` in pixels, when known. Supplying it replaces
            the crude flat-angle test with the correct one -- a lane marking
            points at the vanishing point, a zebra crossing does not -- and is
            the difference between rejecting a crossing and painting it as lane.

    Returns:
        A :class:`LaneMaskResult`. The mask is directly consumable by
        ``models.lane_seg.fitting.LaneFitter.fit``.
    """
    h, w = frame.shape[:2]
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
    hue, light, sat = hls[..., 0], hls[..., 1], hls[..., 2]

    white = light >= white_lightness
    yellow = (
        (hue >= YELLOW_HUE[0]) & (hue <= YELLOW_HUE[1]) & (sat >= yellow_saturation)
    )
    colour = white | yellow

    grey = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    sobel = np.abs(cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=5))
    if sobel.max() > 0:
        sobel = (sobel / sobel.max() * 255).astype(np.uint8)
    else:
        sobel = sobel.astype(np.uint8)
    gradient = sobel >= gradient_threshold

    roi = _roi((h, w), roi_top, roi_bottom) > 0
    combined = ((colour | gradient) & roi).astype(np.uint8) * 255

    # A marking is a thin stripe. Opening removes speckle; the width test then
    # discards broad bright regions -- a sunlit wall or a white van flank --
    # which otherwise survive every threshold above and dominate the fit.
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    # Thinness is measured with a rotated rectangle, NOT the axis-aligned
    # bounding box. A lane marking seen in perspective runs diagonally, and its
    # bounding box is then wide even though the stripe is a few pixels across --
    # so a bounding-box width test rejects exactly the angled markings that
    # carry the most information about where the lane goes. minAreaRect gives
    # the true stripe width whatever its orientation.
    max_width = max(3, int(MAX_STRIPE_FRAC * w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    keep = np.zeros_like(combined)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 12:
            continue
        pts = cv2.findNonZero((labels == i).astype(np.uint8))
        if pts is None:
            continue
        (_, _), (rw, rh), angle = cv2.minAreaRect(pts)
        if min(rw, rh) > max_width:
            continue
        # Angle of the LONG axis, measured from horizontal. OpenCV reports the
        # rectangle angle in (-90, 0]; which side is "long" depends on whether
        # width or height is larger, so both cases are handled explicitly rather
        # than assumed.
        long_axis_angle = angle if rw >= rh else angle + 90.0

        if vanishing_point is not None:
            # Does this stripe point at the vanishing point? Compare its own
            # direction with the direction from its centroid to the VP. Both are
            # undirected lines, so the comparison folds to 0-90 degrees.
            cy_, cx_ = (stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2,
                        stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2)
            to_vp = np.degrees(np.arctan2(vanishing_point[1] - cy_,
                                          vanishing_point[0] - cx_))
            delta = abs(((long_axis_angle - to_vp + 90.0) % 180.0) - 90.0)
            if delta > MAX_VP_MISALIGNMENT_DEG:
                continue    # lies across the road, not along it
        else:
            from_horizontal = abs(((long_axis_angle + 90.0) % 180.0) - 90.0)
            if from_horizontal < MIN_STRIPE_ANGLE_DEG:
                continue    # fallback: obviously transverse
        keep[labels == i] = 255

    roi_area = int(roi.sum())
    coverage = float((keep > 0).sum()) / max(1, roi_area)
    return LaneMaskResult(
        mask=keep,
        coverage=coverage,
        from_colour=int((colour & roi).sum()),
        from_gradient=int((gradient & roi).sum()),
    )


class ClassicalLaneSegmenter:
    """Stateless wrapper, matching the interface a learned segmenter would use."""

    def __init__(
        self,
        roi_top: float = ROI_TOP,
        roi_bottom: float = ROI_BOTTOM,
        white_lightness: int = WHITE_LIGHTNESS,
        yellow_saturation: int = YELLOW_SATURATION,
        gradient_threshold: int = GRADIENT_THRESHOLD,
        vanishing_point: tuple[float, float] | None = None,
    ) -> None:
        self.vanishing_point = vanishing_point
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.white_lightness = white_lightness
        self.yellow_saturation = yellow_saturation
        self.gradient_threshold = gradient_threshold

    @property
    def name(self) -> str:
        return "lane-classical"

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Binary lane mask for one frame."""
        return lane_mask(
            frame, self.roi_top, self.roi_bottom, self.white_lightness,
            self.yellow_saturation, self.gradient_threshold,
            self.vanishing_point,
        ).mask

    def infer_detailed(self, frame: np.ndarray) -> LaneMaskResult:
        """As :meth:`infer`, keeping the evidence for diagnostics."""
        return lane_mask(
            frame, self.roi_top, self.roi_bottom, self.white_lightness,
            self.yellow_saturation, self.gradient_threshold,
            self.vanishing_point,
        )
