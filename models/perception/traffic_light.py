"""Read the state of a detected traffic light from its pixels.

The detector finds the lamp housing; this decides what it is showing. Those are
deliberately separate problems. A housing looks the same whatever it displays, so
asking one network to do both means every colour needs its own training examples
in every lighting condition -- whereas the lit lamp is, by construction, the
brightest saturated thing inside the box.

So this is classical computer vision, not a model: no weights, no training data,
and no dataset dependency. Two signals are combined.

**Hue.** The lit lamp is a strongly saturated red, amber or green region. Red
wraps around the hue circle, which is the usual source of bugs here and is
handled explicitly.

**Vertical position.** A traffic light housing is ordered: red at the top, amber
in the middle, green at the bottom. That ordering is near-universal and survives
the colour being washed out at night or in glare -- exactly when hue is least
reliable. It is used as a prior, not as an override, because horizontal housings
exist and a sideways light must not be confidently misread.

``unknown`` is a real answer. A lamp too small, too dim or too blown out to read
returns it rather than a guess, because a wrong ``green`` is far worse than no
reading at all: it would suppress a red-light warning.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: Minimum box height in pixels before a colour is attempted at all. Below this
#: the lamp is a couple of pixels and hue is meaningless.
MIN_HEIGHT_PX = 10

#: Saturation and value floors for a pixel to count as "lit". A traffic lamp
#: against a dark housing is both saturated and bright; the housing is neither.
MIN_SATURATION = 90
MIN_VALUE = 90

#: Fraction of the box that must be lit before any reading is trusted.
MIN_LIT_FRACTION = 0.02

#: OpenCV hue bands (0-179). Red occupies both ends of the circle.
#:
#: Amber is kept deliberately narrow. A wide amber band swallows two very common
#: things that are not lamps: sunlit tan or brass housings, and the warm halo
#: around a red lamp at night. Both were observed misreading green lights as
#: amber before the band was tightened and the scoring made brightness-weighted.
HUE_BANDS: dict[str, tuple[tuple[int, int], ...]] = {
    "red": ((0, 9), (168, 179)),
    "amber": ((16, 30),),
    "green": ((45, 95),),
}

#: Expected vertical centre of each lamp within the housing, as a fraction of
#: box height. Used as a soft prior.
EXPECTED_POSITION: dict[str, float] = {"red": 0.2, "amber": 0.5, "green": 0.8}

#: How strongly position is allowed to influence the decision. Deliberately
#: modest: horizontal housings exist, and position must inform rather than rule.
POSITION_WEIGHT = 0.35


@dataclass(frozen=True)
class LightState:
    """What a traffic light is showing.

    Attributes:
        colour: ``red``, ``amber``, ``green`` or ``unknown``.
        confidence: 0-1. Combines how much of the box is lit with how cleanly one
            hue band dominates.
        lit_fraction: Share of box pixels that were lit at all.
        scores: Per-colour score, for auditing a surprising reading.
    """

    colour: str = "unknown"
    confidence: float = 0.0
    lit_fraction: float = 0.0
    scores: dict[str, float] | None = None

    @property
    def is_red(self) -> bool:
        return self.colour == "red"

    @property
    def is_actionable(self) -> bool:
        """Whether this reading is solid enough to raise a warning on."""
        return self.colour != "unknown" and self.confidence >= 0.4


def _band_mask(hue: np.ndarray, bands: tuple[tuple[int, int], ...]) -> np.ndarray:
    mask = np.zeros(hue.shape, dtype=bool)
    for lo, hi in bands:
        mask |= (hue >= lo) & (hue <= hi)
    return mask


def classify_light(crop: np.ndarray) -> LightState:
    """Classify one traffic-light crop, given in BGR.

    Args:
        crop: Pixels inside the detected box.

    Returns:
        The lamp state. ``unknown`` whenever the evidence is too weak, which is a
        deliberate answer rather than a failure.
    """
    if crop is None or crop.size == 0:
        return LightState()
    h, w = crop.shape[:2]
    if h < MIN_HEIGHT_PX or w < 3:
        return LightState(lit_fraction=0.0)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (sat >= MIN_SATURATION) & (val >= MIN_VALUE)
    lit_fraction = float(lit.mean())
    if lit_fraction < MIN_LIT_FRACTION:
        return LightState(lit_fraction=lit_fraction)

    rows = np.arange(h, dtype=np.float64)[:, None]
    # Weight each pixel by how strongly it is *emitting*, not merely by being
    # present. A lit lamp is far brighter and more saturated than a housing the
    # sun happens to be falling on, so counting pixels lets a large dull region
    # outvote the small bright one that actually matters. This is the fix for
    # green lamps in tan housings reading as amber.
    energy = (val.astype(np.float64) / 255.0) ** 2 * (sat.astype(np.float64) / 255.0)
    energy = np.where(lit, energy, 0.0)
    total_energy = float(energy.sum())
    if total_energy <= 1e-9:
        return LightState(lit_fraction=lit_fraction)

    scores: dict[str, float] = {}
    for colour, bands in HUE_BANDS.items():
        mask = _band_mask(hue, bands) & lit
        if not mask.any():
            scores[colour] = 0.0
            continue
        band_energy = float(energy[mask].sum())
        hue_score = band_energy / total_energy
        # Where the emitting pixels sit vertically, against where this lamp
        # belongs in the housing. Weighted by energy for the same reason.
        w = energy[mask]
        centre = float(
            (np.broadcast_to(rows, mask.shape)[mask] * w).sum()
            / max(1e-9, w.sum()) / max(1, h - 1)
        )
        position_score = 1.0 - min(1.0, abs(centre - EXPECTED_POSITION[colour]) / 0.5)
        scores[colour] = float(
            (1.0 - POSITION_WEIGHT) * hue_score + POSITION_WEIGHT * position_score * hue_score
        )

    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    if total <= 1e-9 or scores[best] <= 1e-9:
        return LightState(lit_fraction=lit_fraction, scores=scores)

    # Confidence rewards one band dominating, and is damped when barely any of
    # the box is lit -- a handful of pixels agreeing proves little.
    dominance = scores[best] / total
    coverage = min(1.0, lit_fraction / 0.15)
    confidence = float(dominance * coverage)
    colour = best if confidence >= 0.25 else "unknown"
    return LightState(colour=colour, confidence=confidence,
                      lit_fraction=lit_fraction, scores=scores)


def classify_lights(
    frame: np.ndarray, boxes_px: np.ndarray, pad: int = 1
) -> list[LightState]:
    """Classify every traffic-light box in a frame."""
    out: list[LightState] = []
    h, w = frame.shape[:2]
    for box in np.asarray(boxes_px, dtype=np.float64).reshape(-1, 4):
        x0 = max(0, int(round(box[0])) - pad)
        y0 = max(0, int(round(box[1])) - pad)
        x1 = min(w, int(round(box[2])) + pad)
        y1 = min(h, int(round(box[3])) + pad)
        out.append(classify_light(frame[y0:y1, x0:x1]) if x1 > x0 and y1 > y0
                   else LightState())
    return out
