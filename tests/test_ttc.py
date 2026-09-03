"""Tests for time-to-collision.

The scale-expansion estimator is checked against an *analytic* ground truth: a
synthetic object closing at constant speed has an exactly known TTC at every
frame, so these assert real numbers rather than "it returned something".

The cases that matter most are the ones where a naive implementation returns a
plausible-looking wrong answer: a receding object, a stationary one, and an
object whose motion does not fit the constant-velocity model at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import TTCConfig
from tracking.ttc import (
    NO_COLLISION,
    combine,
    range_rate_ttc,
    scale_expansion_ttc,
)

FOCAL_TIMES_HEIGHT = 900.0 * 1.5      # f * real object height, in px*m
FPS = 25.0


def closing_history(z0: float, speed_mps: float, n: int, fps: float = FPS):
    """``(frame, t, height_px)`` for an object closing at constant speed.

    Height follows the pinhole relation h = f*H/Z exactly, so the TTC implied at
    any frame is analytically Z/speed.
    """
    out = []
    for i in range(n):
        t = i / fps
        z = z0 - speed_mps * t
        out.append((i, t, FOCAL_TIMES_HEIGHT / z))
    return out


class TestScaleExpansion:
    def test_matches_analytic_ttc_for_constant_closing_speed(self) -> None:
        hist = closing_history(z0=30.0, speed_mps=10.0, n=15)
        ttc, n, r2 = scale_expansion_ttc(hist, window=15, min_samples=4)
        t_last = hist[-1][1]
        expected = (30.0 - 10.0 * t_last) / 10.0
        assert ttc == pytest.approx(expected, rel=1e-6)
        assert r2 == pytest.approx(1.0, abs=1e-9)
        assert n == 15

    @pytest.mark.parametrize("speed", [5.0, 10.0, 25.0])
    def test_accurate_across_closing_speeds(self, speed: float) -> None:
        hist = closing_history(z0=60.0, speed_mps=speed, n=12)
        ttc, _, _ = scale_expansion_ttc(hist, window=12, min_samples=4)
        expected = (60.0 - speed * hist[-1][1]) / speed
        assert ttc == pytest.approx(expected, rel=1e-6)

    def test_a_receding_object_never_collides(self) -> None:
        """Box shrinking: 1/h rising. Must be infinity, not a negative time."""
        hist = [(i, i / FPS, FOCAL_TIMES_HEIGHT / (20.0 + 10.0 * i / FPS)) for i in range(10)]
        ttc, _, _ = scale_expansion_ttc(hist, 10, 4)
        assert ttc == NO_COLLISION

    def test_a_stationary_object_never_collides(self) -> None:
        hist = [(i, i / FPS, 100.0) for i in range(10)]
        assert scale_expansion_ttc(hist, 10, 4)[0] == NO_COLLISION

    def test_too_few_samples_gives_no_estimate(self) -> None:
        hist = closing_history(30.0, 10.0, 3)
        assert scale_expansion_ttc(hist, 15, min_samples=4)[0] == NO_COLLISION

    def test_zero_height_is_rejected_rather_than_dividing(self) -> None:
        hist = [(i, i / FPS, 0.0) for i in range(6)]
        ttc, _, _ = scale_expansion_ttc(hist, 6, 4)
        assert ttc == NO_COLLISION

    def test_erratic_motion_scores_low_fit_quality(self) -> None:
        """A box jumping around does not fit the model, and must say so."""
        rng = np.random.default_rng(0)
        hist = [(i, i / FPS, 100.0 + rng.uniform(-40, 40)) for i in range(15)]
        _, _, r2 = scale_expansion_ttc(hist, 15, 4)
        assert r2 < 0.5

    def test_ttc_shrinks_as_the_object_approaches(self) -> None:
        hist = closing_history(40.0, 10.0, 30)
        early, _, _ = scale_expansion_ttc(hist[:10], 10, 4)
        late, _, _ = scale_expansion_ttc(hist[:25], 15, 4)
        assert late < early

    def test_never_returns_a_negative_time(self) -> None:
        """Past the extrapolated contact point the answer is 0, not negative."""
        hist = [(i, i / FPS, 1e6 / max(1e-3, 2.0 - 1.0 * i / FPS)) for i in range(12)]
        ttc, _, _ = scale_expansion_ttc(hist, 12, 4)
        assert ttc >= 0.0


class TestRangeRate:
    def test_constant_closing_range_gives_correct_ttc(self) -> None:
        samples = [(i / FPS, 50.0 - 12.0 * (i / FPS)) for i in range(15)]
        ttc, d_now, rate = range_rate_ttc(samples, 15, 4)
        expected_d = 50.0 - 12.0 * samples[-1][0]
        assert d_now == pytest.approx(expected_d, abs=1e-6)
        assert rate == pytest.approx(-12.0, abs=1e-6)
        assert ttc == pytest.approx(expected_d / 12.0, rel=1e-6)

    def test_a_widening_gap_never_collides(self) -> None:
        samples = [(i / FPS, 20.0 + 5.0 * (i / FPS)) for i in range(10)]
        assert range_rate_ttc(samples, 10, 4)[0] == NO_COLLISION

    def test_constant_range_never_collides(self) -> None:
        """Following at a fixed distance is the common motorway case."""
        samples = [(i / FPS, 25.0) for i in range(10)]
        assert range_rate_ttc(samples, 10, 4)[0] == NO_COLLISION

    def test_too_few_samples_still_reports_current_distance(self) -> None:
        ttc, d, rate = range_rate_ttc([(0.0, 31.0)], 10, 4)
        assert ttc == NO_COLLISION and d == 31.0 and rate is None


class TestCombine:
    CFG = TTCConfig()

    def test_range_is_preferred_close_in(self) -> None:
        e = combine(scale_ttc=3.0, range_ttc=2.0, distance_m=10.0, range_rate=-5.0,
                    n_samples=10, fit_quality=1.0, min_fit_quality=0.5,
                    prefer_range_within_m=30.0)
        assert e.source == "range" and e.ttc_s == 2.0

    def test_scale_is_preferred_far_out(self) -> None:
        """Beyond the range limit, pixel quantisation makes metres unreliable."""
        e = combine(3.0, 2.0, distance_m=45.0, range_rate=-5.0, n_samples=10,
                    fit_quality=1.0, min_fit_quality=0.5, prefer_range_within_m=30.0)
        assert e.source == "scale" and e.ttc_s == 3.0

    def test_a_poor_fit_discards_the_scale_estimate(self) -> None:
        e = combine(3.0, NO_COLLISION, distance_m=None, range_rate=None,
                    n_samples=10, fit_quality=0.1, min_fit_quality=0.5,
                    prefer_range_within_m=30.0)
        assert e.ttc_s == NO_COLLISION and e.source == "none"

    def test_disagreement_is_recorded_not_averaged(self) -> None:
        """Averaging would hide exactly the case worth flagging."""
        e = combine(6.0, 2.0, distance_m=10.0, range_rate=-5.0, n_samples=10,
                    fit_quality=1.0, min_fit_quality=0.5, prefer_range_within_m=30.0)
        assert e.disagreement_s == pytest.approx(4.0)
        assert e.ttc_s in (2.0, 6.0), "must pick one, never the mean"

    def test_falls_back_to_range_when_scale_is_unusable(self) -> None:
        e = combine(NO_COLLISION, 2.0, distance_m=50.0, range_rate=-5.0,
                    n_samples=10, fit_quality=1.0, min_fit_quality=0.5,
                    prefer_range_within_m=30.0)
        assert e.source == "range" and e.ttc_s == 2.0

    def test_no_evidence_means_no_collision(self) -> None:
        e = combine(NO_COLLISION, NO_COLLISION, None, None, 0, 0.0, 0.5, 30.0)
        assert e.ttc_s == NO_COLLISION
        assert not e.is_approaching
