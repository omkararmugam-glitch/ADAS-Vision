"""Tests for road-split (fork / exit ramp) detection.

Built on synthetic lane geometry where the answer is known by construction: a
parallel road must never report a split, and a boundary that peels away must.

The failure this guards is silent. Across a gore the widening gap reads as one
very wide lane, so lane_width_m grows past anything plausible and the ego offset
the lane-departure rule depends on stops meaning anything -- with nothing
raised. The system simply believes it is centred in a nine-metre lane.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.types import LaneFit, LaneSet, LaneStatus
from models.lane_seg.split import RoadSplit, SplitSide, SplitTracker, detect_split

NEAR, FAR = 6.0, 34.0


def boundary(role: str, c: float, b: float = 0.0, a: float = 0.0) -> LaneFit:
    """x = a*y^2 + b*y + c, over a range wide enough to test divergence."""
    return LaneFit(coeffs=np.array([a, b, c]), role=role,
                   n_pixels=400, y_range_m=(NEAR, FAR))


def lane_set(*fits: LaneFit, status: LaneStatus = LaneStatus.OK) -> LaneSet:
    ego = [f for f in fits if f.role.startswith("ego")]
    adj = tuple(f for f in fits if not f.role.startswith("ego"))
    return LaneSet(
        status=status,
        ego_left=next((f for f in ego if "left" in f.role), None),
        ego_right=next((f for f in ego if "right" in f.role), None),
        adjacent=adj, eval_distance_m=10.0,
    )


class TestNoFalsePositives:
    def test_a_parallel_road_reports_no_split(self) -> None:
        lanes = lane_set(boundary("ego_left", -1.75), boundary("ego_right", 1.75))
        assert detect_split(lanes) is None

    def test_a_gently_curving_road_reports_no_split(self) -> None:
        """Both boundaries curve together, so separation stays constant."""
        lanes = lane_set(
            boundary("ego_left", -1.75, b=0.02, a=0.001),
            boundary("ego_right", 1.75, b=0.02, a=0.001),
        )
        assert detect_split(lanes) is None

    def test_a_single_boundary_cannot_diverge(self) -> None:
        assert detect_split(lane_set(boundary("ego_left", -1.75))) is None

    def test_no_lane_means_no_split(self) -> None:
        assert detect_split(LaneSet(status=LaneStatus.NOT_DETECTED)) is None

    def test_slight_lane_widening_is_tolerated(self) -> None:
        """Real lanes widen a little near junctions; that is not a fork."""
        lanes = lane_set(boundary("ego_left", -1.75, b=-0.01),
                         boundary("ego_right", 1.75, b=0.01))
        assert detect_split(lanes) is None


class TestDetection:
    def test_a_diverging_pair_is_detected(self) -> None:
        """Right boundary peels away: the classic exit-ramp gore."""
        lanes = lane_set(boundary("ego_left", -1.75),
                         boundary("ego_right", 1.75, b=0.12))
        split = detect_split(lanes)
        assert split is not None
        assert split.divergence_ratio > 1.6
        assert split.widening_m > 1.5

    def test_the_diverging_side_is_reported(self) -> None:
        right = detect_split(lane_set(boundary("ego_left", -1.75),
                                      boundary("ego_right", 1.75, b=0.12)))
        left = detect_split(lane_set(boundary("ego_left", -1.75, b=-0.12),
                                     boundary("ego_right", 1.75)))
        assert right is not None and left is not None
        assert right.side is SplitSide.RIGHT
        assert left.side is SplitSide.LEFT

    def test_the_gore_distance_is_within_the_fitted_range(self) -> None:
        split = detect_split(lane_set(boundary("ego_left", -1.75),
                                      boundary("ego_right", 1.75, b=0.12)))
        assert split is not None
        assert NEAR <= split.gore_distance_m <= FAR

    def test_separations_are_reported_and_ordered(self) -> None:
        split = detect_split(lane_set(boundary("ego_left", -1.75),
                                      boundary("ego_right", 1.75, b=0.12)))
        assert split is not None
        assert split.far_separation_m > split.near_separation_m
        assert split.divergence_ratio == pytest.approx(
            split.far_separation_m / split.near_separation_m)

    def test_the_strongest_divergence_wins(self) -> None:
        """With several boundaries, the most clearly forking pair is reported."""
        lanes = lane_set(
            boundary("ego_left", -1.75), boundary("ego_right", 1.75),
            boundary("adjacent_right", 5.25, b=0.25),
        )
        split = detect_split(lanes)
        assert split is not None
        assert "adjacent_right" in (split.left_role, split.right_role)

    def test_boundaries_already_far_apart_are_not_a_fork(self) -> None:
        """Two separate carriageways are not a road dividing."""
        lanes = lane_set(boundary("ego_left", -9.0),
                         boundary("ego_right", 9.0, b=0.2))
        assert detect_split(lanes, max_plausible_near_m=6.0) is None


class TestDebouncing:
    def _split(self, ratio: float = 2.0) -> RoadSplit:
        return RoadSplit(side=SplitSide.RIGHT, near_separation_m=3.5,
                         far_separation_m=3.5 * ratio, divergence_ratio=ratio,
                         gore_distance_m=20.0, left_role="ego_left",
                         right_role="ego_right")

    def test_one_frame_does_not_confirm(self) -> None:
        """A single bad lane fit has exactly the signature of a fork."""
        t = SplitTracker(min_frames=4, window=8)
        assert t.update(self._split()) is None

    def test_sustained_divergence_confirms(self) -> None:
        t = SplitTracker(min_frames=4, window=8)
        out = [t.update(self._split()) for _ in range(6)]
        assert sum(o is not None for o in out) == 1

    def test_it_reports_once_not_every_frame(self) -> None:
        t = SplitTracker(min_frames=3, window=8, cooldown=20)
        fired = sum(t.update(self._split()) is not None for _ in range(30))
        assert fired == 1

    def test_intermittent_noise_never_confirms(self) -> None:
        t = SplitTracker(min_frames=5, window=8)
        out = [t.update(self._split() if i % 3 == 0 else None) for i in range(24)]
        assert all(o is None for o in out)

    def test_the_clearest_frame_is_reported(self) -> None:
        """The latest frame is not necessarily the best view of the fork."""
        t = SplitTracker(min_frames=3, window=8)
        t.update(self._split(1.8))
        t.update(self._split(3.4))
        got = t.update(self._split(1.7))
        assert got is not None and got.divergence_ratio == pytest.approx(3.4)

    def test_an_impossible_config_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="never be met"):
            SplitTracker(min_frames=10, window=4)

    def test_reset_clears_state(self) -> None:
        t = SplitTracker(min_frames=2, window=4)
        t.update(self._split())
        t.reset()
        assert t.update(self._split()) is None


class TestEngineIntegration:
    def _lanes_forking(self) -> LaneSet:
        return lane_set(boundary("ego_left", -1.75),
                        boundary("ego_right", 1.75, b=0.12))

    def test_a_sustained_fork_raises_one_alert(self) -> None:
        from core.config import FusionConfig, LaneChangeConfig
        from fusion import AlertType, DecisionEngine, Severity

        cfg = FusionConfig(lane_change=LaneChangeConfig(
            split_min_frames=3, split_window=6, split_cooldown_frames=20))
        engine = DecisionEngine(cfg)
        fired = []
        for i in range(20):
            d = engine.update(i, i / 30, [], self._lanes_forking())
            fired += [a for a in d.alerts if a.type is AlertType.ROAD_SPLIT]
        assert len(fired) == 1
        assert fired[0].severity is Severity.INFO
        assert "splits" in fired[0].message.lower()
        assert fired[0].evidence["side"] == "right"

    def test_a_straight_road_raises_nothing(self) -> None:
        from core.config import FusionConfig
        from fusion import AlertType, DecisionEngine

        engine = DecisionEngine(FusionConfig())
        lanes = lane_set(boundary("ego_left", -1.75), boundary("ego_right", 1.75))
        for i in range(20):
            d = engine.update(i, i / 30, [], lanes)
            assert not [a for a in d.alerts if a.type is AlertType.ROAD_SPLIT]

    def test_it_can_be_disabled(self) -> None:
        from core.config import FusionConfig, LaneChangeConfig
        from fusion import AlertType, DecisionEngine

        engine = DecisionEngine(FusionConfig(
            lane_change=LaneChangeConfig(detect_split=False)))
        for i in range(20):
            d = engine.update(i, i / 30, [], self._lanes_forking())
            assert not [a for a in d.alerts if a.type is AlertType.ROAD_SPLIT]


class TestSideConsistency:
    """The reported side must not contradict what the frames actually show.

    Reporting the single strongest frame let the alert say "right branch" while
    the HUD, reading the current frame, said "left" -- two answers to the same
    question on one screen.
    """

    @staticmethod
    def _s(side: SplitSide, ratio: float) -> RoadSplit:
        return RoadSplit(side=side, near_separation_m=3.5,
                         far_separation_m=3.5 * ratio, divergence_ratio=ratio,
                         gore_distance_m=20.0, left_role="ego_left",
                         right_role="ego_right")

    def test_the_majority_side_wins_over_the_strongest_frame(self) -> None:
        """One very strong outlier must not overrule three agreeing frames."""
        t = SplitTracker(min_frames=4, window=8)
        t.update(self._s(SplitSide.LEFT, 1.9))
        t.update(self._s(SplitSide.LEFT, 2.0))
        t.update(self._s(SplitSide.RIGHT, 9.0))     # strongest, but outnumbered
        got = t.update(self._s(SplitSide.LEFT, 2.1))
        assert got is not None
        assert got.side is SplitSide.LEFT

    def test_the_clearest_frame_of_the_majority_is_reported(self) -> None:
        t = SplitTracker(min_frames=3, window=8)
        t.update(self._s(SplitSide.LEFT, 1.8))
        t.update(self._s(SplitSide.LEFT, 3.1))
        got = t.update(self._s(SplitSide.LEFT, 2.0))
        assert got is not None
        assert got.divergence_ratio == pytest.approx(3.1)

    def test_a_unanimous_side_is_unchanged(self) -> None:
        t = SplitTracker(min_frames=3, window=8)
        for _ in range(3):
            got = t.update(self._s(SplitSide.RIGHT, 2.0))
        assert got is not None and got.side is SplitSide.RIGHT
