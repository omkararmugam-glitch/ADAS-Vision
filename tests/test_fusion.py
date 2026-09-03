"""Tests for lane-change detection and the rules-based decision engine.

These are the rules a driver would actually be shown, so they are tested
directly rather than through the pipeline: no video, no model, no GPU. The
cases chosen are the ones where a plausible implementation is silently wrong --
a stationary pedestrian (infinite TTC, still a hazard), a wobbling lane fit that
must not fire, and weather that has to move the threshold in the right
direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import FusionConfig, LaneChangeConfig, TTCConfig
from core.types import LaneFit, LaneSet, LaneStatus
from fusion import AlertType, DecisionEngine, LaneChangeDetector, LaneEvent, Severity
from tracking.collision import CollisionEstimator, ObjectState
from tracking.ttc import NO_COLLISION, TTCEstimate


def lane_set(offset_m: float | None, width_m: float | None = 3.5,
             status: LaneStatus = LaneStatus.OK) -> LaneSet:
    """A LaneSet carrying just the scalars the detector reads."""
    def fit(role: str, c: float) -> LaneFit:
        return LaneFit(coeffs=np.array([0.0, 0.0, c]), role=role,
                       n_pixels=400, y_range_m=(5.0, 30.0))

    half = (width_m / 2.0) if width_m else 1.75
    return LaneSet(
        status=status,
        ego_left=fit("ego_left", -half) if status is LaneStatus.OK else None,
        ego_right=fit("ego_right", half) if status is LaneStatus.OK else None,
        ego_offset_m=offset_m, lane_width_m=width_m, eval_distance_m=10.0,
    )


def obj(track_id=1, name="vehicle", ttc_s=NO_COLLISION, distance_m=None,
        in_path=True, lateral=0.0) -> ObjectState:
    return ObjectState(
        track_id=track_id, class_name=name,
        box_px=np.array([100.0, 100, 200, 200]), score=0.9,
        distance_m=distance_m, lateral_offset_m=lateral,
        ttc=TTCEstimate(ttc_s=ttc_s, source="scale", distance_m=distance_m,
                        fit_quality=1.0, n_samples=10),
        in_path=in_path,
    )


# --------------------------------------------------------------------------- #
# lane events
# --------------------------------------------------------------------------- #


class TestLaneDeparture:
    def test_centred_driving_raises_nothing(self) -> None:
        d = LaneChangeDetector(LaneChangeConfig())
        for _ in range(20):
            assert d.update(lane_set(0.05)).event is LaneEvent.NONE

    def test_a_sustained_drift_right_fires_once(self) -> None:
        cfg = LaneChangeConfig(min_departure_frames=3)
        d = LaneChangeDetector(cfg)
        events = [d.update(lane_set(1.4)).event for _ in range(10)]
        assert events.count(LaneEvent.DEPARTURE_RIGHT) == 1, "must not re-fire every frame"
        assert events.index(LaneEvent.DEPARTURE_RIGHT) == 2

    def test_a_sustained_drift_left_fires_left(self) -> None:
        d = LaneChangeDetector(LaneChangeConfig(min_departure_frames=3))
        events = [d.update(lane_set(-1.4)).event for _ in range(6)]
        assert LaneEvent.DEPARTURE_LEFT in events

    def test_a_single_frame_wobble_does_not_fire(self) -> None:
        """One bad lane fit must never raise a departure warning."""
        d = LaneChangeDetector(LaneChangeConfig(min_departure_frames=3))
        d.update(lane_set(0.0))
        assert d.update(lane_set(1.6)).event is LaneEvent.NONE
        assert d.update(lane_set(0.0)).event is LaneEvent.NONE

    def test_hysteresis_prevents_retriggering_at_the_threshold(self) -> None:
        """Sitting near the limit must not fire repeatedly."""
        cfg = LaneChangeConfig(min_departure_frames=2, departure_threshold=0.65,
                               recover_threshold=0.45)
        d = LaneChangeDetector(cfg)
        half = 3.5 / 2
        fires = 0
        for u in [0.7, 0.7, 0.7, 0.6, 0.7, 0.6, 0.7, 0.7]:
            if d.update(lane_set(u * half)).event is LaneEvent.DEPARTURE_RIGHT:
                fires += 1
        assert fires == 1

    def test_an_implausible_lane_width_is_rejected(self) -> None:
        """A mis-locked fit must not produce a confident offset."""
        d = LaneChangeDetector(LaneChangeConfig())
        for _ in range(8):
            st = d.update(lane_set(1.4, width_m=12.0))
            assert st.event is LaneEvent.NONE
            assert st.normalised_offset is None

    def test_no_lane_fit_yields_no_offset(self) -> None:
        d = LaneChangeDetector(LaneChangeConfig())
        st = d.update(lane_set(None, None, LaneStatus.NOT_DETECTED))
        assert st.normalised_offset is None and st.event is LaneEvent.NONE

    def test_a_brief_gap_does_not_clear_a_departure(self) -> None:
        """Markings vanish for a few frames constantly; the drift is still real."""
        cfg = LaneChangeConfig(min_departure_frames=3, max_gap_frames=5)
        d = LaneChangeDetector(cfg)
        for _ in range(4):
            d.update(lane_set(1.4))
        st = d.update(lane_set(None, None, LaneStatus.MASK_TOO_SPARSE))
        assert st.departing


class TestLaneChange:
    def test_a_crossing_to_the_right_is_detected(self) -> None:
        """Drift right, then the fitter re-locks and the offset flips sign."""
        cfg = LaneChangeConfig(min_departure_frames=3, cross_threshold=0.5)
        d = LaneChangeDetector(cfg)
        half = 3.5 / 2
        events = []
        for u in [0.1, 0.4, 0.7, 0.85, 0.95, -0.9, -0.6, -0.2]:
            events.append(d.update(lane_set(u * half)).event)
        assert LaneEvent.CHANGE_RIGHT in events

    def test_a_crossing_to_the_left_is_detected(self) -> None:
        cfg = LaneChangeConfig(min_departure_frames=3, cross_threshold=0.5)
        d = LaneChangeDetector(cfg)
        half = 3.5 / 2
        events = []
        for u in [-0.1, -0.4, -0.7, -0.85, -0.95, 0.9, 0.6, 0.2]:
            events.append(d.update(lane_set(u * half)).event)
        assert LaneEvent.CHANGE_LEFT in events

    def test_a_sign_flip_without_a_departure_is_not_a_change(self) -> None:
        """Noise around the centre must never read as a lane change."""
        d = LaneChangeDetector(LaneChangeConfig(min_departure_frames=3))
        half = 3.5 / 2
        events = [d.update(lane_set(u * half)).event
                  for u in [0.1, -0.1, 0.15, -0.15, 0.1, -0.1]]
        assert all(e is LaneEvent.NONE for e in events)

    def test_returning_to_centre_is_a_recovery_not_a_change(self) -> None:
        """Drifting out and correcting back is not a lane change."""
        cfg = LaneChangeConfig(min_departure_frames=3, cross_threshold=0.5)
        d = LaneChangeDetector(cfg)
        half = 3.5 / 2
        events = [d.update(lane_set(u * half)).event
                  for u in [0.2, 0.7, 0.9, 0.9, 0.6, 0.3, 0.05]]
        assert LaneEvent.CHANGE_RIGHT not in events
        assert LaneEvent.CHANGE_LEFT not in events

    def test_cooldown_prevents_an_immediate_second_change(self) -> None:
        cfg = LaneChangeConfig(min_departure_frames=2, cooldown_frames=10)
        d = LaneChangeDetector(cfg)
        half = 3.5 / 2
        seq = [0.7, 0.9, 0.95, -0.9, -0.95, 0.9, -0.9]
        events = [d.update(lane_set(u * half)).event for u in seq]
        changes = [e for e in events if e in (LaneEvent.CHANGE_LEFT, LaneEvent.CHANGE_RIGHT)]
        assert len(changes) == 1


# --------------------------------------------------------------------------- #
# decision engine
# --------------------------------------------------------------------------- #


class TestForwardCollision:
    def test_a_closing_vehicle_warns_then_escalates(self) -> None:
        e = DecisionEngine(FusionConfig(ttc_warning_s=2.5, ttc_critical_s=1.2))
        warn = e.update(0, 0.0, [obj(ttc_s=2.0, distance_m=20)], lane_set(0.0))
        crit = e.update(1, 0.04, [obj(ttc_s=0.9, distance_m=10)], lane_set(0.0))
        assert warn.alerts[0].type is AlertType.FORWARD_COLLISION
        assert warn.alerts[0].severity is Severity.WARNING
        assert crit.alerts[0].severity is Severity.CRITICAL

    def test_a_distant_vehicle_raises_nothing(self) -> None:
        e = DecisionEngine(FusionConfig())
        d = e.update(0, 0.0, [obj(ttc_s=9.0, distance_m=90)], lane_set(0.0))
        assert d.alerts == ()

    def test_a_non_approaching_vehicle_raises_nothing(self) -> None:
        """Following at a steady distance is the common motorway case."""
        e = DecisionEngine(FusionConfig())
        d = e.update(0, 0.0, [obj(ttc_s=NO_COLLISION, distance_m=15)], lane_set(0.0))
        assert d.alerts == ()

    def test_an_object_outside_the_corridor_is_ignored(self) -> None:
        """A car in the next lane with a low TTC is not our collision."""
        e = DecisionEngine(FusionConfig())
        d = e.update(0, 0.0, [obj(ttc_s=0.5, distance_m=8, in_path=False)], lane_set(0.0))
        assert d.alerts == ()

    def test_the_alert_carries_auditable_evidence(self) -> None:
        e = DecisionEngine(FusionConfig())
        d = e.update(0, 0.0, [obj(ttc_s=1.0, distance_m=9.0)], lane_set(0.0))
        ev = d.alerts[0].evidence
        assert ev["ttc_s"] == pytest.approx(1.0)
        assert ev["distance_m"] == pytest.approx(9.0)
        assert d.alerts[0].track_id == 1


class TestPedestrianRule:
    def test_a_stationary_pedestrian_in_path_still_alerts(self) -> None:
        """Infinite TTC -- nothing is closing -- but this must warn."""
        e = DecisionEngine(FusionConfig(pedestrian_distance_m=12.0))
        d = e.update(0, 0.0, [obj(name="person", ttc_s=NO_COLLISION, distance_m=8.0)],
                     lane_set(0.0))
        assert d.alerts[0].type is AlertType.PEDESTRIAN_IN_PATH
        assert d.alerts[0].severity is Severity.CRITICAL

    def test_a_far_stationary_pedestrian_does_not_alert(self) -> None:
        e = DecisionEngine(FusionConfig(pedestrian_distance_m=12.0))
        d = e.update(0, 0.0, [obj(name="person", ttc_s=NO_COLLISION, distance_m=40.0)],
                     lane_set(0.0))
        assert d.alerts == ()

    def test_an_approaching_pedestrian_alerts_on_ttc(self) -> None:
        e = DecisionEngine(FusionConfig())
        d = e.update(0, 0.0, [obj(name="person", ttc_s=2.0, distance_m=30.0)],
                     lane_set(0.0))
        assert d.alerts[0].type is AlertType.PEDESTRIAN_IN_PATH

    @pytest.mark.parametrize("name", ["person", "bicycle", "motorcycle"])
    def test_all_vulnerable_classes_use_the_proximity_rule(self, name: str) -> None:
        e = DecisionEngine(FusionConfig(pedestrian_distance_m=12.0))
        d = e.update(0, 0.0, [obj(name=name, ttc_s=NO_COLLISION, distance_m=6.0)],
                     lane_set(0.0))
        assert d.alerts[0].type is AlertType.PEDESTRIAN_IN_PATH

    def test_a_vehicle_does_not_use_the_proximity_rule(self) -> None:
        """A car stopped ahead at a junction is normal; a person there is not."""
        e = DecisionEngine(FusionConfig(pedestrian_distance_m=12.0))
        d = e.update(0, 0.0, [obj(name="vehicle", ttc_s=NO_COLLISION, distance_m=6.0)],
                     lane_set(0.0))
        assert d.alerts == ()


class TestWeatherAdaptation:
    def test_rain_widens_the_warning_threshold(self) -> None:
        e = DecisionEngine(FusionConfig(ttc_warning_s=2.5))
        e.set_weather("clear")
        assert e.ttc_thresholds()[0] == pytest.approx(2.5)
        e.set_weather("rain")
        assert e.ttc_thresholds()[0] > 2.5

    def test_a_ttc_safe_in_the_dry_alerts_in_the_rain(self) -> None:
        """The whole point: the same closing speed is more dangerous when wet."""
        cfg = FusionConfig(ttc_warning_s=2.5, ttc_critical_s=1.2)
        e = DecisionEngine(cfg)
        e.set_weather("clear")
        assert e.update(0, 0.0, [obj(ttc_s=3.2, distance_m=25)], lane_set(0.0)).alerts == ()
        e.set_weather("rain")
        assert e.update(1, 0.04, [obj(ttc_s=3.2, distance_m=25)], lane_set(0.0)).alerts != ()

    def test_an_unknown_weather_label_is_neutral(self) -> None:
        e = DecisionEngine(FusionConfig(ttc_warning_s=2.5))
        e.set_weather("hurricane")
        assert e.ttc_thresholds()[0] == pytest.approx(2.5)

    def test_snow_is_treated_as_worse_than_rain(self) -> None:
        e = DecisionEngine(FusionConfig())
        e.set_weather("rain")
        rain = e.ttc_thresholds()[0]
        e.set_weather("snow")
        assert e.ttc_thresholds()[0] > rain


class TestDecisionRecord:
    def test_the_worst_alert_is_surfaced_first(self) -> None:
        e = DecisionEngine(FusionConfig())
        d = e.update(0, 0.0, [obj(1, "vehicle", ttc_s=2.0, distance_m=20),
                              obj(2, "person", ttc_s=NO_COLLISION, distance_m=5)],
                     lane_set(0.0))
        assert d.alerts[0].severity is Severity.CRITICAL
        assert d.worst.type is AlertType.PEDESTRIAN_IN_PATH

    def test_the_record_serialises_to_json_safe_types(self) -> None:
        import json
        e = DecisionEngine(FusionConfig())
        d = e.update(3, 0.12, [obj(ttc_s=1.0, distance_m=9.0)], lane_set(0.2))
        blob = json.dumps(d.to_dict())
        assert '"forward_collision"' in blob
        assert json.loads(blob)["frame_index"] == 3

    def test_infinite_ttc_serialises_as_null_not_infinity(self) -> None:
        """json.dumps emits bare Infinity, which is not valid JSON."""
        import json
        e = DecisionEngine(FusionConfig(pedestrian_distance_m=12.0))
        d = e.update(0, 0.0, [obj(name="person", ttc_s=NO_COLLISION, distance_m=5.0)],
                     lane_set(0.0))
        text = json.dumps(d.to_dict())
        assert "Infinity" not in text
        assert json.loads(text)["alerts"][0]["evidence"]["ttc_s"] is None

    def test_counts_are_reported(self) -> None:
        e = DecisionEngine(FusionConfig())
        d = e.update(0, 0.0, [obj(1, in_path=True), obj(2, in_path=False)], lane_set(0.0))
        assert d.n_objects == 2 and d.n_in_path == 1


class TestCorridor:
    def test_the_corridor_widens_with_distance(self) -> None:
        """Lateral error grows with range, so the corridor must too."""
        c = CollisionEstimator(TTCConfig(corridor_half_width_m=1.6, corridor_widening=0.02))
        assert c.in_path(lateral_m=2.0, forward_m=50.0)     # inside at range
        assert not c.in_path(lateral_m=2.0, forward_m=5.0)  # outside up close

    def test_without_calibration_nothing_is_excluded(self) -> None:
        """Silently dropping objects would be the dangerous failure."""
        c = CollisionEstimator(TTCConfig())
        assert c.in_path(None, None)
