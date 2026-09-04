"""Tests for traffic-light state reading and the red-light rule.

The asymmetry matters throughout: a missed red is a nuisance, a false green is a
safety failure. So the tests concentrate on `unknown` being returned whenever the
evidence is weak, and on red never being silently downgraded.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.config import FusionConfig
from core.types import ROAD_USER_CLASSES, CLASS_IDS, CLASS_NAMES
from fusion import AlertType, DecisionEngine, Severity
from models.perception.traffic_light import (
    LightState,
    classify_light,
    classify_lights,
)
from tracking.collision import ObjectState
from tracking.ttc import NO_COLLISION, TTCEstimate


def lamp(colour: str, h: int = 44, w: int = 18, housing: int = 18) -> np.ndarray:
    """A synthetic housing with one lamp lit at its correct position."""
    img = np.full((h, w, 3), housing, np.uint8)
    y = {"red": 0.2, "amber": 0.5, "green": 0.8}[colour]
    bgr = {"red": (0, 0, 255), "amber": (0, 170, 255), "green": (0, 230, 0)}[colour]
    cv2.circle(img, (w // 2, int(y * h)), max(2, w // 3), bgr, -1)
    return img


class TestColourReading:
    @pytest.mark.parametrize("colour", ["red", "amber", "green"])
    def test_each_lamp_is_read_correctly(self, colour: str) -> None:
        st = classify_light(lamp(colour))
        assert st.colour == colour
        assert st.is_actionable

    def test_red_is_flagged(self) -> None:
        assert classify_light(lamp("red")).is_red
        assert not classify_light(lamp("green")).is_red

    def test_a_dark_housing_is_unknown_not_a_guess(self) -> None:
        """An unlit light must never resolve to a colour."""
        st = classify_light(np.full((44, 18, 3), 12, np.uint8))
        assert st.colour == "unknown"
        assert not st.is_actionable

    def test_a_tiny_box_is_unknown(self) -> None:
        assert classify_light(np.zeros((5, 4, 3), np.uint8)).colour == "unknown"

    def test_an_empty_crop_is_unknown(self) -> None:
        assert classify_light(np.zeros((0, 0, 3), np.uint8)).colour == "unknown"
        assert classify_light(None).colour == "unknown"

    def test_a_sunlit_tan_housing_does_not_read_as_amber(self) -> None:
        """The observed failure: a bright housing outvoting the actual lamp."""
        img = np.full((44, 18, 3), 18, np.uint8)
        img[:, :] = (150, 190, 215)                       # warm sunlit metal
        cv2.circle(img, (9, 35), 6, (0, 230, 0), -1)      # green lamp, lit
        assert classify_light(img).colour == "green"

    def test_position_disambiguates_a_washed_out_lamp(self) -> None:
        """Hue is least reliable at night; position still carries information."""
        top = classify_light(lamp("red"))
        bottom = classify_light(lamp("green"))
        assert top.colour == "red" and bottom.colour == "green"

    def test_confidence_is_reported(self) -> None:
        st = classify_light(lamp("red"))
        assert 0.0 <= st.confidence <= 1.0
        assert st.scores is not None and set(st.scores) == {"red", "amber", "green"}


class TestBatchReading:
    def test_boxes_are_read_in_order(self) -> None:
        frame = np.full((200, 200, 3), 15, np.uint8)
        frame[10:54, 20:38] = lamp("red")
        frame[10:54, 60:78] = lamp("green")
        out = classify_lights(frame, np.array([[20, 10, 38, 54], [60, 10, 78, 54]]))
        assert [s.colour for s in out] == ["red", "green"]

    def test_a_box_outside_the_frame_is_unknown(self) -> None:
        frame = np.full((100, 100, 3), 15, np.uint8)
        assert classify_lights(frame, np.array([[400, 400, 450, 450]]))[0].colour == "unknown"

    def test_no_boxes_gives_no_readings(self) -> None:
        assert classify_lights(np.zeros((50, 50, 3), np.uint8), np.zeros((0, 4))) == []


def light_obj(colour: str, conf: float = 0.9, distance=20.0, rate=-5.0) -> ObjectState:
    return ObjectState(
        track_id=7, class_name="traffic_light",
        box_px=np.array([100.0, 40, 118, 84]), score=0.8,
        distance_m=distance, lateral_offset_m=0.2,
        ttc=TTCEstimate(ttc_s=NO_COLLISION, source="range", range_rate_mps=rate),
        in_path=True,
        light_state=LightState(colour=colour, confidence=conf),
    )


class TestRedLightRule:
    def test_an_approaching_red_warns(self) -> None:
        e = DecisionEngine(FusionConfig())
        d = e.update(0, 0.0, [light_obj("red")], _lanes())
        assert d.alerts[0].type is AlertType.RED_LIGHT_RUNNING
        assert d.alerts[0].severity is Severity.CRITICAL

    def test_green_does_not_warn(self) -> None:
        e = DecisionEngine(FusionConfig())
        assert e.update(0, 0.0, [light_obj("green")], _lanes()).alerts == ()

    def test_an_unreadable_light_does_not_warn(self) -> None:
        """`unknown` must never be treated as red."""
        e = DecisionEngine(FusionConfig())
        assert e.update(0, 0.0, [light_obj("unknown", conf=0.0)], _lanes()).alerts == ()

    def test_a_low_confidence_red_does_not_warn(self) -> None:
        e = DecisionEngine(FusionConfig())
        assert e.update(0, 0.0, [light_obj("red", conf=0.1)], _lanes()).alerts == ()

    def test_a_distant_red_does_not_warn(self) -> None:
        e = DecisionEngine(FusionConfig(red_light_distance_m=45.0))
        assert e.update(0, 0.0, [light_obj("red", distance=120.0)], _lanes()).alerts == ()

    def test_a_red_already_being_stopped_for_does_not_warn(self) -> None:
        """Braking for a red is normal driving, and warning about it is noise."""
        e = DecisionEngine(FusionConfig())
        assert e.update(0, 0.0, [light_obj("red", rate=-0.2)], _lanes()).alerts == ()

    def test_the_rule_can_be_disabled(self) -> None:
        e = DecisionEngine(FusionConfig(warn_on_red_light=False))
        assert e.update(0, 0.0, [light_obj("red")], _lanes()).alerts == ()

    def test_the_alert_records_its_evidence(self) -> None:
        e = DecisionEngine(FusionConfig())
        ev = e.update(0, 0.0, [light_obj("red")], _lanes()).alerts[0].evidence
        assert ev["light_colour"] == "red"
        assert ev["distance_m"] == pytest.approx(20.0)


class TestSignalsAreNotObstacles:
    def test_traffic_control_is_excluded_from_collision(self) -> None:
        """Otherwise the system warns about hitting a signal gantry."""
        e = DecisionEngine(FusionConfig())
        obj = ObjectState(
            track_id=1, class_name="traffic_light",
            box_px=np.array([0.0, 0, 10, 10]), score=0.9,
            distance_m=6.0, lateral_offset_m=0.0,
            ttc=TTCEstimate(ttc_s=0.4, source="range"), in_path=True,
            light_state=LightState(colour="green", confidence=0.9),
        )
        types = {a.type for a in e.update(0, 0.0, [obj], _lanes()).alerts}
        assert AlertType.FORWARD_COLLISION not in types

    def test_the_road_user_set_excludes_traffic_control(self) -> None:
        assert "traffic_light" not in ROAD_USER_CLASSES
        assert "traffic_sign" not in ROAD_USER_CLASSES
        assert ROAD_USER_CLASSES < set(CLASS_NAMES)

    def test_traffic_classes_are_appended_not_inserted(self) -> None:
        """Existing ids must keep their meaning across the schema bump."""
        assert CLASS_IDS["vehicle"] == 0
        assert CLASS_IDS["person"] == 1
        assert CLASS_IDS["traffic_light"] == 4
        assert CLASS_IDS["traffic_sign"] == 5


def _lanes():
    from core.types import LaneSet, LaneStatus
    return LaneSet(status=LaneStatus.NOT_DETECTED)
