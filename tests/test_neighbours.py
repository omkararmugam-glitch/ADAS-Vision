"""Tests for lane enumeration and lane-change availability.

The output here is an *invitation*: orange says "you may move into this lane".
That makes the asymmetry of errors extreme. Calling an occupied lane free
invites a collision; calling a free lane occupied merely withholds a suggestion.
Most of these tests therefore push on the first kind, and several of them encode
bugs found by rendering real frames and looking at them:

* a lane painted to the horizon but checked for occupants only over the few
  metres its paint survived -- orange laid across three visible cars;
* the same at the near end, where a fit starting at 5.6 m ignored a car at
  4.3 m, the single most important vehicle in the frame for a lane change;
* a traffic light's box back-projected to the road, putting a phantom vehicle
  in whichever lane the gantry crossed.
"""

from __future__ import annotations

import numpy as np

from core.autocalib import focal_from_fov, homography_from_pose
from core.calibrate import CameraModel
from core.types import LaneFit, LaneSet, LaneStatus
from models.lane_seg.neighbours import (
    Lane,
    LaneKind,
    change_options,
    enumerate_lanes,
)

W, H = 1280, 720
NEAR, FAR = 5.0, 40.0


def camera() -> CameraModel:
    Hm = homography_from_pose(focal_from_fov(W, 90.0), (W / 2, H / 2),
                              np.deg2rad(6.0), 0.0, 1.35)
    return CameraModel(H_image_to_ground=Hm, image_size=(W, H))


def boundary(role: str, c: float, b: float = 0.0, a: float = 0.0,
             y_range=(NEAR, FAR)) -> LaneFit:
    return LaneFit(coeffs=np.array([a, b, c]), role=role, n_pixels=400,
                   y_range_m=y_range)


def lane_set(*fits: LaneFit) -> LaneSet:
    ego = [f for f in fits if f.role.startswith("ego")]
    return LaneSet(
        status=LaneStatus.OK,
        ego_left=next((f for f in ego if "left" in f.role), None),
        ego_right=next((f for f in ego if "right" in f.role), None),
        adjacent=tuple(f for f in fits if not f.role.startswith("ego")),
        lane_width_m=3.5, eval_distance_m=15.0, parallel_model=True)


class Obj:
    """Stand-in for ObjectState: only box_px and class_name are read."""

    def __init__(self, box, class_name: str = "vehicle") -> None:
        self.box_px = np.asarray(box, dtype=np.float64)
        self.class_name = class_name


def car_at(cam: CameraModel, x_m: float, y_m: float, width_m: float = 1.8,
           height_px: int = 60, class_name: str = "vehicle") -> Obj:
    """A detection box whose bottom edge sits at ground point (x, y)."""
    left = cam.ground_to_image(np.array([[x_m - width_m / 2, y_m]]))[0]
    right = cam.ground_to_image(np.array([[x_m + width_m / 2, y_m]]))[0]
    v = float(left[1])
    return Obj([float(left[0]), v - height_px, float(right[0]), v], class_name)


def three_lane_road() -> LaneSet:
    """Ego lane with one same-direction lane either side."""
    return lane_set(
        boundary("ego_left", -1.75), boundary("ego_right", 1.75),
        boundary("adjacent_left", -5.25), boundary("adjacent_right", 5.25))


class TestEnumeration:
    def test_a_three_lane_road_yields_three_lanes(self) -> None:
        got = enumerate_lanes(three_lane_road())
        assert len(got) == 3
        assert [ln.side for ln in got] == ["left", "ego", "right"]

    def test_the_ego_lane_is_identified(self) -> None:
        got = enumerate_lanes(three_lane_road())
        ego = [ln for ln in got if ln.kind is LaneKind.EGO]
        assert len(ego) == 1
        assert ego[0].side == "ego"

    def test_lanes_come_out_ordered_left_to_right(self) -> None:
        got = enumerate_lanes(three_lane_road())
        xs = [ln.centre_x_at(15.0) for ln in got]
        assert xs == sorted(xs)

    def test_no_lane_means_no_answer(self) -> None:
        assert enumerate_lanes(LaneSet(status=LaneStatus.NOT_DETECTED)) == []

    def test_a_single_boundary_bounds_nothing(self) -> None:
        assert enumerate_lanes(lane_set(boundary("ego_left", -1.75))) == []

    def test_a_median_is_not_a_lane(self) -> None:
        """Six metres between boundaries is a central reservation."""
        lanes = lane_set(boundary("ego_left", -1.75), boundary("ego_right", 1.75),
                         boundary("adjacent_left", -7.75))
        sides = [ln.side for ln in enumerate_lanes(lanes)]
        assert sides == ["ego"]

    def test_a_gore_is_not_a_lane(self) -> None:
        """A gap that opens with distance is a fork, not a lane alongside."""
        lanes = lane_set(boundary("ego_left", -1.75), boundary("ego_right", 1.75),
                         boundary("adjacent_right", 5.25, b=0.14))
        sides = [ln.side for ln in enumerate_lanes(lanes)]
        assert sides == ["ego"]


class TestOccupancy:
    def test_an_empty_neighbour_is_free(self) -> None:
        got = enumerate_lanes(three_lane_road(), camera(), [])
        assert all(ln.kind in (LaneKind.EGO, LaneKind.FREE) for ln in got)
        assert len(change_options(got)) == 2

    def test_a_car_in_a_neighbour_makes_it_occupied(self) -> None:
        cam = camera()
        got = enumerate_lanes(three_lane_road(), cam, [car_at(cam, 3.5, 20.0)])
        right = next(ln for ln in got if ln.side == "right")
        assert right.kind is LaneKind.OCCUPIED
        assert right.occupants == (0,)
        assert [ln.side for ln in change_options(got)] == ["left"]

    def test_a_car_beyond_the_fitted_range_still_occupies(self) -> None:
        """The lane is painted to the horizon; it must be checked that far too.

        Found by rendering: orange "lane change available" over visible cars.
        """
        cam = camera()
        lanes = lane_set(
            boundary("ego_left", -1.75, y_range=(5.0, 18.0)),
            boundary("ego_right", 1.75, y_range=(5.0, 18.0)),
            boundary("adjacent_right", 5.25, y_range=(5.0, 18.0)))
        far_car = car_at(cam, 3.5, 34.0)
        assert 34.0 > 18.0
        got = enumerate_lanes(lanes, cam, [far_car])
        assert next(ln for ln in got if ln.side == "right").kind is LaneKind.OCCUPIED

    def test_a_car_nearer_than_the_fitted_range_still_occupies(self) -> None:
        """The car directly alongside is the one that matters most."""
        cam = camera()
        lanes = lane_set(
            boundary("ego_left", -1.75, y_range=(9.0, 40.0)),
            boundary("ego_right", 1.75, y_range=(9.0, 40.0)),
            boundary("adjacent_right", 5.25, y_range=(9.0, 40.0)))
        got = enumerate_lanes(lanes, cam, [car_at(cam, 3.5, 5.0)])
        assert next(ln for ln in got if ln.side == "right").kind is LaneKind.OCCUPIED

    def test_a_car_in_the_ego_lane_does_not_block_the_neighbour(self) -> None:
        cam = camera()
        got = enumerate_lanes(three_lane_road(), cam, [car_at(cam, 0.0, 18.0)])
        assert len(change_options(got)) == 2

    def test_a_traffic_light_is_not_an_occupant(self) -> None:
        """It hangs over the road; back-projecting its box invents a vehicle."""
        cam = camera()
        light = car_at(cam, 3.5, 20.0, class_name="traffic_light")
        got = enumerate_lanes(three_lane_road(), cam, [light])
        assert next(ln for ln in got if ln.side == "right").kind is LaneKind.FREE

    def test_a_pedestrian_does_occupy(self) -> None:
        cam = camera()
        person = car_at(cam, 3.5, 20.0, width_m=0.6, class_name="person")
        got = enumerate_lanes(three_lane_road(), cam, [person])
        assert next(ln for ln in got if ln.side == "right").kind is LaneKind.OCCUPIED

    def test_without_a_camera_nothing_can_be_placed(self) -> None:
        got = enumerate_lanes(three_lane_road(), None, [Obj([0, 0, 10, 10])])
        assert all(not ln.occupants for ln in got)

    def test_occupancy_uses_the_ground_contact_not_the_box_centre(self) -> None:
        """A tall box centred over one lane can have its wheels in another."""
        cam = camera()
        tall = car_at(cam, 3.5, 12.0, width_m=2.4, height_px=300)
        got = enumerate_lanes(three_lane_road(), cam, [tall])
        assert next(ln for ln in got if ln.side == "right").kind is LaneKind.OCCUPIED
        assert next(ln for ln in got if ln.side == "ego").kind is LaneKind.EGO


class TestNeverOffersTheWrongLane:
    def test_the_oncoming_carriageway_is_never_a_change_option(self) -> None:
        """The worst possible output: inviting a move into head-on traffic."""
        from models.lane_seg.carriageway import DividerKind, DualCarriageway

        cw = DualCarriageway(
            divider_role="ego_left", divider_x_m=-1.75, divider_side="left",
            kind=DividerKind.YELLOW_LINE, yellow_fraction=0.9,
            median_width_m=None, ego_lanes=1, oncoming_lanes=1,
            confidence=0.9, eval_distance_m=15.0)
        got = enumerate_lanes(three_lane_road(), camera(), [], carriageway=cw)
        left = next(ln for ln in got if ln.side == "left")
        assert left.kind is LaneKind.ONCOMING
        assert not left.kind.is_change_option
        assert [ln.side for ln in change_options(got)] == ["right"]

    def test_an_exit_branch_is_not_offered_as_a_lane_change(self) -> None:
        from models.lane_seg.split import RoadSplit, SplitSide

        split = RoadSplit(side=SplitSide.RIGHT, near_separation_m=3.5,
                          far_separation_m=7.0, divergence_ratio=2.0,
                          gore_distance_m=24.0, left_role="ego_right",
                          right_role="adjacent_right")
        got = enumerate_lanes(three_lane_road(), camera(), [], split=split)
        right = next(ln for ln in got if ln.side == "right")
        assert right.kind is LaneKind.BRANCH
        assert not right.kind.is_change_option

    def test_oncoming_beats_free_when_both_could_apply(self) -> None:
        """An empty oncoming lane looks exactly like an empty free one."""
        from models.lane_seg.carriageway import DividerKind, DualCarriageway

        cw = DualCarriageway(
            divider_role="ego_left", divider_x_m=-1.75, divider_side="left",
            kind=DividerKind.YELLOW_LINE, yellow_fraction=1.0,
            median_width_m=None, ego_lanes=1, oncoming_lanes=1,
            confidence=0.9, eval_distance_m=15.0)
        got = enumerate_lanes(three_lane_road(), camera(), [], carriageway=cw)
        assert next(ln for ln in got if ln.side == "left").kind is LaneKind.ONCOMING


class TestLaneGeometry:
    def test_contains_respects_the_lateral_bounds(self) -> None:
        lane = Lane(left=boundary("a", 1.75), right=boundary("b", 5.25),
                    kind=LaneKind.FREE, side="right", width_m=3.5,
                    y_range_m=(NEAR, FAR))
        assert lane.contains(3.5, 20.0)
        assert not lane.contains(0.0, 20.0)
        assert not lane.contains(7.0, 20.0)

    def test_contains_respects_the_range_unless_overridden(self) -> None:
        lane = Lane(left=boundary("a", 1.75), right=boundary("b", 5.25),
                    kind=LaneKind.FREE, side="right", width_m=3.5,
                    y_range_m=(10.0, 20.0))
        assert not lane.contains(3.5, 40.0)
        assert lane.contains(3.5, 40.0, far_m=60.0)
        assert not lane.contains(3.5, 2.0)
        assert lane.contains(3.5, 2.0, near_m=0.0)

    def test_the_centre_is_between_the_bounds(self) -> None:
        lane = Lane(left=boundary("a", 1.75), right=boundary("b", 5.25),
                    kind=LaneKind.FREE, side="right", width_m=3.5,
                    y_range_m=(NEAR, FAR))
        assert lane.centre_x_at(15.0) == 3.5
