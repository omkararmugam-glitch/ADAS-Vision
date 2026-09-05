"""Tests for annotated-output saving and the overlay additions.

The regression this guards is the one that prompted the change: training and
evaluation reported scalars and silently wrote no pictures. A falling loss is
consistent with a model putting boxes in confidently wrong places, and only the
frames distinguish those. So the tests check that files are actually written,
and that the fixed sample really is fixed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.config import OverlayConfig
from core.overlay import draw_boxes, draw_status_corner
from core.types import CLASS_IDS, FrameResult
from eval.preview import contact_sheet, pick_fixed_frames


def result(n: int = 2) -> FrameResult:
    blank = np.zeros((240, 320), np.uint8)
    return FrameResult(
        boxes_px=np.array([[20.0 + 60 * i, 60, 90.0 + 60 * i, 150] for i in range(n)],
                          np.float32),
        class_ids=np.full(n, CLASS_IDS["vehicle"], np.int32),
        scores=np.full(n, 0.85, np.float32),
        drivable_mask=blank, lane_mask=blank.copy(),
    )


class TestTrackIdsOnBoxes:
    def test_a_track_id_is_drawn_when_supplied(self) -> None:
        img = np.full((240, 320, 3), 40, np.uint8)
        plain = draw_boxes(img, result(1), OverlayConfig())
        tagged = draw_boxes(img, result(1), OverlayConfig(), track_ids={0: 7})
        assert not np.array_equal(plain, tagged), "the id must change the pixels"

    def test_no_ids_leaves_the_frame_unchanged(self) -> None:
        img = np.full((240, 320, 3), 40, np.uint8)
        a = draw_boxes(img, result(1), OverlayConfig())
        b = draw_boxes(img, result(1), OverlayConfig(), track_ids=None)
        assert np.array_equal(a, b)

    def test_an_unmatched_detection_simply_gets_no_id(self) -> None:
        """A partial mapping must not raise; untracked boxes are normal."""
        img = np.full((240, 320, 3), 40, np.uint8)
        out = draw_boxes(img, result(2), OverlayConfig(), track_ids={0: 3})
        assert out.shape == img.shape

    def test_the_source_frame_is_not_modified(self) -> None:
        img = np.full((240, 320, 3), 40, np.uint8)
        before = img.copy()
        draw_boxes(img, result(2), OverlayConfig(), track_ids={0: 1, 1: 2})
        assert np.array_equal(img, before)


class TestStatusCorner:
    def test_labels_are_drawn(self) -> None:
        img = np.full((240, 320, 3), 40, np.uint8)
        out = draw_status_corner(img, OverlayConfig(), weather="rain")
        assert not np.array_equal(img, out)

    @pytest.mark.parametrize("value", ["unknown", "none", "undefined", "", None])
    def test_an_absent_classification_draws_nothing(self, value) -> None:
        """A corner permanently reading 'unknown' trains the eye to skip it."""
        img = np.full((240, 320, 3), 40, np.uint8)
        out = draw_status_corner(img, OverlayConfig(), weather=value)
        assert np.array_equal(img, out)

    def test_both_labels_render_together(self) -> None:
        img = np.full((240, 320, 3), 40, np.uint8)
        one = draw_status_corner(img, OverlayConfig(), weather="rain")
        two = draw_status_corner(img, OverlayConfig(), weather="rain",
                                 road_condition="pothole")
        assert not np.array_equal(one, two)

    def test_it_stays_inside_the_frame(self) -> None:
        img = np.full((120, 160, 3), 40, np.uint8)
        out = draw_status_corner(img, OverlayConfig(), weather="snow")
        assert out.shape == img.shape


class TestFixedSampling:
    def _images(self, tmp_path: Path, n: int) -> Path:
        d = tmp_path / "images"
        d.mkdir()
        for i in range(n):
            cv2.imwrite(str(d / f"{i:04d}.jpg"), np.full((32, 32, 3), i % 255, np.uint8))
        return d

    def test_the_sample_is_stable_across_calls(self, tmp_path: Path) -> None:
        """Comparing epoch_005 with epoch_030 is meaningless if the frames move."""
        d = self._images(tmp_path, 50)
        assert pick_fixed_frames(d, 8) == pick_fixed_frames(d, 8)

    def test_it_spreads_across_the_listing(self, tmp_path: Path) -> None:
        d = self._images(tmp_path, 50)
        picked = pick_fixed_frames(d, 5)
        assert len(picked) == 5
        assert len(set(picked)) == 5
        assert picked[0] != picked[-1]

    def test_a_small_set_returns_everything(self, tmp_path: Path) -> None:
        d = self._images(tmp_path, 3)
        assert len(pick_fixed_frames(d, 8)) == 3

    def test_an_empty_directory_is_not_an_error(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        assert pick_fixed_frames(d, 8) == []


class TestContactSheet:
    def test_it_writes_a_file(self, tmp_path: Path) -> None:
        paths = []
        for i in range(5):
            p = tmp_path / f"{i}.jpg"
            cv2.imwrite(str(p), np.full((90, 160, 3), 40 * i % 255, np.uint8))
            paths.append(p)
        out = contact_sheet(paths, tmp_path / "sheet.jpg", cols=3, width=160)
        assert out is not None and out.is_file()
        img = cv2.imread(str(out))
        assert img is not None and img.shape[1] == 480

    def test_no_inputs_writes_nothing(self, tmp_path: Path) -> None:
        assert contact_sheet([], tmp_path / "sheet.jpg") is None

    def test_a_ragged_last_row_is_padded_not_dropped(self, tmp_path: Path) -> None:
        paths = []
        for i in range(5):
            p = tmp_path / f"{i}.jpg"
            cv2.imwrite(str(p), np.full((90, 160, 3), 60, np.uint8))
            paths.append(p)
        out = contact_sheet(paths, tmp_path / "sheet.jpg", cols=4, width=160)
        img = cv2.imread(str(out))
        assert img.shape[0] > 90, "the fifth image must occupy a second row"


class TestDistanceLines:
    """Lines from the ego to each object, labelled in metres."""

    @staticmethod
    def _obj(distance, box=(300.0, 300, 400, 400), name="vehicle", in_path=True):
        from tracking.collision import ObjectState
        from tracking.ttc import NO_COLLISION, TTCEstimate
        return ObjectState(
            track_id=1, class_name=name, box_px=np.array(box), score=0.9,
            distance_m=distance, lateral_offset_m=0.0,
            ttc=TTCEstimate(ttc_s=NO_COLLISION), in_path=in_path,
        )

    @staticmethod
    def _camera():
        from core.calibrate import CameraModel
        return CameraModel(H_image_to_ground=np.eye(3), image_size=(640, 480))

    def _blank(self):
        return np.full((480, 640, 3), 40, np.uint8)

    def test_a_line_is_drawn_for_a_measured_object(self) -> None:
        from core.overlay import draw_distance_lines
        img = self._blank()
        out = draw_distance_lines(img, [self._obj(12.0)], self._camera(), OverlayConfig())
        assert not np.array_equal(img, out)

    def test_an_object_with_no_distance_draws_nothing(self) -> None:
        """Without calibration there is no metre value to put on the line."""
        from core.overlay import draw_distance_lines
        img = self._blank()
        out = draw_distance_lines(img, [self._obj(None)], self._camera(), OverlayConfig())
        assert np.array_equal(img, out)

    def test_no_objects_draws_nothing(self) -> None:
        from core.overlay import draw_distance_lines
        img = self._blank()
        assert np.array_equal(img, draw_distance_lines(img, [], self._camera(), OverlayConfig()))

    def test_distant_objects_are_dropped(self) -> None:
        """A confidently-labelled 90 m line implies precision that does not exist."""
        from core.overlay import draw_distance_lines
        img = self._blank()
        out = draw_distance_lines(img, [self._obj(90.0)], self._camera(),
                                  OverlayConfig(), max_distance_m=60.0)
        assert np.array_equal(img, out)

    def test_only_the_nearest_few_are_drawn(self) -> None:
        """Twenty lines at a junction is an unreadable fan."""
        from core.overlay import draw_distance_lines
        objs = [self._obj(float(5 + i), box=(50.0 + 20 * i, 300, 90.0 + 20 * i, 380))
                for i in range(10)]
        few = draw_distance_lines(self._blank(), objs, self._camera(),
                                  OverlayConfig(), max_lines=2)
        many = draw_distance_lines(self._blank(), objs, self._camera(),
                                   OverlayConfig(), max_lines=8)
        assert int((few != 40).sum()) < int((many != 40).sum())

    def test_in_path_filter_is_honoured(self) -> None:
        from core.overlay import draw_distance_lines
        img = self._blank()
        out = draw_distance_lines(img, [self._obj(10.0, in_path=False)], self._camera(),
                                  OverlayConfig(), in_path_only=True)
        assert np.array_equal(img, out)

    def test_the_source_frame_is_not_modified(self) -> None:
        from core.overlay import draw_distance_lines
        img = self._blank()
        before = img.copy()
        draw_distance_lines(img, [self._obj(10.0)], self._camera(), OverlayConfig())
        assert np.array_equal(img, before)

    def test_a_non_finite_distance_is_ignored(self) -> None:
        from core.overlay import draw_distance_lines
        img = self._blank()
        out = draw_distance_lines(img, [self._obj(float("inf"))], self._camera(),
                                  OverlayConfig())
        assert np.array_equal(img, out)


class TestSmoothSplitRendering:
    """Long smooth curves along each diverging boundary.

    The earlier treatment marked WHERE the split was with a diamond but drew the
    boundaries as short segments, so the fork itself -- two curves parting
    company -- was not visible. Showing the divergence is the point of detecting
    it.
    """

    @staticmethod
    def _camera():
        from core.calibrate import (
            CameraModel, rectangle_world_points, solve_ground_homography,
        )
        W, H, F, CH, P = 1280, 720, 900.0, 1.5, np.deg2rad(-4.0)

        def project(pts):
            pts = np.asarray(pts, float).reshape(-1, 2)
            x, y = pts[:, 0], pts[:, 1]
            z = -CH
            c, s = np.cos(P), np.sin(P)
            yc, zc = c * y - s * z, s * y + c * z
            return np.column_stack([F * (x / yc) + W / 2, F * (-zc / yc) + H / 2])

        world = rectangle_world_points(3.5, 6.0, 30.0)
        Hm, _ = solve_ground_homography(project(world), world)
        return CameraModel(H_image_to_ground=Hm, image_size=(W, H))

    @staticmethod
    def _lanes():
        from core.types import LaneFit, LaneSet, LaneStatus
        def fit(role, c, b=0.0):
            # 6-34 m: the same span the split tests use. A shorter range gives a
            # divergence ratio below the detection gate, so no split is found and
            # the rendering under test never runs.
            return LaneFit(coeffs=np.array([0.0, b, c]), role=role,
                           n_pixels=400, y_range_m=(6.0, 34.0))
        return LaneSet(status=LaneStatus.OK,
                       ego_left=fit("ego_left", -1.75),
                       ego_right=fit("ego_right", 1.75, 0.12),
                       eval_distance_m=10.0)

    def _split(self):
        from models.lane_seg.split import detect_split
        return detect_split(self._lanes())

    def test_a_split_is_drawn(self) -> None:
        from core.overlay import draw_split_smooth
        img = np.full((720, 1280, 3), 45, np.uint8)
        out = draw_split_smooth(img, self._split(), self._lanes(),
                                self._camera(), OverlayConfig())
        assert not np.array_equal(img, out)

    def test_no_split_draws_nothing(self) -> None:
        from core.overlay import draw_split_smooth
        img = np.full((720, 1280, 3), 45, np.uint8)
        assert np.array_equal(
            img, draw_split_smooth(img, None, self._lanes(),
                                   self._camera(), OverlayConfig()))

    def test_the_curve_extends_past_the_fitted_range(self) -> None:
        """A fit supported by a short stub of dashes still has to reach the road."""
        from core.overlay import draw_smooth_curve
        from models.lane_seg.fitting import fit_to_image_points
        cam = self._camera()
        fit = self._lanes().ego_left
        plain = len(fit_to_image_points(fit, cam))
        img = np.full((720, 1280, 3), 45, np.uint8)
        out = draw_smooth_curve(img.copy(), fit, cam, (0, 255, 0))
        painted = int((out != 45).any(axis=2).sum())
        assert painted > plain, "the extended curve must cover more than the fit alone"

    def test_the_source_frame_is_not_modified(self) -> None:
        from core.overlay import draw_split_smooth
        img = np.full((720, 1280, 3), 45, np.uint8)
        before = img.copy()
        draw_split_smooth(img, self._split(), self._lanes(),
                          self._camera(), OverlayConfig())
        assert np.array_equal(img, before)
