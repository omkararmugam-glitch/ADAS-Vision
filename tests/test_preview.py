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
