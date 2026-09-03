"""Tests for the typed per-frame records.

``FrameResult`` validates in ``__post_init__`` rather than trusting its caller:
mismatched array lengths and degenerate boxes are the failure modes that would
otherwise surface much later as a confusing index error deep in the overlay.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.types import CLASS_IDS, FrameResult


class TestFrameResult:
    def test_rejects_mismatched_lengths(self) -> None:
        blank = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="same length"):
            FrameResult(
                boxes_px=np.zeros((2, 4), dtype=np.float32),
                class_ids=np.zeros((1,), dtype=np.int32),
                scores=np.zeros((2,), dtype=np.float32),
                drivable_mask=blank,
                lane_mask=blank.copy(),
            )

    def test_rejects_degenerate_boxes(self) -> None:
        blank = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="degenerate"):
            FrameResult(
                boxes_px=np.array([[10.0, 10.0, 5.0, 20.0]], dtype=np.float32),
                class_ids=np.zeros((1,), dtype=np.int32),
                scores=np.ones((1,), dtype=np.float32),
                drivable_mask=blank,
                lane_mask=blank.copy(),
            )

    def test_rejects_mismatched_masks(self) -> None:
        with pytest.raises(ValueError, match="share a shape"):
            FrameResult(
                boxes_px=np.zeros((0, 4), dtype=np.float32),
                class_ids=np.zeros((0,), dtype=np.int32),
                scores=np.zeros((0,), dtype=np.float32),
                drivable_mask=np.zeros((10, 10), dtype=np.uint8),
                lane_mask=np.zeros((5, 5), dtype=np.uint8),
            )

    def test_bottom_centres_are_the_ground_contact_points(self) -> None:
        blank = np.zeros((10, 10), dtype=np.uint8)
        r = FrameResult(
            boxes_px=np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32),
            class_ids=np.array([CLASS_IDS["vehicle"]], dtype=np.int32),
            scores=np.array([0.9], dtype=np.float32),
            drivable_mask=blank,
            lane_mask=blank.copy(),
        )
        assert r.bottom_centres_px().tolist() == [[200.0, 400.0]]

    def test_filter_keeps_masks(self) -> None:
        blank = np.zeros((10, 10), dtype=np.uint8)
        r = FrameResult(
            boxes_px=np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0]], dtype=np.float32),
            class_ids=np.array([0, 1], dtype=np.int32),
            scores=np.array([0.9, 0.2], dtype=np.float32),
            drivable_mask=blank,
            lane_mask=blank.copy(),
        )
        kept = r.filter(r.scores > 0.5)
        assert len(kept) == 1
        assert kept.drivable_mask.shape == blank.shape

    def test_empty_result_has_the_requested_size(self) -> None:
        r = FrameResult.empty((640, 480))
        assert len(r) == 0
        assert r.drivable_mask.shape == (480, 640)
        assert r.size == (640, 480)
