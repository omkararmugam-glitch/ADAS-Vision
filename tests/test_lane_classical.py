"""Tests for weightless lane-marking segmentation.

Exists because the only lane-producing model (YOLOPv2) has no weights, so the
lane path was silently dead: the detector emits a blank mask and the fitter
dutifully found nothing in it.

The cases here are the ones that made the first version wrong on real frames --
the ego bonnet, and broad bright regions that are not paint.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from models.lane_seg.classical import (
    ROI_BOTTOM,
    ROI_TOP,
    ClassicalLaneSegmenter,
    lane_mask,
)

H, W = 720, 1280


def road(with_markings: bool = True, bonnet: bool = False) -> np.ndarray:
    img = np.full((H, W, 3), 70, np.uint8)
    img[: int(0.42 * H)] = (170, 160, 150)
    if with_markings:
        for x0, x1 in ((470, 500), (790, 820)):
            cv2.line(img, (x0, H - 120), (x1 - 180 + 180, int(0.5 * H)),
                     (245, 245, 245), 8)
    if bonnet:
        # Bright saturated streaks low in the frame, as reflections appear.
        for i in range(14):
            x = 60 + i * 85
            cv2.line(img, (x, H - 40), (x + 40, H - 30), (255, 240, 255), 9)
    return img


class TestSegmentation:
    def test_markings_are_found(self) -> None:
        out = lane_mask(road())
        assert out.mask.sum() > 0
        assert out.coverage > 0

    def test_a_blank_road_yields_almost_nothing(self) -> None:
        out = lane_mask(road(with_markings=False))
        assert out.coverage < 0.01

    def test_the_mask_matches_the_frame_size(self) -> None:
        assert lane_mask(road()).mask.shape == (H, W)

    def test_the_mask_is_binary(self) -> None:
        assert set(np.unique(lane_mask(road()).mask)) <= {0, 255}

    def test_nothing_above_the_horizon_is_marked(self) -> None:
        """Sky is never road, and marking it would drag the fit upward."""
        m = lane_mask(road()).mask
        assert m[: int(ROI_TOP * H) - 1].sum() == 0

    def test_the_ego_bonnet_is_excluded(self) -> None:
        """The regression: at night the bonnet mirrors signs and tail lights,
        and those reflections sit exactly where the sliding-window fit starts,
        so they anchor the whole lane to the bonnet."""
        m = lane_mask(road(bonnet=True)).mask
        assert m[int(ROI_BOTTOM * H) + 2:].sum() == 0

    def test_a_broad_bright_region_is_rejected(self) -> None:
        """A white van flank or sunlit wall is not a lane marking."""
        img = road(with_markings=False)
        cv2.rectangle(img, (300, 420), (900, 600), (250, 250, 250), -1)
        assert lane_mask(img).coverage < 0.02


class TestPlausibility:
    def test_a_reasonable_mask_is_plausible(self) -> None:
        assert lane_mask(road()).is_plausible

    def test_an_empty_mask_is_not_plausible(self) -> None:
        assert not lane_mask(road(with_markings=False)).is_plausible

    def test_a_saturated_mask_is_not_plausible(self) -> None:
        """A dense mask would produce a confident and entirely wrong lane."""
        img = np.full((H, W, 3), 250, np.uint8)
        assert not lane_mask(img).is_plausible


class TestSegmenterInterface:
    def test_infer_returns_a_mask(self) -> None:
        seg = ClassicalLaneSegmenter()
        assert seg.infer(road()).shape == (H, W)

    def test_it_names_itself(self) -> None:
        assert ClassicalLaneSegmenter().name == "lane-classical"

    def test_detailed_keeps_the_evidence(self) -> None:
        out = ClassicalLaneSegmenter().infer_detailed(road())
        assert out.from_colour >= 0 and out.from_gradient >= 0

    def test_the_mask_feeds_the_fitter(self) -> None:
        """The whole point: it must be consumable by LaneFitter unchanged."""
        from core.config import BEVConfig, LaneFitConfig
        from core.calibrate import CameraModel
        from models.lane_seg.fitting import LaneFitter

        cam = CameraModel(H_image_to_ground=np.eye(3), image_size=(W, H))
        fitter = LaneFitter(cam, BEVConfig(), LaneFitConfig())
        lanes = fitter.fit(ClassicalLaneSegmenter().infer(road()))
        assert lanes is not None and lanes.status is not None
