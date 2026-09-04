"""Tests for road-hazard configuration and graceful degradation.

The models need data that has not been supplied, so what is testable is the part
that must not fail: the configuration invariants, and that a missing checkpoint
degrades to None rather than taking the collision pipeline down with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import Config, PotholeConfig
from models.pothole_det import RoadHazard, build_hazard_detector


class TestConfig:
    def test_it_is_in_the_cache_key(self) -> None:
        """It runs a network per frame, so its output is expensive and cached."""
        assert "pothole" in Config.CACHE_KEY_SECTIONS

    def test_disabled_by_default(self) -> None:
        assert PotholeConfig().enabled is False

    def test_precision_is_favoured_by_default(self) -> None:
        """An over-eager hazard warning trains the driver to ignore all alerts."""
        assert PotholeConfig().conf_threshold >= 0.5

    def test_an_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="detector.*classifier"):
            PotholeConfig(mode="magic")

    def test_a_stride_wider_than_the_patch_is_rejected(self) -> None:
        """Otherwise the sweep leaves strips of road it never looks at."""
        with pytest.raises(ValueError, match="gaps"):
            PotholeConfig(patch_px=64, patch_stride_px=96)

    def test_an_inverted_distance_band_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_distance_m"):
            PotholeConfig(min_distance_m=40.0, max_distance_m=10.0)

    @pytest.mark.parametrize("mode", ["detector", "classifier"])
    def test_both_modes_are_accepted(self, mode: str) -> None:
        assert PotholeConfig(mode=mode).mode == mode


class TestGracefulDegradation:
    def test_disabled_builds_nothing(self) -> None:
        assert build_hazard_detector(PotholeConfig(enabled=False)) is None

    @pytest.mark.parametrize("mode", ["detector", "classifier"])
    def test_missing_weights_degrade_to_none(self, mode: str) -> None:
        """Road hazards are advisory; losing them must not break collision logic."""
        cfg = PotholeConfig(enabled=True, mode=mode, weights="does/not/exist.pt")
        assert build_hazard_detector(cfg) is None


class TestRoadHazard:
    def test_a_detection_is_marked_localised(self) -> None:
        h = RoadHazard(box_px=np.array([0.0, 0, 10, 10]), score=0.8)
        assert h.localised

    def test_a_classifier_patch_is_marked_unlocalised(self) -> None:
        """A grid cell is not a measured position and must not read as one."""
        h = RoadHazard(box_px=np.array([0.0, 0, 96, 96]), score=0.7, localised=False)
        assert not h.localised
