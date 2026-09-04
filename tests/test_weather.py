"""Tests for weather smoothing and the fusion label mapping.

The classifier itself needs a trained checkpoint, but the two parts the pipeline
actually depends on do not, and they are the parts that can be quietly wrong:
the temporal smoother that stops the collision threshold oscillating, and the
mapping from BDD100K's weather vocabulary onto the fusion engine's.
"""

from __future__ import annotations

import pytest

from core.config import FusionConfig, WeatherConfig
from models.weather_cls import TO_FUSION_LABEL, WEATHER_CLASSES, WeatherSmoother


class TestSmoother:
    def test_the_first_prediction_is_adopted_immediately(self) -> None:
        assert WeatherSmoother(window=10).update("rainy") == "rainy"

    def test_a_single_outlier_does_not_switch_the_label(self) -> None:
        """One flickering frame must not move the collision threshold."""
        s = WeatherSmoother(window=10, switch_fraction=0.6)
        for _ in range(9):
            s.update("clear")
        assert s.update("rainy") == "clear"

    def test_a_sustained_change_does_switch(self) -> None:
        s = WeatherSmoother(window=10, switch_fraction=0.6)
        for _ in range(10):
            s.update("clear")
        out = [s.update("rainy") for _ in range(10)]
        assert out[-1] == "rainy"

    def test_switching_lags_deliberately(self) -> None:
        """It should take a clear majority, not one or two frames."""
        s = WeatherSmoother(window=10, switch_fraction=0.6)
        for _ in range(10):
            s.update("clear")
        assert s.update("rainy") == "clear"
        assert s.update("rainy") == "clear"

    def test_alternating_input_does_not_oscillate(self) -> None:
        """The failure this class exists to prevent."""
        s = WeatherSmoother(window=20, switch_fraction=0.6)
        labels = [s.update("clear" if i % 2 == 0 else "rainy") for i in range(40)]
        assert len(set(labels[5:])) == 1

    def test_reset_clears_state(self) -> None:
        s = WeatherSmoother(window=5)
        s.update("snowy")
        s.reset()
        assert s.current is None
        assert s.update("clear") == "clear"

    def test_a_switch_fraction_at_or_below_half_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="oscillate"):
            WeatherSmoother(window=10, switch_fraction=0.5)

    def test_window_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            WeatherSmoother(window=0)


class TestFusionMapping:
    def test_every_class_maps_somewhere(self) -> None:
        for c in WEATHER_CLASSES:
            assert c in TO_FUSION_LABEL

    def test_every_mapped_label_is_known_to_fusion(self) -> None:
        """A label fusion does not recognise silently means factor 1.0."""
        factors = FusionConfig().weather_ttc_factor
        for src, dst in TO_FUSION_LABEL.items():
            assert dst in factors, f"'{src}' -> '{dst}' is not in weather_ttc_factor"

    def test_cloud_cover_does_not_change_the_threshold(self) -> None:
        """Cloud does not affect stopping distance; rain and snow do."""
        factors = FusionConfig().weather_ttc_factor
        for label in ("clear", "cloudy", "partly cloudy"):
            assert factors[TO_FUSION_LABEL[label]] == 1.0

    @pytest.mark.parametrize("label", ["rainy", "snowy", "foggy"])
    def test_hazardous_weather_widens_the_threshold(self, label: str) -> None:
        factors = FusionConfig().weather_ttc_factor
        assert factors[TO_FUSION_LABEL[label]] > 1.0

    def test_snow_is_treated_as_worse_than_rain(self) -> None:
        f = FusionConfig().weather_ttc_factor
        assert f[TO_FUSION_LABEL["snowy"]] > f[TO_FUSION_LABEL["rainy"]]


class TestConfigIntegration:
    def test_the_smoother_accepts_the_configured_values(self) -> None:
        cfg = WeatherConfig()
        s = WeatherSmoother(cfg.smoothing_window, cfg.switch_fraction)
        assert s.window == cfg.smoothing_window

    def test_a_missing_checkpoint_fails_with_a_useful_message(self) -> None:
        from models.weather_cls.classifier import WeatherClassifier
        cfg = WeatherConfig(enabled=True, weights="does/not/exist.pt")
        with pytest.raises(FileNotFoundError, match="weather.enabled=false"):
            WeatherClassifier(cfg)
