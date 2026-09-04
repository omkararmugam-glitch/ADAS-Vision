"""Weather classification, feeding the fusion engine's threshold adaptation.

Its only job is to pick which stopping-distance multiplier applies, so stability
matters more than per-frame accuracy: a classifier flickering between `clear` and
`rain` would swing the collision threshold several times a second, which is worse
than one that is slightly wrong but consistent. :class:`WeatherSmoother` is
therefore the part the pipeline actually depends on.
"""

from models.weather_cls.classifier import (
    TO_FUSION_LABEL,
    WEATHER_CLASSES,
    WeatherClassifier,
    WeatherPrediction,
    WeatherSmoother,
    build_model,
)

__all__ = [
    "WeatherClassifier",
    "WeatherSmoother",
    "WeatherPrediction",
    "WEATHER_CLASSES",
    "TO_FUSION_LABEL",
    "build_model",
]
