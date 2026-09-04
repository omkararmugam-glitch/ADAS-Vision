"""Weather and time-of-day classification from a single frame.

Feeds the fusion engine's threshold adaptation: wet tarmac roughly doubles
stopping distance, so a TTC threshold that is right in the dry fires far too late
in the rain. The classifier's only job is to say which multiplier applies.

That framing keeps the requirements modest, and it should. The output is used to
pick one of half a dozen scalars, not to steer anything, so **stability matters
more than per-frame accuracy**. A classifier that flickers between `clear` and
`rain` every other frame would swing the collision threshold back and forth,
which is worse than one that is slightly wrong but consistent -- hence the
temporal smoothing in :class:`WeatherSmoother`, which is the part that actually
gets used.

ResNet18 over EfficientNet-B0 as the default backbone: at this task's difficulty
both saturate, and ResNet18 is faster, smaller and has fewer surprises when
exported to ONNX. The choice is a config flag, and the comparison belongs in
`docs/EVAL_RESULTS.md` once there is data to run it on.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import WeatherConfig
from core.logging_utils import get_logger, log_kv

_log = get_logger(__name__)

#: BDD100K's weather vocabulary, minus `undefined`. Kept identical to the
#: dataset's own labels so the adapter never has to invent a mapping.
WEATHER_CLASSES: tuple[str, ...] = (
    "clear", "cloudy", "foggy", "overcast", "partly cloudy", "rainy", "snowy",
)

#: Maps a predicted label onto the keys `FusionConfig.weather_ttc_factor` uses.
#: Several BDD100K labels describe cloud cover, which does not change stopping
#: distance at all -- they collapse to `clear` rather than inventing a factor.
TO_FUSION_LABEL: dict[str, str] = {
    "clear": "clear",
    "cloudy": "clear",
    "partly cloudy": "clear",
    "overcast": "cloudy",
    "rainy": "rain",
    "snowy": "snow",
    "foggy": "fog",
}


@dataclass(frozen=True)
class WeatherPrediction:
    """One frame's classification.

    Attributes:
        label: Raw predicted class.
        confidence: Softmax probability of ``label``.
        fusion_label: ``label`` mapped onto the fusion engine's vocabulary.
        probabilities: Full distribution, for auditing a surprising decision.
    """

    label: str
    confidence: float
    fusion_label: str
    probabilities: dict[str, float]


class WeatherSmoother:
    """Majority vote over a sliding window, with a switching threshold.

    Weather does not change between frames. Any per-frame flicker is model noise,
    and letting it through would modulate the collision threshold several times a
    second. A new label is adopted only once it holds a clear majority of the
    window, so switching is deliberate and lagged rather than instant.
    """

    def __init__(self, window: int = 30, switch_fraction: float = 0.6) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if not 0.5 < switch_fraction <= 1.0:
            raise ValueError(
                f"switch_fraction must be in (0.5, 1.0] -- below a majority the "
                f"label would oscillate, got {switch_fraction}"
            )
        self.window = window
        self.switch_fraction = switch_fraction
        self._history: deque[str] = deque(maxlen=window)
        self._current: str | None = None

    def update(self, label: str) -> str:
        """Add one prediction and return the smoothed label."""
        self._history.append(label)
        if self._current is None:
            self._current = label
            return self._current
        counts = Counter(self._history)
        top, n = counts.most_common(1)[0]
        if top != self._current and n >= self.switch_fraction * len(self._history):
            log_kv(_log, logging.INFO, "weather.switch",
                   frm=self._current, to=top, support=f"{n}/{len(self._history)}")
            self._current = top
        return self._current

    @property
    def current(self) -> str | None:
        return self._current

    def reset(self) -> None:
        self._history.clear()
        self._current = None


def build_model(architecture: str, n_classes: int, pretrained: bool = True):
    """Construct a torchvision backbone with a fresh classification head.

    ImageNet initialisation is not optional in practice: the labelled weather
    data available here is a few thousand frames, which is far too few to learn
    low-level filters from scratch.
    """
    import torch.nn as nn
    import torchvision.models as tv

    if architecture == "resnet18":
        model = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif architecture == "efficientnet_b0":
        model = tv.efficientnet_b0(
            weights=tv.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    else:
        raise ValueError(
            f"unknown weather architecture '{architecture}'; "
            f"expected 'resnet18' or 'efficientnet_b0'"
        )
    return model


class WeatherClassifier:
    """Runs a trained weather model over frames."""

    #: ImageNet statistics, matching the pretrained backbones.
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, cfg: WeatherConfig, classes: tuple[str, ...] = WEATHER_CLASSES) -> None:
        import torch

        self.cfg = cfg
        self.classes = classes
        self._torch = torch
        self.device = torch.device(
            "cuda" if cfg.device == "auto" and torch.cuda.is_available()
            else ("cpu" if cfg.device == "auto" else cfg.device)
        )
        self.model = build_model(cfg.architecture, len(classes), pretrained=False)
        weights = Path(cfg.weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"weather weights not found: {weights}. Train one with "
                f"`python models/weather_cls/train.py`, or set weather.enabled=false."
            )
        state = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(state.get("model", state))
        self.model.to(self.device).eval()
        self.smoother = WeatherSmoother(cfg.smoothing_window, cfg.switch_fraction)
        log_kv(_log, logging.INFO, "weather.loaded",
               architecture=cfg.architecture, classes=len(classes), device=str(self.device))

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame -> normalised CHW float32 at the model's input size.

        The frame is resized, not cropped. Weather evidence lives in the sky and
        the road surface -- the top and bottom of the frame -- and a centre crop
        would discard exactly the pixels that carry the signal.
        """
        import cv2

        size = self.cfg.input_size
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - self.MEAN) / self.STD
        return np.transpose(arr, (2, 0, 1))

    def infer(self, frame: np.ndarray) -> WeatherPrediction:
        torch = self._torch
        batch = torch.from_numpy(self.preprocess(frame)[None]).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        label = self.classes[idx]
        return WeatherPrediction(
            label=label,
            confidence=float(probs[idx]),
            fusion_label=TO_FUSION_LABEL.get(label, "clear"),
            probabilities={c: float(p) for c, p in zip(self.classes, probs)},
        )

    def infer_smoothed(self, frame: np.ndarray) -> WeatherPrediction:
        """Classify and apply temporal smoothing. This is what the pipeline uses."""
        raw = self.infer(frame)
        smoothed = self.smoother.update(raw.label)
        if smoothed == raw.label:
            return raw
        return WeatherPrediction(
            label=smoothed,
            confidence=raw.probabilities.get(smoothed, 0.0),
            fusion_label=TO_FUSION_LABEL.get(smoothed, "clear"),
            probabilities=raw.probabilities,
        )

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
