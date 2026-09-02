"""Typed configuration loaded from YAML under ``configs/``.

Config objects carry no coordinate-frame semantics themselves; modules that
consume them state their own frames. Unknown keys are rejected rather than
ignored, so a typo in YAML fails loudly instead of silently taking a default.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IOConfig(_Strict):
    """Frame-source behaviour.

    Attributes:
        image_extensions: Suffixes (lowercase, with dot) treated as frames when
            reading an image directory.
        image_dir_fps: Nominal capture rate for image-directory sources. There is
            no timing information in a folder of stills, so this must be stated
            explicitly; opening an image directory without it raises.
        start_frame: Index of the first frame to emit, counted in source frames
            before ``stride`` is applied.
        stride: Emit every Nth source frame. 1 means every frame.
        max_frames: Stop after this many emitted frames. None means no limit.
    """

    image_extensions: list[str] = Field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    image_dir_fps: float | None = None
    start_frame: int = 0
    stride: int = 1
    max_frames: int | None = None

    @field_validator("image_extensions")
    @classmethod
    def _lowercase_with_dot(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("image_extensions must not be empty")
        bad = [e for e in v if not e.startswith(".") or e != e.lower()]
        if bad:
            raise ValueError(f"image_extensions must be lowercase and dot-prefixed: {bad}")
        return v

    @field_validator("image_dir_fps")
    @classmethod
    def _positive_fps(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"image_dir_fps must be > 0, got {v}")
        return v

    @field_validator("start_frame")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"start_frame must be >= 0, got {v}")
        return v

    @field_validator("stride")
    @classmethod
    def _positive_stride(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"stride must be >= 1, got {v}")
        return v

    @field_validator("max_frames")
    @classmethod
    def _positive_max(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"max_frames must be >= 1 or null, got {v}")
        return v


class Config(_Strict):
    """Root config. Sections are added as pipeline stages land."""

    io: IOConfig = Field(default_factory=IOConfig)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is not a YAML mapping.
        pydantic.ValidationError: if any field is missing, mistyped or unknown.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}: {p}")
    return Config.model_validate(raw)
