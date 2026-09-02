"""Typed configuration loaded from YAML under ``configs/``.

Config objects carry no coordinate-frame semantics themselves; modules that
consume them state their own frames. Unknown keys are rejected rather than
ignored, so a typo in YAML fails loudly instead of silently taking a default.

Cache invalidation depends on :meth:`Config.cache_key`, which hashes only the
sections that actually influence cached per-frame results. Recolouring the
overlay must not throw away an hour of inference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

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
        canonical_size: ``[width, height]`` every emitted frame is resized to.
            None means "adopt the first frame's size". Frames arriving at any
            other size are resized to match, because the homography is defined
            for exactly one resolution.
        use_container_timestamps: For video, take each frame's time from the
            container's presentation timestamp rather than ``index / fps``. This
            is what makes variable-frame-rate recordings come out with correct
            timing. Falls back to ``index / fps`` when the container reports
            nothing usable.
        start_frame: Index of the first frame to emit, counted in source frames
            before ``stride`` is applied.
        stride: Emit every Nth source frame. 1 means every frame.
        max_frames: Stop after this many emitted frames. None means no limit.
    """

    image_extensions: list[str] = Field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    image_dir_fps: float | None = None
    canonical_size: tuple[int, int] | None = None
    use_container_timestamps: bool = True
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

    @field_validator("canonical_size")
    @classmethod
    def _positive_size(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        if v is not None and (v[0] < 1 or v[1] < 1):
            raise ValueError(f"canonical_size must be positive, got {v}")
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


class ChessboardConfig(_Strict):
    """Calibration target geometry.

    Attributes:
        cols: Inner corners per row (9 for a standard 10x7 board).
        rows: Inner corners per column.
        square_size_m: Physical square edge length in metres. This scales the
            extrinsic translations only; the camera matrix and distortion
            coefficients are unaffected by it.
    """

    cols: int = 9
    rows: int = 6
    square_size_m: float = 0.024

    @field_validator("cols", "rows")
    @classmethod
    def _at_least_two(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"chessboard needs >= 2 inner corners per side, got {v}")
        return v

    @field_validator("square_size_m")
    @classmethod
    def _positive_square(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"square_size_m must be > 0, got {v}")
        return v


class CalibrationConfig(_Strict):
    """Where calibration lives and how it is applied.

    Attributes:
        intrinsics_path: Output/input path for the chessboard calibration.
        homography_path: Output/input path for the ground-plane homography.
        undistort: Whether frames are undistorted before anything else. When
            False the homography absorbs whatever lens distortion is present,
            which is accurate near the image centre and degrades toward the
            edges -- exactly where distant vehicles and lane markings sit.
            Whichever way this is set, the choice is recorded in the homography
            file and in the cache manifest, so results are never ambiguous.
        chessboard: Calibration target geometry.
    """

    intrinsics_path: Path = Path("calib/intrinsics.yaml")
    homography_path: Path = Path("calib/homography.yaml")
    undistort: bool = False
    chessboard: ChessboardConfig = Field(default_factory=ChessboardConfig)


class BEVConfig(_Strict):
    """Bird's-eye rendering window, in ego-vehicle metres (X right, Y forward).

    Attributes:
        x_min_m, x_max_m: Lateral extent; 0 is the camera's ground projection.
        y_min_m, y_max_m: Longitudinal extent ahead of the camera. ``y_min_m``
            must be > 0: the ground plane meets the image only ahead of the car.
        px_per_m: Resolution of the rendered bird's-eye image.
        grid_step_m: Spacing of the metre grid drawn by ``calibrate verify``.
    """

    x_min_m: float = -6.0
    x_max_m: float = 6.0
    y_min_m: float = 3.0
    y_max_m: float = 40.0
    px_per_m: float = 20.0
    grid_step_m: float = 5.0

    @field_validator("px_per_m", "grid_step_m")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"must be > 0, got {v}")
        return v

    def model_post_init(self, _ctx: Any) -> None:
        if self.x_max_m <= self.x_min_m:
            raise ValueError(f"x_max_m must exceed x_min_m, got {self.x_min_m}..{self.x_max_m}")
        if self.y_max_m <= self.y_min_m:
            raise ValueError(f"y_max_m must exceed y_min_m, got {self.y_min_m}..{self.y_max_m}")
        if self.y_min_m <= 0:
            raise ValueError(
                f"y_min_m must be > 0 (the road plane is only imaged ahead of the "
                f"camera), got {self.y_min_m}"
            )

    @property
    def size_px(self) -> tuple[int, int]:
        """``(width, height)`` of the rendered bird's-eye image."""
        w = int(round((self.x_max_m - self.x_min_m) * self.px_per_m))
        h = int(round((self.y_max_m - self.y_min_m) * self.px_per_m))
        return w, h


class CacheConfig(_Strict):
    """Two-stage cache location and identity.

    Attributes:
        root: Directory holding one subdirectory per run.
        model_version: Free-form identifier for whatever produced the per-frame
            results. Bump it whenever the model or its weights change; a cache
            written under a different value is treated as stale.
        run_id: Fixed run directory name, or None to derive one from the source
            name and the config hash (so identical settings reuse the cache).
    """

    root: Path = Path("outputs")
    model_version: str = "stub-0.1"
    run_id: str | None = None


class DetectorConfig(_Strict):
    """Settings for the Stage 1 stub detector.

    Replaced wholesale when the real YOLOPv2 backbone lands; it exists only to
    prove the pipeline end to end.

    Attributes:
        kind: Which detector implementation to run. Only ``"stub"`` exists now.
        score_threshold: Detections below this confidence are dropped.
        seed: Makes the fake detections deterministic, so cache reuse is testable.
    """

    kind: str = "stub"
    score_threshold: float = 0.3
    seed: int = 0

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, v: str) -> str:
        if v != "stub":
            raise ValueError(
                f"unknown detector kind '{v}'. Only 'stub' exists at this stage; "
                f"the real backbone arrives with Stage 2."
            )
        return v


class OverlayConfig(_Strict):
    """Annotation appearance. Excluded from the cache key by design.

    Attributes:
        box_thickness: Outline width in pixels for detection boxes.
        lane_thickness: Polyline width in pixels for lane boundaries.
        mask_alpha: Blend weight for translucent masks, in ``[0, 1]``.
        font_scale: OpenCV Hershey font scale for labels and HUD text.
        hud: Whether to draw the text HUD panel.
        hud_width_px: Width of the HUD panel.
    """

    box_thickness: int = 2
    lane_thickness: int = 4
    mask_alpha: float = 0.4
    font_scale: float = 0.5
    hud: bool = True
    hud_width_px: int = 340

    @field_validator("mask_alpha")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"mask_alpha must be in [0, 1], got {v}")
        return v


class VideoConfig(_Strict):
    """Annotated-output encoding. Excluded from the cache key by design.

    Attributes:
        fourcc: OpenCV FourCC code for the writer.
        fps: Output frame rate, or None to inherit the emitted frame rate
            (source fps divided by ``io.stride``).
    """

    fourcc: str = "mp4v"
    fps: float | None = None

    @field_validator("fourcc")
    @classmethod
    def _four_chars(cls, v: str) -> str:
        if len(v) != 4:
            raise ValueError(f"fourcc must be exactly 4 characters, got '{v}'")
        return v


class Config(_Strict):
    """Root config. Sections are added as pipeline stages land."""

    io: IOConfig = Field(default_factory=IOConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    bev: BEVConfig = Field(default_factory=BEVConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)

    #: Sections whose values change cached per-frame results. Presentation-only
    #: sections are deliberately absent: restyling the overlay must not
    #: invalidate an expensive cache.
    CACHE_KEY_SECTIONS: ClassVar[tuple[str, ...]] = ("io", "calibration", "detector")

    def cache_key_payload(self) -> dict[str, Any]:
        """The config subset that cached results depend on."""
        dumped = self.model_dump(mode="json")
        return {k: dumped[k] for k in self.CACHE_KEY_SECTIONS}

    def cache_key(self) -> str:
        """Stable 16-hex-char digest of the cache-relevant config."""
        blob = json.dumps(self.cache_key_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


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
