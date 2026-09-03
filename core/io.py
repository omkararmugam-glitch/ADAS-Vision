"""Frame sources: read recorded video files or directories of stills.

This is the only place the pipeline touches raw media. Everything downstream
receives :class:`Frame` objects.

Coordinate frame: images are OpenCV BGR ``uint8`` arrays of shape ``(H, W, 3)``
in **image pixel** coordinates -- origin at the top-left pixel centre, ``+x``
right, ``+y`` down.

Variable resolution and variable frame rate are both handled, deliberately and
visibly:

  * **Resolution.** Every emitted frame is delivered at one canonical size, taken
    from ``io.canonical_size`` or else from the first frame. A frame arriving at
    any other size is resized to match and flagged on the :class:`Frame`, and the
    count of resized frames is logged. The homography is defined for exactly one
    resolution, so this keeps metric output coherent -- but a resize is still a
    resample, and metric accuracy degrades with the scale factor.
  * **Frame rate.** Video timestamps come from the container's presentation
    time, so variable-frame-rate recordings keep their true timing. Only if the
    container reports nothing usable does it fall back to ``index / fps``, and
    that fallback is logged.

Failures are still loud: a missing path, an unreadable file, a video with no
usable frame rate, or an image directory with no declared capture rate all raise.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from core.config import IOConfig
from core.logging_utils import get_logger, log_kv

_log = get_logger(__name__)

VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"})


@dataclass(frozen=True)
class Frame:
    """One decoded frame, delivered at the source's canonical size.

    Attributes:
        index: Position in the *emitted* sequence, 0-based and contiguous.
        source_index: Position in the underlying source, before ``start_frame``
            and ``stride`` were applied.
        timestamp_s: Seconds since the start of the source. Taken from the
            container's presentation timestamp where available, so this is
            correct for variable-frame-rate media and is *not* simply
            ``index / fps``.
        image: BGR ``uint8`` array of shape ``(H, W, 3)`` in image pixel
            coordinates, always at the canonical size.
        native_size: ``(width, height)`` the frame was decoded at, before any
            resize. Equal to :attr:`size` unless ``was_resized``.
        was_resized: True if this frame's native size differed from the canonical
            size and it had to be resampled.
        path: File the frame came from, for image-directory sources; None for video.
    """

    index: int
    source_index: int
    timestamp_s: float
    image: np.ndarray
    native_size: tuple[int, int]
    was_resized: bool = False
    path: Path | None = None

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels, after any canonicalising resize."""
        h, w = self.image.shape[:2]
        return w, h


class FrameSource(ABC):
    """Base class for offline frame sources.

    Subclasses provide :meth:`_raw_frames`, yielding every frame of the source in
    capture order together with its container timestamp where one exists. The
    base class applies ``start_frame``, ``stride`` and ``max_frames``, resolves
    timestamps, and canonicalises frame size.
    """

    def __init__(self, path: Path, cfg: IOConfig) -> None:
        self.path = path
        self.cfg = cfg
        self._canonical_size: tuple[int, int] | None = (
            None if cfg.canonical_size is None else (int(cfg.canonical_size[0]), int(cfg.canonical_size[1]))
        )
        self.n_resized = 0
        self.n_timestamp_fallback = 0

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal capture rate in frames per second. Always > 0."""

    @property
    @abstractmethod
    def source_frame_count(self) -> int | None:
        """Total frames in the source, or None when it cannot be known cheaply."""

    @property
    def canonical_size(self) -> tuple[int, int] | None:
        """``(width, height)`` every emitted frame is delivered at.

        Known upfront if configured, otherwise adopted from the first frame read.
        """
        return self._canonical_size

    @abstractmethod
    def _raw_frames(self) -> Iterator[tuple[int, np.ndarray, float | None, Path | None]]:
        """Yield ``(source_index, bgr_image, container_time_s_or_None, path_or_None)``."""

    @abstractmethod
    def close(self) -> None:
        """Release any underlying handles."""

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _canonicalise(
        self, image: np.ndarray, source_index: int
    ) -> tuple[np.ndarray, tuple[int, int], bool]:
        """Resize ``image`` to the canonical size if it differs.

        Returns:
            ``(image, native_size, was_resized)``.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"expected an (H, W, 3) BGR frame at source index {source_index}, "
                f"got shape {image.shape} in {self.path}"
            )
        h, w = image.shape[:2]
        native = (w, h)
        if self._canonical_size is None:
            self._canonical_size = native
            return image, native, False
        if native == self._canonical_size:
            return image, native, False

        target_w, target_h = self._canonical_size
        # INTER_AREA is the right filter for shrinking; it averages rather than
        # point-samples, which matters for thin lane markings.
        interp = cv2.INTER_AREA if (w > target_w or h > target_h) else cv2.INTER_LINEAR
        resized = cv2.resize(image, (target_w, target_h), interpolation=interp)
        if self.n_resized == 0:
            log_kv(
                _log,
                logging.WARNING,
                "source.resolution_change",
                source=self.path.name,
                source_index=source_index,
                native=f"{w}x{h}",
                canonical=f"{target_w}x{target_h}",
                note="frames resampled to the canonical size; metric accuracy degrades",
            )
        self.n_resized += 1
        return resized, native, True

    def _resolve_timestamp(self, source_index: int, container_time_s: float | None) -> float:
        """Prefer the container's presentation time; fall back to ``index / fps``."""
        if (
            self.cfg.use_container_timestamps
            and container_time_s is not None
            and np.isfinite(container_time_s)
            and container_time_s >= 0.0
        ):
            return float(container_time_s)
        self.n_timestamp_fallback += 1
        return source_index / self.fps

    def __iter__(self) -> Iterator[Frame]:
        emitted = 0
        last_t = -np.inf
        for source_index, image, container_time_s, path in self._raw_frames():
            if source_index < self.cfg.start_frame:
                continue
            if (source_index - self.cfg.start_frame) % self.cfg.stride != 0:
                continue
            if self.cfg.max_frames is not None and emitted >= self.cfg.max_frames:
                break
            image, native, was_resized = self._canonicalise(image, source_index)
            t = self._resolve_timestamp(source_index, container_time_s)
            if t < last_t:
                raise ValueError(
                    f"timestamps went backwards at source index {source_index}: "
                    f"{t:.6f}s after {last_t:.6f}s in {self.path}. The container's "
                    f"presentation times are inconsistent; set "
                    f"io.use_container_timestamps: false to derive timing from the "
                    f"frame index instead."
                )
            last_t = t
            yield Frame(
                index=emitted,
                source_index=source_index,
                timestamp_s=t,
                image=image,
                native_size=native,
                was_resized=was_resized,
                path=path,
            )
            emitted += 1
        log_kv(
            _log,
            logging.INFO,
            "source.exhausted",
            source=self.path.name,
            emitted=emitted,
            resized=self.n_resized,
            timestamp_fallbacks=self.n_timestamp_fallback,
        )


class VideoSource(FrameSource):
    """Frames decoded sequentially from a single video file.

    Seeking is done by decoding and discarding, not by ``CAP_PROP_POS_FRAMES``,
    because frame-accurate seeking is unreliable across containers and codecs.
    Each frame's presentation time is read from ``CAP_PROP_POS_MSEC`` before the
    frame is decoded, which is what makes variable-frame-rate input come out with
    correct timing.
    """

    def __init__(self, path: Path, cfg: IOConfig) -> None:
        super().__init__(path, cfg)
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise OSError(f"could not open video: {path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            self._cap.release()
            raise ValueError(
                f"video reports an unusable frame rate ({fps}): {path}. "
                f"Re-encode the file or supply a source with valid timing."
            )
        self._fps = fps
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._count: int | None = count if count > 0 else None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log_kv(
            _log,
            logging.INFO,
            "source.open",
            kind="video",
            source=path.name,
            fps=f"{fps:.3f}",
            frames=self._count if self._count is not None else "unknown",
            size=f"{w}x{h}",
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def source_frame_count(self) -> int | None:
        return self._count

    def _raw_frames(self) -> Iterator[tuple[int, np.ndarray, float | None, Path | None]]:
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            # POS_MSEC must be read *after* the decode: OpenCV then reports the
            # presentation time of the frame just returned. Verified against both
            # MJPG/AVI and H.264/MP4 -- reading it before the decode yields the
            # previous frame's time and shifts every timestamp by one frame.
            pos_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
            container_time_s = pos_ms / 1000.0 if np.isfinite(pos_ms) and pos_ms >= 0 else None
            yield index, image, container_time_s, None
            index += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def _natural_key(path: Path) -> tuple[object, ...]:
    """Sort key splitting digit runs, so ``frame2`` precedes ``frame10``."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


class ImageDirSource(FrameSource):
    """Frames read from a directory of stills, ordered by natural filename sort.

    A directory carries no timing, so ``IOConfig.image_dir_fps`` must be set;
    it is used solely to derive :attr:`Frame.timestamp_s`.
    """

    def __init__(self, path: Path, cfg: IOConfig) -> None:
        super().__init__(path, cfg)
        if cfg.image_dir_fps is None:
            raise ValueError(
                f"image directory source requires io.image_dir_fps to be set: {path}. "
                f"A folder of stills has no intrinsic frame rate, and derived "
                f"velocities and time-to-collision depend on it."
            )
        self._fps = cfg.image_dir_fps
        exts = set(cfg.image_extensions)
        self._files = sorted(
            (p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts),
            key=_natural_key,
        )
        if not self._files:
            raise FileNotFoundError(f"no images with extensions {sorted(exts)} in {path}")
        log_kv(
            _log,
            logging.INFO,
            "source.open",
            kind="image_dir",
            source=path.name,
            fps=f"{self._fps:.3f}",
            frames=len(self._files),
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def source_frame_count(self) -> int | None:
        return len(self._files)

    def _raw_frames(self) -> Iterator[tuple[int, np.ndarray, float | None, Path | None]]:
        for index, file in enumerate(self._files):
            image = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"could not decode image: {file}")
            # Stills have no container timing; the base class derives it from fps.
            yield index, image, None, file

    def close(self) -> None:
        return None


def open_source(path: str | Path, cfg: IOConfig) -> FrameSource:
    """Open ``path`` as a video file or a directory of stills.

    Args:
        path: Video file, or directory containing frames.
        cfg: IO settings; ``image_dir_fps`` is required for directories.

    Returns:
        An open :class:`FrameSource`. Use it as a context manager.

    Raises:
        FileNotFoundError: if ``path`` does not exist, or a directory holds no images.
        ValueError: for an unrecognised file suffix or an unusable frame rate.
        OSError: if the media cannot be opened or decoded.
    """
    p = Path(path)
    if p.is_dir():
        return ImageDirSource(p, cfg)
    if p.is_file():
        if p.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(
                f"unrecognised video extension '{p.suffix}': {p}. "
                f"Known extensions: {sorted(VIDEO_EXTENSIONS)}"
            )
        return VideoSource(p, cfg)
    raise FileNotFoundError(f"source not found: {p}")
