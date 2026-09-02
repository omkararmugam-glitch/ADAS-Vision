"""Frame sources: read recorded video files or directories of stills.

This is the only place the pipeline touches raw media. Everything downstream
receives :class:`Frame` objects.

Coordinate frame: images are OpenCV BGR ``uint8`` arrays of shape ``(H, W, 3)``
in **image pixel** coordinates -- origin at the top-left pixel centre, ``+x``
right, ``+y`` down. Frames are never resized here: the ground-plane homography
is defined against the native capture resolution, so silently rescaling would
invalidate every metric downstream.

Failures are loud. A missing path, an unreadable file, a video with no usable
frame rate, an image directory with no stated capture rate, or a mid-sequence
resolution change all raise rather than degrade.
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

from src.config import IOConfig
from src.logging_utils import get_logger, log_kv

_log = get_logger(__name__)

VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"})


@dataclass(frozen=True)
class Frame:
    """One decoded frame.

    Attributes:
        index: Position in the *emitted* sequence, 0-based and contiguous.
        source_index: Position in the underlying source, before ``start_frame``
            and ``stride`` were applied.
        timestamp_s: Seconds since the start of the source, computed as
            ``source_index / fps``. Recorded media only; there is no wall clock.
        image: BGR ``uint8`` array of shape ``(H, W, 3)`` in image pixel coordinates.
        path: File the frame came from, for image-directory sources; None for video.
    """

    index: int
    source_index: int
    timestamp_s: float
    image: np.ndarray
    path: Path | None = None

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        h, w = self.image.shape[:2]
        return w, h


class FrameSource(ABC):
    """Base class for offline frame sources.

    Subclasses provide :meth:`_raw_frames`, yielding every frame of the source in
    capture order. The base class applies ``start_frame``, ``stride`` and
    ``max_frames``, assigns timestamps, and enforces a constant frame size.
    """

    def __init__(self, path: Path, cfg: IOConfig) -> None:
        self.path = path
        self.cfg = cfg
        self._size: tuple[int, int] | None = None

    @property
    @abstractmethod
    def fps(self) -> float:
        """Capture rate in frames per second. Always > 0."""

    @property
    @abstractmethod
    def source_frame_count(self) -> int | None:
        """Total frames in the source, or None when it cannot be known cheaply."""

    @property
    def size(self) -> tuple[int, int] | None:
        """``(width, height)`` of the frames, known once the first frame is read."""
        return self._size

    @abstractmethod
    def _raw_frames(self) -> Iterator[tuple[int, np.ndarray, Path | None]]:
        """Yield ``(source_index, bgr_image, path_or_none)`` for every source frame."""

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

    def _check_size(self, image: np.ndarray, source_index: int) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"expected an (H, W, 3) BGR frame at source index {source_index}, "
                f"got shape {image.shape} in {self.path}"
            )
        h, w = image.shape[:2]
        if self._size is None:
            self._size = (w, h)
        elif self._size != (w, h):
            raise ValueError(
                f"frame size changed mid-source at index {source_index}: "
                f"{self._size} -> {(w, h)} in {self.path}. The homography is defined "
                f"for a single resolution; mixed sizes cannot be processed together."
            )

    def __iter__(self) -> Iterator[Frame]:
        fps = self.fps
        emitted = 0
        for source_index, image, path in self._raw_frames():
            if source_index < self.cfg.start_frame:
                continue
            if (source_index - self.cfg.start_frame) % self.cfg.stride != 0:
                continue
            if self.cfg.max_frames is not None and emitted >= self.cfg.max_frames:
                break
            self._check_size(image, source_index)
            yield Frame(
                index=emitted,
                source_index=source_index,
                timestamp_s=source_index / fps,
                image=image,
                path=path,
            )
            emitted += 1
        log_kv(
            _log,
            logging.INFO,
            "source.exhausted",
            source=self.path.name,
            emitted=emitted,
            fps=f"{fps:.3f}",
        )


class VideoSource(FrameSource):
    """Frames decoded sequentially from a single video file.

    Seeking is done by decoding and discarding, not by ``CAP_PROP_POS_FRAMES``,
    because frame-accurate seeking is unreliable across containers and codecs.
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
        log_kv(
            _log,
            logging.INFO,
            "source.open",
            kind="video",
            source=path.name,
            fps=f"{fps:.3f}",
            frames=self._count if self._count is not None else "unknown",
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def source_frame_count(self) -> int | None:
        return self._count

    def _raw_frames(self) -> Iterator[tuple[int, np.ndarray, Path | None]]:
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            yield index, image, None
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

    def _raw_frames(self) -> Iterator[tuple[int, np.ndarray, Path | None]]:
        for index, file in enumerate(self._files):
            image = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"could not decode image: {file}")
            yield index, image, file

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
