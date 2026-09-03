"""Tests for frame sources, using synthetic video and synthetic stills.

All fixtures are generated in a tmp_path; nothing here touches real datasets.
Synthetic frames encode their own index as a uniform grey level so that ordering
and stride can be asserted on decoded content, not just on frame count.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.config import IOConfig, load_config
from core.io import Frame, ImageDirSource, VideoSource, open_source

WIDTH, HEIGHT = 64, 48


def _synthetic_frame(index: int, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Uniform grey BGR frame encoding ``index`` as its level, for identity checks.

    Grey levels spaced 20 apart survive MJPG round-tripping, where a saturated
    single-channel marker would not: chroma subsampling shifts it by a few counts.
    """
    level = 10 + 20 * index
    if level > 255:
        raise ValueError(f"index {index} exceeds the encodable range")
    return np.full((height, width, 3), level, dtype=np.uint8)


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """Ten stills named out of lexical order (frame2.png sorts before frame10.png)."""
    d = tmp_path / "stills"
    d.mkdir()
    for i in range(10):
        assert cv2.imwrite(str(d / f"frame{i}.png"), _synthetic_frame(i))
    return d


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    """Ten-frame lossless-ish MJPG video at 20 fps."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (WIDTH, HEIGHT))
    assert writer.isOpened(), "OpenCV build cannot write MJPG AVI"
    for i in range(10):
        writer.write(_synthetic_frame(i))
    writer.release()
    return path


def _marker(frame: Frame) -> int:
    """Recover the index encoded in a synthetic frame, tolerant of codec noise."""
    return int(round((float(frame.image.mean()) - 10.0) / 20.0))


class TestImageDirSource:
    def test_reads_all_frames_in_natural_order(self, image_dir: Path) -> None:
        with open_source(image_dir, IOConfig(image_dir_fps=10.0)) as src:
            frames = list(src)
        assert isinstance(src, ImageDirSource)
        assert isinstance(frames[0], Frame)
        assert len(frames) == 10
        assert [_marker(f) for f in frames] == list(range(10))
        assert [f.index for f in frames] == list(range(10))
        assert [f.path.name for f in frames[:3]] == ["frame0.png", "frame1.png", "frame2.png"]

    def test_timestamps_follow_declared_fps(self, image_dir: Path) -> None:
        with open_source(image_dir, IOConfig(image_dir_fps=10.0)) as src:
            times = [f.timestamp_s for f in src]
        assert times == pytest.approx([i / 10.0 for i in range(10)])

    def test_size_and_count_reported(self, image_dir: Path) -> None:
        with open_source(image_dir, IOConfig(image_dir_fps=10.0)) as src:
            assert src.source_frame_count == 10
            assert src.canonical_size is None  # adopted from the first frame read
            next(iter(src))
            assert src.canonical_size == (WIDTH, HEIGHT)

    def test_missing_fps_raises(self, image_dir: Path) -> None:
        with pytest.raises(ValueError, match="image_dir_fps"):
            open_source(image_dir, IOConfig())

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="no images"):
            open_source(tmp_path / "empty", IOConfig(image_dir_fps=10.0))

    def test_non_matching_extensions_are_ignored(self, image_dir: Path) -> None:
        (image_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
        with open_source(image_dir, IOConfig(image_dir_fps=10.0)) as src:
            assert len(list(src)) == 10

    def test_undecodable_image_raises(self, image_dir: Path) -> None:
        (image_dir / "frame99.png").write_bytes(b"not a png")
        with pytest.raises(OSError, match="could not decode"):
            with open_source(image_dir, IOConfig(image_dir_fps=10.0)) as src:
                list(src)

    def test_odd_sized_frame_is_resized_to_canonical_and_flagged(
        self, image_dir: Path
    ) -> None:
        """A mid-sequence size change is absorbed, not hidden: the frame says so."""
        cv2.imwrite(str(image_dir / "frame5b.png"), _synthetic_frame(5, width=32, height=24))
        with open_source(image_dir, IOConfig(image_dir_fps=10.0)) as src:
            frames = list(src)
        assert {f.size for f in frames} == {(WIDTH, HEIGHT)}
        odd = [f for f in frames if f.was_resized]
        assert len(odd) == 1 and odd[0].native_size == (32, 24)
        assert src.n_resized == 1


class TestVideoSource:
    def test_reads_all_frames(self, video_file: Path) -> None:
        with open_source(video_file, IOConfig()) as src:
            frames = list(src)
        assert isinstance(src, VideoSource)
        assert len(frames) == 10
        assert [_marker(f) for f in frames] == list(range(10))
        assert all(f.path is None for f in frames)

    def test_fps_and_timestamps_come_from_the_container(self, video_file: Path) -> None:
        with open_source(video_file, IOConfig()) as src:
            assert src.fps == pytest.approx(20.0)
            times = [f.timestamp_s for f in src]
        assert times == pytest.approx([i / 20.0 for i in range(10)])

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="source not found"):
            open_source(tmp_path / "absent.mp4", IOConfig())

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        odd = tmp_path / "clip.bin"
        odd.write_bytes(b"\x00\x01")
        with pytest.raises(ValueError, match="unrecognised video extension"):
            open_source(odd, IOConfig())

    def test_unopenable_video_raises(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")
        with pytest.raises(OSError, match="could not open video"):
            open_source(broken, IOConfig())


class TestFrameSelection:
    """start_frame / stride / max_frames apply identically to both source kinds."""

    def test_start_frame_skips_leading_frames(self, image_dir: Path) -> None:
        with open_source(image_dir, IOConfig(image_dir_fps=10.0, start_frame=4)) as src:
            frames = list(src)
        assert [_marker(f) for f in frames] == [4, 5, 6, 7, 8, 9]
        assert [f.index for f in frames] == [0, 1, 2, 3, 4, 5]
        assert [f.source_index for f in frames] == [4, 5, 6, 7, 8, 9]

    def test_stride_subsamples_relative_to_start(self, image_dir: Path) -> None:
        cfg = IOConfig(image_dir_fps=10.0, start_frame=1, stride=3)
        with open_source(image_dir, cfg) as src:
            frames = list(src)
        assert [_marker(f) for f in frames] == [1, 4, 7]

    def test_timestamps_track_source_index_not_emitted_index(self, image_dir: Path) -> None:
        cfg = IOConfig(image_dir_fps=10.0, stride=2)
        with open_source(image_dir, cfg) as src:
            frames = list(src)
        assert [f.timestamp_s for f in frames] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])

    def test_max_frames_caps_output(self, image_dir: Path) -> None:
        cfg = IOConfig(image_dir_fps=10.0, max_frames=3)
        with open_source(image_dir, cfg) as src:
            assert len(list(src)) == 3

    def test_selection_applies_to_video_too(self, video_file: Path) -> None:
        cfg = IOConfig(start_frame=2, stride=2, max_frames=3)
        with open_source(video_file, cfg) as src:
            frames = list(src)
        assert [_marker(f) for f in frames] == [2, 4, 6]
        assert [f.timestamp_s for f in frames] == pytest.approx([0.10, 0.20, 0.30])


class TestFrame:
    def test_size_is_width_height(self) -> None:
        frame = Frame(
            index=0,
            source_index=0,
            timestamp_s=0.0,
            image=_synthetic_frame(0),
            native_size=(WIDTH, HEIGHT),
        )
        assert frame.size == (WIDTH, HEIGHT)
        assert frame.was_resized is False


class TestIOConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"stride": 0},
            {"start_frame": -1},
            {"max_frames": 0},
            {"image_dir_fps": 0.0},
            {"image_dir_fps": -5.0},
            {"image_extensions": []},
            {"image_extensions": [".PNG"]},
            {"image_extensions": ["png"]},
            {"unknown_key": 1},
        ],
    )
    def test_invalid_values_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(Exception):
            IOConfig(**kwargs)

    def test_shipped_default_config_loads(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
        assert cfg.io.stride == 1
        assert cfg.io.image_dir_fps is None

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="config not found"):
            load_config(tmp_path / "nope.yaml")

    def test_non_mapping_config_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_config(bad)
