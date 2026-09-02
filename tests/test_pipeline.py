"""End-to-end tests: variable resolution/fps handling, the stub, and run.py.

These exercise the whole Stage 1 path on synthetic media -- ingest, calibration,
cache, overlay, MP4 -- without touching the real sample video, so the suite stays
fast and hermetic.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import run as run_cli
from src.cache import RunCache
from src.calibrate import CameraModel, rectangle_world_points, solve_ground_homography
from src.config import Config, DetectorConfig, IOConfig
from src.io import open_source
from src.stubs import StubDetector
from tests.test_calibrate import HEIGHT, WIDTH, project_ground_to_image


@pytest.fixture
def camera() -> CameraModel:
    world = rectangle_world_points(3.5, 6.0, 30.0)
    H, _ = solve_ground_homography(project_ground_to_image(world), world)
    return CameraModel(H_image_to_ground=H, image_size=(WIDTH, HEIGHT))


@pytest.fixture
def road_video(tmp_path: Path) -> Path:
    """A short synthetic clip at the calibration resolution."""
    path = tmp_path / "road.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (WIDTH, HEIGHT))
    assert w.isOpened()
    for i in range(12):
        frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
        frame[: HEIGHT // 2] = 150  # crude sky/road split
        cv2.putText(frame, str(i), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        w.write(frame)
    w.release()
    return path


# --------------------------------------------------------------------------- #
# io: variable resolution and variable frame rate
# --------------------------------------------------------------------------- #


class TestVariableResolution:
    def _mixed_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "mixed"
        d.mkdir()
        for i, size in enumerate([(640, 480), (640, 480), (320, 240), (640, 480)]):
            img = np.full((size[1], size[0], 3), 10 + i * 30, dtype=np.uint8)
            cv2.imwrite(str(d / f"f{i}.png"), img)
        return d

    def test_frames_are_delivered_at_one_canonical_size(self, tmp_path: Path) -> None:
        d = self._mixed_dir(tmp_path)
        with open_source(d, IOConfig(image_dir_fps=10.0)) as src:
            frames = list(src)
        assert {f.size for f in frames} == {(640, 480)}, "sizes were not canonicalised"
        assert [f.native_size for f in frames] == [
            (640, 480), (640, 480), (320, 240), (640, 480)
        ]
        assert [f.was_resized for f in frames] == [False, False, True, False]
        assert src.n_resized == 1

    def test_explicit_canonical_size_resizes_everything(self, tmp_path: Path) -> None:
        d = self._mixed_dir(tmp_path)
        cfg = IOConfig(image_dir_fps=10.0, canonical_size=(160, 120))
        with open_source(d, cfg) as src:
            frames = list(src)
        assert {f.size for f in frames} == {(160, 120)}
        assert all(f.was_resized for f in frames)
        assert src.canonical_size == (160, 120)

    def test_canonical_size_known_before_reading_when_configured(self, tmp_path: Path) -> None:
        d = self._mixed_dir(tmp_path)
        cfg = IOConfig(image_dir_fps=10.0, canonical_size=(200, 100))
        with open_source(d, cfg) as src:
            assert src.canonical_size == (200, 100)

    def test_a_resize_is_visible_on_the_frame_not_hidden(self, tmp_path: Path) -> None:
        """The caller can always tell a frame was resampled."""
        d = self._mixed_dir(tmp_path)
        with open_source(d, IOConfig(image_dir_fps=10.0)) as src:
            resized = [f for f in src if f.was_resized]
        assert len(resized) == 1
        assert resized[0].native_size != resized[0].size


class TestVariableFrameRate:
    def test_video_timestamps_come_from_the_container(self, road_video: Path) -> None:
        with open_source(road_video, IOConfig()) as src:
            frames = list(src)
        assert [f.timestamp_s for f in frames] == pytest.approx(
            [i / 25.0 for i in range(12)], abs=1e-3
        )
        assert src.n_timestamp_fallback == 0

    def test_falling_back_to_index_over_fps_is_counted(self, road_video: Path) -> None:
        cfg = IOConfig(use_container_timestamps=False)
        with open_source(road_video, cfg) as src:
            frames = list(src)
        assert src.n_timestamp_fallback == len(frames)
        assert [f.timestamp_s for f in frames] == pytest.approx([i / 25.0 for i in range(12)])

    def test_image_dirs_always_use_the_declared_rate(self, tmp_path: Path) -> None:
        d = tmp_path / "stills"
        d.mkdir()
        for i in range(4):
            cv2.imwrite(str(d / f"{i}.png"), np.zeros((20, 20, 3), dtype=np.uint8))
        with open_source(d, IOConfig(image_dir_fps=8.0)) as src:
            times = [f.timestamp_s for f in src]
        assert times == pytest.approx([0.0, 0.125, 0.25, 0.375])

    def test_timestamps_reflect_stride_not_emission_order(self, road_video: Path) -> None:
        with open_source(road_video, IOConfig(stride=3)) as src:
            times = [f.timestamp_s for f in src]
        assert times == pytest.approx([0.0, 0.12, 0.24, 0.36], abs=1e-3)


# --------------------------------------------------------------------------- #
# stub detector
# --------------------------------------------------------------------------- #


class TestStubDetector:
    def _frame(self, index: int, t: float):
        from src.io import Frame

        return Frame(
            index=index,
            source_index=index,
            timestamp_s=t,
            image=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
            native_size=(WIDTH, HEIGHT),
        )

    def test_detections_land_at_the_distance_they_encode(self, camera: CameraModel) -> None:
        """The stub's whole point: its output survives the geometry round-trip."""
        det = StubDetector(camera, DetectorConfig())
        result = det.detect(self._frame(0, 2.0))
        assert result.detections, "stub produced no detection"
        truth_m = result.extras["stub_vehicle_distance_m"]
        contact = np.array([result.detections[0].bottom_centre_px])
        recovered = camera.image_to_ground(contact)[0]
        assert recovered[1] == pytest.approx(truth_m, abs=1e-3)

    def test_lateral_position_also_round_trips(self, camera: CameraModel) -> None:
        det = StubDetector(camera, DetectorConfig())
        result = det.detect(self._frame(0, 5.0))
        contact = np.array([result.detections[0].bottom_centre_px])
        recovered = camera.image_to_ground(contact)[0]
        assert recovered[0] == pytest.approx(result.extras["stub_vehicle_lateral_m"], abs=1e-3)

    def test_vehicle_approaches_over_time(self, camera: CameraModel) -> None:
        det = StubDetector(camera, DetectorConfig())
        distances = [
            det.detect(self._frame(i, t)).extras["stub_vehicle_distance_m"]
            for i, t in enumerate([0.0, 2.0, 4.0, 6.0])
        ]
        assert distances == sorted(distances, reverse=True), "vehicle should close in"

    def test_boxes_grow_as_the_vehicle_nears(self, camera: CameraModel) -> None:
        det = StubDetector(camera, DetectorConfig())
        far = det.detect(self._frame(0, 0.0)).detections[0].bbox_px
        near = det.detect(self._frame(1, 8.0)).detections[0].bbox_px
        assert (near[2] - near[0]) > (far[2] - far[0])

    def test_lane_lines_are_produced_with_marking_types(self, camera: CameraModel) -> None:
        result = StubDetector(camera, DetectorConfig()).detect(self._frame(0, 1.0))
        roles = {line.role: line.kind for line in result.lanes}
        assert roles == {"ego_left": "solid", "ego_right": "dashed"}

    def test_lane_lines_sit_at_the_configured_offset(self, camera: CameraModel) -> None:
        det = StubDetector(camera, DetectorConfig(), lane_half_width_m=1.75)
        result = det.detect(self._frame(0, 1.0))
        left = next(line for line in result.lanes if line.role == "ego_left")
        ground = camera.image_to_ground(left.points_px)
        assert ground[:, 0] == pytest.approx(-1.75, abs=1e-4)

    def test_output_is_deterministic(self, camera: CameraModel) -> None:
        a = StubDetector(camera, DetectorConfig(seed=0)).detect(self._frame(3, 1.5))
        b = StubDetector(camera, DetectorConfig(seed=0)).detect(self._frame(3, 1.5))
        assert a.to_dict() == b.to_dict()

    def test_score_threshold_filters(self, camera: CameraModel) -> None:
        det = StubDetector(camera, DetectorConfig(score_threshold=0.99))
        assert det.detect(self._frame(0, 0.0)).detections == []

    def test_drivable_mask_covers_the_lane_and_nothing_above_the_horizon(
        self, camera: CameraModel
    ) -> None:
        det = StubDetector(camera, DetectorConfig())
        mask = det.drivable_mask(HEIGHT, WIDTH)
        assert mask.any(), "drivable mask is empty"
        horizon = float(camera.horizon_v(WIDTH / 2)[0])
        assert not mask[: int(horizon)].any(), "drivable area leaked above the horizon"


# --------------------------------------------------------------------------- #
# run.py end to end
# --------------------------------------------------------------------------- #


@pytest.fixture
def project(tmp_path: Path, camera: CameraModel, road_video: Path) -> tuple[Path, Path, Path]:
    """A config + saved homography + video, wired to write under tmp_path."""
    hom = tmp_path / "homography.yaml"
    camera.save(hom)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "calibration:",
                f"  homography_path: {hom.as_posix()}",
                "cache:",
                f"  root: {(tmp_path / 'outputs').as_posix()}",
                "  model_version: stub-test",
            ]
        ),
        encoding="utf-8",
    )
    return cfg_path, road_video, tmp_path


class TestRunCLI:
    def test_process_writes_a_cache_and_an_mp4(self, project) -> None:
        cfg_path, video, tmp_path = project
        out = tmp_path / "annotated.mp4"
        assert run_cli.main(
            ["--config", str(cfg_path), "process", "--source", str(video), "--out", str(out)]
        ) == 0

        assert out.is_file() and out.stat().st_size > 0
        cap = cv2.VideoCapture(str(out))
        try:
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
        finally:
            cap.release()

        from src.config import load_config

        cache = RunCache.open(load_config(cfg_path), video)
        assert cache.is_reusable()
        assert cache.existing_manifest().frame_count == 12
        assert cache.existing_manifest().model_version == "stub-test"
        assert cache.existing_manifest().extra["undistort_applied"] is False

    def test_second_run_reuses_the_cache(self, project, caplog) -> None:
        cfg_path, video, tmp_path = project
        argv = ["--config", str(cfg_path), "process", "--source", str(video),
                "--out", str(tmp_path / "a.mp4")]
        run_cli.main(argv)
        from src.config import load_config

        cache = RunCache.open(load_config(cfg_path), video)
        first = cache.existing_manifest().created_utc

        run_cli.main(argv)
        assert cache.existing_manifest().created_utc == first, "cache was rebuilt unnecessarily"

    def test_refresh_rebuilds_the_cache(self, project) -> None:
        cfg_path, video, tmp_path = project
        argv = ["--config", str(cfg_path), "process", "--source", str(video),
                "--out", str(tmp_path / "a.mp4")]
        run_cli.main(argv)
        from src.config import load_config

        cache = RunCache.open(load_config(cfg_path), video)
        (cache.run_dir / "sentinel.txt").write_text("x", encoding="utf-8")
        run_cli.main([*argv, "--refresh"])
        assert not (cache.run_dir / "sentinel.txt").exists(), "refresh did not clear the run dir"

    def test_missing_homography_raises_rather_than_defaulting(self, project) -> None:
        cfg_path, video, tmp_path = project
        cfg_path.write_text(
            f"calibration:\n  homography_path: {(tmp_path / 'nope.yaml').as_posix()}\n",
            encoding="utf-8",
        )
        with pytest.raises(FileNotFoundError, match="homography not found"):
            run_cli.main(["--config", str(cfg_path), "process", "--source", str(video)])

    def test_max_frames_override_is_applied(self, project) -> None:
        cfg_path, video, tmp_path = project
        run_cli.main(
            ["--config", str(cfg_path), "process", "--source", str(video),
             "--out", str(tmp_path / "short.mp4"), "--max-frames", "4"]
        )
        cap = cv2.VideoCapture(str(tmp_path / "short.mp4"))
        try:
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
        finally:
            cap.release()

    def test_cached_results_survive_an_overlay_change(self, project) -> None:
        """Restyling re-renders from cache; it must not re-run the detector."""
        cfg_path, video, tmp_path = project
        base = cfg_path.read_text(encoding="utf-8")
        run_cli.main(["--config", str(cfg_path), "process", "--source", str(video),
                      "--out", str(tmp_path / "a.mp4")])
        from src.config import load_config

        cache = RunCache.open(load_config(cfg_path), video)
        created = cache.existing_manifest().created_utc

        cfg_path.write_text(base + "\noverlay:\n  box_thickness: 6\n  hud: false\n", "utf-8")
        run_cli.main(["--config", str(cfg_path), "process", "--source", str(video),
                      "--out", str(tmp_path / "b.mp4")])
        assert cache.existing_manifest().created_utc == created
        assert (tmp_path / "b.mp4").is_file()
