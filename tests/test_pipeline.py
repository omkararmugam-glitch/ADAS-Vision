"""End-to-end tests: io behaviour, the stub detector, and the predict/render CLI.

These run the whole Stage 2 path on synthetic media with the stub detector, so
the suite stays fast and needs neither weights nor a GPU. The real backbone is
exercised separately, on real footage.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import run as run_cli
from src.cache import RunCache
from src.calibrate import CameraModel, rectangle_world_points, solve_ground_homography
from src.config import DetectorConfig, IOConfig, load_config
from src.detect import Detector, build_detector
from src.io import open_source
from src.lanes import LaneFitter
from src.stubs import StubDetector
from src.types import LaneStatus
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
        frame[: HEIGHT // 2] = 150
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
            cv2.imwrite(str(d / f"f{i}.png"), np.full((size[1], size[0], 3), 10 + i * 30, np.uint8))
        return d

    def test_frames_are_delivered_at_one_canonical_size(self, tmp_path: Path) -> None:
        d = self._mixed_dir(tmp_path)
        with open_source(d, IOConfig(image_dir_fps=10.0)) as src:
            frames = list(src)
        assert {f.size for f in frames} == {(640, 480)}
        assert [f.was_resized for f in frames] == [False, False, True, False]
        assert src.n_resized == 1

    def test_explicit_canonical_size_resizes_everything(self, tmp_path: Path) -> None:
        d = self._mixed_dir(tmp_path)
        cfg = IOConfig(image_dir_fps=10.0, canonical_size=(160, 120))
        with open_source(d, cfg) as src:
            frames = list(src)
        assert {f.size for f in frames} == {(160, 120)}
        assert all(f.was_resized for f in frames)

    def test_a_resize_is_visible_not_hidden(self, tmp_path: Path) -> None:
        d = self._mixed_dir(tmp_path)
        with open_source(d, IOConfig(image_dir_fps=10.0)) as src:
            resized = [f for f in src if f.was_resized]
        assert len(resized) == 1 and resized[0].native_size != resized[0].size


class TestVariableFrameRate:
    def test_video_timestamps_come_from_the_container(self, road_video: Path) -> None:
        with open_source(road_video, IOConfig()) as src:
            times = [f.timestamp_s for f in src]
        assert times == pytest.approx([i / 25.0 for i in range(12)], abs=1e-3)
        assert src.n_timestamp_fallback == 0

    def test_fallback_to_index_over_fps_is_counted(self, road_video: Path) -> None:
        with open_source(road_video, IOConfig(use_container_timestamps=False)) as src:
            frames = list(src)
        assert src.n_timestamp_fallback == len(frames)

    def test_timestamps_reflect_stride(self, road_video: Path) -> None:
        with open_source(road_video, IOConfig(stride=3)) as src:
            times = [f.timestamp_s for f in src]
        assert times == pytest.approx([0.0, 0.12, 0.24, 0.36], abs=1e-3)


# --------------------------------------------------------------------------- #
# stub detector
# --------------------------------------------------------------------------- #


class TestStubDetector:
    def test_implements_the_detector_interface(self, camera: CameraModel) -> None:
        det = StubDetector(camera, DetectorConfig(kind="stub"))
        assert isinstance(det, Detector)
        assert det.name == "stub"
        assert det.weights_hash

    def test_built_by_the_factory(self, camera: CameraModel) -> None:
        det = build_detector(DetectorConfig(kind="stub"), camera)
        assert isinstance(det, StubDetector)

    def test_factory_requires_a_camera_for_the_stub(self) -> None:
        with pytest.raises(ValueError, match="requires a CameraModel"):
            build_detector(DetectorConfig(kind="stub"), None)

    def test_detections_land_at_the_distance_they_encode(self, camera: CameraModel) -> None:
        det = StubDetector(camera, DetectorConfig(kind="stub"))
        det.set_time(2.0)
        result = det.infer(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
        assert len(result) == 1
        truth = det.ground_truth()["stub_vehicle_distance_m"]
        recovered = camera.image_to_ground(result.bottom_centres_px())[0]
        assert recovered[1] == pytest.approx(truth, abs=1e-3)

    def test_vehicle_approaches_over_time(self, camera: CameraModel) -> None:
        det = StubDetector(camera, DetectorConfig(kind="stub"))
        widths = []
        for t in (0.0, 3.0, 6.0, 9.0):
            det.set_time(t)
            box = det.infer(np.zeros((HEIGHT, WIDTH, 3), np.uint8)).boxes_px[0]
            widths.append(box[2] - box[0])
        assert widths == sorted(widths), "boxes must grow as the vehicle closes"

    def test_masks_are_produced_at_frame_resolution(self, camera: CameraModel) -> None:
        det = StubDetector(camera, DetectorConfig(kind="stub"))
        det.set_time(1.0)
        r = det.infer(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
        assert r.drivable_mask.shape == (HEIGHT, WIDTH)
        assert r.lane_mask.shape == (HEIGHT, WIDTH)
        assert r.drivable_mask.any() and r.lane_mask.any()

    def test_native_masks_are_smaller_than_the_frame(self, camera: CameraModel) -> None:
        det = StubDetector(camera, DetectorConfig(kind="stub"))
        det.set_time(1.0)
        r = det.infer(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
        assert r.native_masks is not None
        assert r.native_masks.drivable.size < r.drivable_mask.size

    def test_lane_fitting_recovers_the_offsets_the_stub_painted(
        self, camera: CameraModel
    ) -> None:
        """Full loop: stub paints markings through calibration, fitter recovers them."""
        from src.config import BEVConfig, LaneFitConfig

        det = StubDetector(camera, DetectorConfig(kind="stub"), lane_half_width_m=1.85)
        det.set_time(0.0)
        result = det.infer(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
        lanes = LaneFitter(camera, BEVConfig(y_min_m=4, y_max_m=40), LaneFitConfig()).fit(
            result.lane_mask
        )
        assert lanes.status is LaneStatus.OK
        assert lanes.lane_width_m == pytest.approx(3.7, abs=0.15)
        assert lanes.ego_offset_m == pytest.approx(0.0, abs=0.10)

    def test_score_threshold_filters(self, camera: CameraModel) -> None:
        det = StubDetector(camera, DetectorConfig(kind="stub", score_threshold=0.99))
        det.set_time(0.0)
        assert len(det.infer(np.zeros((HEIGHT, WIDTH, 3), np.uint8))) == 0

    def test_output_is_deterministic(self, camera: CameraModel) -> None:
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        a = StubDetector(camera, DetectorConfig(kind="stub"))
        b = StubDetector(camera, DetectorConfig(kind="stub"))
        a.set_time(1.5)
        b.set_time(1.5)
        assert a.infer(frame).to_dict() == b.infer(frame).to_dict()


# --------------------------------------------------------------------------- #
# run.py: predict and render
# --------------------------------------------------------------------------- #


@pytest.fixture
def project(tmp_path: Path, camera: CameraModel, road_video: Path):
    """A config + saved homography + video, all under tmp_path."""
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
                "detector:",
                "  kind: stub",
                "bev:",
                "  y_min_m: 4.0",
                "  y_max_m: 40.0",
            ]
        ),
        encoding="utf-8",
    )
    return cfg_path, road_video, tmp_path


def _cache_for(cfg_path: Path, video: Path) -> RunCache:
    from src.calibrate import file_sha256_of_text

    cfg = load_config(cfg_path)
    return RunCache.open(
        cfg, video,
        calibration_hash=file_sha256_of_text(cfg.calibration.homography_path),
        model_name="stub", weights_hash=StubDetector.VERSION, class_names=("vehicle",),
    )


class TestPredictAndRender:
    def test_predict_populates_the_cache(self, project) -> None:
        cfg_path, video, _ = project
        assert run_cli.main(["--config", str(cfg_path), "predict", "--source", str(video)]) == 0
        cache = _cache_for(cfg_path, video)
        assert cache.is_reusable()
        m = cache.existing_manifest()
        assert m.frame_count == 12
        assert m.model_name == "stub"
        assert m.class_names == ["vehicle"]
        assert len(list(cache.masks_dir.glob("*.png"))) == 24

    def test_render_writes_an_mp4_from_the_cache(self, project) -> None:
        cfg_path, video, tmp_path = project
        run_cli.main(["--config", str(cfg_path), "predict", "--source", str(video)])
        out = tmp_path / "annotated.mp4"
        assert run_cli.main(
            ["--config", str(cfg_path), "render", "--source", str(video), "--out", str(out)]
        ) == 0
        assert out.is_file() and out.stat().st_size > 0
        cap = cv2.VideoCapture(str(out))
        try:
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
        finally:
            cap.release()

    def test_render_works_with_the_model_weights_absent(self, project, monkeypatch) -> None:
        """render must never touch a model. Proven by making loading one fail."""
        cfg_path, video, tmp_path = project
        run_cli.main(["--config", str(cfg_path), "predict", "--source", str(video)])

        import src.detect as detect_mod

        def explode(*_a, **_k):
            raise AssertionError("render must not build a detector")

        monkeypatch.setattr(detect_mod, "build_detector", explode)
        monkeypatch.setattr(run_cli, "build_detector", explode)
        import torch

        monkeypatch.setattr(
            torch.jit, "load", lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("render must not load weights")
            )
        )
        out = tmp_path / "no_weights.mp4"
        assert run_cli.main(
            ["--config", str(cfg_path), "render", "--source", str(video), "--out", str(out)]
        ) == 0
        assert out.is_file()

    def test_render_without_a_cache_says_what_to_run(self, project) -> None:
        cfg_path, video, _ = project
        with pytest.raises(FileNotFoundError, match="run.py predict"):
            run_cli.main(["--config", str(cfg_path), "render", "--source", str(video)])

    def test_second_predict_reuses_the_cache(self, project) -> None:
        cfg_path, video, _ = project
        argv = ["--config", str(cfg_path), "predict", "--source", str(video)]
        run_cli.main(argv)
        cache = _cache_for(cfg_path, video)
        created = cache.existing_manifest().created_utc
        run_cli.main(argv)
        assert cache.existing_manifest().created_utc == created

    def test_refresh_rebuilds(self, project) -> None:
        cfg_path, video, _ = project
        argv = ["--config", str(cfg_path), "predict", "--source", str(video)]
        run_cli.main(argv)
        cache = _cache_for(cfg_path, video)
        (cache.run_dir / "sentinel.txt").write_text("x", encoding="utf-8")
        run_cli.main([*argv, "--refresh"])
        assert not (cache.run_dir / "sentinel.txt").exists()

    def test_overlay_restyle_does_not_re_run_prediction(self, project) -> None:
        cfg_path, video, tmp_path = project
        base = cfg_path.read_text(encoding="utf-8")
        run_cli.main(["--config", str(cfg_path), "predict", "--source", str(video)])
        cache = _cache_for(cfg_path, video)
        created = cache.existing_manifest().created_utc

        cfg_path.write_text(base + "\noverlay:\n  box_thickness: 6\n  hud: false\n", "utf-8")
        run_cli.main(["--config", str(cfg_path), "render", "--source", str(video),
                      "--out", str(tmp_path / "restyled.mp4")])
        assert cache.existing_manifest().created_utc == created
        assert (tmp_path / "restyled.mp4").is_file()

    def test_missing_homography_raises_rather_than_defaulting(self, project) -> None:
        cfg_path, video, tmp_path = project
        cfg_path.write_text(
            f"calibration:\n  homography_path: {(tmp_path / 'nope.yaml').as_posix()}\n",
            encoding="utf-8",
        )
        with pytest.raises(FileNotFoundError, match="homography not found"):
            run_cli.main(["--config", str(cfg_path), "predict", "--source", str(video)])

    def test_max_frames_override(self, project) -> None:
        cfg_path, video, tmp_path = project
        run_cli.main(["--config", str(cfg_path), "predict", "--source", str(video),
                      "--max-frames", "4"])
        run_cli.main(["--config", str(cfg_path), "render", "--source", str(video),
                      "--max-frames", "4", "--out", str(tmp_path / "short.mp4")])
        cap = cv2.VideoCapture(str(tmp_path / "short.mp4"))
        try:
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
        finally:
            cap.release()

    def test_lane_geometry_reaches_the_cache(self, project) -> None:
        cfg_path, video, _ = project
        run_cli.main(["--config", str(cfg_path), "predict", "--source", str(video)])
        records = list(_cache_for(cfg_path, video).read(with_masks=False))
        assert all(r.lanes.status is LaneStatus.OK for r in records)
        assert all(r.lanes.lane_width_m == pytest.approx(3.7, abs=0.2) for r in records)
