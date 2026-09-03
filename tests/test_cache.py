"""Tests for the two-stage cache: round-tripping, staleness, and resume.

The staleness tests carry the weight here. The cache exists so that retuning
downstream logic never re-runs a network; the flip side is that it must refuse
to serve predictions produced by a different model, calibration, config or code.
Each identity field gets its own test, because a hash that is recorded but never
compared is worse than no hash at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.cache import Manifest, RunCache, SourceFingerprint, code_version, default_run_id
from core.config import Config
from core.types import CLASS_IDS, FrameRecord, FrameResult, LaneFit, LaneSet, LaneStatus

MASK_H, MASK_W = 180, 320
FRAME_SIZE = (640, 360)

IDENTITY = dict(
    calibration_hash="cal0",
    model_name="yolopv2",
    weights_hash="w0",
    class_names=("vehicle", "person"),
)


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"pretend this is a video" * 100)
    return p


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config.model_validate({"cache": {"root": str(tmp_path / "outputs")}})


def _open(cfg: Config, source: Path, **overrides) -> RunCache:
    kwargs = {**IDENTITY, **overrides}
    return RunCache.open(cfg, source, extra={"frame_size": list(FRAME_SIZE)}, **kwargs)


def _result(i: int) -> FrameResult:
    drivable = np.zeros((MASK_H, MASK_W), dtype=np.uint8)
    drivable[MASK_H // 2 :, :] = 1
    lane = np.zeros((MASK_H, MASK_W), dtype=np.uint8)
    lane[:, 40 + i : 46 + i] = 1
    from core.types import NativeMasks

    return FrameResult(
        boxes_px=np.array([[10.0, 20.0, 110.0, 140.0]], dtype=np.float32),
        class_ids=np.array([CLASS_IDS["vehicle"]], dtype=np.int32),
        scores=np.array([0.87], dtype=np.float32),
        drivable_mask=np.zeros(FRAME_SIZE[::-1], dtype=np.uint8),
        lane_mask=np.zeros(FRAME_SIZE[::-1], dtype=np.uint8),
        native_masks=NativeMasks(drivable=drivable, lane=lane, original_size=FRAME_SIZE),
    )


def _record(i: int) -> FrameRecord:
    return FrameRecord(
        frame_index=i,
        source_index=i * 2,
        timestamp_s=i * 0.04,
        result=_result(i),
        lanes=LaneSet(
            status=LaneStatus.OK,
            ego_left=LaneFit(np.array([1e-4, 0.01, -1.75]), "ego_left", 800, (5.0, 35.0)),
            ego_right=LaneFit(np.array([1e-4, 0.01, 1.75]), "ego_right", 700, (5.0, 30.0)),
            curvature_radius_m=850.0,
            ego_offset_m=-0.12,
            lane_width_m=3.5,
            eval_distance_m=10.0,
        ),
    )


def _populate(cfg: Config, source: Path, n: int = 4, **overrides) -> RunCache:
    cache = _open(cfg, source, **overrides)
    with cache.writer() as w:
        for i in range(n):
            w.write(_record(i))
    return cache


class TestRoundTrip:
    def test_records_read_back_identically(self, cfg: Config, source_file: Path) -> None:
        cache = _populate(cfg, source_file)
        got = list(cache.read())
        assert [r.frame_index for r in got] == [0, 1, 2, 3]
        assert got[2].result.boxes_px[0] == pytest.approx([10.0, 20.0, 110.0, 140.0])
        assert got[2].result.class_ids[0] == CLASS_IDS["vehicle"]
        assert got[2].result.scores[0] == pytest.approx(0.87, abs=1e-4)
        assert got[2].timestamp_s == pytest.approx(0.08)

    def test_lane_geometry_round_trips(self, cfg: Config, source_file: Path) -> None:
        got = list(_populate(cfg, source_file).read())
        lanes = got[0].lanes
        assert lanes.status is LaneStatus.OK
        assert lanes.lane_width_m == pytest.approx(3.5)
        assert lanes.ego_offset_m == pytest.approx(-0.12)
        assert lanes.curvature_radius_m == pytest.approx(850.0)
        assert lanes.ego_left.coeffs == pytest.approx([1e-4, 0.01, -1.75])
        assert lanes.ego_left.n_pixels == 800

    def test_masks_are_stored_at_native_resolution_and_upsampled_on_read(
        self, cfg: Config, source_file: Path
    ) -> None:
        cache = _populate(cfg, source_file)
        stored = sorted(cache.masks_dir.glob("*.png"))
        assert len(stored) == 8, "two masks per frame"

        import cv2

        native = cv2.imread(str(stored[0]), cv2.IMREAD_GRAYSCALE)
        assert native.shape == (MASK_H, MASK_W), "must persist at native, not frame, size"

        record = next(iter(cache.read()))
        assert record.result.drivable_mask.shape == FRAME_SIZE[::-1], "upsampled on read"

    def test_mask_content_survives(self, cfg: Config, source_file: Path) -> None:
        cache = _populate(cfg, source_file)
        record = next(iter(cache.read()))
        # The bottom half was marked drivable; that must still hold after upsampling.
        h = record.result.drivable_mask.shape[0]
        assert record.result.drivable_mask[int(h * 0.75)].any()
        assert not record.result.drivable_mask[int(h * 0.25)].any()

    def test_reading_without_masks_is_allowed(self, cfg: Config, source_file: Path) -> None:
        cache = _populate(cfg, source_file)
        records = list(cache.read(with_masks=False))
        assert len(records) == 4
        assert records[0].lanes.status is LaneStatus.OK

    def test_size_report_is_measured(self, cfg: Config, source_file: Path) -> None:
        report = _populate(cfg, source_file).size_report()
        assert report["frames"] == 4
        assert report["masks_bytes_per_frame"] > 0
        assert report["json_bytes_per_frame"] > 0
        assert report["raw_fullres_bytes_per_frame"] > report["raw_native_bytes_per_frame"]


class TestStaleness:
    """Every identity field must actually gate reuse, not merely be recorded."""

    def test_fresh_cache_is_reusable(self, cfg: Config, source_file: Path) -> None:
        _populate(cfg, source_file)
        assert _open(cfg, source_file).staleness_reason() is None

    def test_absent_cache_reports_so(self, cfg: Config, source_file: Path) -> None:
        assert _open(cfg, source_file).staleness_reason() == "no cache present"

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("calibration_hash", "cal-CHANGED", "calibration hash"),
            ("model_name", "hybridnets", "model name"),
            ("weights_hash", "w-CHANGED", "weights hash"),
        ],
    )
    def test_identity_change_invalidates(
        self, cfg: Config, source_file: Path, field: str, value: str, expected: str
    ) -> None:
        _populate(cfg, source_file)
        reason = _open(cfg, source_file, **{field: value}).staleness_reason()
        assert reason is not None and expected in reason

    def test_code_version_change_invalidates(self, cfg: Config, source_file: Path) -> None:
        cache = _populate(cfg, source_file)
        stored = json.loads(cache.manifest_path.read_text("utf-8"))
        stored["code_version"] = "deadbee-from-last-year"
        cache.manifest_path.write_text(json.dumps(stored), encoding="utf-8")
        reason = _open(cfg, source_file).staleness_reason()
        assert reason is not None and "code version" in reason

    def test_config_change_invalidates(self, cfg: Config, source_file: Path) -> None:
        _populate(cfg, source_file)
        changed = cfg.model_copy(update={"io": cfg.io.model_copy(update={"stride": 4})})
        changed = changed.model_copy(
            update={"cache": changed.cache.model_copy(
                update={"run_id": default_run_id(source_file, cfg.cache_key())})}
        )
        reason = _open(changed, source_file).staleness_reason()
        assert reason is not None and "config hash" in reason

    def test_lane_config_participates_in_the_key(self, cfg: Config, source_file: Path) -> None:
        """Lane thresholds change cached LaneSets, so they must invalidate."""
        changed = cfg.model_copy(
            update={"lanes": cfg.lanes.model_copy(update={"n_windows": 20})}
        )
        assert changed.cache_key() != cfg.cache_key()

    def test_presentation_config_does_NOT_invalidate(self, cfg: Config, source_file: Path) -> None:
        _populate(cfg, source_file)
        restyled = cfg.model_copy(
            update={
                "overlay": cfg.overlay.model_copy(update={"box_thickness": 9, "hud": False}),
                "video": cfg.video.model_copy(update={"fourcc": "avc1"}),
            }
        )
        assert restyled.cache_key() == cfg.cache_key()
        assert _open(restyled, source_file).staleness_reason() is None

    def test_changed_source_invalidates(self, cfg: Config, source_file: Path) -> None:
        _populate(cfg, source_file)
        source_file.write_bytes(b"a completely different video file")
        reason = _open(cfg, source_file).staleness_reason()
        assert reason is not None and "source changed" in reason

    def test_schema_version_mismatch_invalidates(self) -> None:
        base = Manifest(
            run_id="r", config_hash="c", calibration_hash="cal", model_name="m",
            weights_hash="w", code_version="v", schema_version=1, class_names=[],
            source=SourceFingerprint("f", 10, 20), source_path="f", complete=True,
        )
        newer = Manifest(**{**base.__dict__, "schema_version": 2})
        assert "schema version" in base.staleness_reason(newer)

    def test_reading_a_stale_cache_raises(self, cfg: Config, source_file: Path) -> None:
        _populate(cfg, source_file)
        with pytest.raises(ValueError, match="refusing to read a stale cache"):
            list(_open(cfg, source_file, weights_hash="different").read())

    def test_interrupted_run_is_not_reusable(self, cfg: Config, source_file: Path) -> None:
        cache = _open(cfg, source_file)
        with pytest.raises(RuntimeError):
            with cache.writer() as w:
                w.write(_record(0))
                raise RuntimeError("inference blew up halfway")
        reason = _open(cfg, source_file).staleness_reason()
        assert reason is not None and "did not finish" in reason

    def test_corrupt_manifest_invalidates(self, cfg: Config, source_file: Path) -> None:
        cache = _populate(cfg, source_file)
        cache.manifest_path.write_text("{not json", encoding="utf-8")
        assert _open(cfg, source_file).staleness_reason() == "no cache present"


class TestResume:
    def test_interrupted_run_is_resumable(self, cfg: Config, source_file: Path) -> None:
        cache = _open(cfg, source_file)
        with pytest.raises(RuntimeError):
            with cache.writer() as w:
                for i in range(3):
                    w.write(_record(i))
                raise RuntimeError("interrupted")
        assert _open(cfg, source_file).resumable_frames() == 3

    def test_resuming_appends_and_completes(self, cfg: Config, source_file: Path) -> None:
        cache = _open(cfg, source_file)
        with pytest.raises(RuntimeError):
            with cache.writer() as w:
                for i in range(3):
                    w.write(_record(i))
                raise RuntimeError("interrupted")

        resumed = _open(cfg, source_file)
        n = resumed.resumable_frames()
        with resumed.writer(resume_after=n) as w:
            for i in range(n, 6):
                w.write(_record(i))

        assert resumed.is_reusable()
        assert [r.frame_index for r in resumed.read(with_masks=False)] == [0, 1, 2, 3, 4, 5]
        assert resumed.existing_manifest().frame_count == 6

    def test_a_different_model_is_not_resumable(self, cfg: Config, source_file: Path) -> None:
        cache = _open(cfg, source_file)
        with pytest.raises(RuntimeError):
            with cache.writer() as w:
                w.write(_record(0))
                raise RuntimeError("interrupted")
        assert _open(cfg, source_file, weights_hash="other").resumable_frames() == 0

    def test_a_record_without_masks_is_not_counted_as_done(
        self, cfg: Config, source_file: Path
    ) -> None:
        """A frame whose masks never landed must be recomputed, not trusted."""
        cache = _open(cfg, source_file)
        with pytest.raises(RuntimeError):
            with cache.writer() as w:
                for i in range(3):
                    w.write(_record(i))
                raise RuntimeError("interrupted")
        cache._mask_path(2, "lane").unlink()
        assert _open(cfg, source_file).resumable_frames() == 2

    def test_truncated_final_line_is_dropped(self, cfg: Config, source_file: Path) -> None:
        cache = _open(cfg, source_file)
        with pytest.raises(RuntimeError):
            with cache.writer() as w:
                for i in range(3):
                    w.write(_record(i))
                raise RuntimeError("interrupted")
        with cache.frames_path.open("a", encoding="utf-8") as fh:
            fh.write('{"frame_index":3,"source_ind')
        assert _open(cfg, source_file).resumable_frames() == 3

    def test_complete_run_reports_all_frames_resumable(
        self, cfg: Config, source_file: Path
    ) -> None:
        _populate(cfg, source_file, n=4)
        assert _open(cfg, source_file).resumable_frames() == 4


class TestOrdering:
    def test_out_of_order_writes_raise(self, cfg: Config, source_file: Path) -> None:
        cache = _open(cfg, source_file)
        with pytest.raises(ValueError, match="increasing order"):
            with cache.writer() as w:
                w.write(_record(5))
                w.write(_record(2))


class TestManifestFile:
    def test_manifest_records_every_identity_field(self, cfg: Config, source_file: Path) -> None:
        cache = _populate(cfg, source_file)
        raw = json.loads(cache.manifest_path.read_text("utf-8"))
        assert set(raw) >= {
            "run_id", "config_hash", "calibration_hash", "model_name", "weights_hash",
            "code_version", "schema_version", "class_names", "source", "frame_count",
            "complete",
        }
        assert raw["class_names"] == ["vehicle", "person"]
        assert raw["model_name"] == "yolopv2"

    def test_code_version_is_reported(self) -> None:
        v = code_version()
        assert isinstance(v, str) and v


class TestSourceFingerprint:
    def test_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="fingerprint"):
            SourceFingerprint.of(tmp_path / "absent.mp4")

    def test_directory_fingerprint_reflects_contents(self, tmp_path: Path) -> None:
        d = tmp_path / "frames"
        d.mkdir()
        (d / "a.png").write_bytes(b"1234")
        before = SourceFingerprint.of(d)
        (d / "b.png").write_bytes(b"5678")
        assert SourceFingerprint.of(d) != before
