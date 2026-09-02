"""Tests for the two-stage cache: round-tripping, ordering, and staleness.

The staleness tests are the important ones. The cache exists so that tuning
event thresholds never re-runs a network; the flip side is that it must refuse
to serve results produced under different settings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.cache import Manifest, RunCache, SourceFingerprint, default_run_id
from src.config import Config
from src.types import Detection, FrameResult, LaneLine


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"pretend this is a video" * 100)
    return p


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config.model_validate({"cache": {"root": str(tmp_path / "outputs")}})


def _result(i: int) -> FrameResult:
    return FrameResult(
        frame_index=i,
        source_index=i * 2,
        timestamp_s=i * 0.04,
        detections=[
            Detection(bbox_px=(10.0, 20.0, 110.0, 140.0), label="car", score=0.9, track_id=3)
        ],
        lanes=[
            LaneLine(
                points_px=np.array([[100.0, 700.0], [600.0, 400.0]]),
                role="ego_left",
                kind="solid",
                score=0.8,
            )
        ],
        extras={"stub_vehicle_distance_m": 30.0 - i},
    )


class TestRoundTrip:
    def test_written_results_read_back_identically(self, cfg: Config, source_file: Path) -> None:
        cache = RunCache.open(cfg, source_file)
        with cache.writer() as w:
            for i in range(5):
                w.write(_result(i))
        assert cache.is_reusable()

        got = list(cache.read())
        assert [r.frame_index for r in got] == list(range(5))
        assert got[2].detections[0].label == "car"
        assert got[2].detections[0].track_id == 3
        assert got[2].lanes[0].role == "ego_left"
        assert got[2].lanes[0].points_px == pytest.approx(
            np.array([[100.0, 700.0], [600.0, 400.0]])
        )
        assert got[2].extras["stub_vehicle_distance_m"] == pytest.approx(28.0)
        assert got[3].timestamp_s == pytest.approx(0.12)

    def test_arrays_round_trip(self, cfg: Config, source_file: Path) -> None:
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:20, 5:30] = True
        cache = RunCache.open(cfg, source_file)
        with cache.writer() as w:
            w.write(_result(0), arrays={"drivable": mask})
        loaded = cache.read_arrays(0)
        assert set(loaded) == {"drivable"}
        assert np.array_equal(loaded["drivable"], mask)

    def test_missing_arrays_return_empty(self, cfg: Config, source_file: Path) -> None:
        cache = RunCache.open(cfg, source_file)
        with cache.writer() as w:
            w.write(_result(0))
        assert cache.read_arrays(0) == {}

    def test_manifest_records_identity(self, cfg: Config, source_file: Path) -> None:
        cache = RunCache.open(cfg, source_file, extra={"note": "hello"})
        with cache.writer() as w:
            for i in range(3):
                w.write(_result(i))
        m = cache.existing_manifest()
        assert m.frame_count == 3
        assert m.complete is True
        assert m.config_hash == cfg.cache_key()
        assert m.model_version == cfg.cache.model_version
        assert m.extra == {"note": "hello"}
        assert m.completed_utc is not None


class TestOrdering:
    def test_out_of_order_writes_raise(self, cfg: Config, source_file: Path) -> None:
        cache = RunCache.open(cfg, source_file)
        with pytest.raises(ValueError, match="increasing order"):
            with cache.writer() as w:
                w.write(_result(5))
                w.write(_result(2))

    def test_duplicate_index_raises(self, cfg: Config, source_file: Path) -> None:
        cache = RunCache.open(cfg, source_file)
        with pytest.raises(ValueError, match="increasing order"):
            with cache.writer() as w:
                w.write(_result(1))
                w.write(_result(1))


class TestStaleness:
    def _populate(self, cfg: Config, source_file: Path) -> RunCache:
        cache = RunCache.open(cfg, source_file)
        with cache.writer() as w:
            for i in range(3):
                w.write(_result(i))
        return cache

    def test_fresh_cache_is_reusable(self, cfg: Config, source_file: Path) -> None:
        self._populate(cfg, source_file)
        assert RunCache.open(cfg, source_file).staleness_reason() is None

    def test_absent_cache_reports_so(self, cfg: Config, source_file: Path) -> None:
        assert RunCache.open(cfg, source_file).staleness_reason() == "no cache present"

    def test_changed_model_version_invalidates(self, cfg: Config, source_file: Path) -> None:
        self._populate(cfg, source_file)
        bumped = cfg.model_copy(
            update={"cache": cfg.cache.model_copy(update={"model_version": "yolopv2-1.0"})}
        )
        # Pin the run_id so we look at the same directory, isolating the model change.
        bumped = bumped.model_copy(
            update={"cache": bumped.cache.model_copy(
                update={"run_id": default_run_id(source_file, cfg.cache_key())}
            )}
        )
        reason = RunCache.open(bumped, source_file).staleness_reason()
        assert reason is not None and "model version" in reason

    def test_changed_cache_relevant_config_invalidates(
        self, cfg: Config, source_file: Path
    ) -> None:
        self._populate(cfg, source_file)
        changed = cfg.model_copy(update={"io": cfg.io.model_copy(update={"stride": 4})})
        changed = changed.model_copy(
            update={"cache": changed.cache.model_copy(
                update={"run_id": default_run_id(source_file, cfg.cache_key())}
            )}
        )
        reason = RunCache.open(changed, source_file).staleness_reason()
        assert reason is not None and "config hash" in reason

    def test_presentation_only_config_does_NOT_invalidate(
        self, cfg: Config, source_file: Path
    ) -> None:
        """Restyling the overlay must not throw away an expensive cache."""
        self._populate(cfg, source_file)
        restyled = cfg.model_copy(
            update={
                "overlay": cfg.overlay.model_copy(update={"box_thickness": 9, "hud": False}),
                "video": cfg.video.model_copy(update={"fourcc": "avc1"}),
            }
        )
        assert restyled.cache_key() == cfg.cache_key()
        assert RunCache.open(restyled, source_file).staleness_reason() is None

    def test_changed_source_invalidates(self, cfg: Config, source_file: Path) -> None:
        self._populate(cfg, source_file)
        source_file.write_bytes(b"a completely different video file")
        reason = RunCache.open(cfg, source_file).staleness_reason()
        assert reason is not None and "source changed" in reason

    def test_interrupted_run_is_never_reused(self, cfg: Config, source_file: Path) -> None:
        cache = RunCache.open(cfg, source_file)
        with pytest.raises(RuntimeError):
            with cache.writer() as w:
                w.write(_result(0))
                raise RuntimeError("inference blew up halfway")
        reason = RunCache.open(cfg, source_file).staleness_reason()
        assert reason is not None and "did not finish" in reason

    def test_missing_frames_file_invalidates(self, cfg: Config, source_file: Path) -> None:
        cache = self._populate(cfg, source_file)
        cache.frames_path.unlink()
        assert "frames.jsonl is missing" in RunCache.open(cfg, source_file).staleness_reason()

    def test_corrupt_manifest_invalidates(self, cfg: Config, source_file: Path) -> None:
        cache = self._populate(cfg, source_file)
        cache.manifest_path.write_text("{not json", encoding="utf-8")
        assert RunCache.open(cfg, source_file).staleness_reason() == "no cache present"

    def test_reading_a_stale_cache_raises(self, cfg: Config, source_file: Path) -> None:
        self._populate(cfg, source_file)
        source_file.write_bytes(b"replaced")
        with pytest.raises(ValueError, match="refusing to read a stale cache"):
            list(RunCache.open(cfg, source_file).read())

    def test_schema_version_mismatch_invalidates(self) -> None:
        base = Manifest(
            run_id="r", config_hash="abc", model_version="m", schema_version=1,
            source=SourceFingerprint("f", 10, 20), source_path="f", complete=True,
        )
        newer = Manifest(
            run_id="r", config_hash="abc", model_version="m", schema_version=2,
            source=SourceFingerprint("f", 10, 20), source_path="f", complete=True,
        )
        assert "schema version" in base.staleness_reason(newer)

    def test_rewriting_clears_the_old_run(self, cfg: Config, source_file: Path) -> None:
        cache = self._populate(cfg, source_file)
        with cache.writer() as w:
            w.write(_result(0))
        assert cache.existing_manifest().frame_count == 1
        assert len(list(cache.read())) == 1


class TestRunId:
    def test_deterministic_for_identical_config(self, cfg: Config, source_file: Path) -> None:
        a = RunCache.open(cfg, source_file)
        b = RunCache.open(cfg, source_file)
        assert a.run_dir == b.run_dir
        assert a.wanted.run_id.startswith("clip__")

    def test_differs_when_cache_relevant_config_differs(
        self, cfg: Config, source_file: Path
    ) -> None:
        other = cfg.model_copy(update={"io": cfg.io.model_copy(update={"stride": 7})})
        assert RunCache.open(cfg, source_file).run_dir != RunCache.open(other, source_file).run_dir

    def test_explicit_run_id_is_honoured(self, cfg: Config, source_file: Path) -> None:
        pinned = cfg.model_copy(update={"cache": cfg.cache.model_copy(update={"run_id": "my-run"})})
        assert RunCache.open(pinned, source_file).run_dir.name == "my-run"


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


class TestManifestFile:
    def test_manifest_is_human_readable_json(self, cfg: Config, source_file: Path) -> None:
        cache = RunCache.open(cfg, source_file)
        with cache.writer() as w:
            w.write(_result(0))
        raw = json.loads(cache.manifest_path.read_text("utf-8"))
        assert set(raw) >= {
            "run_id", "config_hash", "model_version", "schema_version",
            "source", "frame_count", "complete",
        }
        assert "\n" in cache.manifest_path.read_text("utf-8")  # indented, not minified
