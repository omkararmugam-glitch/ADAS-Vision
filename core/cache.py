"""Two-stage cache: expensive per-frame model output on disk, cheap rules on top.

This is the piece that makes threshold tuning survivable. Per-frame results are
written once under ``outputs/<run_id>/``; the rule engine, the lane analysis and
the overlay read them back without loading a model at all.

Layout of a run directory::

    outputs/<run_id>/
        manifest.json      identity, staleness fields, completion state
        frames.jsonl       one FrameRecord per line, ordered by frame_index
        masks/             <frame>_drivable.png and <frame>_lane.png

Masks are stored as **paletted PNG at the network's native resolution**, not as
full-resolution arrays. Two reasons: the network predicts at its letterboxed
input size and upsampling adds interpolated pixels but no information; and PNG's
lossless run-length-friendly compression suits binary masks far better than raw
bytes. Both effects are measured and reported by ``cache_size_report``.

Staleness is decided by the manifest, never by timestamps. A cache is reusable
only if every identity field matches -- config hash, calibration hash, model
name, weights hash, code version, schema version and source fingerprint -- and
the run was marked complete. Any mismatch raises on read.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from core.config import Config
from core.letterbox import LetterboxTransform
from core.logging_utils import get_logger, log_kv
from core.types import SCHEMA_VERSION, FrameRecord, FrameResult

_log = get_logger(__name__)

MANIFEST_NAME = "manifest.json"
FRAMES_NAME = "frames.jsonl"
MASKS_DIR = "masks"


def code_version() -> str:
    """Short git description of the working tree, or ``"unknown"``.

    Recorded so a cache produced by different code is not silently reused after
    a change to post-processing. A dirty tree is marked, because the commit hash
    alone would then be a lie.
    """
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{rev}-dirty" if dirty else rev
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return "unknown"


@dataclass(frozen=True)
class SourceFingerprint:
    """Cheap identity for an input source, without hashing gigabytes of pixels.

    Size and mtime catch the realistic case -- the file was replaced or
    re-encoded. They will not catch a byte-level edit preserving both, a trade
    made deliberately: hashing a 25 MB video every run to guard an implausible
    case is not worth the seconds.
    """

    name: str
    size_bytes: int
    mtime_ns: int

    @classmethod
    def of(cls, path: str | Path) -> SourceFingerprint:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"cannot fingerprint a missing source: {p}")
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.is_file())
            total = sum(f.stat().st_size for f in files)
            newest = max((f.stat().st_mtime_ns for f in files), default=0)
            return cls(name=p.name, size_bytes=total, mtime_ns=newest)
        st = p.stat()
        return cls(name=p.name, size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)


@dataclass
class Manifest:
    """Identity and completion state of one cached run.

    Every field except ``frame_count``, ``complete`` and the timestamps takes
    part in staleness: a difference in any of them means the cached predictions
    were produced by a different system and must not be reused.
    """

    run_id: str
    config_hash: str
    calibration_hash: str
    model_name: str
    weights_hash: str
    code_version: str
    schema_version: int
    class_names: list[str]
    source: SourceFingerprint
    source_path: str
    frame_count: int = 0
    complete: bool = False
    created_utc: str = ""
    completed_utc: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    #: Fields compared when deciding whether a cache may be reused.
    IDENTITY_FIELDS = (
        ("schema_version", "schema version"),
        ("config_hash", "config hash"),
        ("calibration_hash", "calibration hash"),
        ("model_name", "model name"),
        ("weights_hash", "weights hash"),
        ("code_version", "code version"),
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source"] = asdict(self.source)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        return cls(
            run_id=d["run_id"],
            config_hash=d["config_hash"],
            calibration_hash=d["calibration_hash"],
            model_name=d["model_name"],
            weights_hash=d["weights_hash"],
            code_version=d["code_version"],
            schema_version=int(d["schema_version"]),
            class_names=list(d["class_names"]),
            source=SourceFingerprint(**d["source"]),
            source_path=d["source_path"],
            frame_count=int(d["frame_count"]),
            complete=bool(d["complete"]),
            created_utc=d.get("created_utc", ""),
            completed_utc=d.get("completed_utc"),
            extra=d.get("extra", {}),
        )

    def staleness_reason(self, other: Manifest, require_complete: bool = True) -> str | None:
        """Why a cache written as ``self`` cannot serve a run wanting ``other``.

        Args:
            other: The identity the caller needs.
            require_complete: When False, an unfinished run is acceptable --
                that is what makes resuming possible.

        Returns:
            None when the cache is usable.
        """
        for attr, label in self.IDENTITY_FIELDS:
            mine, theirs = getattr(self, attr), getattr(other, attr)
            if mine != theirs:
                return f"{label} {mine!r} != {theirs!r}"
        if self.source != other.source:
            return f"source changed ({self.source.name}: size or mtime differs)"
        if require_complete and not self.complete:
            return "previous run did not finish (partial cache)"
        return None


def default_run_id(source_path: str | Path, config_hash: str) -> str:
    """Deterministic run directory name: ``<source stem>__<config hash>``.

    Deterministic on purpose -- rerunning with identical settings finds the
    existing cache instead of piling up timestamped directories.
    """
    return f"{Path(source_path).stem}__{config_hash}"


class RunCache:
    """Read/write access to one run directory."""

    def __init__(self, run_dir: Path, wanted: Manifest) -> None:
        self.run_dir = run_dir
        self.wanted = wanted

    @classmethod
    def open(
        cls,
        cfg: Config,
        source_path: str | Path,
        calibration_hash: str,
        model_name: str,
        weights_hash: str,
        class_names: tuple[str, ...] = (),
        extra: dict[str, Any] | None = None,
    ) -> RunCache:
        """Locate the run directory implied by this config, model and source."""
        config_hash = cfg.cache_key()
        run_id = cfg.cache.run_id or default_run_id(source_path, config_hash)
        wanted = Manifest(
            run_id=run_id,
            config_hash=config_hash,
            calibration_hash=calibration_hash,
            model_name=model_name,
            weights_hash=weights_hash,
            code_version=code_version(),
            schema_version=SCHEMA_VERSION,
            class_names=list(class_names),
            source=SourceFingerprint.of(source_path),
            source_path=str(source_path),
            extra=extra or {},
        )
        return cls(Path(cfg.cache.root) / run_id, wanted)

    # -- paths -------------------------------------------------------------- #

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / MANIFEST_NAME

    @property
    def frames_path(self) -> Path:
        return self.run_dir / FRAMES_NAME

    @property
    def masks_dir(self) -> Path:
        return self.run_dir / MASKS_DIR

    def _mask_path(self, frame_index: int, which: str) -> Path:
        return self.masks_dir / f"{frame_index:08d}_{which}.png"

    # -- staleness ---------------------------------------------------------- #

    def existing_manifest(self) -> Manifest | None:
        """The manifest already on disk, or None if there is no readable cache."""
        if not self.manifest_path.is_file():
            return None
        try:
            return Manifest.from_dict(json.loads(self.manifest_path.read_text("utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log_kv(
                _log, logging.WARNING, "cache.manifest_unreadable",
                path=str(self.manifest_path), error=type(exc).__name__,
            )
            return None

    def staleness_reason(self, require_complete: bool = True) -> str | None:
        """Why the cache on disk cannot be reused, or None if it can."""
        existing = self.existing_manifest()
        if existing is None:
            return "no cache present"
        if not self.frames_path.is_file():
            return "manifest present but frames.jsonl is missing"
        return existing.staleness_reason(self.wanted, require_complete=require_complete)

    def is_reusable(self) -> bool:
        """True when the cached run matches this config, model, calibration and source."""
        reason = self.staleness_reason()
        if reason is None:
            log_kv(
                _log, logging.INFO, "cache.hit",
                run_id=self.wanted.run_id, frames=self.existing_manifest().frame_count,
            )
            return True
        log_kv(_log, logging.INFO, "cache.miss", run_id=self.wanted.run_id, reason=reason)
        return False

    def resumable_frames(self) -> int:
        """How many leading frames of a matching, unfinished run can be kept.

        Returns 0 when nothing is reusable -- a different config, model or source
        means the partial results describe a different computation.
        """
        reason = self.staleness_reason(require_complete=False)
        if reason is not None:
            return 0
        existing = self.existing_manifest()
        assert existing is not None
        if existing.complete:
            return existing.frame_count
        n = self._count_intact_records()
        log_kv(
            _log, logging.INFO, "cache.resumable",
            run_id=self.wanted.run_id, frames=n,
            note="partial run matches; resuming after these frames",
        )
        return n

    def _count_intact_records(self) -> int:
        """Frames with a complete JSON line *and* both mask files present.

        An interrupted run can leave a truncated final line or a record whose
        masks were never written; those frames must be recomputed.
        """
        if not self.frames_path.is_file():
            return 0
        n = 0
        with self.frames_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    idx = int(d["frame_index"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    break
                if not (
                    self._mask_path(idx, "drivable").is_file()
                    and self._mask_path(idx, "lane").is_file()
                ):
                    break
                n += 1
        return n

    def invalidate(self) -> None:
        """Delete the run directory. Called before rewriting a stale cache."""
        if self.run_dir.exists():
            log_kv(_log, logging.WARNING, "cache.invalidate", run_dir=str(self.run_dir))
            shutil.rmtree(self.run_dir)

    # -- writing ------------------------------------------------------------ #

    def writer(self, resume_after: int = 0) -> CacheWriter:
        """Open a writer.

        Args:
            resume_after: Keep this many existing leading frames and append after
                them. 0 clears the directory and starts fresh.
        """
        return CacheWriter(self, resume_after=resume_after)

    # -- reading ------------------------------------------------------------ #

    def read_masks(self, frame_index: int, original_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Load one frame's masks and upsample them to the original frame size."""
        out = []
        for which in ("drivable", "lane"):
            path = self._mask_path(frame_index, which)
            if not path.is_file():
                raise FileNotFoundError(f"cached mask missing: {path}")
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise OSError(f"could not decode cached mask: {path}")
            out.append(LetterboxTransform.upsample_cached_mask(mask, original_size))
        return out[0], out[1]

    def read(self, with_masks: bool = True) -> Iterator[FrameRecord]:
        """Yield cached per-frame records in frame order.

        Args:
            with_masks: Load and upsample the masks. Set False when only the
                detections and lane geometry are needed -- it is much faster.

        Raises:
            FileNotFoundError: if there is nothing cached here.
            ValueError: if the cache is stale. Callers must rebuild rather than
                silently mixing results from different models or configs.
        """
        reason = self.staleness_reason()
        if reason is not None:
            raise ValueError(
                f"refusing to read a stale cache at {self.run_dir}: {reason}. "
                f"Rebuild it, or call invalidate() first."
            )
        size = tuple(self.existing_manifest().extra.get("frame_size", (0, 0)))
        with self.frames_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if with_masks:
                    drivable, lane = self.read_masks(int(d["frame_index"]), size)
                else:
                    blank = np.zeros((1, 1), dtype=np.uint8)
                    drivable, lane = blank, blank.copy()
                yield FrameRecord.from_dict(d, drivable, lane)

    # -- reporting ---------------------------------------------------------- #

    def size_report(self) -> dict[str, float]:
        """Measured bytes per frame, broken down by component.

        Also reports what full-resolution raw storage *would* have cost, which is
        the comparison that justifies storing masks at native resolution.
        """
        manifest = self.existing_manifest()
        n = max(1, manifest.frame_count if manifest else 1)
        masks = sorted(self.masks_dir.glob("*.png")) if self.masks_dir.is_dir() else []
        mask_bytes = sum(p.stat().st_size for p in masks)
        json_bytes = self.frames_path.stat().st_size if self.frames_path.is_file() else 0

        report = {
            "frames": float(n),
            "masks_bytes_per_frame": mask_bytes / n,
            "json_bytes_per_frame": json_bytes / n,
            "total_bytes_per_frame": (mask_bytes + json_bytes) / n,
            "total_mb": (mask_bytes + json_bytes) / 1e6,
        }
        if masks:
            sample = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
            w, h = tuple(manifest.extra.get("frame_size", (0, 0))) if manifest else (0, 0)
            report["mask_native_shape"] = float(sample.shape[0])
            report["raw_native_bytes_per_frame"] = float(2 * sample.size)
            report["raw_fullres_bytes_per_frame"] = float(2 * w * h)
        return report


class CacheWriter:
    """Append-only writer for one run. Use as a context manager.

    The manifest is only marked complete on clean exit, so an interrupted run
    leaves a cache that :meth:`RunCache.is_reusable` rejects -- but that
    :meth:`RunCache.resumable_frames` can still continue from.
    """

    def __init__(self, cache: RunCache, resume_after: int = 0) -> None:
        self._cache = cache
        self._manifest = cache.wanted
        self._fh: Any = None
        self._count = resume_after
        self._resume_after = resume_after
        self._last_index = resume_after - 1

    def __enter__(self) -> CacheWriter:
        if self._resume_after > 0:
            self._truncate_to(self._resume_after)
            self._fh = self._cache.frames_path.open("a", encoding="utf-8")
            log_kv(
                _log, logging.INFO, "cache.write.resume",
                run_dir=str(self._cache.run_dir), after_frame=self._resume_after,
            )
        else:
            self._cache.invalidate()
            self._cache.run_dir.mkdir(parents=True, exist_ok=True)
            self._fh = self._cache.frames_path.open("w", encoding="utf-8")
            log_kv(
                _log, logging.INFO, "cache.write.start",
                run_dir=str(self._cache.run_dir),
                config_hash=self._manifest.config_hash,
                model=self._manifest.model_name,
                weights=self._manifest.weights_hash[:12],
                code=self._manifest.code_version,
            )
        self._manifest.created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._manifest.complete = False
        self._write_manifest()
        return self

    def _truncate_to(self, n: int) -> None:
        """Drop any records past frame ``n - 1``, so appends stay contiguous."""
        kept: list[str] = []
        with self._cache.frames_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(kept) >= n:
                    break
                if line.strip():
                    kept.append(line.rstrip("\n"))
        self._cache.frames_path.write_text(
            "".join(x + "\n" for x in kept), encoding="utf-8"
        )

    def _write_manifest(self) -> None:
        self._cache.run_dir.mkdir(parents=True, exist_ok=True)
        self._manifest.frame_count = self._count
        self._cache.manifest_path.write_text(
            json.dumps(self._manifest.to_dict(), indent=2), encoding="utf-8"
        )

    def write(self, record: FrameRecord, result: FrameResult | None = None) -> None:
        """Append one frame's record, writing its masks alongside.

        Masks are taken from ``result.native_masks`` when present, so what lands
        on disk is what the network predicted rather than an upsampled copy.

        Raises:
            ValueError: if frames arrive out of order; the reader relies on the
                file being ordered by ``frame_index``.
        """
        if record.frame_index <= self._last_index:
            raise ValueError(
                f"frames must be written in increasing order: got "
                f"{record.frame_index} after {self._last_index}"
            )
        result = result or record.result
        native = result.native_masks
        drivable = native.drivable if native is not None else result.drivable_mask
        lane = native.lane if native is not None else result.lane_mask

        self._cache.masks_dir.mkdir(parents=True, exist_ok=True)
        for which, mask in (("drivable", drivable), ("lane", lane)):
            # Scale {0,1} to {0,255}: a 0/1 PNG is technically fine but unviewable
            # in any image tool, which makes debugging a cached run needlessly hard.
            ok = cv2.imwrite(
                str(self._cache._mask_path(record.frame_index, which)),
                (np.asarray(mask) > 0).astype(np.uint8) * 255,
                [cv2.IMWRITE_PNG_COMPRESSION, 9],
            )
            if not ok:
                raise OSError(f"failed to write cached {which} mask for frame {record.frame_index}")

        self._fh.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
        self._last_index = record.frame_index
        self._count += 1

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._fh is not None:
            self._fh.close()
        if exc_type is None:
            self._manifest.complete = True
            self._manifest.completed_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            log_kv(
                _log, logging.INFO, "cache.write.done",
                run_dir=str(self._cache.run_dir), frames=self._count,
            )
        else:
            log_kv(
                _log, logging.ERROR, "cache.write.aborted",
                run_dir=str(self._cache.run_dir), frames=self._count,
                error=exc_type.__name__,
                note="manifest left incomplete; resume will continue from here",
            )
        self._write_manifest()
