"""Two-stage cache: expensive per-frame model output on disk, cheap rules on top.

This is the piece that makes threshold tuning survivable. Per-frame results are
written once under ``outputs/<run_id>/``; the rule engine and the overlay read
them back without touching a network. Re-running with different event thresholds
must never re-run inference.

Layout of a run directory::

    outputs/<run_id>/
        manifest.json      identity, staleness fields, and completion state
        frames.jsonl       one FrameResult per line, ordered by frame_index
        arrays/            optional per-frame .npz for masks and other big arrays

Staleness is decided by the manifest, not by timestamps. A cache is reusable only
if the config hash, model version, schema version and source fingerprint all
match, and the run was marked complete. Anything else is reported with a specific
reason and rebuilt.

JSONL rather than one file per frame: a 1260-frame run is one append-only file,
sequential to write and to scan, instead of 1260 directory entries. Masks stay
out of it, in ``.npz``, because base64 in JSON would be both slow and enormous.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.config import Config
from src.logging_utils import get_logger, log_kv
from src.types import SCHEMA_VERSION, FrameResult

_log = get_logger(__name__)

MANIFEST_NAME = "manifest.json"
FRAMES_NAME = "frames.jsonl"
ARRAYS_DIR = "arrays"


@dataclass(frozen=True)
class SourceFingerprint:
    """Cheap identity for an input source, without hashing gigabytes of pixels.

    Size and mtime catch the realistic case -- the file was replaced or re-encoded.
    They will not catch a byte-level edit that preserves both, which is a trade
    made deliberately: hashing a 25 MB video on every run to guard against an
    implausible case is not worth the seconds.
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

    Attributes:
        run_id: Name of the run directory.
        config_hash: :meth:`~src.config.Config.cache_key` at write time.
        model_version: Identifier of whatever produced the per-frame results.
        schema_version: Version of the :mod:`src.types` serialisation format.
        source: Fingerprint of the input media.
        source_path: Where the input came from, for human readers.
        frame_count: Frames actually written.
        complete: False until the writer closes cleanly. An interrupted run is
            never reused -- a half-written cache is worse than no cache.
        created_utc: When the run started.
        completed_utc: When the writer closed, or None.
        extra: Free-form provenance, e.g. the calibration in force.
    """

    run_id: str
    config_hash: str
    model_version: str
    schema_version: int
    source: SourceFingerprint
    source_path: str
    frame_count: int = 0
    complete: bool = False
    created_utc: str = ""
    completed_utc: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source"] = asdict(self.source)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        return cls(
            run_id=d["run_id"],
            config_hash=d["config_hash"],
            model_version=d["model_version"],
            schema_version=int(d["schema_version"]),
            source=SourceFingerprint(**d["source"]),
            source_path=d["source_path"],
            frame_count=int(d["frame_count"]),
            complete=bool(d["complete"]),
            created_utc=d.get("created_utc", ""),
            completed_utc=d.get("completed_utc"),
            extra=d.get("extra", {}),
        )

    def staleness_reason(self, other: Manifest) -> str | None:
        """Why a cache written as ``self`` cannot serve a run wanting ``other``.

        Returns None when the cache is reusable.
        """
        if not self.complete:
            return "previous run did not finish (partial cache)"
        if self.schema_version != other.schema_version:
            return f"schema version {self.schema_version} != {other.schema_version}"
        if self.config_hash != other.config_hash:
            return f"config hash {self.config_hash} != {other.config_hash}"
        if self.model_version != other.model_version:
            return f"model version '{self.model_version}' != '{other.model_version}'"
        if self.source != other.source:
            return f"source changed ({self.source.name}: size or mtime differs)"
        return None


def default_run_id(source_path: str | Path, config_hash: str) -> str:
    """Deterministic run directory name: ``<source stem>__<config hash>``.

    Deterministic on purpose -- rerunning with identical settings finds the
    existing cache instead of piling up timestamped directories.
    """
    return f"{Path(source_path).stem}__{config_hash}"


class RunCache:
    """Read/write access to one run directory.

    Typical use::

        cache = RunCache.open(cfg, source_path)
        if cache.is_reusable():
            results = list(cache.read())
        else:
            with cache.writer() as w:
                for frame in source: w.write(compute(frame))
    """

    def __init__(self, run_dir: Path, wanted: Manifest) -> None:
        self.run_dir = run_dir
        self.wanted = wanted

    @classmethod
    def open(
        cls, cfg: Config, source_path: str | Path, extra: dict[str, Any] | None = None
    ) -> RunCache:
        """Locate the run directory implied by ``cfg`` and this source."""
        config_hash = cfg.cache_key()
        run_id = cfg.cache.run_id or default_run_id(source_path, config_hash)
        wanted = Manifest(
            run_id=run_id,
            config_hash=config_hash,
            model_version=cfg.cache.model_version,
            schema_version=SCHEMA_VERSION,
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
    def arrays_dir(self) -> Path:
        return self.run_dir / ARRAYS_DIR

    # -- staleness ---------------------------------------------------------- #

    def existing_manifest(self) -> Manifest | None:
        """The manifest already on disk, or None if there is no readable cache."""
        if not self.manifest_path.is_file():
            return None
        try:
            return Manifest.from_dict(json.loads(self.manifest_path.read_text("utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log_kv(
                _log,
                logging.WARNING,
                "cache.manifest_unreadable",
                path=str(self.manifest_path),
                error=type(exc).__name__,
            )
            return None

    def staleness_reason(self) -> str | None:
        """Why the cache on disk cannot be reused, or None if it can."""
        existing = self.existing_manifest()
        if existing is None:
            return "no cache present"
        if not self.frames_path.is_file():
            return "manifest present but frames.jsonl is missing"
        return existing.staleness_reason(self.wanted)

    def is_reusable(self) -> bool:
        """True when the cached run matches this config, model and source."""
        reason = self.staleness_reason()
        if reason is None:
            log_kv(
                _log, logging.INFO, "cache.hit",
                run_id=self.wanted.run_id, frames=self.existing_manifest().frame_count,
            )
            return True
        log_kv(_log, logging.INFO, "cache.miss", run_id=self.wanted.run_id, reason=reason)
        return False

    def invalidate(self) -> None:
        """Delete the run directory. Called before rewriting a stale cache."""
        if self.run_dir.exists():
            log_kv(_log, logging.WARNING, "cache.invalidate", run_dir=str(self.run_dir))
            shutil.rmtree(self.run_dir)

    # -- writing ------------------------------------------------------------ #

    def writer(self) -> CacheWriter:
        """Open a writer, clearing any stale cache in this directory first."""
        return CacheWriter(self)

    # -- reading ------------------------------------------------------------ #

    def read(self) -> Iterator[FrameResult]:
        """Yield cached per-frame results in frame order.

        Raises:
            FileNotFoundError: if there is nothing cached here.
            ValueError: if the cache is stale -- callers must rebuild rather than
                silently mixing results from different configs.
        """
        reason = self.staleness_reason()
        if reason is not None:
            raise ValueError(
                f"refusing to read a stale cache at {self.run_dir}: {reason}. "
                f"Rebuild it, or call invalidate() first."
            )
        with self.frames_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield FrameResult.from_dict(json.loads(line))

    def read_arrays(self, frame_index: int) -> dict[str, np.ndarray]:
        """Load the per-frame arrays for ``frame_index``; empty dict if none."""
        path = self.arrays_dir / f"{frame_index:08d}.npz"
        if not path.is_file():
            return {}
        with np.load(path) as data:
            return {k: data[k] for k in data.files}


class CacheWriter:
    """Append-only writer for one run. Use as a context manager.

    The manifest is only marked complete on clean exit, so an interrupted run
    leaves a cache that :meth:`RunCache.is_reusable` will reject.
    """

    def __init__(self, cache: RunCache) -> None:
        self._cache = cache
        self._manifest = cache.wanted
        self._fh: Any = None
        self._count = 0
        self._last_index = -1

    def __enter__(self) -> CacheWriter:
        self._cache.invalidate()
        self._cache.run_dir.mkdir(parents=True, exist_ok=True)
        self._manifest.created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._manifest.complete = False
        self._manifest.frame_count = 0
        self._write_manifest()
        self._fh = self._cache.frames_path.open("w", encoding="utf-8")
        log_kv(
            _log, logging.INFO, "cache.write.start",
            run_dir=str(self._cache.run_dir),
            config_hash=self._manifest.config_hash,
            model_version=self._manifest.model_version,
        )
        return self

    def _write_manifest(self) -> None:
        self._cache.manifest_path.write_text(
            json.dumps(self._manifest.to_dict(), indent=2), encoding="utf-8"
        )

    def write(self, result: FrameResult, arrays: dict[str, np.ndarray] | None = None) -> None:
        """Append one frame's result, plus optional arrays such as masks.

        Raises:
            ValueError: if frames arrive out of order; the reader relies on the
                file being ordered by ``frame_index``.
        """
        if result.frame_index <= self._last_index:
            raise ValueError(
                f"frames must be written in increasing order: got "
                f"{result.frame_index} after {self._last_index}"
            )
        self._fh.write(json.dumps(result.to_dict(), separators=(",", ":")) + "\n")
        self._last_index = result.frame_index
        self._count += 1
        if arrays:
            self._cache.arrays_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self._cache.arrays_dir / f"{result.frame_index:08d}.npz", **arrays
            )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._fh is not None:
            self._fh.close()
        self._manifest.frame_count = self._count
        if exc_type is None:
            self._manifest.complete = True
            self._manifest.completed_utc = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            log_kv(
                _log, logging.INFO, "cache.write.done",
                run_dir=str(self._cache.run_dir), frames=self._count,
            )
        else:
            log_kv(
                _log, logging.ERROR, "cache.write.aborted",
                run_dir=str(self._cache.run_dir), frames=self._count,
                error=exc_type.__name__,
                note="manifest left incomplete; this cache will not be reused",
            )
        self._write_manifest()
