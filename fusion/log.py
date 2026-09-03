"""Structured alert logs: per-frame JSON, and one CSV row per alert.

Two formats because they answer different questions. The JSON keeps every frame,
including the quiet ones, so "what did the system believe at t=41.2s?" is
answerable and the run can be replayed. The CSV keeps only alerts, one row each,
so it drops straight into a spreadsheet or pandas for scoring against ground
truth in ``eval/``.

Alerts are also collapsed into *episodes*: a collision warning that holds for
90 frames is one event a driver experienced, not 90. Scoring per frame would
make a single long warning dominate the metrics, so the episode view is what
``eval/`` compares against the labelled intervals in ``DATASET_FORMAT.md``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from fusion.alerts import Alert, AlertType, FrameDecision, Severity


@dataclass(frozen=True)
class Episode:
    """A run of consecutive frames carrying the same alert type.

    Attributes:
        type: Which alert.
        peak_severity: Worst severity reached during the episode.
        start_frame, end_frame: Inclusive frame bounds.
        start_s, end_s: Inclusive time bounds.
        n_frames: Frames the alert was present on.
        track_id: Object responsible, if the episode stayed with one.
    """

    type: AlertType
    peak_severity: Severity
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    n_frames: int
    track_id: int | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "peak_severity": self.peak_severity.value,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_s": round(self.start_s, 4),
            "end_s": round(self.end_s, 4),
            "duration_s": round(self.duration_s, 4),
            "n_frames": self.n_frames,
            "track_id": self.track_id,
        }


def to_episodes(decisions: Iterable[FrameDecision], max_gap_frames: int = 5) -> list[Episode]:
    """Collapse per-frame alerts into episodes.

    An alert of the same type on the same track is treated as continuing while
    the gap is at most ``max_gap_frames``. The tolerance matters because a
    detector that drops one frame would otherwise split one warning into two,
    which reads as two separate events in the metrics.
    """
    open_: dict[tuple[AlertType, int | None], dict[str, Any]] = {}
    done: list[Episode] = []

    for d in decisions:
        present = set()
        for a in d.alerts:
            key = (a.type, a.track_id)
            present.add(key)
            cur = open_.get(key)
            if cur is None or d.frame_index - cur["last_frame"] > max_gap_frames:
                if cur is not None:
                    done.append(_close(key, cur))
                open_[key] = {
                    "start_frame": a.frame_index, "start_s": a.timestamp_s,
                    "last_frame": a.frame_index, "last_s": a.timestamp_s,
                    "peak": a.severity, "n": 1,
                }
            else:
                cur["last_frame"] = a.frame_index
                cur["last_s"] = a.timestamp_s
                cur["n"] += 1
                if a.severity.rank > cur["peak"].rank:
                    cur["peak"] = a.severity

        for key, cur in list(open_.items()):
            if key not in present and d.frame_index - cur["last_frame"] > max_gap_frames:
                done.append(_close(key, cur))
                del open_[key]

    done.extend(_close(k, v) for k, v in open_.items())
    done.sort(key=lambda e: (e.start_frame, e.type.value))
    return done


def _close(key: tuple[AlertType, int | None], cur: dict[str, Any]) -> Episode:
    return Episode(
        type=key[0], peak_severity=cur["peak"], track_id=key[1],
        start_frame=cur["start_frame"], end_frame=cur["last_frame"],
        start_s=cur["start_s"], end_s=cur["last_s"], n_frames=cur["n"],
    )


class AlertLog:
    """Collects per-frame decisions and writes them out."""

    def __init__(self) -> None:
        self.decisions: list[FrameDecision] = []

    def add(self, decision: FrameDecision) -> None:
        self.decisions.append(decision)

    # ------------------------------------------------------------------ #

    @property
    def alerts(self) -> list[Alert]:
        return [a for d in self.decisions for a in d.alerts]

    def episodes(self, max_gap_frames: int = 5) -> list[Episode]:
        return to_episodes(self.decisions, max_gap_frames)

    def counts_by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.alerts:
            out[a.type.value] = out.get(a.type.value, 0) + 1
        return out

    def write_json(self, path: str | Path, source: str = "", max_gap_frames: int = 5) -> Path:
        """Every frame, plus the episode summary."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": source,
            "n_frames": len(self.decisions),
            "n_alerts": len(self.alerts),
            "counts_by_type": self.counts_by_type(),
            "episodes": [e.to_dict() for e in self.episodes(max_gap_frames)],
            "frames": [d.to_dict() for d in self.decisions],
        }
        # allow_nan=False so an infinite TTC that escaped Alert.to_dict raises
        # here rather than silently emitting bare `Infinity`, which is not JSON
        # and which pandas and most parsers reject.
        p.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        return p

    def write_csv(self, path: str | Path) -> Path:
        """One row per alert, for scoring in ``eval/``."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "frame_index", "timestamp_s", "type", "severity", "track_id",
            "message", "ttc_s", "distance_m", "lateral_offset_m", "class",
        ]
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for a in self.alerts:
                row = {
                    "frame_index": a.frame_index,
                    "timestamp_s": round(a.timestamp_s, 4),
                    "type": a.type.value,
                    "severity": a.severity.value,
                    "track_id": a.track_id,
                    "message": a.message,
                }
                row.update({k: a.evidence.get(k) for k in
                            ("ttc_s", "distance_m", "lateral_offset_m", "class")})
                w.writerow(row)
        return p

    def write_episodes_csv(self, path: str | Path, max_gap_frames: int = 5) -> Path:
        """One row per episode -- the view ``eval/`` scores against ground truth."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "type", "peak_severity", "start_frame", "end_frame",
                "start_s", "end_s", "duration_s", "n_frames", "track_id"])
            w.writeheader()
            for e in self.episodes(max_gap_frames):
                w.writerow(e.to_dict())
        return p
