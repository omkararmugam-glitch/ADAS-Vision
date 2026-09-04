"""Tests for alert scoring against labelled event intervals.

The rules that matter are the ones that decide whether a system looks good or
bad: one labelled event can be claimed only once (so repeated firing is
penalised, as a driver would experience it), and lead time is measured from the
alert to the event it warned about.
"""

from __future__ import annotations

import pytest

from eval.score_alerts import Interval, score


def iv(t, a, b):
    return Interval(type=t, start_s=a, end_s=b)


class TestMatching:
    def test_an_overlapping_alert_is_a_true_positive(self) -> None:
        r = score([iv("fc", 9.0, 12.0)], [iv("fc", 10.0, 13.0)])
        assert r["per_type"]["fc"]["tp"] == 1
        assert r["per_type"]["fc"]["fp"] == 0

    def test_a_missed_event_is_a_false_negative(self) -> None:
        r = score([], [iv("fc", 10.0, 13.0)])
        assert r["per_type"]["fc"]["fn"] == 1
        assert r["per_type"]["fc"]["recall"] == 0.0

    def test_an_alert_with_no_event_is_a_false_positive(self) -> None:
        r = score([iv("fc", 40.0, 41.0)], [])
        assert r["per_type"]["fc"]["fp"] == 1
        assert r["per_type"]["fc"]["precision"] == 0.0

    def test_each_event_can_only_be_claimed_once(self) -> None:
        """Ten alerts over one event is one TP and nine FPs, not ten TPs."""
        preds = [iv("fc", 10.0 + i * 0.1, 10.5 + i * 0.1) for i in range(10)]
        r = score(preds, [iv("fc", 10.0, 13.0)])
        assert r["per_type"]["fc"]["tp"] == 1
        assert r["per_type"]["fc"]["fp"] == 9

    def test_types_are_scored_separately(self) -> None:
        """A lane departure must not be credited against a collision event."""
        r = score([iv("lane_departure", 10.0, 11.0)], [iv("forward_collision", 10.0, 11.0)])
        assert r["per_type"]["lane_departure"]["fp"] == 1
        assert r["per_type"]["forward_collision"]["fn"] == 1

    def test_tolerance_admits_a_near_miss(self) -> None:
        assert score([iv("fc", 8.0, 9.0)], [iv("fc", 9.5, 11.0)],
                     tolerance_s=1.0)["per_type"]["fc"]["tp"] == 1

    def test_zero_tolerance_rejects_a_gap(self) -> None:
        assert score([iv("fc", 8.0, 9.0)], [iv("fc", 9.5, 11.0)],
                     tolerance_s=0.0)["per_type"]["fc"]["fp"] == 1


class TestLeadTime:
    def test_lead_time_is_measured_to_the_event_start(self) -> None:
        r = score([iv("fc", 8.0, 12.0)], [iv("fc", 10.5, 13.0)])
        assert r["per_type"]["fc"]["mean_lead_s"] == pytest.approx(2.5)

    def test_a_late_alert_has_negative_lead(self) -> None:
        """Firing after the event began is worth recording as such."""
        r = score([iv("fc", 11.0, 13.0)], [iv("fc", 10.0, 13.0)])
        assert r["per_type"]["fc"]["mean_lead_s"] == pytest.approx(-1.0)

    def test_min_lead_is_reported_alongside_the_mean(self) -> None:
        preds = [iv("fc", 1.0, 3.0), iv("fc", 20.0, 22.0)]
        truth = [iv("fc", 3.0, 4.0), iv("fc", 20.5, 23.0)]
        r = score(preds, truth)
        assert r["per_type"]["fc"]["min_lead_s"] == pytest.approx(0.5)
        assert r["per_type"]["fc"]["mean_lead_s"] == pytest.approx(1.25)

    def test_no_true_positives_means_no_lead_time(self) -> None:
        assert score([iv("fc", 40.0, 41.0)], [])["per_type"]["fc"]["mean_lead_s"] is None


class TestAggregate:
    def test_overall_sums_across_types(self) -> None:
        r = score([iv("a", 1, 2), iv("b", 5, 6)], [iv("a", 1, 2), iv("b", 5, 6)])
        assert r["overall"]["tp"] == 2
        assert r["overall"]["precision"] == 1.0
        assert r["overall"]["recall"] == 1.0

    def test_perfect_prediction_scores_one(self) -> None:
        r = score([iv("fc", 10, 12)], [iv("fc", 10, 12)])
        assert r["per_type"]["fc"]["f1"] == pytest.approx(1.0)

    def test_empty_inputs_do_not_divide_by_zero(self) -> None:
        r = score([], [])
        assert r["overall"] == {"tp": 0, "fp": 0, "fn": 0,
                                "precision": 0.0, "recall": 0.0}
