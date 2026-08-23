"""Tests for Stage 2 grouped summaries and severity contrasts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "scripts"))

from summarize_multigap_benchmark import (
    GroupAccumulator,
    _severity_contrasts,
    group_key,
    summarize_group,
)


def test_group_key_keeps_gap_count_and_width_separate():
    row = {
        "split": "development",
        "dataset": "boat",
        "variant_status": "primary_noncontiguous",
        "gap_count": 2,
        "gap_width": 3,
    }
    assert group_key(row, "punctuation", "taln") == (
        "development",
        "boat",
        "primary_noncontiguous",
        2,
        3,
        "punctuation",
        "taln",
    )


def test_candidate_burden_uses_successful_taln_population():
    accumulator = GroupAccumulator(
        document_ids=["doc-1", "doc-1", "doc-2"],
        support=[True, False, True],
        localization=[True, False, True],
        no_matched_token_path=[False, True, False],
        runtime_ms=[1.0, 2.0, 3.0],
        truncated=1,
        burden_document_ids=["doc-1", "doc-2"],
        candidate_counts=[1, 100_000],
        burden_truncated=[False, True],
        unique_interval_counts=[1, 3],
        multiple_location=[False, True],
    )
    summary = summarize_group(accumulator, n_resamples=100, seed=7)
    assert summary["full_lexical_support_count"] == 2
    assert summary["no_matched_token_path_count"] == 1
    assert summary["candidate_burden"]["tasks"] == 2
    assert (
        summary["candidate_burden"]["truncated_candidate_count_tasks"] == 1
    )
    assert summary["candidate_burden"]["multiple_source_location_count"] == 1


def test_severity_contrast_pairs_by_base_record_not_row_order():
    outcomes = {
        ("development", "boat", "taln", "record-b", 1, 1): ("doc-2", True),
        ("development", "boat", "taln", "record-a", 1, 1): ("doc-1", True),
        ("development", "boat", "taln", "record-a", 3, 1): ("doc-1", False),
        ("development", "boat", "taln", "record-b", 3, 1): ("doc-2", True),
    }
    spec = {
        "statistics": {
            "severity_contrasts": [
                {
                    "name": "three-vs-one",
                    "reference": {"gap_count": 1, "gap_width": 1},
                    "comparison": {"gap_count": 3, "gap_width": 1},
                    "interpretation": "test contrast",
                }
            ],
            "severity_tokenizer": "punctuation",
        }
    }
    summary = _severity_contrasts(
        outcomes, spec=spec, n_resamples=100, seed=7
    )["development::boat::taln::three-vs-one"]
    assert summary["paired_records"] == 2
    assert summary["paired_counts"] == {
        "both_succeed": 1,
        "reference_only": 1,
        "comparison_only": 0,
        "neither_succeeds": 0,
    }
    assert summary["clustered_difference_ci"]["estimate"] == -0.5
