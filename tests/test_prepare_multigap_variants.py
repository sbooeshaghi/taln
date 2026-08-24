"""Tests for deterministic Stage 2 multi-gap construction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "scripts"))

from prepare_multigap_variants import (
    COMMON_COHORT_MIN_CHUNKS,
    deletion_blocks,
    literal_occurrences,
    retained_segment_lengths,
    variants_for_record,
)
from run_corrected_baseline_matrix import sha256_text


def _record(source=None):
    target = "one two three four five six seven eight nine ten eleven twelve thirteen"
    source = source or f"prefix {target} suffix"
    start = source.index(target)
    return {
        "record_id": "record-1",
        "dataset": "boat",
        "document_id": "doc-1",
        "split": "development",
        "source": source,
        "source_sha256": sha256_text(source),
        "target": target,
        "target_sha256": sha256_text(target),
        "gold_intervals_original": [
            {"start": start, "end": start + len(target)}
        ],
    }


def test_balanced_placement_matches_stage1_and_separates_gaps():
    assert COMMON_COHORT_MIN_CHUNKS == 13
    assert retained_segment_lengths(13, 1, 1) == (6, 6)
    assert deletion_blocks(13, 1, 1) == ((6, 7),)
    assert retained_segment_lengths(13, 3, 3) == (1, 1, 1, 1)
    assert deletion_blocks(13, 3, 3) == ((1, 4), (5, 8), (9, 12))


def test_full_factorial_variants_are_stable_and_reconstructable():
    first = variants_for_record(_record())
    second = variants_for_record(_record())
    assert first == second
    assert len(first) == 10
    assert len({row["variant_id"] for row in first}) == 10
    assert {(row["gap_count"], row["gap_width"]) for row in first} == {
        (0, 0),
        *((gap_count, gap_width) for gap_count in range(1, 4) for gap_width in range(1, 4)),
    }

    original_chunks = _record()["target"].split()
    for row in first[1:]:
        removed = row["removed_target_chunk_indices"]
        assert removed == sorted(set(removed))
        assert removed[0] > 0
        assert removed[-1] < len(original_chunks) - 1
        retained = row["retained_target_chunk_indices"]
        assert row["target"] == " ".join(original_chunks[index] for index in retained)
        assert len(row["removed_blocks"]) == row["gap_count"]
        assert all(block["width"] == row["gap_width"] for block in row["removed_blocks"])


def test_short_targets_are_ineligible():
    record = _record()
    record["target"] = "one two three four five six seven eight nine ten eleven twelve"
    record["target_sha256"] = sha256_text(record["target"])
    record["source"] = record["target"]
    record["source_sha256"] = sha256_text(record["source"])
    record["gold_intervals_original"] = [
        {"start": 0, "end": len(record["target"])}
    ]
    assert variants_for_record(record) == []


def test_repeated_original_occurrences_are_all_gold():
    target = _record()["target"]
    source = f"{target}; then {target}."
    variants = variants_for_record(_record(source))
    assert len(literal_occurrences(source, target)) == 2
    assert all(row["gold_occurrence_count"] == 2 for row in variants)
    assert all(len(row["gold_intervals_original"]) == 2 for row in variants)


def test_shortened_exact_substrings_are_labeled_controls():
    record = _record()
    variants = variants_for_record(record)
    one_gap = next(
        row for row in variants if row["gap_count"] == 1 and row["gap_width"] == 1
    )
    record["source"] += " and " + one_gap["target"]
    record["source_sha256"] = sha256_text(record["source"])
    rebuilt = variants_for_record(record)
    control = next(
        row for row in rebuilt if row["gap_count"] == 1 and row["gap_width"] == 1
    )
    assert control["variant_status"] == "still_contiguous_control"
    assert control["remains_exact_substring_after_deletion"]
    assert not control["primary_noncontiguous"]
