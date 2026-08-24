"""Tests for the frozen Stage 1 matrix runner."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "scripts"))

from run_corrected_baseline_matrix import (
    DEFAULT_INPUT_ROOT,
    central_one_gap,
    evaluate_matrix_task,
    evaluate_one,
    iter_tasks,
    load_and_validate_freeze,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_central_one_gap_matches_the_frozen_middle_chunk_rule():
    gap = central_one_gap("one two three four five")
    assert gap == {
        "target": "one two four five",
        "removed_chunk": "three",
        "removed_chunk_index": 2,
        "original_target_chunk_count": 5,
        "removed_target_interval": [8, 13],
    }
    assert central_one_gap("one two") is None


def test_still_contiguous_deletions_are_separate_controls(tmp_path):
    record = {
        "record_id": "boat-repeat",
        "dataset": "boat",
        "split": "development",
        "document_id": "doc-repeat",
        "source": "alpha beta gamma and alpha gamma",
        "source_sha256": "hash",
        "target": "alpha beta gamma",
        "gold_intervals_original": [{"start": 0, "end": 16}],
        "generation_method": "human",
    }
    _write_jsonl(tmp_path / "boat_records.jsonl", [record])
    _write_jsonl(tmp_path / "bioboat_records.jsonl", [])
    _write_jsonl(tmp_path / "llmarkers_inventory.jsonl", [])

    tasks = list(iter_tasks(tmp_path, "development"))

    assert tasks[1]["condition"] == "one_gap_still_contiguous_control"
    assert tasks[1]["remains_exact_substring_after_deletion"]


def test_iter_tasks_expands_conditions_and_filters_positive_llmarkers(tmp_path):
    boat = {
        "record_id": "boat-1",
        "dataset": "boat",
        "split": "development",
        "document_id": "doc-1",
        "source": "one two three",
        "source_sha256": "source-hash",
        "target": "one two three",
        "gold_intervals_original": [{"start": 0, "end": 13}],
        "generation_method": "human",
    }
    bioboat = dict(boat, record_id="bio-1", dataset="bioboat")
    positive = {
        "record_id": "llm-1",
        "dataset": "llmarkers",
        "dataset_id": "set-1",
        "split": "development",
        "document_id": "llmarkers:set-1",
        "source_rationale": "T cells express CD3D.",
        "source_sha256": "llm-hash",
        "group_label": "T cells",
        "feature_label": "CD3D",
        "all_verified": True,
        "inventory_status": "not_independent_gold",
    }
    negative = dict(positive, record_id="llm-2", all_verified=False)
    _write_jsonl(tmp_path / "boat_records.jsonl", [boat])
    _write_jsonl(tmp_path / "bioboat_records.jsonl", [bioboat])
    _write_jsonl(tmp_path / "llmarkers_inventory.jsonl", [positive, negative])

    tasks = list(iter_tasks(tmp_path, "development"))

    assert len(tasks) == 6
    assert [task["condition"] for task in tasks].count("contiguous") == 2
    assert [task["condition"] for task in tasks].count("one_gap") == 2
    assert [task["condition"] for task in tasks].count("positive_label") == 2
    assert {task.get("label_field") for task in tasks[-2:]} == {
        "group_label",
        "feature_label",
    }


def test_matrix_uses_exact_once_and_symmetric_sequence_methods():
    task = {
        "schema_version": "test",
        "task_id": "task-1",
        "base_record_id": "record-1",
        "dataset": "boat",
        "condition": "one_gap",
        "split": "development",
        "document_id": "doc-1",
        "source": "alpha removed beta",
        "source_sha256": "hash",
        "target": "alpha beta",
        "target_sha256": "hash",
        "gold_intervals": [{"start": 0, "end": 18}],
    }
    result = evaluate_matrix_task(
        task,
        tokenizers=["whitespace"],
        methods=["taln", "lcs", "difflib", "semi_global_exact"],
        candidate_cap=100,
    )

    assert not result["exact"]["full_lexical_support"]
    assert set(result["sequence"]) == {"whitespace"}
    assert set(result["sequence"]["whitespace"]) == {
        "taln",
        "lcs",
        "difflib",
        "semi_global_exact",
    }
    assert all(
        method["full_lexical_support"]
        for method in result["sequence"]["whitespace"].values()
    )


def test_method_errors_remain_in_localization_denominator():
    task = {
        "task_id": "invalid-empty-target",
        "source": "alpha",
        "target": "",
        "gold_intervals": [{"start": 0, "end": 0}],
    }
    result = evaluate_one(
        task,
        method="taln",
        tokenizer="whitespace",
        candidate_cap=100,
    )
    assert result["error"] is not None
    assert result["localization_evaluable"]
    assert result["localization"] is False


@pytest.mark.skipif(
    not (DEFAULT_INPUT_ROOT / "boat_records.jsonl").exists(),
    reason="frozen input records are not distributed with the repository; "
    "regenerate them with analysis/scripts/prepare_revision_inputs.py",
)
def test_frozen_input_hashes_and_document_splits_validate():
    splits, hashes = load_and_validate_freeze(DEFAULT_INPUT_ROOT)
    assert len(hashes) == 3
    assert splits[("llmarkers", "llmarkers:bone_He2021")] == "heldout_test"
