"""Tests for the frozen Stage 2 benchmark runner."""

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "scripts"))

from common_evaluator import DEFAULT_CANDIDATE_CAP
from run_multigap_benchmark import (
    METHODS,
    TOKENIZERS,
    strict_task_ids,
    task_from_variant,
    validate_heldout_arguments,
    validate_heldout_paths,
)


def test_variant_joins_to_frozen_source_without_copying_generation_details():
    base = {
        "record_id": "record-1",
        "dataset": "boat",
        "document_id": "doc-1",
        "split": "development",
        "source": "alpha removed beta",
        "source_sha256": "source-hash",
    }
    variant = {
        "variant_id": "variant-1",
        "base_record_id": "record-1",
        "dataset": "boat",
        "document_id": "doc-1",
        "split": "development",
        "source_sha256": "source-hash",
        "target": "alpha beta",
        "target_sha256": "target-hash",
        "condition": "multigap",
        "variant_status": "primary_noncontiguous",
        "cohort": "common_3x3",
        "gold_intervals_original": [{"start": 0, "end": 18}],
        "gold_occurrence_count": 1,
        "gap_count": 1,
        "gap_width": 1,
        "original_target_chunk_count": 3,
        "primary_noncontiguous": True,
        "remains_exact_substring_after_deletion": False,
        "placement_rule": "balanced_retained_segments_v1",
    }
    task = task_from_variant(variant, {"record-1": base})
    assert task["task_id"] == "variant-1"
    assert task["source"] == "alpha removed beta"
    assert task["gold_intervals"] == [{"start": 0, "end": 18}]
    assert "removed_blocks" not in task


def test_task_join_rejects_changed_source_identity():
    base = {
        "record_id": "record-1",
        "dataset": "boat",
        "document_id": "doc-1",
        "split": "development",
        "source": "alpha beta",
        "source_sha256": "base-hash",
    }
    variant = {
        "variant_id": "variant-1",
        "base_record_id": "record-1",
        "dataset": "boat",
        "document_id": "doc-1",
        "split": "development",
        "source_sha256": "different-hash",
    }
    with pytest.raises(ValueError, match="source_sha256 mismatch"):
        task_from_variant(variant, {"record-1": base})


def test_resume_index_rejects_duplicate_task_ids(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"task_id": "duplicate"})
        + "\n"
        + json.dumps({"task_id": "duplicate"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate task_id"):
        strict_task_ids(path)


def _heldout_args(**updates):
    values = {
        "split": "heldout_test",
        "limit": None,
        "tokenizers": list(TOKENIZERS),
        "methods": list(METHODS),
        "candidate_cap": DEFAULT_CANDIDATE_CAP,
        "overwrite": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_heldout_gate_rejects_parameter_changes():
    validate_heldout_arguments(_heldout_args())
    with pytest.raises(ValueError, match="--limit"):
        validate_heldout_arguments(_heldout_args(limit=1))
    with pytest.raises(ValueError, match="tokenizers"):
        validate_heldout_arguments(_heldout_args(tokenizers=["punctuation"]))
    with pytest.raises(ValueError, match="methods"):
        validate_heldout_arguments(_heldout_args(methods=["taln"]))
    with pytest.raises(ValueError, match="candidate cap"):
        validate_heldout_arguments(_heldout_args(candidate_cap=10))
    with pytest.raises(ValueError, match="overwritten"):
        validate_heldout_arguments(_heldout_args(overwrite=True))


def test_heldout_gate_requires_canonical_paths(tmp_path):
    args = _heldout_args()
    with pytest.raises(ValueError, match="canonical input root"):
        validate_heldout_paths(
            args,
            input_root=tmp_path,
            variants_path=Path("data/revision_2026/stage2/multigap_variants.jsonl").resolve(),
            output_root=Path("data/revision_2026/stage2").resolve(),
        )
