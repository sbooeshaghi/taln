"""Tests for Stage 1 clustered summaries and discrepancy indexing."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "scripts"))

from run_corrected_baseline_matrix import (
    RESULT_FIELDS,
    compact_matrix_row,
    evaluate_matrix_task,
)
from summarize_corrected_baseline_matrix import paired_counts, summarize


def test_paired_counts_partition_every_task():
    counts = paired_counts(
        [True, True, False, False],
        [True, False, True, False],
    )
    assert counts == {
        "both_succeed": 1,
        "reference_only": 1,
        "comparison_only": 1,
        "neither_succeeds": 1,
    }


def test_summarizer_writes_grouped_outputs(tmp_path):
    base = {
        "schema_version": "test",
        "base_record_id": "record-1",
        "dataset": "boat",
        "split": "development",
        "document_id": "doc-1",
        "source": "alpha removed beta.",
        "source_sha256": "source-hash",
        "gold_intervals": [{"start": 0, "end": 19}],
    }
    tasks = [
        {
            **base,
            "task_id": "task-contiguous",
            "condition": "contiguous",
            "target": "alpha removed beta",
            "target_sha256": "target-hash-1",
        },
        {
            **base,
            "task_id": "task-gap",
            "condition": "one_gap",
            "target": "alpha beta",
            "target_sha256": "target-hash-2",
        },
        {
            **base,
            "task_id": "task-llmarkers",
            "base_record_id": "record-llmarkers",
            "dataset": "llmarkers",
            "condition": "positive_label",
            "label_field": "feature_label",
            "target": "beta",
            "target_sha256": "target-hash-3",
        },
    ]
    rows = [
        evaluate_matrix_task(
            task,
            tokenizers=["whitespace", "punctuation"],
            methods=["taln", "lcs", "difflib", "semi_global_exact"],
            candidate_cap=100,
        )
        for task in tasks
    ]
    tokenizers = ["whitespace", "punctuation"]
    methods = ["taln", "lcs", "difflib", "semi_global_exact"]
    compact_rows = [
        compact_matrix_row(row, tokenizers=tokenizers, methods=methods)
        for row in rows
    ]
    records = tmp_path / "records.jsonl"
    records.write_text(
        "".join(json.dumps(row) + "\n" for row in compact_rows), encoding="utf-8"
    )
    (tmp_path / "corrected_baseline_run_config.json").write_text(
        json.dumps(
            {
                "tokenizers": tokenizers,
                "methods": methods,
                "record_encoding": {"result_fields": list(RESULT_FIELDS)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    (input_root / "boat_records.jsonl").write_text(
        json.dumps({"record_id": "record-1", "source": base["source"]}) + "\n",
        encoding="utf-8",
    )
    (input_root / "bioboat_records.jsonl").write_text("", encoding="utf-8")
    (input_root / "llmarkers_inventory.jsonl").write_text(
        json.dumps(
            {
                "record_id": "record-llmarkers",
                "source_rationale": base["source"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "summary"
    result = summarize(
        argparse.Namespace(
            records=[records],
            output_dir=output,
            input_root=input_root,
            write_combined_records=False,
            bootstrap_resamples=100,
            seed=7,
            sample_per_discrepancy=2,
        )
    )

    assert result["rows"] == 3
    assert result["unique_task_ids"] == 3
    assert result["groups"]
    assert result["primary_contrasts"]
    assert result["stratified_groups"]
    assert result["stratified_primary_contrasts"]
    assert (output / "corrected_baseline_summary.json").exists()
    assert (output / "tokenizer_by_method_table.tsv").exists()
    assert (output / "corrected_baseline_discrepancy_report.md").exists()
    assert result["discrepancy_records"] is not None
    assert (output / "corrected_baseline_discrepancies.jsonl").exists()

    skipped_output = tmp_path / "summary-without-discrepancy-index"
    skipped_output.mkdir()
    stale_index = skipped_output / "corrected_baseline_discrepancies.jsonl"
    stale_index.write_text("stale\n", encoding="utf-8")
    skipped_result = summarize(
        argparse.Namespace(
            records=[records],
            output_dir=skipped_output,
            input_root=input_root,
            write_combined_records=False,
            skip_discrepancy_records=True,
            bootstrap_resamples=100,
            seed=7,
            sample_per_discrepancy=2,
        )
    )
    assert skipped_result["discrepancy_records"] is None
    assert not stale_index.exists()
