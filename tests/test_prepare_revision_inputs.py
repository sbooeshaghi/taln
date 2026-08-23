"""Unit tests for deterministic Stage 0 input preparation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "scripts"))

from prepare_revision_inputs import boat_records, shifted_target, split_by_blind_hash


def test_blind_hash_split_is_order_independent_and_exact_size():
    document_ids = [f"paper-{index}" for index in range(12)]
    forward = split_by_blind_hash(document_ids, heldout_count=3)
    reverse = split_by_blind_hash(list(reversed(document_ids)), heldout_count=3)
    assert forward == reverse
    assert sum(value == "heldout_test" for value in forward.values()) == 3


def test_contextual_space_does_not_replace_canonical_target():
    source = "prefix target phrase suffix"
    start = source.index("target")
    contextual, contextual_start = shifted_target(source, "target phrase", start)
    assert contextual == " target phrase"
    assert contextual_start == start - 1


def test_boat_records_keep_canonical_and_contextual_intervals(tmp_path):
    source = "prefix target phrase here suffix"
    start = source.index("target")
    squad = {
        "data": [
            {
                "title": "Example",
                "paragraphs": [
                    {
                        "context": source,
                        "qas": [
                            {
                                "id": "q1",
                                "is_impossible": False,
                                "answers": [
                                    {
                                        "text": "target phrase here",
                                        "answer_start": start,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    path = tmp_path / "squad.json"
    path.write_text(json.dumps(squad), encoding="utf-8")

    records, counts = boat_records(path, "dev")

    assert counts["final_records"] == 1
    assert records[0]["target"] == "target phrase here"
    assert records[0]["contextual_target"] == " target phrase here"
    assert records[0]["gold_intervals_original"] == [
        {"start": start, "end": start + len("target phrase here")}
    ]
    assert records[0]["contextual_gold_intervals_original"] == [
        {"start": start - 1, "end": start + len("target phrase here")}
    ]
    assert records[0]["split"] == "heldout_test"
