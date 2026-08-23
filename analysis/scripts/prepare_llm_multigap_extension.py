"""Freeze the rebalanced multi-gap sample for the LLM lexical baseline.

Implements the 2026-08-16 rebalance amendment in llm_baseline_spec.json: every
held-out multi-gap document contributes at most RECORD_CAP base records,
selected deterministically by SHA-256 of the record identifier, so that no
single document dominates the document-clustered interval. Every primary
non-contiguous variant of each selected record is included.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from run_corrected_baseline_matrix import sha256_file, write_json
from run_selection_assessment import iter_stage2_tasks

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = REPO_ROOT / "data" / "revision_2026"
OUTPUT_ROOT = INPUT_ROOT / "llm_baseline"
SPEC_PATH = (
    REPO_ROOT / "analysis" / "config" / "revision_2026" / "llm_baseline_spec.json"
)

RECORD_CAP = 10
SELECTION_SALT = "revision-2026-llm-multigap-balanced-v1"


def main() -> None:
    tasks_path = OUTPUT_ROOT / "heldout_multigap_tasks.jsonl"
    manifest_path = OUTPUT_ROOT / "llm_baseline_multigap_manifest.json"

    variants_by_record: dict[str, list[dict]] = defaultdict(list)
    records_by_document: dict[tuple[str, str], set[str]] = defaultdict(set)
    for task in iter_stage2_tasks(INPUT_ROOT, "heldout_test"):
        if not task["primary_noncontiguous"]:
            continue
        variants_by_record[task["base_record_id"]].append(task)
        records_by_document[(task["dataset"], task["document_id"])].add(
            task["base_record_id"]
        )

    selected: list[dict] = []
    documents: dict[str, list[str]] = {"boat": [], "bioboat": []}
    record_counts: dict[str, int] = {"boat": 0, "bioboat": 0}
    for (dataset, document), records in sorted(records_by_document.items()):
        ranked = sorted(
            records,
            key=lambda record: hashlib.sha256(
                f"{SELECTION_SALT}\x1f{record}".encode()
            ).hexdigest(),
        )
        picked = ranked[:RECORD_CAP]
        documents[dataset].append(document)
        record_counts[dataset] += len(picked)
        for record in picked:
            selected.extend(variants_by_record[record])

    task_ids = [task["task_id"] for task in selected]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate variant in rebalanced sample")

    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with tasks_path.open("w", encoding="utf-8") as handle:
        for task in selected:
            row = {
                "stage": "stage2",
                "task_id": task["task_id"],
                "dataset": task["dataset"],
                "condition": task["condition"],
                "split": task["split"],
                "document_id": task["document_id"],
                "gold_intervals": [
                    interval
                    if not isinstance(interval, dict)
                    else [interval["start"], interval["end"]]
                    for interval in task["gold_intervals"]
                ],
                "source": task["source"],
                "target": task["target"],
                "source_sha256": hashlib.sha256(
                    task["source"].encode()
                ).hexdigest(),
                "target_sha256": hashlib.sha256(
                    task["target"].encode()
                ).hexdigest(),
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    manifest = {
        "schema_version": "llm-baseline-multigap-manifest-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": "all held-out multi-gap documents, capped per document",
        "record_cap_per_document": RECORD_CAP,
        "selection_salt": SELECTION_SALT,
        "documents_used": {
            dataset: len(values) for dataset, values in documents.items()
        },
        "documents": documents,
        "records": record_counts,
        "tasks": len(selected),
        "path": str(tasks_path.relative_to(REPO_ROOT))
        if tasks_path.is_relative_to(REPO_ROOT)
        else str(tasks_path),
        "sha256": sha256_file(tasks_path),
        "spec": {
            "path": str(SPEC_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(SPEC_PATH),
        },
        "command": sys.argv,
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "documents": manifest["documents_used"],
                "records": record_counts,
                "tasks": len(selected),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
