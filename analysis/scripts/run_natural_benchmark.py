"""Evaluate alignment methods on human-curated natural marker records.

Population: the human-curated marker corpus in the sibling `llmarkers`
repository, restricted to records whose evidence is manuscript prose
(`source_type == "text"`). A curator copied each cell-type and marker-gene
label out of the passage recorded with it, so lexical support is a property of
how the corpus was constructed rather than an annotation added afterwards.

Each record contributes two label tasks: the cell-type (group) label and the
marker-gene (feature) label, each aligned against the recorded passage. Labels
carry leading or trailing whitespace from manual selection; outer whitespace is
stripped before evaluation and the count of affected labels is reported.

The corpus contains no curator-marked negatives, so this benchmark measures
recovery only. It supports no precision or false-acceptance claim.

The sibling repository is read and never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from common_evaluator import DEFAULT_CANDIDATE_CAP, evaluate_task
from run_corrected_baseline_matrix import package_version, write_json
from run_selection_assessment import bootstrap_ci, repo_relative

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLMARKERS_ROOT = REPO_ROOT.parent / "llmarkers" / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "revision_2026" / "natural"

STUDIES = [
    "adipose_Emont2022",
    "adipose_Hildreth2021",
    "bone_He2021",
    "eye_Gautam2021",
    "lung_Adams2020",
    "ovary_Wagner2020",
    "testis_Shamis2020",
]
LABEL_FIELDS = [("group_label", "cell_type"), ("feature_label", "marker_gene")]
TOKENIZERS = ("whitespace", "punctuation", "cl100k_base", "scibert", "pubmedbert")
SEQUENCE_METHODS = ("taln", "lcs", "difflib", "semi_global_exact")
BOOTSTRAP_SEED = 20260816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llmarkers-root", type=Path, default=DEFAULT_LLMARKERS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_tasks(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one task per (record, label field) over text-evidence records."""
    tasks: list[dict[str, Any]] = []
    stats = {
        "records_total": 0,
        "records_text": 0,
        "labels_whitespace_padded": 0,
        "labels_empty_after_strip": 0,
        "records_per_study": {},
    }
    seen: set[tuple[str, str, str]] = set()
    for study in STUDIES:
        path = root / study / "evidence_human" / "extracted.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        text_rows = [row for row in rows if row.get("source_type") == "text"]
        stats["records_total"] += len(rows)
        stats["records_text"] += len(text_rows)
        stats["records_per_study"][study] = len(text_rows)
        for index, row in enumerate(text_rows):
            source = str(row.get("source_rationale") or "")
            if not source.strip():
                continue
            for field, label_type in LABEL_FIELDS:
                raw = str(row.get(field) or "")
                label = raw.strip()
                if raw != label:
                    stats["labels_whitespace_padded"] += 1
                if not label:
                    stats["labels_empty_after_strip"] += 1
                    continue
                key = (study, label, sha256_text(source))
                duplicate = key in seen
                seen.add(key)
                tasks.append(
                    {
                        "task_id": f"natural:{study}:{index}:{field}",
                        "study": study,
                        "document_id": f"llmarkers:{study}",
                        "record_index": index,
                        "label_type": label_type,
                        "label": label,
                        "label_raw": raw,
                        "source": source,
                        "source_sha256": sha256_text(source),
                        "duplicate_of_earlier_task": duplicate,
                    }
                )
    stats["tasks"] = len(tasks)
    stats["unique_label_passage_pairs"] = len(seen)
    return tasks, stats


def contiguity(task: dict[str, Any]) -> str:
    """Classify a task by whether exact substring matching can find the label."""
    result = evaluate_task(
        task_id=task["task_id"],
        source=task["source"],
        target=task["label"],
        method="exact",
        tokenizer="whitespace",
        candidate_cap=DEFAULT_CANDIDATE_CAP,
        include_candidates=False,
    )
    return "contiguous" if result["full_lexical_support"] else "non_contiguous"


def evaluate(task: dict[str, Any]) -> dict[str, Any]:
    outcome: dict[str, Any] = {"exact": None, "sequence": {}}
    exact = evaluate_task(
        task_id=task["task_id"],
        source=task["source"],
        target=task["label"],
        method="exact",
        tokenizer="whitespace",
        candidate_cap=DEFAULT_CANDIDATE_CAP,
        include_candidates=False,
    )
    outcome["exact"] = bool(exact["full_lexical_support"])
    for tokenizer in TOKENIZERS:
        outcome["sequence"][tokenizer] = {}
        for method in SEQUENCE_METHODS:
            try:
                result = evaluate_task(
                    task_id=task["task_id"],
                    source=task["source"],
                    target=task["label"],
                    method=method,
                    tokenizer=tokenizer,
                    candidate_cap=DEFAULT_CANDIDATE_CAP,
                    include_candidates=False,
                )
                outcome["sequence"][tokenizer][method] = {
                    "support": bool(result["full_lexical_support"]),
                    "candidates": int(result["candidate_count"]),
                    "unique_intervals": int(result["unique_source_interval_count"]),
                }
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                outcome["sequence"][tokenizer][method] = {
                    "support": False,
                    "candidates": 0,
                    "unique_intervals": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    return outcome


def rate_block(
    records: list[dict[str, Any]],
    predicate,
    rng: np.random.Generator,
) -> dict[str, Any]:
    by_document: dict[str, tuple[int, int]] = {}
    hits = 0
    for record in records:
        hit = int(bool(predicate(record)))
        hits += hit
        prior = by_document.get(record["document_id"], (0, 0))
        by_document[record["document_id"]] = (prior[0] + hit, prior[1] + 1)
    n = len(records)
    low, high = bootstrap_ci(by_document, rng)
    return {
        "hits": hits,
        "n": n,
        "rate": hits / n if n else None,
        "ci95": [low, high],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata: dict[str, list[dict[str, Any]]] = {"all": records}
    for label_type in ("cell_type", "marker_gene"):
        strata[label_type] = [
            record for record in records if record["label_type"] == label_type
        ]
    for contig in ("contiguous", "non_contiguous"):
        strata[contig] = [
            record for record in records if record["contiguity"] == contig
        ]

    payload: dict[str, Any] = {"strata": {}}
    for name, group in strata.items():
        block: dict[str, Any] = {
            "tasks": len(group),
            "exact": rate_block(group, lambda r: r["exact"], rng),
            "methods": {},
        }
        for tokenizer in TOKENIZERS:
            for method in SEQUENCE_METHODS:
                block["methods"][f"{tokenizer}:{method}"] = rate_block(
                    group,
                    lambda r, t=tokenizer, m=method: r["sequence"][t][m]["support"],
                    rng,
                )
        payload["strata"][name] = block

    payload["by_study"] = {}
    for study in STUDIES:
        group = [record for record in records if record["study"] == study]
        if not group:
            continue
        payload["by_study"][study] = {
            "tasks": len(group),
            "exact_hits": sum(record["exact"] for record in group),
            "taln_punctuation_hits": sum(
                record["sequence"]["punctuation"]["taln"]["support"]
                for record in group
            ),
        }
    return payload


def main() -> None:
    args = parse_args()
    root = args.llmarkers_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    records_path = output_root / "natural_benchmark_records.jsonl"
    summary_path = output_root / "natural_benchmark_summary.json"
    metadata_path = output_root / "natural_benchmark_run_metadata.json"
    if records_path.exists() and not args.overwrite:
        raise FileExistsError(records_path)

    tasks, stats = load_tasks(root)
    started = datetime.now(timezone.utc)

    records: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            record = {
                "schema_version": "natural-benchmark-record-v1",
                "task_id": task["task_id"],
                "study": task["study"],
                "document_id": task["document_id"],
                "label_type": task["label_type"],
                "label": task["label"],
                "duplicate_of_earlier_task": task["duplicate_of_earlier_task"],
                "contiguity": contiguity(task),
                **evaluate(task),
            }
            records.append(record)
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    summary = summarize(records)
    summary["schema_version"] = "natural-benchmark-summary-v1"
    summary["population"] = stats
    summary["boundary"] = (
        "Human-curated marker records with manuscript-prose evidence. Labels "
        "were copied from the recorded passage during curation, so the corpus "
        "contains no curator-marked negatives and supports no precision claim."
    )
    summary["contiguity_counts"] = dict(
        Counter(record["contiguity"] for record in records)
    )
    write_json(summary_path, summary)

    ended = datetime.now(timezone.utc)
    write_json(
        metadata_path,
        {
            "schema_version": "natural-benchmark-run-metadata-v1",
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "llmarkers_root": str(root),
            "source_files": {
                f"{study}/evidence_human/extracted.json": hashlib.sha256(
                    (root / study / "evidence_human" / "extracted.json").read_bytes()
                ).hexdigest()
                for study in STUDIES
            },
            "records_path": repo_relative(records_path),
            "summary_path": repo_relative(summary_path),
            "tokenizers": list(TOKENIZERS),
            "methods": list(SEQUENCE_METHODS),
            "population": stats,
            "software": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "taln": package_version("taln"),
            },
            "command": sys.argv,
        },
    )
    print(json.dumps({"population": stats, "contiguity": summary["contiguity_counts"]}, indent=2))


if __name__ == "__main__":
    main()
