"""Freeze the document-clustered task sample for the LLM lexical baseline.

Implements the sampling section of
analysis/config/revision_2026/llm_baseline_spec.json. The sample is drawn
deterministically (seed 20260814) from the frozen split manifest before any
model call. Held-out and development documents are disjoint by construction of
the frozen splits. The manifest records the sampled task identities and input
hashes so the later runner cannot silently change the sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from run_corrected_baseline_matrix import (
    iter_tasks,
    load_and_validate_freeze,
    sha256_file,
    write_json,
)
from run_selection_assessment import iter_stage2_tasks

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "revision_2026"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "llm_baseline"
SPEC_PATH = (
    REPO_ROOT / "analysis" / "config" / "revision_2026" / "llm_baseline_spec.json"
)

SEED = 20260814
HELDOUT_ONE_GAP_TARGET = 500
HELDOUT_CONTIGUOUS_TARGET = 250
DEVELOPMENT_TASK_TARGET = 200
SEVERE_CONDITION = "gaps3_width3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stage1_tasks_by_document(
    input_root: Path,
    split: str,
    frozen_splits: dict[tuple[str, str], str],
) -> dict[tuple[str, str], dict[str, list[dict[str, Any]]]]:
    """Group Stage 1 BOAT/BIO-BOAT tasks by (dataset, document) and condition."""
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for task in iter_tasks(input_root, split, frozen_splits):
        if task["dataset"] not in ("boat", "bioboat"):
            continue
        if task["condition"] not in ("contiguous", "one_gap"):
            continue
        key = (task["dataset"], task["document_id"])
        grouped.setdefault(key, {"contiguous": [], "one_gap": []})
        grouped[key][task["condition"]].append(
            {
                "stage": "stage1",
                "task_id": task["task_id"],
                "dataset": task["dataset"],
                "condition": task["condition"],
                "split": task["split"],
                "document_id": task["document_id"],
                "source_sha256": task["source_sha256"],
                "target_sha256": task["target_sha256"],
                "gold_intervals": task["gold_intervals"],
                "source": task["source"],
                "target": task["target"],
            }
        )
    return grouped


def severe_stage2_tasks_by_document(
    input_root: Path, split: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in iter_stage2_tasks(input_root, split):
        if task["condition"] != SEVERE_CONDITION:
            continue
        key = (task["dataset"], task["document_id"])
        grouped.setdefault(key, []).append(
            {
                "stage": "stage2",
                "task_id": task["task_id"],
                "dataset": task["dataset"],
                "condition": task["condition"],
                "split": task["split"],
                "document_id": task["document_id"],
                "source_sha256": sha256_text(task["source"]),
                "target_sha256": sha256_text(task["target"]),
                "gold_intervals": [
                    interval
                    if not isinstance(interval, dict)
                    else [interval["start"], interval["end"]]
                    for interval in task["gold_intervals"]
                ],
                "source": task["source"],
                "target": task["target"],
                "primary_noncontiguous": task["primary_noncontiguous"],
            }
        )
    return grouped


def sample_heldout(
    stage1: dict[tuple[str, str], dict[str, list[dict[str, Any]]]],
    stage2: dict[tuple[str, str], list[dict[str, Any]]],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Whole-document sampling until per-dataset condition targets are met."""
    tasks: list[dict[str, Any]] = []
    sampled_documents: dict[str, list[str]] = {"boat": [], "bioboat": []}
    counts = {
        dataset: {"one_gap": 0, "contiguous": 0, "stage2_severe": 0}
        for dataset in ("boat", "bioboat")
    }
    for dataset in ("boat", "bioboat"):
        documents = sorted(
            document for data, document in stage1 if data == dataset
        )
        order = rng.permutation(len(documents))
        for index in order:
            document = documents[index]
            need_one_gap = counts[dataset]["one_gap"] < HELDOUT_ONE_GAP_TARGET
            need_contiguous = (
                counts[dataset]["contiguous"] < HELDOUT_CONTIGUOUS_TARGET
            )
            if not (need_one_gap or need_contiguous):
                break
            grouped = stage1[(dataset, document)]
            took_any = False
            if need_one_gap and grouped["one_gap"]:
                tasks.extend(grouped["one_gap"])
                counts[dataset]["one_gap"] += len(grouped["one_gap"])
                took_any = True
            if need_contiguous and grouped["contiguous"]:
                tasks.extend(grouped["contiguous"])
                counts[dataset]["contiguous"] += len(grouped["contiguous"])
                took_any = True
            if took_any:
                sampled_documents[dataset].append(document)
                severe = stage2.get((dataset, document), [])
                tasks.extend(severe)
                counts[dataset]["stage2_severe"] += len(severe)
    return tasks, {"documents": sampled_documents, "counts": counts}


def sample_development(
    stage1: dict[tuple[str, str], dict[str, list[dict[str, Any]]]],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Small disjoint development sample for prompt development only."""
    tasks: list[dict[str, Any]] = []
    sampled_documents: dict[str, list[str]] = {"boat": [], "bioboat": []}
    for dataset in ("boat", "bioboat"):
        documents = sorted(
            document for data, document in stage1 if data == dataset
        )
        order = rng.permutation(len(documents))
        for index in order:
            if len(tasks) >= DEVELOPMENT_TASK_TARGET:
                break
            document = documents[index]
            grouped = stage1[(dataset, document)]
            document_tasks = grouped["one_gap"] + grouped["contiguous"]
            if not document_tasks:
                continue
            tasks.extend(document_tasks)
            sampled_documents[dataset].append(document)
    return tasks[:DEVELOPMENT_TASK_TARGET], {"documents": sampled_documents}


def write_tasks(path: Path, tasks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = output_root / "llm_baseline_sample_manifest.json"
    heldout_path = output_root / "heldout_sample_tasks.jsonl"
    development_path = output_root / "development_sample_tasks.jsonl"
    for path in (manifest_path, heldout_path, development_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} already exists")

    frozen_splits, input_hashes = load_and_validate_freeze(input_root)

    heldout_stage1 = stage1_tasks_by_document(
        input_root, "heldout_test", frozen_splits
    )
    heldout_stage2 = severe_stage2_tasks_by_document(input_root, "heldout_test")
    development_stage1 = stage1_tasks_by_document(
        input_root, "development", frozen_splits
    )

    rng = np.random.default_rng(SEED)
    heldout_tasks, heldout_info = sample_heldout(
        heldout_stage1, heldout_stage2, rng
    )
    development_tasks, development_info = sample_development(
        development_stage1, rng
    )

    heldout_ids = {task["task_id"] for task in heldout_tasks}
    development_ids = {task["task_id"] for task in development_tasks}
    if heldout_ids & development_ids:
        raise ValueError("Held-out and development samples overlap")
    if len(heldout_ids) != len(heldout_tasks):
        raise ValueError("Duplicate task in held-out sample")
    if len(development_ids) != len(development_tasks):
        raise ValueError("Duplicate task in development sample")

    write_tasks(heldout_path, heldout_tasks)
    write_tasks(development_path, development_tasks)

    manifest = {
        "schema_version": "llm-baseline-sample-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "targets": {
            "heldout_one_gap_per_dataset": HELDOUT_ONE_GAP_TARGET,
            "heldout_contiguous_per_dataset": HELDOUT_CONTIGUOUS_TARGET,
            "development_tasks": DEVELOPMENT_TASK_TARGET,
            "severe_condition": SEVERE_CONDITION,
        },
        "heldout": {
            **heldout_info,
            "tasks": len(heldout_tasks),
            "path": str(heldout_path.relative_to(REPO_ROOT))
            if heldout_path.is_relative_to(REPO_ROOT)
            else str(heldout_path),
            "sha256": sha256_file(heldout_path),
        },
        "development": {
            **development_info,
            "tasks": len(development_tasks),
            "path": str(development_path.relative_to(REPO_ROOT))
            if development_path.is_relative_to(REPO_ROOT)
            else str(development_path),
            "sha256": sha256_file(development_path),
        },
        "input_hashes": input_hashes,
        "spec": {
            "path": str(SPEC_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(SPEC_PATH),
        },
        "command": sys.argv,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest["heldout"]["counts"], indent=2))
    print(f"heldout tasks: {len(heldout_tasks)}")
    print(f"development tasks: {len(development_tasks)}")


if __name__ == "__main__":
    main()
