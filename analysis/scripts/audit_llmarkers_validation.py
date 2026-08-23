"""Audit llmarkers extraction records for real-output validation.

This script summarizes manually curated and LLM-produced marker records from
the sibling `llmarkers` project without dropping failed or partially verified
records. It is intended as the first reproducible step before deciding which
subset should become a manuscript benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "llmarkers" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Path to llmarkers/data (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional path for JSON summary output",
    )
    parser.add_argument(
        "--records-output",
        type=Path,
        help="Optional path for per-record CSV output",
    )
    parser.add_argument(
        "--include-hca",
        action="store_true",
        help="Include the larger hca/manuscripts marker corpus",
    )
    parser.add_argument(
        "--print-datasets",
        action="store_true",
        help="Print per-dataset details to stdout",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalized(value: Any) -> str:
    return " ".join(str(value or "").split())


def exact_reconstruction(label: Any, reconstructed: Any) -> bool | None:
    label_text = normalized(label)
    reconstructed_text = normalized(reconstructed)
    if not label_text or not reconstructed_text:
        return None
    return label_text == reconstructed_text


def verification_class(record: dict[str, Any]) -> str:
    verification = record.get("_verification") or {}
    if not verification:
        return "verification_absent"

    source_found = verification.get("source_rationale_found")
    group_found = verification.get("group_label_found")
    feature_found = verification.get("feature_label_found")

    if verification.get("all_verified") is True:
        group_exact = exact_reconstruction(
            record.get("group_label"), verification.get("group_label_reconstructed")
        )
        feature_exact = exact_reconstruction(
            record.get("feature_label"), verification.get("feature_label_reconstructed")
        )
        if group_exact is False or feature_exact is False:
            return "all_verified_partial_reconstruction"
        return "all_verified_exact_reconstruction"

    missing = []
    if source_found is False:
        missing.append("source")
    if group_found is False:
        missing.append("group_label")
    if feature_found is False:
        missing.append("feature_label")

    if missing:
        return "missing_" + "_and_".join(missing)

    return "not_all_verified_inconsistent_metadata"


def flatten_record(
    path: Path,
    dataset_kind: str,
    dataset_id: str,
    record_index: int,
    record: dict[str, Any],
) -> dict[str, Any]:
    verification = record.get("_verification") or {}
    return {
        "dataset_kind": dataset_kind,
        "dataset_id": dataset_id,
        "record_index": record_index,
        "path": str(path),
        "source_id": record.get("source_id"),
        "data_id": record.get("data_id"),
        "organism": record.get("organism"),
        "group_label": record.get("group_label"),
        "group_name": record.get("group_name"),
        "feature_label": record.get("feature_label"),
        "feature_name": record.get("feature_name"),
        "source_type": record.get("source_type"),
        "source_rationale": record.get("source_rationale"),
        "source_rationale_found": verification.get("source_rationale_found"),
        "source_rationale_method": verification.get("source_rationale_method"),
        "group_label_found": verification.get("group_label_found"),
        "group_label_reconstructed": verification.get("group_label_reconstructed"),
        "group_label_exact_reconstruction": exact_reconstruction(
            record.get("group_label"), verification.get("group_label_reconstructed")
        ),
        "feature_label_found": verification.get("feature_label_found"),
        "feature_label_reconstructed": verification.get("feature_label_reconstructed"),
        "feature_label_exact_reconstruction": exact_reconstruction(
            record.get("feature_label"), verification.get("feature_label_reconstructed")
        ),
        "all_verified": verification.get("all_verified"),
        "verification_class": verification_class(record),
    }


def iter_marker_paths(data_root: Path, include_hca: bool) -> list[tuple[str, str, Path]]:
    paths: list[tuple[str, str, Path]] = []

    for path in sorted(data_root.glob("manual_papers/*/markers.json")):
        paths.append(("manual_papers", path.parent.name, path))

    for path in sorted(data_root.glob("*/evidence_llm/extracted_txt.json")):
        paths.append(("evidence_llm", path.parent.parent.name, path))

    if include_hca:
        for path in sorted(data_root.glob("hca/manuscripts/*/markers.json")):
            paths.append(("hca_manuscripts", path.parent.name, path))

    return paths


def load_records(data_root: Path, include_hca: bool) -> list[dict[str, Any]]:
    rows = []
    for dataset_kind, dataset_id, path in iter_marker_paths(data_root, include_hca):
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
        for record_index, record in enumerate(data):
            if not isinstance(record, dict):
                continue
            rows.append(flatten_record(path, dataset_kind, dataset_id, record_index, record))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, Counter] = defaultdict(Counter)
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    verification_classes = Counter()

    for row in rows:
        keys = [row["dataset_kind"], f"{row['dataset_kind']}:{row['dataset_id']}"]
        for key in keys:
            counter = by_kind[key] if ":" not in key else by_dataset[key]
            counter["records"] += 1
            counter["all_verified"] += row["all_verified"] is True
            counter["source_rationale_found"] += row["source_rationale_found"] is True
            counter["group_label_found"] += row["group_label_found"] is True
            counter["feature_label_found"] += row["feature_label_found"] is True
            counter["group_label_exact_reconstruction"] += (
                row["group_label_exact_reconstruction"] is True
            )
            counter["feature_label_exact_reconstruction"] += (
                row["feature_label_exact_reconstruction"] is True
            )
        verification_classes[row["verification_class"]] += 1

    return {
        "total_records": len(rows),
        "dataset_kinds": {key: dict(value) for key, value in sorted(by_kind.items())},
        "datasets": {key: dict(value) for key, value in sorted(by_dataset.items())},
        "verification_classes": dict(verification_classes),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    rows = load_records(data_root, include_hca=args.include_hca)
    summary = summarize(rows)

    stdout_summary = summary
    if not args.print_datasets:
        stdout_summary = {
            "total_records": summary["total_records"],
            "dataset_kinds": summary["dataset_kinds"],
            "verification_classes": summary["verification_classes"],
        }

    print(json.dumps(stdout_summary, indent=2, sort_keys=True))

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.records_output:
        if not rows:
            raise ValueError("No rows to write")
        write_csv(args.records_output, rows)


if __name__ == "__main__":
    main()
