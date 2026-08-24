"""Run real-world alignment validation on the curated llmarkers benchmark.

The benchmark uses the curated llmarkers slices only:

- manual_papers/*/markers.json
- */evidence_llm/extracted_txt.json

It intentionally excludes the larger hca/manuscripts corpus so the manuscript
validation is based on a compact, auditable benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from audit_llmarkers_validation import DEFAULT_DATA_ROOT, load_records, normalized

from taln.taln_aln import (
    align_difflib,
    align_lcs,
    align_ng,
    norm_text,
    reconstruct_target_by_token,
)

DEFAULT_SUMMARY_OUTPUT = Path("data/curated_real_world_validation_summary.json")


AlignmentFn = Callable[[str, str, str], list[list[dict[str, Any]]]]


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
        default=DEFAULT_SUMMARY_OUTPUT,
        help=f"Path for JSON summary output (default: {DEFAULT_SUMMARY_OUTPUT})",
    )
    parser.add_argument(
        "--records-output",
        type=Path,
        help="Optional path for per-target JSONL output",
    )
    parser.add_argument(
        "--tokenization-types",
        nargs="+",
        default=["whitespace"],
        choices=["whitespace", "token"],
        help="Tokenization types for sequence alignment methods",
    )
    return parser.parse_args()


def clean_compare(value: Any) -> str:
    return normalized(norm_text(value)).strip()


def naive_align(source: str, target: str) -> list[list[dict[str, Any]]]:
    source_norm = norm_text(source)
    target_norm = norm_text(target)
    start = source_norm.find(target_norm)
    if start == -1:
        return []
    return [
        [
            {
                "token": target_norm,
                "enc_token": target_norm,
                "start_idx": start,
                "end_idx": start + len(target_norm),
            }
        ]
    ]


def best_reconstruction(alns: list[list[dict[str, Any]]]) -> str:
    if not alns:
        return ""
    best = max(alns, key=len)
    return reconstruct_target_by_token("", best)


def evaluate_method(
    source: str,
    target: str,
    verifier_reconstruction: str | None,
    method: str,
    tokenization: str,
    aligner: AlignmentFn | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    error = None
    try:
        if method == "naive":
            alns = naive_align(source, target)
        else:
            if aligner is None:
                raise ValueError(f"No aligner supplied for method {method}")
            alns = aligner(source, target, tokenization)
    except Exception as exc:
        alns = []
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - start) * 1000

    reconstruction = best_reconstruction(alns)
    full_reconstruction = bool(
        reconstruction and clean_compare(reconstruction) == clean_compare(target)
    )
    verifier_match = bool(
        verifier_reconstruction
        and reconstruction
        and clean_compare(reconstruction) == clean_compare(verifier_reconstruction)
    )

    return {
        "method": method,
        "tokenization": tokenization,
        "num_alignments": len(alns),
        "any_alignment": bool(alns),
        "best_reconstruction": reconstruction,
        "full_reconstruction": full_reconstruction,
        "verifier_reconstruction_match": verifier_match,
        "runtime_ms": elapsed_ms,
        "error": error,
    }


def target_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for target_kind in ["group_label", "feature_label"]:
            target = record[target_kind]
            if not target:
                continue
            rows.append(
                {
                    "dataset_kind": record["dataset_kind"],
                    "dataset_id": record["dataset_id"],
                    "record_index": record["record_index"],
                    "target_kind": target_kind,
                    "source_rationale": record["source_rationale"],
                    "target": target,
                    "expected_found": record[f"{target_kind}_found"] is True,
                    "verifier_reconstruction": record.get(
                        f"{target_kind}_reconstructed"
                    ),
                    "verifier_exact_reconstruction": record.get(
                        f"{target_kind}_exact_reconstruction"
                    ),
                    "record_all_verified": record["all_verified"] is True,
                    "verification_class": record["verification_class"],
                }
            )
    return rows


def method_results(rows: list[dict[str, Any]], tokenization_types: list[str]) -> list[dict[str, Any]]:
    aligners: dict[str, AlignmentFn] = {
        "taln": align_ng,
        "lcs": align_lcs,
        "difflib": align_difflib,
    }

    results = []
    for row in rows:
        source = row["source_rationale"] or ""
        target = row["target"] or ""
        verifier_reconstruction = row["verifier_reconstruction"]

        naive = evaluate_method(
            source=source,
            target=target,
            verifier_reconstruction=verifier_reconstruction,
            method="naive",
            tokenization="exact",
        )
        results.append({**row, **naive})

        for tokenization in tokenization_types:
            for method, aligner in aligners.items():
                result = evaluate_method(
                    source=source,
                    target=target,
                    verifier_reconstruction=verifier_reconstruction,
                    method=method,
                    tokenization=tokenization,
                    aligner=aligner,
                )
                results.append({**row, **result})

    return results


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = [row["runtime_ms"] for row in rows if row["error"] is None]
    expected_found = sum(row["expected_found"] for row in rows)
    expected_not_found = len(rows) - expected_found
    any_alignment = sum(row["any_alignment"] for row in rows)
    full_reconstruction = sum(row["full_reconstruction"] for row in rows)
    verifier_match = sum(row["verifier_reconstruction_match"] for row in rows)
    any_alignment_expected_found = sum(
        row["any_alignment"] and row["expected_found"] for row in rows
    )
    full_reconstruction_expected_found = sum(
        row["full_reconstruction"] and row["expected_found"] for row in rows
    )
    verifier_match_expected_found = sum(
        row["verifier_reconstruction_match"] and row["expected_found"] for row in rows
    )
    no_alignment_expected_not_found = sum(
        (not row["any_alignment"]) and (not row["expected_found"]) for row in rows
    )
    false_accept = sum(row["any_alignment"] and not row["expected_found"] for row in rows)
    false_reject = sum((not row["any_alignment"]) and row["expected_found"] for row in rows)
    errors = Counter(row["error"] for row in rows if row["error"])

    return {
        "targets": len(rows),
        "expected_found": expected_found,
        "expected_not_found": expected_not_found,
        "any_alignment": any_alignment,
        "full_reconstruction": full_reconstruction,
        "verifier_reconstruction_match": verifier_match,
        "any_alignment_expected_found": any_alignment_expected_found,
        "full_reconstruction_expected_found": full_reconstruction_expected_found,
        "verifier_reconstruction_match_expected_found": verifier_match_expected_found,
        "no_alignment_expected_not_found": no_alignment_expected_not_found,
        "false_accept_vs_expected_found": false_accept,
        "false_reject_vs_expected_found": false_reject,
        "any_alignment_rate": rate(any_alignment, len(rows)),
        "full_reconstruction_rate": rate(full_reconstruction, len(rows)),
        "verifier_reconstruction_match_rate": rate(verifier_match, len(rows)),
        "any_alignment_recall_vs_expected_found": rate(
            any_alignment_expected_found, expected_found
        ),
        "full_reconstruction_rate_vs_expected_found": rate(
            full_reconstruction_expected_found, expected_found
        ),
        "verifier_reconstruction_match_rate_vs_expected_found": rate(
            verifier_match_expected_found, expected_found
        ),
        "no_alignment_rate_vs_expected_not_found": rate(
            no_alignment_expected_not_found, expected_not_found
        ),
        "false_accept_rate_vs_expected_not_found": rate(false_accept, expected_not_found),
        "false_reject_rate_vs_expected_found": rate(false_reject, expected_found),
        "mean_runtime_ms": statistics.mean(runtimes) if runtimes else None,
        "median_runtime_ms": statistics.median(runtimes) if runtimes else None,
        "max_runtime_ms": max(runtimes) if runtimes else None,
        "errors": dict(errors),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(result["method"], result["tokenization"])].append(result)
        groups[(result["method"], result["tokenization"], result["target_kind"])].append(result)
        groups[
            (
                result["method"],
                result["tokenization"],
                result["dataset_kind"],
                result["target_kind"],
            )
        ].append(result)

    return {
        "benchmark": {
            "name": "curated_llmarkers_real_world_validation",
            "included_slices": [
                "manual_papers/*/markers.json",
                "*/evidence_llm/extracted_txt.json",
            ],
            "excluded_slices": ["hca/manuscripts/*/markers.json"],
        },
        "records": len(
            {
                (row["dataset_kind"], row["dataset_id"], row["record_index"])
                for row in results
            }
        ),
        "targets": len(
            {
                (
                    row["dataset_kind"],
                    row["dataset_id"],
                    row["record_index"],
                    row["target_kind"],
                )
                for row in results
            }
        ),
        "overall": {
            "::".join(key): summarize_group(value)
            for key, value in sorted(groups.items())
            if len(key) == 2
        },
        "by_target_kind": {
            "::".join(key): summarize_group(value)
            for key, value in sorted(groups.items())
            if len(key) == 3
        },
        "by_dataset_kind_and_target_kind": {
            "::".join(key): summarize_group(value)
            for key, value in sorted(groups.items())
            if len(key) == 4
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    records = load_records(args.data_root.expanduser().resolve(), include_hca=False)
    rows = target_rows(records)
    results = method_results(rows, args.tokenization_types)
    summary = summarize(results)

    print(json.dumps(summary["overall"], indent=2, sort_keys=True))

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.records_output:
        write_jsonl(args.records_output, results)


if __name__ == "__main__":
    main()
