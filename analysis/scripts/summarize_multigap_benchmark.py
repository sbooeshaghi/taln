"""Summarize Stage 2 multi-gap recovery, localization, and candidate burden."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from revision_statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_SEED,
    document_clustered_binary_ci,
    paired_document_clustered_difference_ci,
)
from run_multigap_benchmark import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VARIANTS,
    STAGE2_SPEC,
    load_and_validate_variants,
)
from summarize_corrected_baseline_matrix import (
    expand_matrix_row,
    paired_counts,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD_ORDER = ("exact", "taln", "lcs", "difflib")
TOKENIZER_ORDER = (
    "character_exact",
    "punctuation",
    "cl100k_base",
    "pubmedbert",
)


@dataclass
class GroupAccumulator:
    document_ids: list[str] = field(default_factory=list)
    support: list[bool] = field(default_factory=list)
    localization: list[bool] = field(default_factory=list)
    no_matched_token_path: list[bool] = field(default_factory=list)
    errors: int = 0
    runtime_ms: list[float] = field(default_factory=list)
    truncated: int = 0
    burden_document_ids: list[str] = field(default_factory=list)
    candidate_counts: list[int] = field(default_factory=list)
    burden_truncated: list[bool] = field(default_factory=list)
    unique_interval_counts: list[int] = field(default_factory=list)
    multiple_location: list[bool] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit development smoke-test records that do not cover a full split.",
    )
    return parser.parse_args()


def result_map(row: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    results = {("character_exact", "exact"): row["exact"]}
    for tokenizer, methods in row["sequence"].items():
        for method, result in methods.items():
            results[(tokenizer, method)] = result
    return results


def group_key(
    row: dict[str, Any], tokenizer: str, method: str
) -> tuple[str, str, str, int, int, str, str]:
    return (
        row["split"],
        row["dataset"],
        row["variant_status"],
        int(row["gap_count"]),
        int(row["gap_width"]),
        tokenizer,
        method,
    )


def _p95(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(round(0.95 * (len(ordered) - 1)), len(ordered) - 1)]


def summarize_group(
    accumulator: GroupAccumulator, *, n_resamples: int, seed: int
) -> dict[str, Any]:
    support_ci = document_clustered_binary_ci(
        accumulator.document_ids,
        accumulator.support,
        n_resamples=n_resamples,
        seed=seed,
    )
    localization_ci = document_clustered_binary_ci(
        accumulator.document_ids,
        accumulator.localization,
        n_resamples=n_resamples,
        seed=seed,
    )
    no_path_ci = document_clustered_binary_ci(
        accumulator.document_ids,
        accumulator.no_matched_token_path,
        n_resamples=n_resamples,
        seed=seed,
    )
    multi_location_ci = (
        document_clustered_binary_ci(
            accumulator.burden_document_ids,
            accumulator.multiple_location,
            n_resamples=n_resamples,
            seed=seed,
        )
        if accumulator.multiple_location
        else None
    )
    return {
        "tasks": len(accumulator.support),
        "documents": len(set(accumulator.document_ids)),
        "full_lexical_support_count": sum(accumulator.support),
        "full_lexical_support": support_ci,
        "localization_count": sum(accumulator.localization),
        "localization": localization_ci,
        "no_matched_token_path_count": sum(accumulator.no_matched_token_path),
        "no_matched_token_path": no_path_ci,
        "errors": accumulator.errors,
        "median_runtime_ms_diagnostic": statistics.median(accumulator.runtime_ms),
        "candidate_count_truncated_tasks": accumulator.truncated,
        "candidate_burden": {
            "population": "full-support punctuation-tokenized taln tasks only",
            "tasks": len(accumulator.candidate_counts),
            "median_candidate_count": statistics.median(
                accumulator.candidate_counts
            )
            if accumulator.candidate_counts
            else None,
            "p95_candidate_count": _p95(accumulator.candidate_counts),
            "max_candidate_count": max(accumulator.candidate_counts)
            if accumulator.candidate_counts
            else None,
            "median_unique_source_interval_count": statistics.median(
                accumulator.unique_interval_counts
            )
            if accumulator.unique_interval_counts
            else None,
            "p95_unique_source_interval_count": _p95(
                accumulator.unique_interval_counts
            ),
            "max_unique_source_interval_count": max(
                accumulator.unique_interval_counts
            )
            if accumulator.unique_interval_counts
            else None,
            "multiple_source_location_count": sum(accumulator.multiple_location),
            "multiple_source_location": multi_location_ci,
            "truncated_candidate_count_tasks": sum(
                accumulator.burden_truncated
            ),
            "count_note": "Counts are lower bounds when truncation is true.",
        },
        "runtime_note": "Diagnostic only; tokenizer caching and serial method order affect timing.",
    }


def validate_row_against_variant(
    row: dict[str, Any], variant: dict[str, Any]
) -> None:
    comparisons = {
        "task_id": variant["variant_id"],
        "variant_id": variant["variant_id"],
        "base_record_id": variant["base_record_id"],
        "dataset": variant["dataset"],
        "document_id": variant["document_id"],
        "split": variant["split"],
        "target_sha256": variant["target_sha256"],
        "variant_status": variant["variant_status"],
        "gap_count": variant["gap_count"],
        "gap_width": variant["gap_width"],
    }
    for field_name, expected in comparisons.items():
        if row.get(field_name) != expected:
            raise ValueError(
                f"Benchmark/variant {field_name} mismatch for {variant['variant_id']}"
            )


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "split",
        "dataset",
        "variant_status",
        "gap_count",
        "gap_width",
        "tokenizer",
        "method",
        "tasks",
        "documents",
        "support_count",
        "support_rate",
        "support_ci_lower",
        "support_ci_upper",
        "localization_count",
        "localization_scope",
        "localization_rate",
        "localization_ci_lower",
        "localization_ci_upper",
        "no_matched_token_path_count",
        "no_matched_token_path_rate",
        "no_matched_token_path_ci_lower",
        "no_matched_token_path_ci_upper",
        "errors",
        "truncated_tasks",
        "median_runtime_ms_diagnostic",
        "burden_tasks",
        "median_candidate_count",
        "p95_candidate_count",
        "max_candidate_count",
        "median_unique_interval_count",
        "p95_unique_interval_count",
        "max_unique_interval_count",
        "multiple_location_rate",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write(
                "\t".join(str(row.get(column, "")) for column in columns) + "\n"
            )


def _severity_contrasts(
    outcomes: dict[tuple[str, str, str, str, int, int], tuple[str, bool]],
    *,
    spec: dict[str, Any],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    summaries = {}
    splits = sorted({key[0] for key in outcomes})
    datasets = sorted({key[1] for key in outcomes})
    methods = sorted({key[2] for key in outcomes})
    for split in splits:
        for dataset in datasets:
            for method in methods:
                for contrast in spec["statistics"]["severity_contrasts"]:
                    comparison = contrast["comparison"]
                    reference = contrast["reference"]
                    reference_rows = {
                        base_id: value
                        for (
                            row_split,
                            row_dataset,
                            row_method,
                            base_id,
                            gap_count,
                            gap_width,
                        ), value in outcomes.items()
                        if row_split == split
                        and row_dataset == dataset
                        and row_method == method
                        and gap_count == reference["gap_count"]
                        and gap_width == reference["gap_width"]
                    }
                    comparison_rows = {
                        base_id: value
                        for (
                            row_split,
                            row_dataset,
                            row_method,
                            base_id,
                            gap_count,
                            gap_width,
                        ), value in outcomes.items()
                        if row_split == split
                        and row_dataset == dataset
                        and row_method == method
                        and gap_count == comparison["gap_count"]
                        and gap_width == comparison["gap_width"]
                    }
                    paired_ids = sorted(
                        reference_rows.keys() & comparison_rows.keys()
                    )
                    if not paired_ids:
                        continue
                    document_ids = []
                    reference_values = []
                    comparison_values = []
                    for base_id in paired_ids:
                        reference_document, reference_value = reference_rows[base_id]
                        comparison_document, comparison_value = comparison_rows[base_id]
                        if reference_document != comparison_document:
                            raise ValueError(f"Document mismatch for paired {base_id}")
                        document_ids.append(reference_document)
                        reference_values.append(reference_value)
                        comparison_values.append(comparison_value)
                    key = f"{split}::{dataset}::{method}::{contrast['name']}"
                    summaries[key] = {
                        "split": split,
                        "dataset": dataset,
                        "name": contrast["name"],
                        "interpretation": contrast["interpretation"],
                        "reference": reference,
                        "comparison": comparison,
                        "tokenizer": spec["statistics"]["severity_tokenizer"],
                        "method": method,
                        "paired_records": len(paired_ids),
                        "paired_counts": paired_counts(
                            reference_values, comparison_values
                        ),
                        "clustered_difference_ci": paired_document_clustered_difference_ci(
                            document_ids,
                            reference_values,
                            comparison_values,
                            n_resamples=n_resamples,
                            seed=seed,
                        ),
                    }
    return summaries


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be positive")
    records_paths = [path.expanduser().resolve() for path in args.records]
    for path in records_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    variants_path = args.variants.expanduser().resolve()
    freeze, variants = load_and_validate_variants(variants_path)
    spec = read_json(STAGE2_SPEC)

    groups: dict[
        tuple[str, str, str, int, int, str, str], GroupAccumulator
    ] = defaultdict(GroupAccumulator)
    agreements: dict[
        tuple[str, str, str, int, int, str, str], dict[str, int]
    ] = defaultdict(
        lambda: {
            "both_succeed": 0,
            "taln_only": 0,
            "comparison_only": 0,
            "neither_succeeds": 0,
        }
    )
    severity_outcomes: dict[
        tuple[str, str, str, str, int, int], tuple[str, bool]
    ] = {}
    seen_ids = set()
    seen_splits = set()

    for records_path in records_paths:
        config_path = records_path.parent / "multigap_benchmark_run_config.json"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        configuration = read_json(config_path)
        if configuration["variant_file_sha256"] != freeze["variant_file"]["sha256"]:
            raise ValueError("Run configuration used a different variant manifest")
        if configuration.get("limit") is not None and not args.allow_incomplete:
            raise ValueError("Limited records require --allow-incomplete")
        for raw_row in read_jsonl(records_path):
            row = expand_matrix_row(raw_row, configuration)
            task_id = row["task_id"]
            if task_id in seen_ids:
                raise ValueError(f"Duplicate task ID across inputs: {task_id}")
            seen_ids.add(task_id)
            seen_splits.add(row["split"])
            try:
                variant = variants[task_id]
            except KeyError as exc:
                raise KeyError(f"Result absent from variant manifest: {task_id}") from exc
            validate_row_against_variant(row, variant)
            results = result_map(row)

            if (
                row["variant_status"] == "primary_noncontiguous"
                and results[("character_exact", "exact")]["full_lexical_support"]
            ):
                raise ValueError(f"Primary variant passed exact substring QA: {task_id}")

            for (tokenizer, method), result in results.items():
                accumulator = groups[group_key(row, tokenizer, method)]
                support = bool(result["full_lexical_support"])
                accumulator.document_ids.append(row["document_id"])
                accumulator.support.append(support)
                accumulator.localization.append(bool(result["localization"]))
                accumulator.no_matched_token_path.append(
                    result["candidate_count"] == 0
                )
                accumulator.errors += result["error"] is not None
                accumulator.runtime_ms.append(float(result["runtime_ms"]))
                accumulator.truncated += bool(result["candidate_count_truncated"])
                if method == "taln" and tokenizer == "punctuation" and support:
                    accumulator.burden_document_ids.append(row["document_id"])
                    accumulator.candidate_counts.append(int(result["candidate_count"]))
                    accumulator.burden_truncated.append(
                        bool(result["candidate_count_truncated"])
                    )
                    unique_count = int(result["unique_source_interval_count"])
                    accumulator.unique_interval_counts.append(unique_count)
                    accumulator.multiple_location.append(unique_count > 1)

            for tokenizer in configuration["tokenizers"]:
                taln_value = bool(results[(tokenizer, "taln")]["full_lexical_support"])
                for comparison_method in ("lcs", "difflib"):
                    comparison_value = bool(
                        results[(tokenizer, comparison_method)][
                            "full_lexical_support"
                        ]
                    )
                    key = (
                        row["split"],
                        row["dataset"],
                        row["variant_status"],
                        int(row["gap_count"]),
                        int(row["gap_width"]),
                        tokenizer,
                        comparison_method,
                    )
                    if taln_value and comparison_value:
                        agreements[key]["both_succeed"] += 1
                    elif taln_value:
                        agreements[key]["taln_only"] += 1
                    elif comparison_value:
                        agreements[key]["comparison_only"] += 1
                    else:
                        agreements[key]["neither_succeeds"] += 1

            if row["variant_status"] == "primary_noncontiguous":
                for method in spec["statistics"]["severity_methods"]:
                    severity_key = (
                        row["split"],
                        row["dataset"],
                        method,
                        row["base_record_id"],
                        int(row["gap_count"]),
                        int(row["gap_width"]),
                    )
                    severity_outcomes[severity_key] = (
                        row["document_id"],
                        bool(
                            results[("punctuation", method)][
                                "full_lexical_support"
                            ]
                        ),
                    )

    expected_ids = {
        variant_id
        for variant_id, variant in variants.items()
        if variant["split"] in seen_splits
    }
    if not args.allow_incomplete and seen_ids != expected_ids:
        raise ValueError(
            f"Summary inputs do not cover frozen split variants: "
            f"missing={len(expected_ids - seen_ids)}, extra={len(seen_ids - expected_ids)}"
        )

    group_summaries = {}
    table_rows = []
    for key, accumulator in sorted(groups.items()):
        split, dataset, status, gap_count, gap_width, tokenizer, method = key
        metric = summarize_group(
            accumulator, n_resamples=args.bootstrap_resamples, seed=args.seed
        )
        serialized_key = "::".join(str(value) for value in key)
        group_summaries[serialized_key] = metric
        support = metric["full_lexical_support"]
        localization = metric["localization"]
        no_path = metric["no_matched_token_path"]
        burden = metric["candidate_burden"]
        multi = burden["multiple_source_location"] or {}
        localization_scope = (
            "any_candidate_gold_interval_coverage"
            if method == "taln"
            else "selected_path_gold_interval_hit"
        )
        metric["localization_scope"] = localization_scope
        table_rows.append(
            {
                "split": split,
                "dataset": dataset,
                "variant_status": status,
                "gap_count": gap_count,
                "gap_width": gap_width,
                "tokenizer": tokenizer,
                "method": method,
                "tasks": metric["tasks"],
                "documents": metric["documents"],
                "support_count": metric["full_lexical_support_count"],
                "support_rate": support["estimate"],
                "support_ci_lower": support["ci_lower"],
                "support_ci_upper": support["ci_upper"],
                "localization_count": metric["localization_count"],
                "localization_scope": localization_scope,
                "localization_rate": localization["estimate"],
                "localization_ci_lower": localization["ci_lower"],
                "localization_ci_upper": localization["ci_upper"],
                "no_matched_token_path_count": metric[
                    "no_matched_token_path_count"
                ],
                "no_matched_token_path_rate": no_path["estimate"],
                "no_matched_token_path_ci_lower": no_path["ci_lower"],
                "no_matched_token_path_ci_upper": no_path["ci_upper"],
                "errors": metric["errors"],
                "truncated_tasks": metric["candidate_count_truncated_tasks"],
                "median_runtime_ms_diagnostic": metric[
                    "median_runtime_ms_diagnostic"
                ],
                "burden_tasks": burden["tasks"],
                "median_candidate_count": burden["median_candidate_count"],
                "p95_candidate_count": burden["p95_candidate_count"],
                "max_candidate_count": burden["max_candidate_count"],
                "median_unique_interval_count": burden[
                    "median_unique_source_interval_count"
                ],
                "p95_unique_interval_count": burden[
                    "p95_unique_source_interval_count"
                ],
                "max_unique_interval_count": burden[
                    "max_unique_source_interval_count"
                ],
                "multiple_location_rate": multi.get("estimate", ""),
            }
        )

    agreement_summaries = {
        "::".join(str(value) for value in key): value
        for key, value in sorted(agreements.items())
    }
    severity = _severity_contrasts(
        severity_outcomes,
        spec=spec,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    summary = {
        "schema_version": "multigap-benchmark-summary-v1",
        "records": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in records_paths
        ],
        "variant_manifest": freeze["variant_file"],
        "rows": len(seen_ids),
        "splits": sorted(seen_splits),
        "complete": seen_ids == expected_ids,
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "code_hashes": {
            str(Path(__file__).resolve().relative_to(REPO_ROOT)): sha256_file(
                Path(__file__).resolve()
            ),
            "analysis/scripts/revision_statistics.py": sha256_file(
                REPO_ROOT / "analysis" / "scripts" / "revision_statistics.py"
            ),
            str(STAGE2_SPEC.relative_to(REPO_ROOT)): sha256_file(STAGE2_SPEC),
        },
        "groups": group_summaries,
        "method_agreement": agreement_summaries,
        "severity_contrasts": severity,
        "interpretation": "Controlled deletion stress test; not a model of natural LLM errors or semantic correctness.",
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "multigap_benchmark_summary.json", summary)
    write_table(output_dir / "multigap_benchmark_table.tsv", table_rows)
    return summary


def main() -> None:
    summary = summarize(parse_args())
    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "complete": summary["complete"],
                "groups": len(summary["groups"]),
                "severity_contrasts": len(summary["severity_contrasts"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
