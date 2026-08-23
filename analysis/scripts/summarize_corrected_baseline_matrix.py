"""Summarize Stage 1 records with clustered intervals and paired contrasts."""

from __future__ import annotations

import argparse
import hashlib
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
    mcnemar_descriptive,
    paired_document_clustered_difference_ci,
    paired_document_sign_flip_test,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "revision_2026" / "stage1"
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "revision_2026"
STAGE1_SPEC = REPO_ROOT / "analysis" / "config" / "revision_2026" / "stage1_spec.json"
METHOD_ORDER = ("exact", "taln", "lcs", "difflib", "semi_global_exact")
TOKENIZER_ORDER = (
    "character_exact",
    "whitespace",
    "boundary_stripped_whitespace",
    "punctuation",
    "cl100k_base",
    "scibert",
    "pubmedbert",
)


@dataclass
class MetricAccumulator:
    document_ids: list[str] = field(default_factory=list)
    support: list[bool] = field(default_factory=list)
    localization_document_ids: list[str] = field(default_factory=list)
    localization: list[bool] = field(default_factory=list)
    errors: int = 0
    truncated: int = 0
    runtime_ms: list[float] = field(default_factory=list)
    candidate_counts: list[int] = field(default_factory=list)
    unique_interval_counts: list[int] = field(default_factory=list)


@dataclass
class PairAccumulator:
    document_ids: list[str] = field(default_factory=list)
    reference: list[bool] = field(default_factory=list)
    comparison: list[bool] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--write-combined-records",
        action="store_true",
        help="Write all input rows to output-dir/corrected_baseline_records.jsonl.",
    )
    parser.add_argument(
        "--skip-discrepancy-records",
        action="store_true",
        help="Compute discrepancy counts and report samples without another JSONL index.",
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-per-discrepancy", type=int, default=10)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_map(row: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    results = {("character_exact", "exact"): row["exact"]}
    for tokenizer, methods in row["sequence"].items():
        for method, result in methods.items():
            results[(tokenizer, method)] = result
    return results


def expand_result(values: list[Any], fields: list[str]) -> dict[str, Any]:
    if len(values) != len(fields):
        raise ValueError(
            f"Compact result has {len(values)} values but config declares {len(fields)}"
        )
    result = dict(zip(fields, values, strict=True))
    result["localization_evaluable"] = result["localization"] is not None
    interval = result.pop("selected_interval")
    selected_localization = result.pop("selected_localization")
    result["selected_candidate"] = (
        {"source_interval": interval, "localization": selected_localization}
        if interval is not None
        else None
    )
    variant_counts = result["candidate_counts_by_variant"]
    result["candidate_counts_by_variant"] = {
        name: {"count": count, "truncated": truncated}
        for name, count, truncated in variant_counts
    }
    return result


def expand_matrix_row(
    row: dict[str, Any], configuration: dict[str, Any]
) -> dict[str, Any]:
    if isinstance(row["sequence"], dict):
        return row
    fields = configuration["record_encoding"]["result_fields"]
    tokenizers = configuration["tokenizers"]
    methods = configuration["methods"]
    if len(row["sequence"]) != len(tokenizers):
        raise ValueError(f"Tokenizer dimension mismatch for {row['task_id']}")
    expanded = dict(row)
    expanded["exact"] = expand_result(row["exact"], fields)
    expanded["sequence"] = {}
    for tokenizer, tokenizer_values in zip(
        tokenizers, row["sequence"], strict=True
    ):
        if len(tokenizer_values) != len(methods):
            raise ValueError(f"Method dimension mismatch for {row['task_id']}")
        expanded["sequence"][tokenizer] = {
            method: expand_result(values, fields)
            for method, values in zip(methods, tokenizer_values, strict=True)
        }
    return expanded


def load_source_lookup(input_root: Path) -> dict[str, str]:
    lookup = {}
    for filename in ("boat_records.jsonl", "bioboat_records.jsonl"):
        for record in read_jsonl(input_root / filename):
            lookup[record["record_id"]] = record["source"]
    for record in read_jsonl(input_root / "llmarkers_inventory.jsonl"):
        lookup[record["record_id"]] = record["source_rationale"]
    return lookup


def group_key(
    row: dict[str, Any], tokenizer: str, method: str
) -> tuple[str, str, str, str, str]:
    return (
        row["split"],
        row["dataset"],
        row["condition"],
        tokenizer,
        method,
    )


def contrast_definitions(
    spec: dict[str, Any], available: set[tuple[str, str]]
) -> list[dict[str, Any]]:
    contrasts = []
    for comparison, reference in spec["primary_contrasts"]["tokenizer_with_taln"]:
        comparison_key = (comparison, "taln")
        reference_key = (reference, "taln")
        if comparison_key in available and reference_key in available:
            contrasts.append(
                {
                    "name": f"taln:{comparison}-vs-{reference}",
                    "comparison": comparison_key,
                    "reference": reference_key,
                    "family": "tokenizer_with_taln",
                }
            )
    tokenizers = sorted({tokenizer for tokenizer, _ in available})
    for comparison, reference in spec["primary_contrasts"]["method_within_tokenizer"]:
        for tokenizer in tokenizers:
            comparison_key = (tokenizer, comparison)
            reference_key = (tokenizer, reference)
            if comparison_key in available and reference_key in available:
                contrasts.append(
                    {
                        "name": f"{tokenizer}:{comparison}-vs-{reference}",
                        "comparison": comparison_key,
                        "reference": reference_key,
                        "family": "method_within_tokenizer",
                    }
                )
    return contrasts


def paired_counts(reference: list[bool], comparison: list[bool]) -> dict[str, int]:
    return {
        "both_succeed": sum(r and c for r, c in zip(reference, comparison, strict=True)),
        "reference_only": sum(r and not c for r, c in zip(reference, comparison, strict=True)),
        "comparison_only": sum(not r and c for r, c in zip(reference, comparison, strict=True)),
        "neither_succeeds": sum(
            not r and not c for r, c in zip(reference, comparison, strict=True)
        ),
    }


def summarize_metric(
    accumulator: MetricAccumulator, *, n_resamples: int, seed: int
) -> dict[str, Any]:
    support_ci = document_clustered_binary_ci(
        accumulator.document_ids,
        accumulator.support,
        n_resamples=n_resamples,
        seed=seed,
    )
    localization_ci = (
        document_clustered_binary_ci(
            accumulator.localization_document_ids,
            accumulator.localization,
            n_resamples=n_resamples,
            seed=seed,
        )
        if accumulator.localization
        else None
    )
    sorted_candidates = sorted(accumulator.candidate_counts)
    p95_index = (
        min(round(0.95 * (len(sorted_candidates) - 1)), len(sorted_candidates) - 1)
        if sorted_candidates
        else None
    )
    return {
        "tasks": len(accumulator.support),
        "documents": len(set(accumulator.document_ids)),
        "full_lexical_support_count": sum(accumulator.support),
        "full_lexical_support": support_ci,
        "localization_evaluable_tasks": len(accumulator.localization),
        "localization_count": sum(accumulator.localization),
        "localization": localization_ci,
        "errors": accumulator.errors,
        "candidate_count_truncated_tasks": accumulator.truncated,
        "median_runtime_ms": statistics.median(accumulator.runtime_ms)
        if accumulator.runtime_ms
        else None,
        "median_candidate_count": statistics.median(accumulator.candidate_counts)
        if accumulator.candidate_counts
        else None,
        "p95_candidate_count": sorted_candidates[p95_index]
        if p95_index is not None
        else None,
        "max_candidate_count": max(accumulator.candidate_counts)
        if accumulator.candidate_counts
        else None,
        "median_unique_source_interval_count": statistics.median(
            accumulator.unique_interval_counts
        )
        if accumulator.unique_interval_counts
        else None,
        "max_unique_source_interval_count": max(accumulator.unique_interval_counts)
        if accumulator.unique_interval_counts
        else None,
    }


def discrepancy_categories(
    row: dict[str, Any], results: dict[tuple[str, str], dict[str, Any]]
) -> list[str]:
    categories = []

    def differs(first: tuple[str, str], second: tuple[str, str]) -> bool:
        return (
            first in results
            and second in results
            and results[first]["full_lexical_support"]
            != results[second]["full_lexical_support"]
        )

    comparisons = [
        (
            "taln:boundary_stripped_whitespace-vs-whitespace",
            ("boundary_stripped_whitespace", "taln"),
            ("whitespace", "taln"),
        ),
        (
            "taln:punctuation-vs-whitespace",
            ("punctuation", "taln"),
            ("whitespace", "taln"),
        ),
        (
            "taln:cl100k_base-vs-punctuation",
            ("cl100k_base", "taln"),
            ("punctuation", "taln"),
        ),
        (
            "taln:scibert-vs-punctuation",
            ("scibert", "taln"),
            ("punctuation", "taln"),
        ),
        (
            "taln:pubmedbert-vs-punctuation",
            ("pubmedbert", "taln"),
            ("punctuation", "taln"),
        ),
    ]
    for name, first, second in comparisons:
        if differs(first, second):
            categories.append(name)
    for tokenizer in TOKENIZER_ORDER[1:]:
        if differs((tokenizer, "difflib"), (tokenizer, "taln")):
            categories.append(f"{tokenizer}:difflib-vs-taln")
        if differs((tokenizer, "lcs"), (tokenizer, "taln")):
            categories.append(f"{tokenizer}:lcs-vs-taln")
        if differs((tokenizer, "semi_global_exact"), (tokenizer, "taln")):
            categories.append(f"{tokenizer}:semi_global_exact-vs-taln")
    if row["dataset"] == "llmarkers" and len(
        {result["full_lexical_support"] for result in results.values()}
    ) > 1:
        categories.append("llmarkers:any-method-or-tokenizer")
    return categories


def discrepancy_payload(
    row: dict[str, Any],
    results: dict[tuple[str, str], dict[str, Any]],
    categories: list[str],
) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "base_record_id": row["base_record_id"],
        "split": row["split"],
        "dataset": row["dataset"],
        "condition": row["condition"],
        "document_id": row["document_id"],
        "label_field": row.get("label_field"),
        "target": row["target"],
        "categories": categories,
        "outcomes": {
            f"{tokenizer}:{method}": result["full_lexical_support"]
            for (tokenizer, method), result in sorted(results.items())
        },
    }


def _sample_rank(task_id: str, category: str) -> str:
    return hashlib.sha256(f"{category}\0{task_id}".encode()).hexdigest()


def render_discrepancy_report(
    path: Path,
    *,
    category_counts: dict[str, int],
    samples: dict[str, list[tuple[str, dict[str, Any]]]],
    llmarkers_disagreements: int,
) -> None:
    lines = [
        "# Stage 1 Discrepancy Audit",
        "",
        "This report indexes lexical-support disagreements. It does not assess semantic correctness.",
        "",
        f"LLMarkers disagreements requiring inspection: **{llmarkers_disagreements}**.",
        "",
        "## Counts",
        "",
        "| Category | Tasks |",
        "|---|---:|",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"| `{category}` | {count} |")
    lines.extend(["", "## Deterministic Samples", ""])
    for category, ranked in sorted(samples.items()):
        lines.extend([f"### {category}", ""])
        for _, item in sorted(ranked):
            source_excerpt = " ".join(item["source"].split())[:500]
            successful = [name for name, value in item["outcomes"].items() if value]
            lines.extend(
                [
                    f"- `{item['task_id']}` ({item['dataset']}, {item['condition']})",
                    f"  Target: `{item['target']}`",
                    f"  Successful: {', '.join(successful) if successful else 'none'}",
                    f"  Source: {source_excerpt}",
                ]
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "split",
        "dataset",
        "condition",
        "stratum",
        "tokenizer",
        "method",
        "tasks",
        "documents",
        "support_count",
        "support_rate",
        "support_ci_lower",
        "support_ci_upper",
        "localization_tasks",
        "localization_count",
        "localization_rate",
        "localization_ci_lower",
        "localization_ci_upper",
        "errors",
        "truncated_tasks",
        "median_runtime_ms",
        "median_candidate_count",
        "p95_candidate_count",
        "max_candidate_count",
        "median_unique_interval_count",
        "max_unique_interval_count",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be positive")
    if args.sample_per_discrepancy < 1:
        raise ValueError("--sample-per-discrepancy must be positive")
    records = [path.resolve() for path in args.records]
    for path in records:
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_lookup = load_source_lookup(args.input_root.resolve())

    combined_path = output_dir / "corrected_baseline_records.jsonl"
    if args.write_combined_records and combined_path in records:
        raise ValueError("combined output must not overwrite an input records file")
    combined_handle = (
        combined_path.open("w", encoding="utf-8")
        if args.write_combined_records
        else None
    )
    discrepancy_path = output_dir / "corrected_baseline_discrepancies.jsonl"
    skip_discrepancy_records = getattr(args, "skip_discrepancy_records", False)
    if skip_discrepancy_records and discrepancy_path.exists():
        discrepancy_path.unlink()
    discrepancy_handle = (
        None
        if skip_discrepancy_records
        else discrepancy_path.open("w", encoding="utf-8")
    )

    spec = read_json(STAGE1_SPEC)
    metrics: dict[tuple[str, str, str, str, str], MetricAccumulator] = defaultdict(
        MetricAccumulator
    )
    stratified_metrics: dict[
        tuple[str, str, str, str, str, str], MetricAccumulator
    ] = defaultdict(MetricAccumulator)
    pairs: dict[tuple[str, str, str, str], PairAccumulator] = defaultdict(
        PairAccumulator
    )
    stratified_pairs: dict[
        tuple[str, str, str, str, str], PairAccumulator
    ] = defaultdict(PairAccumulator)
    contrast_metadata: dict[str, dict[str, Any]] = {}
    seen_task_ids = set()
    category_counts: dict[str, int] = defaultdict(int)
    samples: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    llmarkers_disagreements = 0
    rows_read = 0

    try:
        for records_path in records:
            config_path = records_path.parent / "corrected_baseline_run_config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Compact matrix records require adjacent run config: {config_path}"
                )
            configuration = read_json(config_path)
            for raw_row in read_jsonl(records_path):
                row = expand_matrix_row(raw_row, configuration)
                task_id = row["task_id"]
                if task_id in seen_task_ids:
                    raise ValueError(f"Duplicate task ID across inputs: {task_id}")
                seen_task_ids.add(task_id)
                rows_read += 1
                if combined_handle is not None:
                    combined_handle.write(
                        json.dumps(raw_row, separators=(",", ":")) + "\n"
                    )

                results = result_map(row)
                for (tokenizer, method), result in results.items():
                    accumulator = metrics[group_key(row, tokenizer, method)]
                    accumulator.document_ids.append(row["document_id"])
                    accumulator.support.append(bool(result["full_lexical_support"]))
                    if result["localization_evaluable"]:
                        accumulator.localization_document_ids.append(row["document_id"])
                        accumulator.localization.append(bool(result["localization"]))
                    accumulator.errors += result["error"] is not None
                    accumulator.truncated += bool(result["candidate_count_truncated"])
                    accumulator.runtime_ms.append(float(result["runtime_ms"]))
                    accumulator.candidate_counts.append(int(result["candidate_count"]))
                    accumulator.unique_interval_counts.append(
                        int(result["unique_source_interval_count"])
                    )
                    if row.get("label_field"):
                        stratified = stratified_metrics[
                            (
                                row["split"],
                                row["dataset"],
                                row["condition"],
                                row["label_field"],
                                tokenizer,
                                method,
                            )
                        ]
                        stratified.document_ids.append(row["document_id"])
                        stratified.support.append(
                            bool(result["full_lexical_support"])
                        )
                        if result["localization_evaluable"]:
                            stratified.localization_document_ids.append(
                                row["document_id"]
                            )
                            stratified.localization.append(
                                bool(result["localization"])
                            )
                        stratified.errors += result["error"] is not None
                        stratified.truncated += bool(
                            result["candidate_count_truncated"]
                        )
                        stratified.runtime_ms.append(float(result["runtime_ms"]))
                        stratified.candidate_counts.append(
                            int(result["candidate_count"])
                        )
                        stratified.unique_interval_counts.append(
                            int(result["unique_source_interval_count"])
                        )

                contrasts = contrast_definitions(spec, set(results))
                for contrast in contrasts:
                    comparison = results[contrast["comparison"]]["full_lexical_support"]
                    reference = results[contrast["reference"]]["full_lexical_support"]
                    key = (
                        row["split"],
                        row["dataset"],
                        row["condition"],
                        contrast["name"],
                    )
                    pair = pairs[key]
                    pair.document_ids.append(row["document_id"])
                    pair.reference.append(bool(reference))
                    pair.comparison.append(bool(comparison))
                    if row.get("label_field"):
                        stratified_pair = stratified_pairs[
                            (
                                row["split"],
                                row["dataset"],
                                row["condition"],
                                row["label_field"],
                                contrast["name"],
                            )
                        ]
                        stratified_pair.document_ids.append(row["document_id"])
                        stratified_pair.reference.append(bool(reference))
                        stratified_pair.comparison.append(bool(comparison))
                    contrast_metadata[contrast["name"]] = contrast

                categories = discrepancy_categories(row, results)
                if categories:
                    try:
                        source = source_lookup[row["base_record_id"]]
                    except KeyError as exc:
                        raise KeyError(
                            f"Missing frozen source for {row['base_record_id']}"
                        ) from exc
                    payload = discrepancy_payload(row, results, categories)
                    if discrepancy_handle is not None:
                        discrepancy_handle.write(
                            json.dumps(payload, separators=(",", ":")) + "\n"
                        )
                    sample_payload = {**payload, "source": source}
                    if "llmarkers:any-method-or-tokenizer" in categories:
                        llmarkers_disagreements += 1
                    for category in categories:
                        category_counts[category] += 1
                        rank = _sample_rank(task_id, category)
                        ranked = samples[category]
                        ranked.append((rank, sample_payload))
                        ranked.sort(key=lambda item: item[0])
                        del ranked[args.sample_per_discrepancy :]
    finally:
        if discrepancy_handle is not None:
            discrepancy_handle.close()
        if combined_handle is not None:
            combined_handle.close()

    grouped_summary = {}
    table_rows = []
    for key, accumulator in sorted(metrics.items()):
        split, dataset, condition, tokenizer, method = key
        summary = summarize_metric(
            accumulator, n_resamples=args.bootstrap_resamples, seed=args.seed
        )
        if split == "heldout_test" and dataset == "llmarkers":
            summary["inference_note"] = (
                "Descriptive: the held-out LLMarkers split contains only two documents."
            )
        serialized_key = "::".join(key)
        grouped_summary[serialized_key] = summary
        support = summary["full_lexical_support"]
        localization = summary["localization"] or {}
        table_rows.append(
            {
                "split": split,
                "dataset": dataset,
                "condition": condition,
                "stratum": "all",
                "tokenizer": tokenizer,
                "method": method,
                "tasks": summary["tasks"],
                "documents": summary["documents"],
                "support_count": summary["full_lexical_support_count"],
                "support_rate": support["estimate"],
                "support_ci_lower": support["ci_lower"],
                "support_ci_upper": support["ci_upper"],
                "localization_tasks": summary["localization_evaluable_tasks"],
                "localization_count": summary["localization_count"],
                "localization_rate": localization.get("estimate", ""),
                "localization_ci_lower": localization.get("ci_lower", ""),
                "localization_ci_upper": localization.get("ci_upper", ""),
                "errors": summary["errors"],
                "truncated_tasks": summary["candidate_count_truncated_tasks"],
                "median_runtime_ms": summary["median_runtime_ms"],
                "median_candidate_count": summary["median_candidate_count"],
                "p95_candidate_count": summary["p95_candidate_count"],
                "max_candidate_count": summary["max_candidate_count"],
                "median_unique_interval_count": summary[
                    "median_unique_source_interval_count"
                ],
                "max_unique_interval_count": summary[
                    "max_unique_source_interval_count"
                ],
            }
        )

    stratified_grouped_summary = {}
    for key, accumulator in sorted(stratified_metrics.items()):
        split, dataset, condition, stratum, tokenizer, method = key
        metric_summary = summarize_metric(
            accumulator, n_resamples=args.bootstrap_resamples, seed=args.seed
        )
        if split == "heldout_test" and dataset == "llmarkers":
            metric_summary["inference_note"] = (
                "Descriptive: the held-out LLMarkers split contains only two documents."
            )
        serialized_key = "::".join(key)
        stratified_grouped_summary[serialized_key] = metric_summary
        support = metric_summary["full_lexical_support"]
        localization = metric_summary["localization"] or {}
        table_rows.append(
            {
                "split": split,
                "dataset": dataset,
                "condition": condition,
                "stratum": stratum,
                "tokenizer": tokenizer,
                "method": method,
                "tasks": metric_summary["tasks"],
                "documents": metric_summary["documents"],
                "support_count": metric_summary["full_lexical_support_count"],
                "support_rate": support["estimate"],
                "support_ci_lower": support["ci_lower"],
                "support_ci_upper": support["ci_upper"],
                "localization_tasks": metric_summary[
                    "localization_evaluable_tasks"
                ],
                "localization_count": metric_summary["localization_count"],
                "localization_rate": localization.get("estimate", ""),
                "localization_ci_lower": localization.get("ci_lower", ""),
                "localization_ci_upper": localization.get("ci_upper", ""),
                "errors": metric_summary["errors"],
                "truncated_tasks": metric_summary[
                    "candidate_count_truncated_tasks"
                ],
                "median_runtime_ms": metric_summary["median_runtime_ms"],
                "median_candidate_count": metric_summary[
                    "median_candidate_count"
                ],
                "p95_candidate_count": metric_summary["p95_candidate_count"],
                "max_candidate_count": metric_summary["max_candidate_count"],
                "median_unique_interval_count": metric_summary[
                    "median_unique_source_interval_count"
                ],
                "max_unique_interval_count": metric_summary[
                    "max_unique_source_interval_count"
                ],
            }
        )

    contrast_summary = {}
    for key, pair in sorted(pairs.items()):
        split, dataset, condition, contrast_name = key
        metadata = contrast_metadata[contrast_name]
        serialized_key = "::".join(key)
        contrast_summary[serialized_key] = {
            "split": split,
            "dataset": dataset,
            "condition": condition,
            "name": contrast_name,
            "family": metadata["family"],
            "reference": ":".join(metadata["reference"]),
            "comparison": ":".join(metadata["comparison"]),
            "paired_counts": paired_counts(pair.reference, pair.comparison),
            "clustered_difference_ci": paired_document_clustered_difference_ci(
                pair.document_ids,
                pair.reference,
                pair.comparison,
                n_resamples=args.bootstrap_resamples,
                seed=args.seed,
            ),
            "document_sign_flip": paired_document_sign_flip_test(
                pair.document_ids,
                pair.reference,
                pair.comparison,
                seed=args.seed,
            ),
            "record_mcnemar_descriptive": mcnemar_descriptive(
                pair.reference, pair.comparison
            ),
            "inference_note": (
                "Descriptive: the held-out LLMarkers split contains only two documents."
                if split == "heldout_test" and dataset == "llmarkers"
                else None
            ),
        }

    stratified_contrast_summary = {}
    for key, pair in sorted(stratified_pairs.items()):
        split, dataset, condition, stratum, contrast_name = key
        metadata = contrast_metadata[contrast_name]
        serialized_key = "::".join(key)
        stratified_contrast_summary[serialized_key] = {
            "split": split,
            "dataset": dataset,
            "condition": condition,
            "stratum": stratum,
            "name": contrast_name,
            "family": metadata["family"],
            "reference": ":".join(metadata["reference"]),
            "comparison": ":".join(metadata["comparison"]),
            "paired_counts": paired_counts(pair.reference, pair.comparison),
            "clustered_difference_ci": paired_document_clustered_difference_ci(
                pair.document_ids,
                pair.reference,
                pair.comparison,
                n_resamples=args.bootstrap_resamples,
                seed=args.seed,
            ),
            "document_sign_flip": paired_document_sign_flip_test(
                pair.document_ids,
                pair.reference,
                pair.comparison,
                seed=args.seed,
            ),
            "record_mcnemar_descriptive": mcnemar_descriptive(
                pair.reference, pair.comparison
            ),
            "inference_note": (
                "Descriptive: the held-out LLMarkers split contains only two documents."
                if split == "heldout_test" and dataset == "llmarkers"
                else None
            ),
        }

    summary = {
        "schema_version": "corrected-baseline-summary-v1",
        "records": [
            {"path": str(path), "sha256": sha256_file(path)} for path in records
        ],
        "rows": rows_read,
        "unique_task_ids": len(seen_task_ids),
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "groups": grouped_summary,
        "stratified_groups": stratified_grouped_summary,
        "primary_contrasts": contrast_summary,
        "stratified_primary_contrasts": stratified_contrast_summary,
        "discrepancy_counts": dict(sorted(category_counts.items())),
        "discrepancy_records": (
            {"path": str(discrepancy_path), "sha256": sha256_file(discrepancy_path)}
            if discrepancy_handle is not None
            else None
        ),
        "llmarkers_disagreements_requiring_manual_inspection": llmarkers_disagreements,
        "combined_records": (
            {"path": str(combined_path), "sha256": sha256_file(combined_path)}
            if combined_handle is not None
            else None
        ),
    }
    write_json(output_dir / "corrected_baseline_summary.json", summary)
    write_table(output_dir / "tokenizer_by_method_table.tsv", table_rows)
    render_discrepancy_report(
        output_dir / "corrected_baseline_discrepancy_report.md",
        category_counts=category_counts,
        samples=samples,
        llmarkers_disagreements=llmarkers_disagreements,
    )
    return summary


def main() -> None:
    summary = summarize(parse_args())
    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "groups": len(summary["groups"]),
                "contrasts": len(summary["primary_contrasts"]),
                "llmarkers_disagreements": summary[
                    "llmarkers_disagreements_requiring_manual_inspection"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
