"""Run the frozen selection-assessment experiment over enumerated alignments.

Implements analysis/config/revision_2026/selection_assessment_spec.json.
Candidates are re-enumerated deterministically with the shared evaluator from
the frozen Stage 1 and Stage 2 inputs. Five frozen parameter-free rules rank
unique original-source intervals. Rules never see gold intervals; localization
is scored afterwards against the frozen gold. Re-enumerated candidate counts
are checked against the frozen per-record outputs and any uncapped mismatch
aborts the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from common_evaluator import DEFAULT_CANDIDATE_CAP, evaluate_task
from run_corrected_baseline_matrix import (
    INPUT_MANIFEST,
    SPLIT_MANIFEST,
    iter_tasks,
    load_and_validate_freeze,
    package_version,
    read_jsonl,
    sha256_file,
    write_json,
)

from taln.taln_aln import norm_text_with_mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "revision_2026"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "selection"
SPEC_PATH = (
    REPO_ROOT
    / "analysis"
    / "config"
    / "revision_2026"
    / "selection_assessment_spec.json"
)

SCHEMA_VERSION = "selection-assessment-record-v1"
TOKENIZERS = ("punctuation", "cl100k_base")
RULES = (
    "input_order",
    "shortest_interval",
    "fewest_gap_characters",
    "matched_token_density",
    "earliest_interval",
)
CAP_SWEEP = (1, 10, 100, 1000, 10000)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260814

RECORD_FIELDS = (
    "full_lexical_support",
    "candidate_count",
    "candidate_count_truncated",
    "unique_interval_count",
    "gold_interval_count",
    "any_gold_interval",
    "lexically_indistinguishable",
    "rule_top_intervals",
    "rule_first_gold_ranks",
    "cap_sweep",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", required=True, choices=("development", "heldout_test")
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--candidate-cap", type=int, default=DEFAULT_CANDIDATE_CAP
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N tasks per stage for pipeline validation.",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--tokenizers", nargs="+", choices=TOKENIZERS, default=list(TOKENIZERS)
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing development output.",
    )
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def validate_heldout_arguments(args: argparse.Namespace) -> None:
    if args.split != "heldout_test":
        return
    if args.limit is not None:
        raise ValueError("Held-out runs cannot use --limit")
    if tuple(args.tokenizers) != TOKENIZERS:
        raise ValueError("Held-out runs must use the frozen tokenizer set")
    if args.candidate_cap != DEFAULT_CANDIDATE_CAP:
        raise ValueError("Held-out runs must use the frozen candidate cap")
    if args.overwrite:
        raise ValueError("Held-out outputs cannot be overwritten")


def load_development_selection(output_root: Path) -> dict[str, Any]:
    """A held-out run requires the recorded development rule choice."""
    path = output_root / "development" / "selection_development_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            "Held-out evaluation requires the development summary with the "
            f"recorded rule selection: {path}"
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    selection = summary.get("development_rule_selection")
    if not selection or selection.get("selected_rule") not in RULES:
        raise ValueError(
            "Development summary does not record a valid selected rule"
        )
    return selection


def load_frozen_taln_counts(
    input_root: Path, split: str, tokenizers: list[str]
) -> dict[tuple[str, str], tuple[str, int, bool, bool]]:
    """Map (task_id, tokenizer) -> frozen taln (stage, count, truncated, support).

    Stage 2 records were produced by the current evaluator revision, so their
    uncapped candidate counts must match exactly. Stage 1 records predate the
    evaluator's repeated-adjacent-token revision; for them only full-support
    agreement is a hard requirement and count drift is recorded as a warning.
    """
    counts: dict[tuple[str, str], tuple[str, int, bool, bool]] = {}
    sources = (
        (
            "stage1",
            input_root / "stage1" / split / "corrected_baseline_records.jsonl",
            input_root / "stage1" / split / "corrected_baseline_run_config.json",
        ),
        (
            "stage2",
            input_root / "stage2" / split / "multigap_benchmark_records.jsonl",
            input_root / "stage2" / split / "multigap_benchmark_run_config.json",
        ),
    )
    for stage, records_path, config_path in sources:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        frozen_tokenizers = list(config["tokenizers"])
        frozen_methods = list(config["methods"])
        taln_index = frozen_methods.index("taln")
        wanted = {
            tokenizer: frozen_tokenizers.index(tokenizer)
            for tokenizer in tokenizers
            if tokenizer in frozen_tokenizers
        }
        for row in read_jsonl(records_path):
            for tokenizer, tokenizer_index in wanted.items():
                result = row["sequence"][tokenizer_index][taln_index]
                counts[(row["task_id"], tokenizer)] = (
                    stage,
                    int(result[4]),
                    bool(result[5]),
                    bool(result[0]),
                )
    return counts


def iter_stage1_tasks(
    input_root: Path,
    split: str,
    frozen_splits: dict[tuple[str, str], str],
):
    """Frozen Stage 1 BOAT/BIO-BOAT one-gap tasks (llmarkers excluded)."""
    for task in iter_tasks(input_root, split, frozen_splits):
        if task["dataset"] not in ("boat", "bioboat"):
            continue
        if task["condition"] not in ("one_gap", "one_gap_still_contiguous_control"):
            continue
        yield {
            "stage": "stage1",
            "task_id": task["task_id"],
            "dataset": task["dataset"],
            "condition": task["condition"],
            "split": task["split"],
            "document_id": task["document_id"],
            "source": task["source"],
            "target": task["target"],
            "gold_intervals": task["gold_intervals"],
            "primary_noncontiguous": task["condition"] == "one_gap",
        }


def iter_stage2_tasks(input_root: Path, split: str):
    """Frozen Stage 2 variants joined to their base-record sources."""
    sources: dict[str, str] = {}
    for filename in ("boat_records.jsonl", "bioboat_records.jsonl"):
        for record in read_jsonl(input_root / filename):
            sources[record["record_id"]] = record["source"]
    for variant in read_jsonl(input_root / "stage2" / "multigap_variants.jsonl"):
        if variant["split"] != split:
            continue
        source = sources.get(variant["base_record_id"])
        if source is None:
            raise ValueError(
                f"Variant without frozen base record: {variant['variant_id']}"
            )
        if sha256_text(source) != variant["source_sha256"]:
            raise ValueError(
                f"Source hash mismatch for {variant['variant_id']}"
            )
        condition = (
            "contiguous_control"
            if variant["condition"] == "contiguous_control"
            else f"gaps{variant['gap_count']}_width{variant['gap_width']}"
        )
        yield {
            "stage": "stage2",
            "task_id": variant["variant_id"],
            "base_record_id": variant["base_record_id"],
            "dataset": variant["dataset"],
            "condition": condition,
            "split": variant["split"],
            "document_id": variant["document_id"],
            "source": source,
            "target": variant["target"],
            "gold_intervals": variant["gold_intervals_original"],
            "primary_noncontiguous": bool(variant["primary_noncontiguous"]),
        }


def coerce_intervals(raw: list[Any]) -> list[tuple[int, int]]:
    intervals = []
    for interval in raw or []:
        if isinstance(interval, dict):
            intervals.append((int(interval["start"]), int(interval["end"])))
        else:
            intervals.append((int(interval[0]), int(interval[1])))
    return intervals


def build_interval_table(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate complete candidates into per-unique-interval statistics."""
    table: dict[tuple[int, int], dict[str, Any]] = {}
    for enum_index, candidate in enumerate(candidates):
        interval = candidate["source_interval"]
        if interval is None:
            continue
        interval = (int(interval[0]), int(interval[1]))
        length = interval[1] - interval[0]
        matched = sum(int(end) - int(start) for start, end in candidate["source_offsets"])
        matched = min(matched, length) if length > 0 else 0
        entry = table.get(interval)
        if entry is None:
            table[interval] = {
                "interval": interval,
                "length": length,
                "first_enum_index": enum_index,
                "max_matched_chars": matched,
            }
        else:
            entry["max_matched_chars"] = max(entry["max_matched_chars"], matched)
    rows = list(table.values())
    for row in rows:
        length = row["length"]
        row["gap_chars"] = max(0, length - row["max_matched_chars"])
        row["density"] = (row["max_matched_chars"] / length) if length > 0 else 0.0
    return rows


def rank_intervals(rows: list[dict[str, Any]], rule: str) -> list[dict[str, Any]]:
    """Frozen deterministic ranking. Ties: earliest start, shortest, enum order."""
    tie = lambda row: (row["interval"][0], row["length"], row["first_enum_index"])  # noqa: E731
    keys = {
        "input_order": lambda row: (row["first_enum_index"], *tie(row)),
        "shortest_interval": lambda row: (row["length"], *tie(row)),
        "fewest_gap_characters": lambda row: (row["gap_chars"], *tie(row)),
        "matched_token_density": lambda row: (-row["density"], *tie(row)),
        "earliest_interval": lambda row: tie(row),
    }
    return sorted(rows, key=keys[rule])


def lexically_indistinguishable(
    source: str,
    rows: list[dict[str, Any]],
    gold: set[tuple[int, int]],
) -> bool:
    """True when a non-gold interval's normalized text equals a gold one's."""
    if not gold or len(rows) < 2:
        return False
    gold_texts = set()
    other_texts = set()
    for row in rows:
        text, _ = norm_text_with_mapping(source[row["interval"][0] : row["interval"][1]])
        if row["interval"] in gold:
            gold_texts.add(text)
        else:
            other_texts.add(text)
    return bool(gold_texts & other_texts)


def evaluate_selection_task(
    task: dict[str, Any], tokenizer: str, candidate_cap: int
) -> dict[str, Any]:
    """Enumerate candidates without gold exposure, then score selection."""
    result = evaluate_task(
        task_id=task["task_id"],
        source=task["source"],
        target=task["target"],
        method="taln",
        tokenizer=tokenizer,
        gold_intervals=None,
        candidate_cap=candidate_cap,
        include_candidates=True,
    )
    complete = [
        candidate
        for candidate in result["candidates"]
        if candidate["full_lexical_support"]
    ]
    gold = set(coerce_intervals(task["gold_intervals"]))
    rows = build_interval_table(complete)
    unique_count = len(rows)
    any_gold = any(row["interval"] in gold for row in rows)

    rule_top_intervals: dict[str, list[int] | None] = {}
    rule_first_gold_ranks: dict[str, int | None] = {}
    rankings: dict[str, list[dict[str, Any]]] = {}
    for rule in RULES:
        ranked = rank_intervals(rows, rule)
        rankings[rule] = ranked
        rule_top_intervals[rule] = list(ranked[0]["interval"]) if ranked else None
        first_gold = next(
            (
                position + 1
                for position, row in enumerate(ranked)
                if row["interval"] in gold
            ),
            None,
        )
        rule_first_gold_ranks[rule] = first_gold

    cap_sweep = {}
    for cap in CAP_SWEEP:
        prefix_rows = build_interval_table(complete[:cap])
        prefix_gold = any(row["interval"] in gold for row in prefix_rows)
        per_rule = []
        for rule in RULES:
            ranked = rank_intervals(prefix_rows, rule)
            per_rule.append(
                bool(ranked) and ranked[0]["interval"] in gold
            )
        cap_sweep[str(cap)] = [
            bool(complete[:cap]),
            prefix_gold,
            per_rule,
        ]

    return {
        "full_lexical_support": bool(result["full_lexical_support"]),
        "candidate_count": int(result["candidate_count"]),
        "candidate_count_truncated": bool(result["candidate_count_truncated"]),
        "unique_interval_count": unique_count,
        "gold_interval_count": len(gold),
        "any_gold_interval": any_gold,
        "lexically_indistinguishable": lexically_indistinguishable(
            task["source"], rows, gold
        ),
        "rule_top_intervals": rule_top_intervals,
        "rule_first_gold_ranks": rule_first_gold_ranks,
        "cap_sweep": cap_sweep,
    }


def compact_record(task: dict[str, Any], per_tokenizer: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task["task_id"],
        "stage": task["stage"],
        "dataset": task["dataset"],
        "condition": task["condition"],
        "split": task["split"],
        "document_id": task["document_id"],
        "primary_noncontiguous": task["primary_noncontiguous"],
        "results": per_tokenizer,
    }


def bootstrap_ci(
    values_by_document: dict[str, tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    """Document-clustered percentile CI for a rate given per-doc (hits, n)."""
    documents = sorted(values_by_document)
    if not documents:
        return None, None
    hits = np.array([values_by_document[doc][0] for doc in documents], dtype=float)
    totals = np.array([values_by_document[doc][1] for doc in documents], dtype=float)
    n_docs = len(documents)
    indices = rng.integers(0, n_docs, size=(BOOTSTRAP_RESAMPLES, n_docs))
    sampled_hits = hits[indices].sum(axis=1)
    sampled_totals = totals[indices].sum(axis=1)
    valid = sampled_totals > 0
    if not valid.any():
        return None, None
    rates = sampled_hits[valid] / sampled_totals[valid]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def summarize(
    records: list[dict[str, Any]],
    tokenizers: list[str],
    split: str,
) -> dict[str, Any]:
    """Per-stratum rule metrics with document-clustered bootstrap intervals."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata: dict[tuple[str, str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    pooled_primary: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        tokenizer: [] for tokenizer in tokenizers
    }
    for record in records:
        for tokenizer in tokenizers:
            result = record["results"][tokenizer]
            key = (record["stage"], record["dataset"], record["condition"], tokenizer)
            strata.setdefault(key, []).append((record, result))
            if record["primary_noncontiguous"]:
                pooled_primary[tokenizer].append((record, result))

    def stratum_summary(
        pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> dict[str, Any]:
        supported = [
            (record, result)
            for record, result in pairs
            if result["full_lexical_support"]
        ]
        counts = sorted(
            result["candidate_count"] for _, result in supported
        )
        uniques = sorted(
            result["unique_interval_count"] for _, result in supported
        )

        def percentile(sorted_values: list[int], q: float) -> int | None:
            if not sorted_values:
                return None
            return int(np.percentile(sorted_values, q, method="nearest"))

        summary: dict[str, Any] = {
            "tasks": len(pairs),
            "tasks_with_full_support": len(supported),
            "tasks_excluded_no_full_support": len(pairs) - len(supported),
            "tasks_capped": sum(
                result["candidate_count_truncated"] for _, result in supported
            ),
            "tasks_lexically_indistinguishable": sum(
                result["lexically_indistinguishable"] for _, result in supported
            ),
            "tasks_multiple_unique_intervals": sum(
                result["unique_interval_count"] > 1 for _, result in supported
            ),
            "candidate_count_percentiles": {
                "median": percentile(counts, 50),
                "p95": percentile(counts, 95),
                "p99": percentile(counts, 99),
                "max": counts[-1] if counts else None,
            },
            "unique_interval_percentiles": {
                "median": percentile(uniques, 50),
                "p95": percentile(uniques, 95),
                "p99": percentile(uniques, 99),
                "max": uniques[-1] if uniques else None,
            },
            "rules": {},
        }
        localizable = [
            (record, result)
            for record, result in supported
            if result["gold_interval_count"] > 0
        ]
        summary["tasks_localization_evaluable"] = len(localizable)
        any_gold_by_doc: dict[str, tuple[int, int]] = {}
        for record, result in localizable:
            hits, total = any_gold_by_doc.get(record["document_id"], (0, 0))
            any_gold_by_doc[record["document_id"]] = (
                hits + int(result["any_gold_interval"]),
                total + 1,
            )
        denominator = len(localizable)
        any_gold_hits = sum(
            int(result["any_gold_interval"]) for _, result in localizable
        )
        summary["any_rank_localization"] = {
            "hits": any_gold_hits,
            "n": denominator,
            "rate": (any_gold_hits / denominator) if denominator else None,
        }
        for rule in RULES:
            top1_by_doc: dict[str, tuple[int, int]] = {}
            top1_hits = 0
            top5_hits = 0
            reciprocal_sum = 0.0
            for record, result in localizable:
                rank = result["rule_first_gold_ranks"][rule]
                top1 = rank == 1
                top1_hits += int(top1)
                top5_hits += int(rank is not None and rank <= 5)
                reciprocal_sum += (1.0 / rank) if rank else 0.0
                hits, total = top1_by_doc.get(record["document_id"], (0, 0))
                top1_by_doc[record["document_id"]] = (hits + int(top1), total + 1)
            low, high = bootstrap_ci(top1_by_doc, rng)
            summary["rules"][rule] = {
                "top1": {
                    "hits": top1_hits,
                    "n": denominator,
                    "rate": (top1_hits / denominator) if denominator else None,
                    "ci95": [low, high],
                },
                "top5": {
                    "hits": top5_hits,
                    "n": denominator,
                    "rate": (top5_hits / denominator) if denominator else None,
                },
                "mean_reciprocal_rank": (
                    reciprocal_sum / denominator if denominator else None
                ),
            }
        cap_summary: dict[str, Any] = {}
        for cap in CAP_SWEEP:
            retained_support = sum(
                result["cap_sweep"][str(cap)][0] for _, result in supported
            )
            retained_any_gold = sum(
                result["cap_sweep"][str(cap)][1] for _, result in localizable
            )
            per_rule = {
                rule: sum(
                    result["cap_sweep"][str(cap)][2][rule_index]
                    for _, result in localizable
                )
                for rule_index, rule in enumerate(RULES)
            }
            cap_summary[str(cap)] = {
                "retained_full_support": retained_support,
                "retained_full_support_n": len(supported),
                "retained_any_gold": retained_any_gold,
                "retained_any_gold_n": denominator,
                "retained_top1_by_rule": per_rule,
            }
        summary["cap_sweep"] = cap_summary
        return summary

    stratum_results = {
        f"{stage}:{dataset}:{condition}:{tokenizer}": stratum_summary(pairs)
        for (stage, dataset, condition, tokenizer), pairs in sorted(strata.items())
    }
    pooled_results = {
        tokenizer: stratum_summary(pairs)
        for tokenizer, pairs in pooled_primary.items()
    }

    payload: dict[str, Any] = {
        "schema_version": "selection-assessment-summary-v1",
        "split": split,
        "rules": list(RULES),
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "document_id",
        },
        "strata": stratum_results,
        "pooled_primary_noncontiguous": pooled_results,
    }

    if split == "development":
        primary = pooled_results["punctuation"]
        ranked_rules = sorted(
            RULES,
            key=lambda rule: (
                -(primary["rules"][rule]["top1"]["rate"] or 0.0),
                RULES.index(rule),
            ),
        )
        selected = ranked_rules[0]
        payload["development_rule_selection"] = {
            "criterion": (
                "highest development top-1 localization on pooled primary "
                "non-contiguous strata under the punctuation tokenizer; ties "
                "broken by frozen rule-list order"
            ),
            "selected_rule": selected,
            "development_top1_rates": {
                rule: primary["rules"][rule]["top1"]["rate"] for rule in RULES
            },
        }
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.candidate_cap < 1:
        raise ValueError("--candidate-cap must be positive")
    validate_heldout_arguments(args)

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    development_selection = (
        load_development_selection(output_root)
        if args.split == "heldout_test"
        else None
    )
    frozen_splits, input_hashes = load_and_validate_freeze(input_root)
    frozen_counts = load_frozen_taln_counts(
        input_root, args.split, list(args.tokenizers)
    )

    output_dir = output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "selection_records.jsonl"
    summary_name = (
        "selection_development_summary.json"
        if args.split == "development"
        else "selection_heldout_summary.json"
    )
    summary_path = output_dir / summary_name
    metadata_path = output_dir / "selection_run_metadata.json"
    if records_path.exists() and not args.overwrite:
        raise FileExistsError(f"{records_path} already exists")

    started_at = datetime.now(timezone.utc)
    task_start = time.perf_counter()
    evaluated = 0
    verified = 0
    stage1_count_warnings: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    def iter_all_tasks():
        for stage_iter in (
            iter_stage1_tasks(input_root, args.split, frozen_splits),
            iter_stage2_tasks(input_root, args.split),
        ):
            stage_count = 0
            for task in stage_iter:
                if args.limit is not None and stage_count >= args.limit:
                    break
                stage_count += 1
                yield task

    with records_path.open("w", encoding="utf-8") as output:
        for task in iter_all_tasks():
            per_tokenizer = {}
            for tokenizer in args.tokenizers:
                result = evaluate_selection_task(
                    task, tokenizer, args.candidate_cap
                )
                frozen = frozen_counts.get((task["task_id"], tokenizer))
                if frozen is not None:
                    stage, frozen_count, frozen_truncated, frozen_support = frozen
                    if frozen_support != result["full_lexical_support"]:
                        raise ValueError(
                            "Full-support disagreement with frozen record for "
                            f"{task['task_id']} ({tokenizer}): frozen="
                            f"{frozen_support} observed="
                            f"{result['full_lexical_support']}"
                        )
                    counts_comparable = (
                        not frozen_truncated
                        and not result["candidate_count_truncated"]
                        and frozen_count != result["candidate_count"]
                    )
                    if counts_comparable and stage == "stage2":
                        raise ValueError(
                            "Re-enumerated candidate count mismatch for "
                            f"{task['task_id']} ({tokenizer}): frozen="
                            f"{frozen_count} observed={result['candidate_count']}"
                        )
                    if counts_comparable and stage == "stage1":
                        stage1_count_warnings.append(
                            {
                                "task_id": task["task_id"],
                                "tokenizer": tokenizer,
                                "frozen_count": frozen_count,
                                "observed_count": result["candidate_count"],
                            }
                        )
                    verified += 1
                per_tokenizer[tokenizer] = result
            record = compact_record(task, per_tokenizer)
            records.append(record)
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            evaluated += 1
            if evaluated % args.progress_every == 0:
                output.flush()
                elapsed = time.perf_counter() - task_start
                rate = evaluated / elapsed if elapsed else 0.0
                print(
                    f"{args.split}: {evaluated} tasks, {rate:.2f} tasks/s, "
                    f"{verified} count checks",
                    flush=True,
                )

    summary = summarize(records, list(args.tokenizers), args.split)
    if development_selection is not None:
        summary["development_rule_selection"] = development_selection
    write_json(summary_path, summary)

    ended_at = datetime.now(timezone.utc)
    code_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "analysis" / "scripts" / "common_evaluator.py",
        REPO_ROOT / "analysis" / "scripts" / "run_corrected_baseline_matrix.py",
        SPEC_PATH,
        INPUT_MANIFEST,
        SPLIT_MANIFEST,
    ]
    metadata = {
        "schema_version": "selection-assessment-run-metadata-v1",
        "split": args.split,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": (ended_at - started_at).total_seconds(),
        "records_path": repo_relative(records_path),
        "records_sha256": sha256_file(records_path),
        "summary_path": repo_relative(summary_path),
        "tasks_evaluated": evaluated,
        "frozen_count_checks": verified,
        "stage1_count_warnings": stage1_count_warnings,
        "stage1_count_warning_total": len(stage1_count_warnings),
        "candidate_cap": args.candidate_cap,
        "tokenizers": list(args.tokenizers),
        "rules": list(RULES),
        "cap_sweep": list(CAP_SWEEP),
        "limit": args.limit,
        "development_rule_selection": development_selection
        or summary.get("development_rule_selection"),
        "input_hashes": input_hashes,
        "code_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in code_paths
        },
        "record_encoding": {"result_fields": list(RECORD_FIELDS)},
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "taln": package_version("taln"),
            "numpy": package_version("numpy"),
            "tiktoken": package_version("tiktoken"),
        },
        "command": sys.argv,
        "complete": args.limit is None,
    }
    write_json(metadata_path, metadata)
    return metadata


def main() -> None:
    metadata = run(parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
