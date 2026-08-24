"""Evaluate simple ranking rules for taln alignment multiplicity.

`taln` intentionally enumerates every order-preserving alignment. That is useful
for auditability, but many valid alignments can be returned for one target. This script evaluates whether lightweight post-processing can
make the first few returned candidates useful without changing the core library.

Generated outputs should go under `data/`, which is intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from taln.taln_aln import align_ng, norm_text, reconstruct_target_by_token

DEFAULT_DATA_PATH = Path("data/bioboat.tdf.json")
DEFAULT_SUMMARY_OUTPUT = Path("data/alignment_ranking_summary.json")


class TimeoutExpired(RuntimeError):
    """Raised when one alignment task exceeds the configured runtime cap."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--records-output", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--min-target-words", type=int, default=3)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["contiguous", "noncontiguous_word_ablation"],
        default=["contiguous", "noncontiguous_word_ablation"],
    )
    parser.add_argument(
        "--tokenization-type",
        choices=["token", "whitespace"],
        default="token",
    )
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    return parser.parse_args()


@contextmanager
def timeout_after(seconds: float) -> Iterator[None]:
    def handler(signum: int, frame: Any) -> None:
        raise TimeoutExpired(f"exceeded {seconds:.2f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def clean_compare(value: Any) -> str:
    return " ".join(norm_text(value).split()).strip()


def target_word_count(target: str) -> int:
    return len(str(target).split())


def make_noncontiguous_target(target: str) -> str | None:
    words = str(target).split()
    if len(words) < 3:
        return None
    remove_idx = len(words) // 2
    prefix = " " if str(target).startswith(" ") else ""
    return prefix + " ".join(words[:remove_idx] + words[remove_idx + 1 :])


def expected_spans(record: dict[str, Any]) -> set[tuple[int, int]]:
    starts = record.get("idx_start", [])
    if isinstance(starts, int):
        starts = [starts]
    target = str(record["target"])
    return {(int(start), int(start) + len(target)) for start in starts}


def sample_records(
    records: list[dict[str, Any]],
    sample_size: int,
    min_target_words: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        record
        for record in records
        if target_word_count(record.get("target", "")) >= min_target_words
    ]
    if sample_size <= 0 or sample_size >= len(eligible):
        return eligible
    rng = random.Random(seed)
    return rng.sample(eligible, sample_size)


def task_rows(records: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    rows = []
    for idx, record in enumerate(records):
        source = str(record["source"])
        target = str(record["target"])
        spans = expected_spans(record)
        if "contiguous" in variants:
            rows.append(
                {
                    "record_index": idx,
                    "variant": "contiguous",
                    "source": source,
                    "target": target,
                    "expected_spans": spans,
                }
            )
        if "noncontiguous_word_ablation" in variants:
            noncontiguous_target = make_noncontiguous_target(target)
            if noncontiguous_target:
                rows.append(
                    {
                        "record_index": idx,
                        "variant": "noncontiguous_word_ablation",
                        "source": source,
                        "target": noncontiguous_target,
                        "expected_spans": spans,
                    }
                )
    return rows


def alignment_span(aln: list[dict[str, Any]]) -> tuple[int, int] | None:
    if not aln:
        return None
    return aln[0]["start_idx"], aln[-1]["end_idx"]


def alignment_token_width(aln: list[dict[str, Any]]) -> int:
    return sum(token["end_idx"] - token["start_idx"] for token in aln)


def alignment_features(aln: list[dict[str, Any]]) -> dict[str, Any]:
    span = alignment_span(aln)
    if span is None:
        return {
            "span": None,
            "span_width": None,
            "gap_width": None,
            "density": None,
            "token_count": 0,
        }
    start, end = span
    span_width = end - start
    token_width = alignment_token_width(aln)
    gap_width = span_width - token_width
    density = token_width / span_width if span_width else 0
    return {
        "span": span,
        "span_width": span_width,
        "gap_width": gap_width,
        "density": density,
        "token_count": len(aln),
    }


def full_reconstruction(aln: list[dict[str, Any]], target: str) -> bool:
    reconstruction = reconstruct_target_by_token("", aln)
    return bool(reconstruction and clean_compare(reconstruction) == clean_compare(target))


def unique_by_span(alns: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    best_by_span: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for aln in alns:
        span = alignment_span(aln)
        if span is None:
            continue
        current = best_by_span.get(span)
        if current is None:
            best_by_span[span] = aln
            continue
        if alignment_features(aln)["gap_width"] < alignment_features(current)["gap_width"]:
            best_by_span[span] = aln
    return list(best_by_span.values())


def rank_alignments(
    alns: list[list[dict[str, Any]]],
    strategy: str,
) -> list[list[dict[str, Any]]]:
    if strategy == "input_order":
        return alns
    if strategy == "unique_shortest_span":
        alns = unique_by_span(alns)

    def key(aln: list[dict[str, Any]]) -> tuple[Any, ...]:
        features = alignment_features(aln)
        span = features["span"] or (10**12, 10**12)
        span_width = features["span_width"] if features["span_width"] is not None else 10**12
        gap_width = features["gap_width"] if features["gap_width"] is not None else 10**12
        density = features["density"] if features["density"] is not None else 0
        token_count = features["token_count"]
        if strategy in {"shortest_span", "unique_shortest_span"}:
            return (span_width, gap_width, span[0], -token_count)
        if strategy == "fewest_gap_chars":
            return (gap_width, span_width, span[0], -token_count)
        if strategy == "highest_density":
            return (-density, span_width, gap_width, span[0], -token_count)
        raise ValueError(f"unknown ranking strategy: {strategy}")

    return sorted(alns, key=key)


def first_correct_rank(
    ranked: list[list[dict[str, Any]]],
    expected: set[tuple[int, int]],
    target: str,
) -> int | None:
    for idx, aln in enumerate(ranked, start=1):
        if not full_reconstruction(aln, target):
            continue
        if alignment_span(aln) in expected:
            return idx
    return None


def evaluate_row(
    row: dict[str, Any],
    tokenization_type: str,
    timeout_sec: float,
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    timed_out = False
    error = None
    alns: list[list[dict[str, Any]]] = []
    try:
        with timeout_after(timeout_sec):
            alns = align_ng(row["source"], row["target"], tokenization_type)
    except TimeoutExpired as exc:
        timed_out = True
        error = str(exc)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    runtime_ms = (time.perf_counter() - start) * 1000

    reconstructing = [aln for aln in alns if full_reconstruction(aln, row["target"])]
    unique_count = len(unique_by_span(alns))
    reconstructing_unique_count = len(unique_by_span(reconstructing))
    base = {
        "record_index": row["record_index"],
        "variant": row["variant"],
        "tokenization_type": tokenization_type,
        "target": row["target"],
        "num_alignments": len(alns),
        "num_unique_spans": unique_count,
        "num_reconstructing_alignments": len(reconstructing),
        "num_reconstructing_unique_spans": reconstructing_unique_count,
        "candidate_reduction_rate": 1 - (unique_count / len(alns)) if alns else None,
        "runtime_ms": runtime_ms,
        "timed_out": timed_out,
        "error": error,
    }

    strategies = [
        "input_order",
        "shortest_span",
        "fewest_gap_chars",
        "highest_density",
        "unique_shortest_span",
    ]
    results = []
    for strategy in strategies:
        ranked = rank_alignments(alns, strategy)
        rank = first_correct_rank(ranked, row["expected_spans"], row["target"])
        top_span = alignment_span(ranked[0]) if ranked else None
        results.append(
            {
                **base,
                "ranking_strategy": strategy,
                "first_correct_rank": rank,
                "top1_localized": rank == 1,
                "top5_localized": bool(rank and rank <= 5),
                "top10_localized": bool(rank and rank <= 10),
                "reciprocal_rank": (1 / rank) if rank else 0,
                "top_span": top_span,
            }
        )
    return results


def evaluate_rows(
    rows: list[dict[str, Any]],
    tokenization_type: str,
    timeout_sec: float,
) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        results.extend(evaluate_row(row, tokenization_type, timeout_sec))
    return results


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(round((pct / 100) * (len(values) - 1)), len(values) - 1)
    return values[idx]


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = [row["runtime_ms"] for row in rows if not row["timed_out"]]
    alignments = [row["num_alignments"] for row in rows if not row["timed_out"]]
    unique_spans = [row["num_unique_spans"] for row in rows if not row["timed_out"]]
    reductions = [
        row["candidate_reduction_rate"]
        for row in rows
        if row["candidate_reduction_rate"] is not None
    ]
    ranks = [row["first_correct_rank"] for row in rows if row["first_correct_rank"]]
    errors = Counter(row["error"] for row in rows if row["error"])

    return {
        "tasks": len(rows),
        "timed_out": sum(row["timed_out"] for row in rows),
        "top1_localized": sum(row["top1_localized"] for row in rows),
        "top5_localized": sum(row["top5_localized"] for row in rows),
        "top10_localized": sum(row["top10_localized"] for row in rows),
        "localized_any_rank": len(ranks),
        "top1_localization_rate": rate(sum(row["top1_localized"] for row in rows), len(rows)),
        "top5_localization_rate": rate(sum(row["top5_localized"] for row in rows), len(rows)),
        "top10_localization_rate": rate(sum(row["top10_localized"] for row in rows), len(rows)),
        "any_rank_localization_rate": rate(len(ranks), len(rows)),
        "mean_reciprocal_rank": statistics.mean(
            row["reciprocal_rank"] for row in rows
        )
        if rows
        else None,
        "median_first_correct_rank": statistics.median(ranks) if ranks else None,
        "p95_first_correct_rank": percentile(ranks, 95),
        "median_alignments": statistics.median(alignments) if alignments else None,
        "p95_alignments": percentile(alignments, 95),
        "max_alignments": max(alignments) if alignments else None,
        "median_unique_spans": statistics.median(unique_spans) if unique_spans else None,
        "p95_unique_spans": percentile(unique_spans, 95),
        "max_unique_spans": max(unique_spans) if unique_spans else None,
        "median_candidate_reduction_rate": statistics.median(reductions)
        if reductions
        else None,
        "median_runtime_ms": statistics.median(runtimes) if runtimes else None,
        "p95_runtime_ms": percentile(runtimes, 95),
        "errors": dict(errors),
    }


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_variant_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_strategy[row["ranking_strategy"]].append(row)
        by_variant_strategy[f"{row['variant']}::{row['ranking_strategy']}"].append(row)

    return {
        "parameters": {
            "data_path": str(args.data_path),
            "seed": args.seed,
            "sample_size": args.sample_size,
            "min_target_words": args.min_target_words,
            "variants": args.variants,
            "tokenization_type": args.tokenization_type,
            "timeout_sec": args.timeout_sec,
        },
        "by_strategy": {
            strategy: summarize_group(rows)
            for strategy, rows in sorted(by_strategy.items())
        },
        "by_variant_strategy": {
            key: summarize_group(rows)
            for key, rows in sorted(by_variant_strategy.items())
        },
    }


def main() -> None:
    args = parse_args()
    records = sample_records(
        load_json(args.data_path),
        sample_size=args.sample_size,
        min_target_words=args.min_target_words,
        seed=args.seed,
    )
    rows = task_rows(records, args.variants)
    results = evaluate_rows(rows, args.tokenization_type, args.timeout_sec)
    summary = summarize(results, args)
    summary["parameters"]["records_loaded"] = len(records)
    summary["parameters"]["tasks_evaluated"] = len(rows)
    write_json(args.summary_output, summary)

    if args.records_output:
        args.records_output.parent.mkdir(parents=True, exist_ok=True)
        with args.records_output.open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row) + "\n")

    print(json.dumps(summary["by_variant_strategy"], indent=2))


if __name__ == "__main__":
    main()
