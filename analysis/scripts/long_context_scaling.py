"""Benchmark alignment behavior as source contexts get longer.

This script builds longer source contexts from the local BIO-BOAT paper corpus
under `tests/papers`. For each sampled source-target pair, it evaluates:

- positive contexts: the original source passage expanded within its paper
- negative controls: unrelated-paper contexts of comparable scale

The output is an aggregate JSON summary plus optional per-task JSONL records.
Generated outputs should go under `data/`, which is intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import signal
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from taln.taln_aln import (
    align_difflib,
    align_lcs,
    align_ng,
    norm_text,
    reconstruct_target_by_token,
)

DEFAULT_PAPERS_ROOT = Path("tests/papers")
DEFAULT_SUMMARY_OUTPUT = Path("data/long_context_scaling_summary.json")

AlignmentFn = Callable[[str, str, str], list[list[dict[str, Any]]]]


class TimeoutExpired(RuntimeError):
    """Raised when one alignment task exceeds the configured runtime cap."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-root", type=Path, default=DEFAULT_PAPERS_ROOT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--records-output", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-papers", type=int, default=4)
    parser.add_argument("--pairs-per-paper", type=int, default=1)
    parser.add_argument("--min-target-words", type=int, default=3)
    parser.add_argument(
        "--scales",
        nargs="+",
        default=["source", "512", "2048", "full"],
        help="Context scales: source, full, or a whitespace-token window size",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["naive", "taln", "lcs", "difflib"],
        choices=["naive", "taln", "lcs", "difflib"],
    )
    parser.add_argument(
        "--tokenization-types",
        nargs="+",
        default=["token"],
        choices=["whitespace", "token"],
        help="Tokenization types for taln, LCS, and difflib",
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


def load_papers(root: Path) -> list[dict[str, Any]]:
    papers = []
    for paper_path in sorted(root.glob("*/paper.md")):
        boat_path = paper_path.with_name("boat.json")
        if not boat_path.exists():
            continue
        try:
            records = load_json(boat_path)
        except json.JSONDecodeError:
            continue
        papers.append(
            {
                "paper_id": paper_path.parent.name,
                "paper_path": paper_path,
                "boat_path": boat_path,
                "paper_text": paper_path.read_text(encoding="utf-8", errors="ignore"),
                "records": records,
            }
        )
    return papers


def sample_records(
    papers: list[dict[str, Any]],
    max_papers: int,
    pairs_per_paper: int,
    min_target_words: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sampled = []
    for paper in papers[:max_papers]:
        eligible = [
            record
            for record in paper["records"]
            if len(str(record.get("target", "")).split()) >= min_target_words
        ]
        rng.shuffle(eligible)
        for record in eligible[:pairs_per_paper]:
            sampled.append({**record, "paper_id": paper["paper_id"]})
    return sampled


def word_spans(text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"\S+", text))


def window_by_words(text: str, center_start: int, center_end: int, target_words: int) -> str:
    words = word_spans(text)
    if len(words) <= target_words:
        return text

    left_idx = 0
    right_idx = len(words) - 1
    for idx, word in enumerate(words):
        if word.start() <= center_start < word.end():
            left_idx = idx
            break
        if word.start() > center_start:
            left_idx = max(idx - 1, 0)
            break
    for idx in range(left_idx, len(words)):
        if words[idx].end() >= center_end:
            right_idx = idx
            break

    source_word_count = right_idx - left_idx + 1
    extra = max(target_words - source_word_count, 0)
    left_extra = extra // 2
    right_extra = extra - left_extra
    out_left = max(left_idx - left_extra, 0)
    out_right = min(right_idx + right_extra, len(words) - 1)

    if out_right - out_left + 1 < target_words:
        out_left = max(out_right - target_words + 1, 0)
    if out_right - out_left + 1 < target_words:
        out_right = min(out_left + target_words - 1, len(words) - 1)

    return text[words[out_left].start() : words[out_right].end()]


def scale_to_context(
    paper_text: str,
    source: str,
    scale: str,
) -> tuple[str, bool]:
    if scale == "source":
        return source, True
    if scale == "full":
        return paper_text, source in paper_text

    target_words = int(scale)
    source_start = paper_text.find(source)
    if source_start == -1:
        return source, False
    source_end = source_start + len(source)
    return window_by_words(paper_text, source_start, source_end, target_words), True


def negative_context(paper_text: str, positive_word_count: int, scale: str) -> str:
    if scale == "full":
        return paper_text
    target_words = positive_word_count if scale == "source" else int(scale)
    words = word_spans(paper_text)
    if not words:
        return paper_text
    out_right = min(target_words, len(words)) - 1
    return paper_text[words[0].start() : words[out_right].end()]


def clean_compare(value: Any) -> str:
    return " ".join(norm_text(value).split()).strip()


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


def run_alignment(
    source: str,
    target: str,
    method: str,
    tokenization: str,
    timeout_sec: float,
) -> dict[str, Any]:
    aligners: dict[str, AlignmentFn] = {
        "taln": align_ng,
        "lcs": align_lcs,
        "difflib": align_difflib,
    }
    start = time.perf_counter()
    error = None
    timed_out = False
    alns: list[list[dict[str, Any]]] = []
    try:
        with timeout_after(timeout_sec):
            if method == "naive":
                alns = naive_align(source, target)
            else:
                alns = aligners[method](source, target, tokenization)
    except TimeoutExpired as exc:
        timed_out = True
        error = str(exc)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - start) * 1000

    reconstruction = best_reconstruction(alns)
    full_reconstruction = bool(
        reconstruction and clean_compare(reconstruction) == clean_compare(target)
    )
    return {
        "method": method,
        "tokenization": tokenization,
        "runtime_ms": elapsed_ms,
        "timed_out": timed_out,
        "error": error,
        "num_alignments": len(alns),
        "any_alignment": bool(alns),
        "full_reconstruction": full_reconstruction,
        "best_reconstruction": reconstruction,
    }


def build_tasks(
    papers: list[dict[str, Any]],
    records: list[dict[str, Any]],
    scales: list[str],
) -> list[dict[str, Any]]:
    by_id = {paper["paper_id"]: paper for paper in papers}
    paper_ids = [paper["paper_id"] for paper in papers]
    tasks = []
    for record in records:
        paper = by_id[record["paper_id"]]
        negative_id = paper_ids[(paper_ids.index(record["paper_id"]) + 1) % len(paper_ids)]
        negative_paper = by_id[negative_id]
        source = record["source"]
        target = record["target"]
        for scale in scales:
            context, source_found = scale_to_context(paper["paper_text"], source, scale)
            positive_words = max(len(context.split()), 1)
            tasks.append(
                {
                    "control": "positive",
                    "paper_id": record["paper_id"],
                    "negative_paper_id": None,
                    "scale": scale,
                    "source_found_in_paper": source_found,
                    "context_words": positive_words,
                    "source": context,
                    "target": target,
                    "target_exactly_present": clean_compare(target) in clean_compare(context),
                }
            )
            neg = negative_context(negative_paper["paper_text"], positive_words, scale)
            tasks.append(
                {
                    "control": "negative",
                    "paper_id": record["paper_id"],
                    "negative_paper_id": negative_id,
                    "scale": scale,
                    "source_found_in_paper": None,
                    "context_words": len(neg.split()),
                    "source": neg,
                    "target": target,
                    "target_exactly_present": clean_compare(target) in clean_compare(neg),
                }
            )
    return tasks


def evaluate_tasks(
    tasks: list[dict[str, Any]],
    methods: list[str],
    tokenization_types: list[str],
    timeout_sec: float,
) -> list[dict[str, Any]]:
    results = []
    for task in tasks:
        for method in methods:
            tokenizations = ["exact"] if method == "naive" else tokenization_types
            for tokenization in tokenizations:
                result = run_alignment(
                    source=task["source"],
                    target=task["target"],
                    method=method,
                    tokenization=tokenization,
                    timeout_sec=timeout_sec,
                )
                output = {key: value for key, value in task.items() if key != "source"}
                output.update(result)
                results.append(output)
    return results


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(round((len(values) - 1) * pct), len(values) - 1)
    return values[idx]


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = [row["runtime_ms"] for row in rows if not row["timed_out"]]
    alignment_counts = [row["num_alignments"] for row in rows if not row["timed_out"]]
    errors = Counter(row["error"] for row in rows if row["error"])
    positive = [row for row in rows if row["control"] == "positive"]
    negative = [row for row in rows if row["control"] == "negative"]
    return {
        "tasks": len(rows),
        "positive_tasks": len(positive),
        "negative_tasks": len(negative),
        "timeouts": sum(row["timed_out"] for row in rows),
        "errors": dict(errors),
        "positive_full_reconstruction": sum(row["full_reconstruction"] for row in positive),
        "positive_full_reconstruction_rate": (
            sum(row["full_reconstruction"] for row in positive) / len(positive)
            if positive
            else None
        ),
        "negative_spurious_full_reconstruction": sum(
            row["full_reconstruction"] for row in negative
        ),
        "negative_exact_target_present": sum(
            row["target_exactly_present"] for row in negative
        ),
        "negative_spurious_full_reconstruction_rate": (
            sum(row["full_reconstruction"] for row in negative) / len(negative)
            if negative
            else None
        ),
        "median_runtime_ms": statistics.median(runtimes) if runtimes else None,
        "p95_runtime_ms": percentile(runtimes, 0.95),
        "p99_runtime_ms": percentile(runtimes, 0.99),
        "max_runtime_ms": max(runtimes) if runtimes else None,
        "median_alignments": statistics.median(alignment_counts)
        if alignment_counts
        else None,
        "p95_alignments": percentile(alignment_counts, 0.95),
        "max_alignments": max(alignment_counts) if alignment_counts else None,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[(row["method"], row["tokenization"])].append(row)
        groups[(row["method"], row["tokenization"], row["scale"])].append(row)
    return {
        "overall": {
            "::".join(key): summarize_group(value)
            for key, value in sorted(groups.items())
            if len(key) == 2
        },
        "by_scale": {
            "::".join(key): summarize_group(value)
            for key, value in sorted(groups.items())
            if len(key) == 3
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    papers = load_papers(args.papers_root)
    records = sample_records(
        papers=papers,
        max_papers=args.max_papers,
        pairs_per_paper=args.pairs_per_paper,
        min_target_words=args.min_target_words,
        seed=args.seed,
    )
    tasks = build_tasks(papers, records, args.scales)
    results = evaluate_tasks(
        tasks=tasks,
        methods=args.methods,
        tokenization_types=args.tokenization_types,
        timeout_sec=args.timeout_sec,
    )
    summary = {
        "parameters": {
            "papers_root": str(args.papers_root),
            "seed": args.seed,
            "max_papers": args.max_papers,
            "pairs_per_paper": args.pairs_per_paper,
            "min_target_words": args.min_target_words,
            "scales": args.scales,
            "methods": args.methods,
            "tokenization_types": args.tokenization_types,
            "timeout_sec": args.timeout_sec,
        },
        "sampled_records": len(records),
        "tasks": len(tasks),
        **summarize(results),
    }

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
