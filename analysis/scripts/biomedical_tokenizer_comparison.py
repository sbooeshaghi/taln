"""Compare tokenizer choices on BIO-BOAT alignment tasks.

The biomedical setting should be evaluated with biomedical tokenizers rather
than only the general-domain cl100k_base tokenizer. This script keeps that comparison
analysis-only: it adapts several tokenizers to the same ordered-alignment
interface without adding Hugging Face dependencies to the core taln package.

Generated outputs should go under `data/`, which is intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import tiktoken

from taln.taln_aln import norm_text

DEFAULT_DATA_PATH = Path("data/bioboat.wdf.json")
DEFAULT_SUMMARY_OUTPUT = Path("data/biomedical_tokenizer_comparison_summary.json")

HF_MODELS = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "bert-base-uncased": "bert-base-uncased",
}

Token = dict[str, Any]
TokenizeFn = Callable[[str], list[Token]]


@dataclass(frozen=True)
class TokenizerAdapter:
    name: str
    tokenize: TokenizeFn
    prefix_space_retry: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--records-output", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of BIO-BOAT records to sample. Use 0 for all records.",
    )
    parser.add_argument("--min-target-words", type=int, default=3)
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=["whitespace", "cl100k_base", "scibert", "pubmedbert"],
        choices=["whitespace", "cl100k_base", *HF_MODELS.keys()],
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Hugging Face tokenizers only from the local cache.",
    )
    parser.add_argument(
        "--alignment-count-cap",
        type=int,
        default=100_000,
        help="Stop counting alignments after this many paths per task.",
    )
    return parser.parse_args()


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


def whitespace_tokenize(text: str) -> list[Token]:
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append(
            {
                "token": match.group(),
                "enc_token": match.group(),
                "start_idx": match.start(),
                "end_idx": match.end(),
            }
        )
    return tokens


def make_tiktoken_tokenizer(encoding_name: str = "cl100k_base") -> TokenizeFn:
    enc = tiktoken.get_encoding(encoding_name)

    def tokenize(text: str) -> list[Token]:
        enc_tokens = enc.encode(text)
        decoded, offsets = enc.decode_with_offsets(enc_tokens)
        if decoded != text:
            raise ValueError("decoded tiktoken text does not match input")
        tokens = []
        for idx, (enc_token, start) in enumerate(zip(enc_tokens, offsets)):
            end = offsets[idx + 1] if idx + 1 < len(offsets) else len(text)
            tokens.append(
                {
                    "token": text[start:end],
                    "enc_token": enc_token,
                    "start_idx": start,
                    "end_idx": end,
                }
            )
        return tokens

    return tokenize


def make_hf_tokenizer(model_id: str, local_files_only: bool = False) -> TokenizeFn:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Hugging Face tokenizers require transformers. Install it in the "
            "analysis environment, e.g. `uv pip install transformers tokenizers "
            "--python .venv/bin/python`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if not tokenizer.is_fast:
        raise ValueError(f"{model_id} did not load a fast tokenizer with offsets")

    def tokenize(text: str) -> list[Token]:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        tokens = []
        for enc_token, (start, end) in zip(
            encoded["input_ids"], encoded["offset_mapping"]
        ):
            if end <= start:
                continue
            tokens.append(
                {
                    "token": text[start:end],
                    "enc_token": int(enc_token),
                    "start_idx": int(start),
                    "end_idx": int(end),
                }
            )
        return tokens

    return tokenize


def build_tokenizers(
    names: list[str], local_files_only: bool = False
) -> list[TokenizerAdapter]:
    adapters = []
    for name in names:
        if name == "whitespace":
            adapters.append(TokenizerAdapter(name=name, tokenize=whitespace_tokenize))
        elif name == "cl100k_base":
            adapters.append(
                TokenizerAdapter(
                    name=name,
                    tokenize=make_tiktoken_tokenizer(name),
                    prefix_space_retry=True,
                )
            )
        else:
            adapters.append(
                TokenizerAdapter(
                    name=name,
                    tokenize=make_hf_tokenizer(HF_MODELS[name], local_files_only),
                )
            )
    return adapters


def target_word_count(target: str) -> int:
    return len(re.findall(r"\S+", target))


def make_noncontiguous_target(target: str) -> str | None:
    words = [match.group() for match in re.finditer(r"\S+", target)]
    if len(words) < 3:
        return None
    remove_idx = len(words) // 2
    kept = words[:remove_idx] + words[remove_idx + 1 :]
    prefix = " " if target.startswith(" ") else ""
    return prefix + " ".join(kept)


def stress_patterns(target: str) -> list[str]:
    patterns = []
    normalized = norm_text(target)
    if re.search(r"[A-Za-z]+[0-9]+|[0-9]+[A-Za-z]+", normalized):
        patterns.append("mixed_alphanumeric")
    if "-" in normalized:
        patterns.append("hyphen")
    if "/" in normalized:
        patterns.append("slash")
    if "(" in normalized or ")" in normalized:
        patterns.append("parentheses")
    if "+" in normalized:
        patterns.append("plus")
    if re.search(r"\b[A-Z0-9][A-Z0-9-]{2,}\b", normalized):
        patterns.append("uppercase_identifier")
    return patterns or ["none"]


def sample_records(
    records: list[dict[str, Any]],
    sample_size: int,
    min_target_words: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        record
        for record in records
        if target_word_count(str(record.get("target", ""))) >= min_target_words
    ]
    if sample_size <= 0 or sample_size >= len(eligible):
        return eligible
    rng = random.Random(seed)
    return rng.sample(eligible, sample_size)


def positions_by_token(source_tokens: list[Token]) -> dict[Any, list[int]]:
    positions: dict[Any, list[int]] = defaultdict(list)
    for idx, token in enumerate(source_tokens):
        positions[token["enc_token"]].append(idx)
    return positions


def ordered_position_lists(
    source_tokens: list[Token],
    target_tokens: list[Token],
) -> list[list[int]]:
    positions = positions_by_token(source_tokens)
    return [positions.get(token["enc_token"], []) for token in target_tokens]


def count_ordered_alignments(
    position_lists: list[list[int]],
    cap: int,
) -> tuple[int, bool]:
    if not position_lists or any(not positions for positions in position_lists):
        return 0, False

    counts = {pos: 1 for pos in position_lists[0]}
    truncated = False
    for positions in position_lists[1:]:
        next_counts = {}
        sorted_prev = sorted(counts.items())
        for pos in positions:
            total = 0
            for prev_pos, prev_count in sorted_prev:
                if prev_pos >= pos:
                    break
                total += prev_count
                if total >= cap:
                    total = cap
                    truncated = True
                    break
            if total:
                next_counts[pos] = total
        counts = next_counts
        if not counts:
            return 0, truncated

    total = sum(counts.values())
    if total >= cap:
        return cap, True
    return total, truncated


def alignment_span_score(source_tokens: list[Token], path: list[int]) -> tuple[int, int]:
    start = source_tokens[path[0]]["start_idx"]
    end = source_tokens[path[-1]]["end_idx"]
    token_width = sum(
        source_tokens[pos]["end_idx"] - source_tokens[pos]["start_idx"] for pos in path
    )
    return end - start, (end - start) - token_width


def best_short_span_alignment(
    source_tokens: list[Token],
    target_tokens: list[Token],
) -> list[int]:
    position_lists = ordered_position_lists(source_tokens, target_tokens)
    if not position_lists or any(not positions for positions in position_lists):
        return []

    states = {pos: [pos] for pos in position_lists[0]}
    for positions in position_lists[1:]:
        next_states = {}
        for pos in positions:
            candidates = [path for prev, path in states.items() if prev < pos]
            if not candidates:
                continue
            best = min(
                candidates,
                key=lambda path: alignment_span_score(source_tokens, path + [pos]),
            )
            next_states[pos] = best + [pos]
        states = next_states
        if not states:
            return []

    return min(
        states.values(),
        key=lambda path: alignment_span_score(source_tokens, path),
    )


def reconstruct_source_tokens(source: str, source_tokens: list[Token], path: list[int]) -> str:
    pieces = []
    prev_end = None
    for pos in path:
        token = source_tokens[pos]
        start = token["start_idx"]
        end = token["end_idx"]
        piece = source[start:end]
        if (
            prev_end is not None
            and start > prev_end
            and pieces
            and not pieces[-1].endswith(" ")
            and not piece.startswith(" ")
        ):
            pieces.append(" ")
        pieces.append(piece)
        prev_end = end
    return "".join(pieces)


def evaluate_one(
    source: str,
    target: str,
    adapter: TokenizerAdapter,
    alignment_count_cap: int,
) -> dict[str, Any]:
    source_norm = norm_text(source)
    target_norm = norm_text(target)
    candidate_targets = [target_norm]
    if adapter.prefix_space_retry and target_norm and not target_norm.startswith(" "):
        candidate_targets.append(" " + target_norm)

    source_tokens = adapter.tokenize(source_norm)
    best_result: dict[str, Any] | None = None
    for candidate_idx, candidate_target in enumerate(candidate_targets):
        target_tokens = adapter.tokenize(candidate_target)
        position_lists = ordered_position_lists(source_tokens, target_tokens)
        alignment_count, count_truncated = count_ordered_alignments(
            position_lists, alignment_count_cap
        )
        path = best_short_span_alignment(source_tokens, target_tokens)
        reconstruction = (
            reconstruct_source_tokens(source_norm, source_tokens, path) if path else ""
        )
        span_width, gap_width = (
            alignment_span_score(source_tokens, path) if path else (None, None)
        )
        result = {
            "tokenizer": adapter.name,
            "target_variant_used": "prefixed_space" if candidate_idx else "original",
            "source_tokens": len(source_tokens),
            "target_tokens": len(target_tokens),
            "token_sequence_recovered": bool(path),
            "full_reconstruction": bool(
                reconstruction and clean_compare(reconstruction) == clean_compare(target)
            ),
            "best_reconstruction": reconstruction,
            "alignment_count": alignment_count,
            "alignment_count_truncated": count_truncated,
            "best_span_width": span_width,
            "best_gap_width": gap_width,
        }
        if best_result is None:
            best_result = result
            continue
        best_key = (
            result["full_reconstruction"],
            result["token_sequence_recovered"],
            -(result["best_span_width"] or 10**9),
            -result["alignment_count"],
        )
        current_key = (
            best_result["full_reconstruction"],
            best_result["token_sequence_recovered"],
            -(best_result["best_span_width"] or 10**9),
            -best_result["alignment_count"],
        )
        if best_key > current_key:
            best_result = result

    if best_result is None:
        raise ValueError("no tokenizer candidates evaluated")
    return best_result


def task_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, record in enumerate(records):
        source = str(record["source"])
        target = str(record["target"])
        rows.append(
            {
                "record_index": idx,
                "variant": "contiguous",
                "source": source,
                "target": target,
                "stress_patterns": stress_patterns(target),
                "naive_exact_found": clean_compare(target) in clean_compare(source),
            }
        )
        noncontiguous_target = make_noncontiguous_target(target)
        if noncontiguous_target:
            rows.append(
                {
                    "record_index": idx,
                    "variant": "noncontiguous_word_ablation",
                    "source": source,
                    "target": noncontiguous_target,
                    "stress_patterns": stress_patterns(noncontiguous_target),
                    "naive_exact_found": clean_compare(noncontiguous_target)
                    in clean_compare(source),
                }
            )
    return rows


def evaluate_rows(
    rows: list[dict[str, Any]],
    adapters: list[TokenizerAdapter],
    alignment_count_cap: int,
) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        for adapter in adapters:
            start = time.perf_counter()
            error = None
            try:
                metrics = evaluate_one(
                    row["source"],
                    row["target"],
                    adapter,
                    alignment_count_cap,
                )
            except Exception as exc:
                metrics = {
                    "tokenizer": adapter.name,
                    "target_variant_used": None,
                    "source_tokens": None,
                    "target_tokens": None,
                    "token_sequence_recovered": False,
                    "full_reconstruction": False,
                    "best_reconstruction": "",
                    "alignment_count": 0,
                    "alignment_count_truncated": False,
                    "best_span_width": None,
                    "best_gap_width": None,
                }
                error = f"{type(exc).__name__}: {exc}"
            elapsed_ms = (time.perf_counter() - start) * 1000
            results.append({**row, **metrics, "runtime_ms": elapsed_ms, "error": error})
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
    runtimes = [row["runtime_ms"] for row in rows if row["error"] is None]
    alignment_counts = [
        row["alignment_count"]
        for row in rows
        if row["error"] is None and row["alignment_count"] is not None
    ]
    source_tokens = [
        row["source_tokens"]
        for row in rows
        if row["error"] is None and row["source_tokens"] is not None
    ]
    target_tokens = [
        row["target_tokens"]
        for row in rows
        if row["error"] is None and row["target_tokens"] is not None
    ]
    span_widths = [
        row["best_span_width"]
        for row in rows
        if row["error"] is None and row["best_span_width"] is not None
    ]
    gap_widths = [
        row["best_gap_width"]
        for row in rows
        if row["error"] is None and row["best_gap_width"] is not None
    ]
    token_sequence_recovered = sum(row["token_sequence_recovered"] for row in rows)
    full_reconstruction = sum(row["full_reconstruction"] for row in rows)
    naive_exact_found = sum(row["naive_exact_found"] for row in rows)
    errors = Counter(row["error"] for row in rows if row["error"])

    return {
        "tasks": len(rows),
        "naive_exact_found": naive_exact_found,
        "token_sequence_recovered": token_sequence_recovered,
        "full_reconstruction": full_reconstruction,
        "zero_alignment": len(rows) - token_sequence_recovered,
        "alignment_count_truncated": sum(row["alignment_count_truncated"] for row in rows),
        "token_sequence_recovery_rate": rate(token_sequence_recovered, len(rows)),
        "full_reconstruction_rate": rate(full_reconstruction, len(rows)),
        "naive_exact_found_rate": rate(naive_exact_found, len(rows)),
        "zero_alignment_rate": rate(len(rows) - token_sequence_recovered, len(rows)),
        "mean_alignment_count": statistics.mean(alignment_counts)
        if alignment_counts
        else None,
        "median_alignment_count": statistics.median(alignment_counts)
        if alignment_counts
        else None,
        "p95_alignment_count": percentile(alignment_counts, 95),
        "p99_alignment_count": percentile(alignment_counts, 99),
        "max_alignment_count": max(alignment_counts) if alignment_counts else None,
        "mean_source_tokens": statistics.mean(source_tokens) if source_tokens else None,
        "mean_target_tokens": statistics.mean(target_tokens) if target_tokens else None,
        "median_span_width": statistics.median(span_widths) if span_widths else None,
        "median_gap_width": statistics.median(gap_widths) if gap_widths else None,
        "median_runtime_ms": statistics.median(runtimes) if runtimes else None,
        "p95_runtime_ms": percentile(runtimes, 95),
        "errors": dict(errors),
    }


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[f"{row['variant']}::{row['tokenizer']}"].append(row)

    by_variant_tokenizer = {
        group: summarize_group(rows) for group, rows in sorted(groups.items())
    }

    by_tokenizer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_tokenizer[row["tokenizer"]].append(row)

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_variant[row["variant"]].append(row)

    by_stress_pattern_variant_tokenizer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        for pattern in row["stress_patterns"]:
            key = f"{pattern}::{row['variant']}::{row['tokenizer']}"
            by_stress_pattern_variant_tokenizer[key].append(row)

    return {
        "parameters": {
            "data_path": str(args.data_path),
            "seed": args.seed,
            "sample_size": args.sample_size,
            "min_target_words": args.min_target_words,
            "tokenizers": args.tokenizers,
            "alignment_count_cap": args.alignment_count_cap,
        },
        "overall": summarize_group(results),
        "by_tokenizer": {
            tokenizer: summarize_group(rows) for tokenizer, rows in sorted(by_tokenizer.items())
        },
        "by_variant": {
            variant: summarize_group(rows) for variant, rows in sorted(by_variant.items())
        },
        "by_variant_tokenizer": by_variant_tokenizer,
        "by_stress_pattern_variant_tokenizer": {
            key: summarize_group(rows)
            for key, rows in sorted(by_stress_pattern_variant_tokenizer.items())
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
    rows = task_rows(records)
    adapters = build_tokenizers(args.tokenizers, args.local_files_only)
    results = evaluate_rows(rows, adapters, args.alignment_count_cap)

    summary = summarize(results, args)
    summary["parameters"]["records_loaded"] = len(records)
    summary["parameters"]["tasks_evaluated"] = len(rows)
    write_json(args.summary_output, summary)

    if args.records_output:
        args.records_output.parent.mkdir(parents=True, exist_ok=True)
        with args.records_output.open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row) + "\n")

    print(json.dumps(summary["by_variant_tokenizer"], indent=2))


if __name__ == "__main__":
    main()
