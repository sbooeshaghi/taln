"""Shared, offset-aware evaluation for lexical text-alignment methods.

This module is deliberately independent of the existing analysis drivers.  It
provides one tokenization and scoring contract for experiments, while retaining
``taln``'s normalization and normalized-to-original offset mapping.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable, Literal, Sequence

from taln.taln_aln import _map_token_offsets, norm_text_with_mapping, tokenize

MethodName = Literal["exact", "taln", "lcs", "difflib", "semi_global_exact"]
TokenizerName = Literal[
    "whitespace",
    "boundary_stripped_whitespace",
    "punctuation",
    "cl100k_base",
    "scibert",
    "pubmedbert",
]

HF_TOKENIZERS = {
    "scibert": {
        "model_id": "allenai/scibert_scivocab_uncased",
        "revision": "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1",
    },
    "pubmedbert": {
        "model_id": (
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        ),
        "revision": "e1354b7a3a09615f6aba48dfad4b7a613eef7062",
    },
}

DEFAULT_CANDIDATE_CAP = 100_000


@dataclass(frozen=True)
class Token:
    """A normalized token with offsets into the original source string."""

    token_id: str | int
    surface: str
    start: int
    end: int
    norm_start: int
    norm_end: int


@dataclass(frozen=True)
class TokenizedText:
    text: str
    normalized_text: str
    normalized_to_original: tuple[tuple[int, int], ...]
    tokens: tuple[Token, ...]


@dataclass(frozen=True)
class TargetVariant:
    name: str
    text: str
    tokenized: TokenizedText


@dataclass(frozen=True)
class Candidate:
    """One ordered correspondence between target and source token positions."""

    target_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    source_offsets: tuple[tuple[int, int], ...]
    target_token_count: int
    target_variant: str
    evidence_interval_original: tuple[int, int] | None = None
    reconstruction_override: str | None = None
    alignment_score: int | None = None

    @property
    def complete(self) -> bool:
        return self.target_indices == tuple(range(self.target_token_count))


@dataclass(frozen=True)
class AlignmentOutput:
    """Candidates plus capped counts for a single target representation."""

    candidates: tuple[Candidate, ...]
    candidate_count: int
    candidate_count_truncated: bool = False
    unique_source_intervals: tuple[tuple[int, int], ...] = ()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tokenized_from_taln(text: str, tokenization_type: str) -> TokenizedText:
    normalized, mapping = norm_text_with_mapping(text)
    tokens, _ = tokenize(text, ttype=tokenization_type)
    return TokenizedText(
        text=text,
        normalized_text=normalized,
        normalized_to_original=tuple(mapping),
        tokens=tuple(
            Token(
                token_id=token["enc_token"],
                surface=token["token"],
                start=token["start_idx"],
                end=token["end_idx"],
                norm_start=token["norm_start_idx"],
                norm_end=token["norm_end_idx"],
            )
            for token in tokens
        ),
    )


def _tokenized_from_normalized_tokens(
    text: str,
    normalized: str,
    mapping: Sequence[tuple[int, int]],
    raw_tokens: list[dict[str, Any]],
) -> TokenizedText:
    mapped = _map_token_offsets(raw_tokens, mapping)
    return TokenizedText(
        text=text,
        normalized_text=normalized,
        normalized_to_original=tuple(mapping),
        tokens=tuple(
            Token(
                token_id=token["enc_token"],
                surface=token["token"],
                start=token["start_idx"],
                end=token["end_idx"],
                norm_start=token["norm_start_idx"],
                norm_end=token["norm_end_idx"],
            )
            for token in mapped
        ),
    )


def _tokenize_boundary_stripped_whitespace(text: str) -> TokenizedText:
    """Strip punctuation only at each whitespace chunk's outer boundaries."""
    normalized, mapping = norm_text_with_mapping(text)
    raw_tokens = []
    for match in re.finditer(r"\S+", normalized):
        surface = match.group()
        left = 0
        right = len(surface)
        while left < right and not re.match(r"\w", surface[left]):
            left += 1
        while right > left and not re.match(r"\w", surface[right - 1]):
            right -= 1
        if left == right:
            continue
        start = match.start() + left
        end = match.start() + right
        token = normalized[start:end]
        raw_tokens.append(
            {
                "token": token,
                "enc_token": token,
                "start_idx": start,
                "end_idx": end,
            }
        )
    return _tokenized_from_normalized_tokens(text, normalized, mapping, raw_tokens)


def _tokenize_punctuation(text: str) -> TokenizedText:
    """Lexical tokenizer that retains punctuation as its own token."""
    normalized, mapping = norm_text_with_mapping(text)
    raw_tokens = [
        {
            "token": match.group(),
            "enc_token": match.group(),
            "start_idx": match.start(),
            "end_idx": match.end(),
        }
        for match in re.finditer(r"\w+|[^\w\s]", normalized)
    ]
    return _tokenized_from_normalized_tokens(text, normalized, mapping, raw_tokens)


@lru_cache(maxsize=None)
def _load_hf_tokenizer(tokenizer: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "SciBERT and PubMedBERT evaluation requires transformers and tokenizers"
        ) from exc

    spec = HF_TOKENIZERS[tokenizer]
    loaded = AutoTokenizer.from_pretrained(
        spec["model_id"],
        revision=spec["revision"],
        use_fast=True,
        local_files_only=True,
    )
    if not loaded.is_fast:
        raise RuntimeError(f"{spec['model_id']} did not load an offset-aware tokenizer")
    return loaded


def _tokenize_hf(text: str, tokenizer: str) -> TokenizedText:
    normalized, mapping = norm_text_with_mapping(text)
    encoded = _load_hf_tokenizer(tokenizer)(
        normalized,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    raw_tokens = []
    for token_id, (start, end) in zip(
        encoded["input_ids"], encoded["offset_mapping"], strict=True
    ):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        raw_tokens.append(
            {
                "token": normalized[start:end],
                # Uncased WordPiece IDs alone would treat case-different or
                # unknown surfaces as equal. Retaining the normalized surface
                # keeps this a verbatim lexical-alignment experiment while
                # still using the biomedical tokenizer's segmentation.
                "enc_token": (int(token_id), normalized[start:end]),
                "start_idx": start,
                "end_idx": end,
            }
        )
    return _tokenized_from_normalized_tokens(text, normalized, mapping, raw_tokens)


@lru_cache(maxsize=8192)
def tokenize_text(text: str, tokenizer: TokenizerName) -> TokenizedText:
    """Apply the evaluator's shared normalization and tokenizer contract."""
    if tokenizer == "whitespace":
        return _tokenized_from_taln(text, "whitespace")
    if tokenizer == "boundary_stripped_whitespace":
        return _tokenize_boundary_stripped_whitespace(text)
    if tokenizer == "cl100k_base":
        return _tokenized_from_taln(text, "token")
    if tokenizer == "punctuation":
        return _tokenize_punctuation(text)
    if tokenizer in HF_TOKENIZERS:
        return _tokenize_hf(text, tokenizer)
    raise ValueError(f"Unsupported tokenizer: {tokenizer}")


@lru_cache(maxsize=8192)
def _target_variants(target: str, tokenizer: TokenizerName) -> tuple[TargetVariant, ...]:
    values = [("standalone", target)]
    if tokenizer == "cl100k_base" and target and not target.startswith(" "):
        values.append(("prefixed_space", " " + target))
    return tuple(
        TargetVariant(name=name, text=value, tokenized=tokenize_text(value, tokenizer))
        for name, value in values
    )


def _candidate(
    target_indices: Iterable[int],
    source_indices: Iterable[int],
    source: TokenizedText,
    target: TargetVariant,
    *,
    evidence_interval_original: tuple[int, int] | None = None,
    reconstruction_override: str | None = None,
    alignment_score: int | None = None,
) -> Candidate:
    target_positions = tuple(target_indices)
    source_positions = tuple(source_indices)
    return Candidate(
        target_indices=target_positions,
        source_indices=source_positions,
        source_offsets=tuple(
            (source.tokens[index].start, source.tokens[index].end)
            for index in source_positions
        ),
        target_token_count=len(target.tokenized.tokens),
        target_variant=target.name,
        evidence_interval_original=evidence_interval_original,
        reconstruction_override=reconstruction_override,
        alignment_score=alignment_score,
    )


def _exact_candidates(source: TokenizedText, target: TargetVariant) -> list[Candidate]:
    if not target.text:
        return []
    candidates = []
    start = source.normalized_text.find(target.tokenized.normalized_text)
    while start != -1:
        end = start + len(target.tokenized.normalized_text)
        source_indices = [
            index
            for index, token in enumerate(source.tokens)
            if token.norm_end > start and token.norm_start < end
        ]
        if start < end and source.normalized_to_original:
            original_start = source.normalized_to_original[start][0]
            original_end = source.normalized_to_original[end - 1][1]
            candidates.append(
                _candidate(
                    range(len(target.tokenized.tokens)),
                    source_indices,
                    source,
                    target,
                    evidence_interval_original=(original_start, original_end),
                    reconstruction_override=source.normalized_text[start:end],
                )
            )
        start = source.normalized_text.find(target.tokenized.normalized_text, start + 1)
    return candidates


def _ordered_position_lists(
    source: TokenizedText, target: TargetVariant
) -> list[list[int]]:
    source_by_id: dict[str | int, list[int]] = {}
    for source_index, token in enumerate(source.tokens):
        source_by_id.setdefault(token.token_id, []).append(source_index)
    return [source_by_id.get(token.token_id, []) for token in target.tokenized.tokens]


def _count_ordered_paths(
    position_lists: Sequence[Sequence[int]], cap: int
) -> tuple[int, bool]:
    if not position_lists or any(not positions for positions in position_lists):
        return 0, False
    saturation = cap + 1
    counts = {position: 1 for position in position_lists[0]}
    for positions in position_lists[1:]:
        previous = sorted(counts.items())
        previous_index = 0
        running = 0
        next_counts: dict[int, int] = {}
        for position in positions:
            while (
                previous_index < len(previous)
                and previous[previous_index][0] < position
            ):
                running = min(
                    saturation, running + previous[previous_index][1]
                )
                previous_index += 1
            if running:
                next_counts[position] = running
        counts = next_counts
        if not counts:
            return 0, False
    total = sum(counts.values())
    if total > cap:
        return cap, True
    return total, False


def _enumerate_ordered_paths(
    position_lists: Sequence[Sequence[int]], cap: int
) -> list[tuple[int, ...]]:
    if not position_lists or any(not positions for positions in position_lists):
        return []
    paths: list[tuple[int, ...]] = []

    def visit(position: int, path: tuple[int, ...]) -> None:
        if len(paths) >= cap:
            return
        if position == len(position_lists):
            paths.append(path)
            return
        previous = path[-1] if path else -1
        for source_index in position_lists[position]:
            if source_index > previous:
                visit(position + 1, path + (source_index,))
            if len(paths) >= cap:
                return

    visit(0, ())
    return paths


def _minimum_gap_path(
    position_lists: Sequence[Sequence[int]],
) -> tuple[int, ...] | None:
    """Return the minimum-gap exact semi-global path with deterministic ties."""
    if not position_lists or any(not positions for positions in position_lists):
        return None
    states = {position: (position,) for position in position_lists[0]}
    for positions in position_lists[1:]:
        previous = sorted(states.items())
        previous_index = 0
        best_path: tuple[int, ...] | None = None
        next_states: dict[int, tuple[int, ...]] = {}
        for position in positions:
            while (
                previous_index < len(previous)
                and previous[previous_index][0] < position
            ):
                path = previous[previous_index][1]
                if best_path is None or (path[0] > best_path[0]) or (
                    path[0] == best_path[0] and path < best_path
                ):
                    best_path = path
                previous_index += 1
            if best_path is not None:
                next_states[position] = best_path + (position,)
        states = next_states
        if not states:
            return None
    return min(
        states.values(),
        key=lambda path: (path[-1] - path[0], path[0], path),
    )


def _ordered_outer_intervals(
    source: TokenizedText,
    target: TargetVariant,
    target_indices: Sequence[int],
    position_lists: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """Deduplicate candidate locations without enumerating internal paths."""
    if not position_lists or any(not positions for positions in position_lists):
        return ()
    starts = list(position_lists[0])
    states = {position: 1 << index for index, position in enumerate(starts)}
    for positions in position_lists[1:]:
        previous = sorted(states.items())
        previous_index = 0
        reachable_starts = 0
        next_states: dict[int, int] = {}
        for position in positions:
            while (
                previous_index < len(previous)
                and previous[previous_index][0] < position
            ):
                reachable_starts |= previous[previous_index][1]
                previous_index += 1
            if reachable_starts:
                next_states[position] = reachable_starts
        states = next_states
        if not states:
            return ()

    intervals = set()
    for end_position, start_mask in states.items():
        for start_index, start_position in enumerate(starts):
            if not start_mask & (1 << start_index):
                continue
            path = (
                (start_position,)
                if len(position_lists) == 1
                else (start_position, end_position)
            )
            candidate = _candidate(target_indices, path, source, target)
            interval = _source_interval(candidate, source, target)
            if interval is not None:
                intervals.add(interval)
    return tuple(sorted(intervals))


def _taln_output(
    source: TokenizedText,
    target: TargetVariant,
    *,
    candidate_cap: int,
    include_candidates: bool,
) -> AlignmentOutput:
    """Enumerate all injective order-preserving lexical token alignments.

    Each repeated target token remains a distinct target position.  This is the
    intended ordered-alignment semantics without the old adjacent-duplicate
    collapse in ``taln.align_target``.
    """
    position_lists = _ordered_position_lists(source, target)
    matched = [
        (target_index, positions)
        for target_index, positions in enumerate(position_lists)
    ]
    matched = [(index, positions) for index, positions in matched if positions]
    if not matched:
        return AlignmentOutput((), 0)

    matched_indices = [target_index for target_index, _ in matched]
    matched_positions = [positions for _, positions in matched]
    candidate_count, truncated = _count_ordered_paths(matched_positions, candidate_cap)
    if include_candidates:
        paths = _enumerate_ordered_paths(matched_positions, candidate_cap)
    else:
        path = _minimum_gap_path(matched_positions)
        paths = [path] if path is not None else []
    candidates = tuple(
        _candidate(matched_indices, path, source, target) for path in paths
    )
    intervals = _ordered_outer_intervals(
        source, target, matched_indices, matched_positions
    )
    return AlignmentOutput(candidates, candidate_count, truncated, intervals)


def _lcs_candidates(source: TokenizedText, target: TargetVariant) -> list[Candidate]:
    source_ids = [token.token_id for token in source.tokens]
    target_ids = [token.token_id for token in target.tokenized.tokens]
    rows = len(source_ids) + 1
    columns = len(target_ids) + 1
    score = [[0] * columns for _ in range(rows)]
    for source_index, source_id in enumerate(source_ids, start=1):
        for target_index, target_id in enumerate(target_ids, start=1):
            if source_id == target_id:
                score[source_index][target_index] = score[source_index - 1][target_index - 1] + 1
            else:
                score[source_index][target_index] = max(
                    score[source_index - 1][target_index], score[source_index][target_index - 1]
                )

    source_indices: list[int] = []
    target_indices: list[int] = []
    source_index = len(source_ids)
    target_index = len(target_ids)
    while source_index and target_index:
        if source_ids[source_index - 1] == target_ids[target_index - 1]:
            source_indices.append(source_index - 1)
            target_indices.append(target_index - 1)
            source_index -= 1
            target_index -= 1
        elif score[source_index - 1][target_index] >= score[source_index][target_index - 1]:
            source_index -= 1
        else:
            target_index -= 1
    if not source_indices:
        return []
    return [_candidate(reversed(target_indices), reversed(source_indices), source, target)]


def _difflib_candidates(source: TokenizedText, target: TargetVariant) -> list[Candidate]:
    source_ids = [token.token_id for token in source.tokens]
    target_ids = [token.token_id for token in target.tokenized.tokens]
    source_indices: list[int] = []
    target_indices: list[int] = []
    matcher = difflib.SequenceMatcher(a=source_ids, b=target_ids, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            source_indices.append(block.a + offset)
            target_indices.append(block.b + offset)
    if not source_indices:
        return []
    return [_candidate(target_indices, source_indices, source, target)]


def _semi_global_exact_candidates(
    source: TokenizedText, target: TargetVariant
) -> list[Candidate]:
    """Return one target-global/source-semi-global exact-token alignment.

    Matches score +1, skipped source tokens score -1, source prefixes and
    suffixes are free, and mismatches or target gaps are forbidden. The chosen
    path therefore minimizes skipped source tokens, then uses the earliest and
    lexicographically smallest exact embedding.
    """
    path = _minimum_gap_path(_ordered_position_lists(source, target))
    if path is None:
        return []
    target_count = len(target.tokenized.tokens)
    skipped = (path[-1] - path[0] + 1) - target_count
    score = target_count - skipped
    return [
        _candidate(
            range(target_count),
            path,
            source,
            target,
            alignment_score=score,
        )
    ]


def _align(
    method: MethodName,
    source: TokenizedText,
    target: TargetVariant,
    *,
    candidate_cap: int,
    include_candidates: bool,
) -> AlignmentOutput:
    def one_path_output(candidates: tuple[Candidate, ...]) -> AlignmentOutput:
        intervals = tuple(
            sorted(
                {
                    interval
                    for candidate in candidates
                    if (interval := _source_interval(candidate, source, target))
                    is not None
                }
            )
        )
        return AlignmentOutput(candidates, len(candidates), False, intervals)

    if method == "exact":
        candidates = tuple(_exact_candidates(source, target))
        return one_path_output(candidates)
    if method == "taln":
        return _taln_output(
            source,
            target,
            candidate_cap=candidate_cap,
            include_candidates=include_candidates,
        )
    if method == "lcs":
        candidates = tuple(_lcs_candidates(source, target))
        return one_path_output(candidates)
    if method == "difflib":
        candidates = tuple(_difflib_candidates(source, target))
        return one_path_output(candidates)
    if method == "semi_global_exact":
        candidates = tuple(_semi_global_exact_candidates(source, target))
        return one_path_output(candidates)
    raise ValueError(f"Unsupported method: {method}")


def _token_reconstruction(
    text: TokenizedText,
    token_indices: Sequence[int],
    tokenizer: TokenizerName,
) -> str:
    selected = [text.tokens[index] for index in token_indices]
    if not selected:
        return ""
    if tokenizer in {"whitespace", "boundary_stripped_whitespace"}:
        return " ".join(token.surface for token in selected)
    if tokenizer == "cl100k_base":
        return "".join(token.surface for token in selected)

    pieces = [selected[0].surface]
    for previous, current in zip(selected, selected[1:]):
        gap = text.normalized_text[previous.norm_end : current.norm_start]
        if any(char.isspace() for char in gap):
            pieces.append(" ")
        pieces.append(current.surface)
    return "".join(pieces)


def _candidate_reconstruction(
    candidate: Candidate,
    source: TokenizedText,
    tokenizer: TokenizerName,
) -> str:
    if candidate.reconstruction_override is not None:
        return candidate.reconstruction_override
    return _token_reconstruction(source, candidate.source_indices, tokenizer)


def _source_interval(
    candidate: Candidate,
    source: TokenizedText,
    target: TargetVariant,
) -> tuple[int, int] | None:
    if candidate.evidence_interval_original is not None:
        start, end = candidate.evidence_interval_original
    elif candidate.source_offsets:
        start, end = candidate.source_offsets[0][0], candidate.source_offsets[-1][1]
    else:
        return None

    if target.name == "prefixed_space" and target.text.startswith(" "):
        while start < end and source.text[start].isspace():
            start += 1
    return start, end


def _ordered_path_at_gold_interval(
    source: TokenizedText,
    target: TargetVariant,
    gold_interval: tuple[int, int],
) -> tuple[int, ...] | None:
    """Find any complete ordered path with the requested outer interval."""
    position_lists = _ordered_position_lists(source, target)
    if not position_lists or any(not positions for positions in position_lists):
        return None
    gold_start, gold_end = gold_interval
    for first in position_lists[0]:
        if len(position_lists) == 1:
            path = (first,)
            candidate = _candidate((0,), path, source, target)
            if _source_interval(candidate, source, target) == gold_interval:
                return path
            continue
        for last in position_lists[-1]:
            if last <= first:
                continue
            path = [first]
            previous = first
            feasible = True
            for positions in position_lists[1:-1]:
                next_position = next(
                    (
                        position
                        for position in positions
                        if previous < position < last
                    ),
                    None,
                )
                if next_position is None:
                    feasible = False
                    break
                path.append(next_position)
                previous = next_position
            if not feasible:
                continue
            path.append(last)
            candidate = _candidate(range(len(path)), path, source, target)
            if _source_interval(candidate, source, target) == (gold_start, gold_end):
                return tuple(path)
    return None


def _coerce_gold_intervals(
    gold_intervals: Sequence[tuple[int, int] | dict[str, int]] | None,
) -> tuple[tuple[int, int], ...]:
    if not gold_intervals:
        return ()
    intervals = []
    for interval in gold_intervals:
        if isinstance(interval, dict):
            start, end = interval["start"], interval["end"]
        else:
            start, end = interval
        if start > end:
            raise ValueError("Gold interval start must not exceed end")
        intervals.append((start, end))
    return tuple(intervals)


def _score_candidate(
    candidate: Candidate,
    source: TokenizedText,
    target: TargetVariant,
    tokenizer: TokenizerName,
    gold_intervals: tuple[tuple[int, int], ...],
) -> dict[str, Any]:
    reconstruction = _candidate_reconstruction(candidate, source, tokenizer)
    target_reconstruction = _token_reconstruction(
        target.tokenized,
        range(len(target.tokenized.tokens)),
        tokenizer,
    )
    interval = _source_interval(candidate, source, target)
    complete = candidate.complete
    full_lexical_support = complete
    localization = (
        interval in gold_intervals
        if gold_intervals and interval is not None
        else None
    )
    payload = asdict(candidate)
    payload.update(
        {
            "complete_token_alignment": complete,
            "full_lexical_support": full_lexical_support,
            "full_lexical_reconstruction": full_lexical_support,
            "reconstruction": reconstruction,
            "target_reconstruction": target_reconstruction,
            "source_interval": interval,
            "localization": localization,
        }
    )
    return payload


def evaluate_task(
    *,
    task_id: str,
    source: str,
    target: str,
    method: MethodName,
    tokenizer: TokenizerName,
    gold_intervals: Sequence[tuple[int, int] | dict[str, int]] | None = None,
    candidate_cap: int = DEFAULT_CANDIDATE_CAP,
    include_candidates: bool = True,
) -> dict[str, Any]:
    """Evaluate one method/tokenizer task under a common, JSON-ready schema.

    ``partial_token_overlap`` means at least one token matched. It is not a
    grounding claim. ``complete_token_alignment`` and ``full_lexical_support``
    require the complete case-preserving target token sequence to match in
    order under the declared tokenizer. Display reconstruction is diagnostic;
    whitespace discarded by a tokenizer cannot make otherwise identical token
    sequences asymmetric. Localization uses original-text intervals only.
    """
    if candidate_cap < 1:
        raise ValueError("candidate_cap must be positive")
    normalized_target, _ = norm_text_with_mapping(target)
    if not normalized_target:
        raise ValueError("target must not normalize to an empty string")
    source_tokenized = tokenize_text(source, tokenizer)
    variants = _target_variants(target, tokenizer)
    if method != "exact" and any(not variant.tokenized.tokens for variant in variants):
        raise ValueError("target must contain at least one token")
    normalized_gold_intervals = _coerce_gold_intervals(gold_intervals)
    scored_candidates = []
    raw_candidate_count = 0
    candidate_count_truncated = False
    candidate_counts_by_variant = {}
    all_unique_source_intervals = set()
    for variant in variants:
        output = _align(
            method,
            source_tokenized,
            variant,
            candidate_cap=candidate_cap,
            include_candidates=include_candidates,
        )
        raw_candidate_count += output.candidate_count
        candidate_count_truncated = (
            candidate_count_truncated
            or output.candidate_count_truncated
            or raw_candidate_count > candidate_cap
        )
        candidate_counts_by_variant[variant.name] = {
            "count": output.candidate_count,
            "truncated": output.candidate_count_truncated,
        }
        all_unique_source_intervals.update(output.unique_source_intervals)
        for candidate in output.candidates:
            scored_candidates.append(
                _score_candidate(
                    candidate,
                    source_tokenized,
                    variant,
                    tokenizer,
                    normalized_gold_intervals,
                )
            )

    localization = (
        any(
            candidate["full_lexical_reconstruction"]
            and candidate["localization"] is True
            for candidate in scored_candidates
        )
        if normalized_gold_intervals
        else None
    )
    if method == "taln" and normalized_gold_intervals and not localization:
        for variant in variants:
            for interval in normalized_gold_intervals:
                path = _ordered_path_at_gold_interval(
                    source_tokenized, variant, interval
                )
                if path is None:
                    continue
                witness = _score_candidate(
                    _candidate(
                        range(len(variant.tokenized.tokens)),
                        path,
                        source_tokenized,
                        variant,
                    ),
                    source_tokenized,
                    variant,
                    tokenizer,
                    normalized_gold_intervals,
                )
                if not witness["full_lexical_support"]:
                    continue
                localization = True
                witness_key = (
                    witness["target_variant"],
                    tuple(witness["source_indices"]),
                )
                existing_keys = {
                    (
                        candidate["target_variant"],
                        tuple(candidate["source_indices"]),
                    )
                    for candidate in scored_candidates
                }
                if witness_key not in existing_keys:
                    scored_candidates.append(witness)
                break
            if localization:
                break

    complete_candidate_count = sum(
        candidate["complete_token_alignment"] for candidate in scored_candidates
    )
    reconstruction_candidate_count = sum(
        candidate["full_lexical_reconstruction"] for candidate in scored_candidates
    )
    unique_source_paths = {
        tuple(candidate["source_indices"]) for candidate in scored_candidates
    }
    materialized_unique_source_intervals = {
        tuple(candidate["source_interval"])
        for candidate in scored_candidates
        if candidate["source_interval"] is not None
    }

    return {
        "schema_version": "common-evaluator-v2",
        "task_id": task_id,
        "method": method,
        "tokenizer": tokenizer,
        "normalization": "taln.norm_text_with_mapping",
        "source_sha256": sha256_text(source),
        "target_sha256": sha256_text(target),
        "source_length": len(source),
        "target_length": len(target),
        "gold_intervals": normalized_gold_intervals,
        "target_variants": [
            {
                "name": variant.name,
                "text": variant.text,
                "token_count": len(variant.tokenized.tokens),
            }
            for variant in variants
        ],
        "candidate_count": min(candidate_cap, raw_candidate_count),
        "candidate_count_truncated": candidate_count_truncated,
        "candidate_counts_by_variant": candidate_counts_by_variant,
        "unique_source_interval_count": len(all_unique_source_intervals),
        "materialized_candidate_count": len(scored_candidates),
        "materialized_unique_source_path_count": len(unique_source_paths),
        "materialized_unique_source_interval_count": len(
            materialized_unique_source_intervals
        ),
        "complete_candidate_count": complete_candidate_count,
        "reconstruction_candidate_count": reconstruction_candidate_count,
        "partial_token_overlap": bool(scored_candidates),
        "complete_token_alignment": complete_candidate_count > 0,
        "full_lexical_support": reconstruction_candidate_count > 0,
        "full_lexical_reconstruction": reconstruction_candidate_count > 0,
        "localization_evaluable": bool(normalized_gold_intervals),
        "localization": localization,
        "candidates": scored_candidates,
    }
