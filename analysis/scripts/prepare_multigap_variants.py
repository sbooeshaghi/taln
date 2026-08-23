"""Generate the frozen Stage 2 controlled multi-gap variant manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from run_corrected_baseline_matrix import (
    DEFAULT_INPUT_ROOT,
    load_and_validate_freeze,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
)

from taln.taln_aln import norm_text_with_mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "stage2"
DEFAULT_VARIANTS = DEFAULT_OUTPUT_DIR / "multigap_variants.jsonl"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "multigap_variant_manifest.json"
STAGE2_SPEC = (
    REPO_ROOT / "analysis" / "config" / "revision_2026" / "stage2_spec.json"
)

SCHEMA_VERSION = "multigap-variant-v1"
PLACEMENT_RULE = "balanced_retained_segments_v1"
GAP_COUNTS = (1, 2, 3)
GAP_WIDTHS = (1, 2, 3)
COMMON_COHORT_MIN_CHUNKS = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def canonical_chunks(target: str) -> list[re.Match[str]]:
    return list(re.finditer(r"\S+", target))


def retained_segment_lengths(
    chunk_count: int, gap_count: int, gap_width: int
) -> tuple[int, ...]:
    """Evenly distribute retained chunks around fixed-width deleted blocks."""
    if gap_count < 1 or gap_width < 1:
        raise ValueError("gap_count and gap_width must be positive")
    retained_count = chunk_count - gap_count * gap_width
    segment_count = gap_count + 1
    if retained_count < segment_count:
        raise ValueError(
            f"{chunk_count} chunks cannot support {gap_count} gaps of width "
            f"{gap_width} with retained separators"
        )
    base, remainder = divmod(retained_count, segment_count)
    return tuple(base + (index < remainder) for index in range(segment_count))


def deletion_blocks(
    chunk_count: int, gap_count: int, gap_width: int
) -> tuple[tuple[int, int], ...]:
    segments = retained_segment_lengths(chunk_count, gap_count, gap_width)
    blocks = []
    cursor = segments[0]
    for gap_index in range(gap_count):
        blocks.append((cursor, cursor + gap_width))
        cursor += gap_width + segments[gap_index + 1]
    if cursor != chunk_count:
        raise AssertionError("deletion placement did not consume every target chunk")
    return tuple(blocks)


def literal_occurrences(source: str, target: str) -> list[dict[str, int]]:
    if not target:
        return []
    intervals = []
    start = source.find(target)
    while start != -1:
        intervals.append({"start": start, "end": start + len(target)})
        start = source.find(target, start + 1)
    return intervals


def normalized_exact_substring(source: str, target: str) -> bool:
    source_normalized, _ = norm_text_with_mapping(source)
    target_normalized, _ = norm_text_with_mapping(target)
    return bool(target_normalized) and target_normalized in source_normalized


def stable_variant_id(
    record_id: str, gap_count: int, gap_width: int, target: str
) -> str:
    key = "\x1f".join(
        [SCHEMA_VERSION, record_id, str(gap_count), str(gap_width), target]
    )
    return f"multigap-variant:{sha256_text(key)[:24]}"


def _base_payload(
    record: dict[str, Any], gold_intervals: list[dict[str, int]]
) -> dict[str, Any]:
    chunks = canonical_chunks(record["target"])
    return {
        "schema_version": SCHEMA_VERSION,
        "base_record_id": record["record_id"],
        "dataset": record["dataset"],
        "document_id": record["document_id"],
        "split": record["split"],
        "source_sha256": record["source_sha256"],
        "original_target": record["target"],
        "original_target_sha256": record["target_sha256"],
        "original_target_chunk_count": len(chunks),
        "gold_intervals_original": gold_intervals,
        "gold_occurrence_count": len(gold_intervals),
        "cohort": "common_3x3",
        "placement_rule": PLACEMENT_RULE,
    }


def variants_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = canonical_chunks(record["target"])
    if len(chunks) < COMMON_COHORT_MIN_CHUNKS:
        return []
    if sha256_text(record["source"]) != record["source_sha256"]:
        raise ValueError(f"Source hash mismatch for {record['record_id']}")

    gold_intervals = literal_occurrences(record["source"], record["target"])
    if not gold_intervals:
        raise ValueError(f"Original target absent from source for {record['record_id']}")
    provided = {
        (interval["start"], interval["end"])
        for interval in record.get("gold_intervals_original", [])
    }
    observed = {(interval["start"], interval["end"]) for interval in gold_intervals}
    if not provided.issubset(observed):
        raise ValueError(
            f"Frozen gold interval is not a literal target occurrence for "
            f"{record['record_id']}"
        )

    base = _base_payload(record, gold_intervals)
    contiguous = {
        **base,
        "variant_id": stable_variant_id(record["record_id"], 0, 0, record["target"]),
        "condition": "contiguous_control",
        "variant_status": "contiguous_control",
        "gap_count": 0,
        "gap_width": 0,
        "target": record["target"],
        "target_sha256": record["target_sha256"],
        "retained_target_chunk_indices": list(range(len(chunks))),
        "removed_target_chunk_indices": [],
        "removed_blocks": [],
        "retained_segment_lengths": [len(chunks)],
        "remains_exact_substring_after_deletion": True,
        "primary_noncontiguous": False,
    }
    variants = [contiguous]

    chunk_text = [chunk.group() for chunk in chunks]
    for gap_count in GAP_COUNTS:
        for gap_width in GAP_WIDTHS:
            blocks = deletion_blocks(len(chunks), gap_count, gap_width)
            removed_indices = [
                index for start, end in blocks for index in range(start, end)
            ]
            removed_set = set(removed_indices)
            retained_indices = [
                index for index in range(len(chunks)) if index not in removed_set
            ]
            target = " ".join(chunk_text[index] for index in retained_indices)
            remains_contiguous = normalized_exact_substring(record["source"], target)
            status = (
                "still_contiguous_control"
                if remains_contiguous
                else "primary_noncontiguous"
            )
            block_payloads = []
            for start, end in blocks:
                block_payloads.append(
                    {
                        "start_chunk_index": start,
                        "end_chunk_index_exclusive": end,
                        "width": end - start,
                        "chunks": chunk_text[start:end],
                        "target_character_intervals": [
                            [chunks[index].start(), chunks[index].end()]
                            for index in range(start, end)
                        ],
                    }
                )
            variants.append(
                {
                    **base,
                    "variant_id": stable_variant_id(
                        record["record_id"], gap_count, gap_width, target
                    ),
                    "condition": "multigap",
                    "variant_status": status,
                    "gap_count": gap_count,
                    "gap_width": gap_width,
                    "target": target,
                    "target_sha256": sha256_text(target),
                    "retained_target_chunk_indices": retained_indices,
                    "removed_target_chunk_indices": removed_indices,
                    "removed_blocks": block_payloads,
                    "retained_segment_lengths": list(
                        retained_segment_lengths(
                            len(chunks), gap_count, gap_width
                        )
                    ),
                    "remains_exact_substring_after_deletion": remains_contiguous,
                    "primary_noncontiguous": not remains_contiguous,
                }
            )
    return variants


def iter_frozen_records(input_root: Path) -> Iterator[dict[str, Any]]:
    for filename, dataset in (
        ("boat_records.jsonl", "boat"),
        ("bioboat_records.jsonl", "bioboat"),
    ):
        for record in read_jsonl(input_root / filename):
            if record["dataset"] != dataset:
                raise ValueError(f"Unexpected dataset in {filename}")
            yield record


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    frozen_splits, input_hashes = load_and_validate_freeze(input_root)

    variants = []
    seen_ids = set()
    for record in iter_frozen_records(input_root):
        expected_split = frozen_splits[(record["dataset"], record["document_id"])]
        if record["split"] != expected_split:
            raise ValueError(f"Split mismatch for {record['record_id']}")
        for variant in variants_for_record(record):
            if variant["variant_id"] in seen_ids:
                raise ValueError(f"Duplicate variant ID: {variant['variant_id']}")
            seen_ids.add(variant["variant_id"])
            variants.append(variant)

    write_jsonl(output, variants)
    counts = Counter(
        (
            variant["dataset"],
            variant["split"],
            variant["variant_status"],
            variant["gap_count"],
            variant["gap_width"],
        )
        for variant in variants
    )
    manifest = {
        "schema_version": "multigap-variant-manifest-v1",
        "variant_file": {
            "path": str(output.relative_to(REPO_ROOT)),
            "sha256": sha256_file(output),
            "rows": len(variants),
        },
        "counts": [
            {
                "dataset": key[0],
                "split": key[1],
                "variant_status": key[2],
                "gap_count": key[3],
                "gap_width": key[4],
                "rows": count,
            }
            for key, count in sorted(counts.items())
        ],
        "unique_base_records": len(
            {variant["base_record_id"] for variant in variants}
        ),
        "unique_documents": len(
            {(variant["dataset"], variant["document_id"]) for variant in variants}
        ),
        "input_hashes": input_hashes,
        "stage2_spec": {
            "path": str(STAGE2_SPEC.relative_to(REPO_ROOT)),
            "sha256": sha256_file(STAGE2_SPEC),
        },
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "command": sys.argv,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    manifest = prepare(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
