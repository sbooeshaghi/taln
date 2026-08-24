"""Run the frozen Stage 1 corrected tokenizer-by-method baseline matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from biomedical_tokenizer_comparison import (
    make_noncontiguous_target as legacy_central_one_gap,
)
from common_evaluator import DEFAULT_CANDIDATE_CAP, HF_TOKENIZERS, evaluate_task

from taln.taln_aln import norm_text_with_mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "revision_2026"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "stage1"
STAGE1_SPEC = REPO_ROOT / "analysis" / "config" / "revision_2026" / "stage1_spec.json"
INPUT_MANIFEST = (
    REPO_ROOT / "analysis" / "config" / "revision_2026" / "input_manifest.json"
)
SPLIT_MANIFEST = (
    REPO_ROOT / "analysis" / "config" / "revision_2026" / "split_manifest.json"
)

TOKENIZERS = (
    "whitespace",
    "boundary_stripped_whitespace",
    "punctuation",
    "cl100k_base",
    "scibert",
    "pubmedbert",
)
SEQUENCE_METHODS = ("taln", "lcs", "difflib", "semi_global_exact")
SCHEMA_VERSION = "corrected-baseline-record-v1"
RESULT_FIELDS = (
    "full_lexical_support",
    "complete_token_alignment",
    "partial_token_overlap",
    "localization",
    "candidate_count",
    "candidate_count_truncated",
    "unique_source_interval_count",
    "runtime_ms",
    "error",
    "selected_interval",
    "selected_localization",
    "candidate_counts_by_variant",
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
        help="Run only the first N tasks for pipeline validation.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--resume", action="store_true", help="Append tasks absent from an existing file."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output. Mutually exclusive with --resume.",
    )
    parser.add_argument(
        "--tokenizers", nargs="+", choices=TOKENIZERS, default=list(TOKENIZERS)
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SEQUENCE_METHODS,
        default=list(SEQUENCE_METHODS),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
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


def central_one_gap(target: str) -> dict[str, Any] | None:
    chunks = list(re.finditer(r"\S+", target))
    if len(chunks) < 3:
        return None
    removed_index = len(chunks) // 2
    retained = [chunk.group() for index, chunk in enumerate(chunks) if index != removed_index]
    removed = chunks[removed_index]
    return {
        "target": " ".join(retained),
        "removed_chunk": removed.group(),
        "removed_chunk_index": removed_index,
        "original_target_chunk_count": len(chunks),
        "removed_target_interval": [removed.start(), removed.end()],
    }


def normalized_exact_substring(source: str, target: str) -> bool:
    source_normalized, _ = norm_text_with_mapping(source)
    target_normalized, _ = norm_text_with_mapping(target)
    return bool(target_normalized) and target_normalized in source_normalized


def _base_task(record: dict[str, Any], condition: str, target: str) -> dict[str, Any]:
    task_id = f"{record['record_id']}:{condition}"
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "base_record_id": record["record_id"],
        "dataset": record["dataset"],
        "condition": condition,
        "split": record["split"],
        "document_id": record["document_id"],
        "source": record["source"],
        "source_sha256": record.get("source_sha256") or sha256_text(record["source"]),
        "target": target,
        "target_sha256": sha256_text(target),
        "gold_intervals": record.get("gold_intervals_original", []),
    }


def boat_tasks(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    contiguous = _base_task(record, "contiguous", record["target"])
    contiguous["generation_method"] = record.get("generation_method")
    yield contiguous

    gap = central_one_gap(record["target"])
    if gap is not None:
        gap_target = gap.pop("target")
        legacy_target = legacy_central_one_gap(record["target"])
        if gap_target != legacy_target:
            raise ValueError(
                f"Stage 1 one-gap target differs from legacy construction for "
                f"{record['record_id']}"
            )
        remains_contiguous = normalized_exact_substring(record["source"], gap_target)
        condition = (
            "one_gap_still_contiguous_control" if remains_contiguous else "one_gap"
        )
        one_gap = _base_task(record, condition, gap_target)
        one_gap["generation_method"] = "central_whitespace_token_deletion"
        one_gap["original_target"] = record["target"]
        one_gap["one_gap_provenance"] = gap
        one_gap["legacy_target_equivalent"] = True
        one_gap["remains_exact_substring_after_deletion"] = remains_contiguous
        yield one_gap


def llmarkers_tasks(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if record.get("all_verified") is not True:
        return
    for field in ("group_label", "feature_label"):
        target = str(record.get(field) or "")
        if not target:
            continue
        task = {
            "schema_version": SCHEMA_VERSION,
            "task_id": f"{record['record_id']}:positive_label:{field}",
            "base_record_id": record["record_id"],
            "dataset": "llmarkers",
            "dataset_id": record["dataset_id"],
            "condition": "positive_label",
            "label_field": field,
            "split": record["split"],
            "document_id": record["document_id"],
            "source": record["source_rationale"],
            "source_sha256": record["source_sha256"],
            "target": target,
            "target_sha256": sha256_text(target),
            "gold_intervals": [],
            "inventory_status": record["inventory_status"],
        }
        yield task


def iter_tasks(
    input_root: Path,
    split: str,
    frozen_splits: dict[tuple[str, str], str] | None = None,
) -> Iterator[dict[str, Any]]:
    for dataset, filename in (
        ("boat", "boat_records.jsonl"),
        ("bioboat", "bioboat_records.jsonl"),
    ):
        for record in read_jsonl(input_root / filename):
            if frozen_splits is not None:
                frozen = frozen_splits.get((dataset, record["document_id"]))
                if frozen is None:
                    raise ValueError(
                        f"Document absent from split manifest: {dataset}:{record['document_id']}"
                    )
                if record["split"] != frozen:
                    raise ValueError(
                        f"Split mismatch for {dataset}:{record['document_id']}: "
                        f"record={record['split']} manifest={frozen}"
                    )
            if record["split"] != split:
                continue
            if record["dataset"] != dataset:
                raise ValueError(f"Unexpected dataset in {filename}: {record['dataset']}")
            yield from boat_tasks(record)

    for record in read_jsonl(input_root / "llmarkers_inventory.jsonl"):
        if frozen_splits is not None:
            key = ("llmarkers", record["document_id"])
            frozen = frozen_splits.get(key)
            if frozen is None:
                raise ValueError(
                    f"Document absent from split manifest: llmarkers:{record['document_id']}"
                )
            if record["split"] != frozen:
                raise ValueError(
                    f"Split mismatch for llmarkers:{record['document_id']}: "
                    f"record={record['split']} manifest={frozen}"
                )
        if record["split"] == split:
            yield from llmarkers_tasks(record)


def _compact_result(result: dict[str, Any], runtime_ms: float) -> dict[str, Any]:
    selected = next(
        (
            candidate
            for candidate in result["candidates"]
            if candidate["full_lexical_support"]
            and candidate["localization"] is True
        ),
        next(
            (
                candidate
                for candidate in result["candidates"]
                if candidate["full_lexical_support"]
            ),
            result["candidates"][0] if result["candidates"] else None,
        ),
    )
    selected_payload = (
        {
            "target_variant": selected["target_variant"],
            "source_interval": selected["source_interval"],
            "reconstruction": selected["reconstruction"],
            "complete_token_alignment": selected["complete_token_alignment"],
            "localization": selected["localization"],
            "alignment_score": selected["alignment_score"],
        }
        if selected is not None
        else None
    )
    return {
        "full_lexical_support": result["full_lexical_support"],
        "complete_token_alignment": result["complete_token_alignment"],
        "partial_token_overlap": result["partial_token_overlap"],
        "localization_evaluable": result["localization_evaluable"],
        "localization": result["localization"],
        "candidate_count": result["candidate_count"],
        "candidate_count_truncated": result["candidate_count_truncated"],
        "candidate_counts_by_variant": result["candidate_counts_by_variant"],
        "unique_source_interval_count": result["unique_source_interval_count"],
        "selected_candidate": selected_payload,
        "runtime_ms": runtime_ms,
        "error": None,
    }


def _failed_result(
    exc: Exception, runtime_ms: float, *, localization_evaluable: bool
) -> dict[str, Any]:
    return {
        "full_lexical_support": False,
        "complete_token_alignment": False,
        "partial_token_overlap": False,
        "localization_evaluable": localization_evaluable,
        "localization": False if localization_evaluable else None,
        "candidate_count": 0,
        "candidate_count_truncated": False,
        "candidate_counts_by_variant": {},
        "unique_source_interval_count": 0,
        "selected_candidate": None,
        "runtime_ms": runtime_ms,
        "error": f"{type(exc).__name__}: {exc}",
    }


def evaluate_one(
    task: dict[str, Any],
    *,
    method: str,
    tokenizer: str,
    candidate_cap: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = evaluate_task(
            task_id=task["task_id"],
            source=task["source"],
            target=task["target"],
            method=method,
            tokenizer=tokenizer,
            gold_intervals=task["gold_intervals"],
            candidate_cap=candidate_cap,
            include_candidates=False,
        )
        runtime_ms = (time.perf_counter() - started) * 1000
        return _compact_result(result, runtime_ms)
    except Exception as exc:
        runtime_ms = (time.perf_counter() - started) * 1000
        return _failed_result(
            exc,
            runtime_ms,
            localization_evaluable=bool(task["gold_intervals"]),
        )


def evaluate_matrix_task(
    task: dict[str, Any],
    *,
    tokenizers: Iterable[str],
    methods: Iterable[str],
    candidate_cap: int,
) -> dict[str, Any]:
    row = dict(task)
    row["exact"] = evaluate_one(
        task,
        method="exact",
        tokenizer="whitespace",
        candidate_cap=candidate_cap,
    )
    row["sequence"] = {}
    for tokenizer in tokenizers:
        row["sequence"][tokenizer] = {}
        for method in methods:
            row["sequence"][tokenizer][method] = evaluate_one(
                task,
                method=method,
                tokenizer=tokenizer,
                candidate_cap=candidate_cap,
            )
    return row


def compact_result(result: dict[str, Any]) -> list[Any]:
    selected = result["selected_candidate"]
    return [
        bool(result["full_lexical_support"]),
        bool(result["complete_token_alignment"]),
        bool(result["partial_token_overlap"]),
        result["localization"],
        int(result["candidate_count"]),
        bool(result["candidate_count_truncated"]),
        int(result["unique_source_interval_count"]),
        round(float(result["runtime_ms"]), 6),
        result["error"],
        list(selected["source_interval"])
        if selected is not None and selected["source_interval"] is not None
        else None,
        selected["localization"] if selected is not None else None,
        [
            [name, int(values["count"]), bool(values["truncated"])]
            for name, values in result["candidate_counts_by_variant"].items()
        ],
    ]


def compact_matrix_row(
    row: dict[str, Any], *, tokenizers: Iterable[str], methods: Iterable[str]
) -> dict[str, Any]:
    omitted = {
        "source",
        "gold_intervals",
        "exact",
        "sequence",
        "original_target",
    }
    compact = {key: value for key, value in row.items() if key not in omitted}
    compact["exact"] = compact_result(row["exact"])
    compact["sequence"] = [
        [compact_result(row["sequence"][tokenizer][method]) for method in methods]
        for tokenizer in tokenizers
    ]
    return compact


def completed_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["task_id"] for row in read_jsonl(path)}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_and_validate_freeze(
    input_root: Path,
) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    input_manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    input_hashes = {}
    filenames = {
        "boat": "boat_records.jsonl",
        "bioboat": "bioboat_records.jsonl",
        "llmarkers": "llmarkers_inventory.jsonl",
    }
    for dataset, filename in filenames.items():
        path = input_root / filename
        expected = input_manifest["datasets"][dataset]["artifact"]
        actual_hash = sha256_file(path)
        if actual_hash != expected["sha256"]:
            raise ValueError(
                f"Frozen input hash mismatch for {dataset}: "
                f"expected {expected['sha256']}, observed {actual_hash}"
            )
        actual_rows = sum(1 for _ in read_jsonl(path))
        if actual_rows != expected["rows"]:
            raise ValueError(
                f"Frozen input row mismatch for {dataset}: "
                f"expected {expected['rows']}, observed {actual_rows}"
            )
        input_hashes[str(path.relative_to(REPO_ROOT))] = actual_hash

    frozen_splits = {}
    for document in split_manifest["documents"]:
        key = (document["dataset"], document["document_id"])
        if key in frozen_splits:
            raise ValueError(f"Duplicate document in split manifest: {key}")
        frozen_splits[key] = document["split"]
    return frozen_splits, input_hashes


def run_configuration(
    args: argparse.Namespace, input_hashes: dict[str, str]
) -> dict[str, Any]:
    code_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "analysis" / "scripts" / "common_evaluator.py",
        REPO_ROOT
        / "analysis"
        / "scripts"
        / "biomedical_tokenizer_comparison.py",
        STAGE1_SPEC,
        REPO_ROOT
        / "analysis"
        / "config"
        / "revision_2026"
        / "stage1_requirements.txt",
        INPUT_MANIFEST,
        SPLIT_MANIFEST,
    ]
    payload = {
        "schema_version": "corrected-baseline-run-config-v1",
        "split": args.split,
        "candidate_cap": args.candidate_cap,
        "tokenizers": list(args.tokenizers),
        "methods": list(args.methods),
        "limit": args.limit,
        "record_encoding": {
            "result_fields": list(RESULT_FIELDS),
            "exact": "One result array.",
            "sequence": "Nested arrays in tokenizers-major, methods-minor order.",
            "source_recovery": "Join base_record_id to the frozen input records; source text is not duplicated in matrix rows.",
        },
        "input_hashes": input_hashes,
        "code_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in code_paths
        },
        "software": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "tiktoken": package_version("tiktoken"),
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
            "huggingface_hub": package_version("huggingface-hub"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["run_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload


def validate_heldout_arguments(args: argparse.Namespace) -> None:
    if args.split != "heldout_test":
        return
    if args.limit is not None:
        raise ValueError("Held-out runs cannot use --limit")
    if tuple(args.tokenizers) != TOKENIZERS:
        raise ValueError("Held-out runs must use the frozen tokenizer matrix")
    if tuple(args.methods) != SEQUENCE_METHODS:
        raise ValueError("Held-out runs must use the frozen method matrix")
    if args.candidate_cap != DEFAULT_CANDIDATE_CAP:
        raise ValueError("Held-out runs must use the frozen candidate cap")
    if args.overwrite:
        raise ValueError("Held-out outputs cannot be overwritten; use a validated resume")


def tokenizer_probe(tokenizer: str) -> dict[str, Any]:
    result = evaluate_task(
        task_id=f"probe:{tokenizer}",
        source="FOX IL-2 TP53,  beta",
        target="IL-2 TP53",
        method="taln",
        tokenizer=tokenizer,
        candidate_cap=100,
        include_candidates=False,
    )
    metadata: dict[str, Any] = {
        "target_variants": result["target_variants"],
        "full_lexical_support": result["full_lexical_support"],
    }
    if tokenizer in HF_TOKENIZERS:
        metadata.update(HF_TOKENIZERS[tokenizer])
    return metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.candidate_cap < 1:
        raise ValueError("--candidate-cap must be positive")
    validate_heldout_arguments(args)

    input_root = args.input_root.expanduser().resolve()
    frozen_splits, input_hashes = load_and_validate_freeze(input_root)
    configuration = run_configuration(args, input_hashes)

    output_dir = args.output_root.resolve() / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "corrected_baseline_records.jsonl"
    config_path = output_dir / "corrected_baseline_run_config.json"
    metadata_path = output_dir / "corrected_baseline_run_metadata.json"
    if records_path.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"{records_path} already exists; pass --resume or --overwrite"
        )
    mode = "a" if args.resume else "w"
    already_done = completed_task_ids(records_path) if args.resume else set()

    if args.resume:
        if not config_path.exists() or not records_path.exists():
            raise FileNotFoundError(
                "A resumable run requires both the run config and records file"
            )
        previous_configuration = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_configuration != configuration:
            raise ValueError(
                "Resume configuration does not match the frozen run fingerprint"
            )
    else:
        write_json(config_path, configuration)

    input_paths = [
        input_root / "boat_records.jsonl",
        input_root / "bioboat_records.jsonl",
        input_root / "llmarkers_inventory.jsonl",
    ]
    for path in [*input_paths, STAGE1_SPEC]:
        if not path.exists():
            raise FileNotFoundError(path)

    probes = {tokenizer: tokenizer_probe(tokenizer) for tokenizer in args.tokenizers}
    started_at = datetime.now(timezone.utc)
    evaluated = 0
    skipped = 0
    errors = 0
    task_start = time.perf_counter()
    generated_task_ids = set()

    with records_path.open(mode, encoding="utf-8") as output:
        for task in iter_tasks(input_root, args.split, frozen_splits):
            if task["task_id"] in generated_task_ids:
                raise ValueError(f"Duplicate generated task ID: {task['task_id']}")
            generated_task_ids.add(task["task_id"])
            if task["task_id"] in already_done:
                skipped += 1
                continue
            if args.limit is not None and evaluated >= args.limit:
                break
            row = evaluate_matrix_task(
                task,
                tokenizers=args.tokenizers,
                methods=args.methods,
                candidate_cap=args.candidate_cap,
            )
            errors += int(row["exact"]["error"] is not None)
            errors += sum(
                result["error"] is not None
                for tokenizer_results in row["sequence"].values()
                for result in tokenizer_results.values()
            )
            compact = compact_matrix_row(
                row, tokenizers=args.tokenizers, methods=args.methods
            )
            output.write(json.dumps(compact, separators=(",", ":")) + "\n")
            evaluated += 1
            if evaluated % args.progress_every == 0:
                output.flush()
                elapsed = time.perf_counter() - task_start
                rate = evaluated / elapsed if elapsed else 0.0
                print(
                    f"{args.split}: {evaluated} tasks, {rate:.2f} tasks/s, "
                    f"{errors} method errors",
                    flush=True,
                )

    ended_at = datetime.now(timezone.utc)
    metadata = {
        "schema_version": "corrected-baseline-run-metadata-v1",
        "split": args.split,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": (ended_at - started_at).total_seconds(),
        "records_path": str(records_path.relative_to(REPO_ROOT)),
        "records_sha256": sha256_file(records_path),
        "run_fingerprint": configuration["run_fingerprint"],
        "tasks_evaluated_this_run": evaluated,
        "tasks_skipped_by_resume": skipped,
        "tasks_in_records_file": len(completed_task_ids(records_path)),
        "tasks_expected_for_split": len(generated_task_ids),
        "method_errors_this_run": errors,
        "limit": args.limit,
        "candidate_cap": args.candidate_cap,
        "tokenizers": args.tokenizers,
        "methods": args.methods,
        "input_files": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in input_paths
        },
        "stage1_spec": {
            "path": str(STAGE1_SPEC.relative_to(REPO_ROOT)),
            "sha256": sha256_file(STAGE1_SPEC),
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "taln": package_version("taln"),
            "numpy": package_version("numpy"),
            "tiktoken": package_version("tiktoken"),
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
            "huggingface_hub": package_version("huggingface-hub"),
        },
        "tokenizer_probes": probes,
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
