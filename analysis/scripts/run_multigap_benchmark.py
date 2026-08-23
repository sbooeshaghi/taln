"""Run the frozen Stage 2 controlled multi-gap benchmark matrix."""

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

from common_evaluator import DEFAULT_CANDIDATE_CAP
from run_corrected_baseline_matrix import (
    INPUT_MANIFEST,
    RESULT_FIELDS,
    SPLIT_MANIFEST,
    compact_matrix_row,
    evaluate_matrix_task,
    load_and_validate_freeze,
    package_version,
    read_jsonl,
    sha256_file,
    tokenizer_probe,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "revision_2026"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "stage2"
DEFAULT_VARIANTS = DEFAULT_OUTPUT_ROOT / "multigap_variants.jsonl"
STAGE2_SPEC = (
    REPO_ROOT / "analysis" / "config" / "revision_2026" / "stage2_spec.json"
)
STAGE2_FREEZE = (
    REPO_ROOT
    / "analysis"
    / "config"
    / "revision_2026"
    / "stage2_variant_freeze.json"
)
STAGE_REQUIREMENTS = (
    REPO_ROOT
    / "analysis"
    / "config"
    / "revision_2026"
    / "stage1_requirements.txt"
)

TOKENIZERS = ("punctuation", "cl100k_base", "pubmedbert")
METHODS = ("taln", "lcs", "difflib")
SCHEMA_VERSION = "multigap-benchmark-record-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", required=True, choices=("development", "heldout_test")
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--candidate-cap", type=int, default=DEFAULT_CANDIDATE_CAP
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--tokenizers", nargs="+", choices=TOKENIZERS, default=list(TOKENIZERS)
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def path_label(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def strict_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    task_ids = set()
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Missing task_id in {path}")
        if task_id in task_ids:
            raise ValueError(f"Duplicate task_id in {path}: {task_id}")
        task_ids.add(task_id)
    return task_ids


def load_base_records(input_root: Path) -> dict[str, dict[str, Any]]:
    lookup = {}
    for filename, dataset in (
        ("boat_records.jsonl", "boat"),
        ("bioboat_records.jsonl", "bioboat"),
    ):
        for record in read_jsonl(input_root / filename):
            if record["dataset"] != dataset:
                raise ValueError(f"Unexpected dataset in {filename}")
            if record["record_id"] in lookup:
                raise ValueError(f"Duplicate base record: {record['record_id']}")
            lookup[record["record_id"]] = record
    return lookup


def load_and_validate_variants(
    variants_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    freeze = read_json(STAGE2_FREEZE)
    expected = freeze["variant_file"]
    if sha256_file(variants_path) != expected["sha256"]:
        raise ValueError("Stage 2 variant file hash does not match the frozen manifest")
    if sha256_file(STAGE2_SPEC) != freeze["stage2_spec"]["sha256"]:
        raise ValueError("Stage 2 specification changed after variant freeze")
    generator_path = REPO_ROOT / freeze["generator"]["path"]
    if sha256_file(generator_path) != freeze["generator"]["sha256"]:
        raise ValueError("Stage 2 variant generator changed after variant freeze")

    variants = {}
    split_counts = {"development": 0, "heldout_test": 0}
    dataset_split_counts: dict[str, int] = {}
    for variant in read_jsonl(variants_path):
        variant_id = variant["variant_id"]
        if variant_id in variants:
            raise ValueError(f"Duplicate frozen variant ID: {variant_id}")
        variants[variant_id] = variant
        split_counts[variant["split"]] += 1
        key = f"{variant['dataset']}::{variant['split']}"
        dataset_split_counts[key] = dataset_split_counts.get(key, 0) + 1

    if len(variants) != expected["rows"]:
        raise ValueError(
            f"Frozen variant row mismatch: expected {expected['rows']}, "
            f"observed {len(variants)}"
        )
    if split_counts != freeze["split_rows"]:
        raise ValueError("Frozen variant split counts do not match")
    if dataset_split_counts != freeze["dataset_split_rows"]:
        raise ValueError("Frozen variant dataset/split counts do not match")
    return freeze, variants


def task_from_variant(
    variant: dict[str, Any], base_records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    try:
        base = base_records[variant["base_record_id"]]
    except KeyError as exc:
        raise KeyError(
            f"Missing base record for {variant['variant_id']}: "
            f"{variant['base_record_id']}"
        ) from exc
    for field in ("dataset", "document_id", "split", "source_sha256"):
        if variant[field] != base[field]:
            raise ValueError(
                f"Variant/base {field} mismatch for {variant['variant_id']}"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": variant["variant_id"],
        "variant_id": variant["variant_id"],
        "base_record_id": variant["base_record_id"],
        "dataset": variant["dataset"],
        "condition": variant["condition"],
        "variant_status": variant["variant_status"],
        "cohort": variant["cohort"],
        "split": variant["split"],
        "document_id": variant["document_id"],
        "source": base["source"],
        "source_sha256": variant["source_sha256"],
        "target": variant["target"],
        "target_sha256": variant["target_sha256"],
        "gold_intervals": variant["gold_intervals_original"],
        "gold_occurrence_count": variant["gold_occurrence_count"],
        "gap_count": variant["gap_count"],
        "gap_width": variant["gap_width"],
        "original_target_chunk_count": variant["original_target_chunk_count"],
        "primary_noncontiguous": variant["primary_noncontiguous"],
        "remains_exact_substring_after_deletion": variant[
            "remains_exact_substring_after_deletion"
        ],
        "placement_rule": variant["placement_rule"],
    }


def validate_heldout_arguments(args: argparse.Namespace) -> None:
    if args.split != "heldout_test":
        return
    if args.limit is not None:
        raise ValueError("Held-out runs cannot use --limit")
    if tuple(args.tokenizers) != TOKENIZERS:
        raise ValueError("Held-out runs must use the frozen Stage 2 tokenizers")
    if tuple(args.methods) != METHODS:
        raise ValueError("Held-out runs must use the frozen Stage 2 methods")
    if args.candidate_cap != DEFAULT_CANDIDATE_CAP:
        raise ValueError("Held-out runs must use the frozen candidate cap")
    if args.overwrite:
        raise ValueError("Held-out outputs cannot be overwritten")


def validate_heldout_paths(
    args: argparse.Namespace,
    *,
    input_root: Path,
    variants_path: Path,
    output_root: Path,
) -> None:
    if args.split != "heldout_test":
        return
    expected = {
        "input root": DEFAULT_INPUT_ROOT.resolve(),
        "variant file": DEFAULT_VARIANTS.resolve(),
        "output root": DEFAULT_OUTPUT_ROOT.resolve(),
    }
    observed = {
        "input root": input_root,
        "variant file": variants_path,
        "output root": output_root,
    }
    for label, expected_path in expected.items():
        if observed[label] != expected_path:
            raise ValueError(
                f"Held-out runs require the canonical {label}: {expected_path}"
            )


def run_configuration(
    args: argparse.Namespace,
    *,
    input_hashes: dict[str, str],
    variant_hash: str,
) -> dict[str, Any]:
    code_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "analysis" / "scripts" / "common_evaluator.py",
        REPO_ROOT / "analysis" / "scripts" / "run_corrected_baseline_matrix.py",
        REPO_ROOT / "analysis" / "scripts" / "prepare_multigap_variants.py",
        STAGE2_SPEC,
        STAGE2_FREEZE,
        STAGE_REQUIREMENTS,
        INPUT_MANIFEST,
        SPLIT_MANIFEST,
    ]
    payload = {
        "schema_version": "multigap-benchmark-run-config-v1",
        "split": args.split,
        "candidate_cap": args.candidate_cap,
        "tokenizers": list(args.tokenizers),
        "methods": list(args.methods),
        "limit": args.limit,
        "variant_file_sha256": variant_hash,
        "record_encoding": {
            "result_fields": list(RESULT_FIELDS),
            "exact": "One result array.",
            "sequence": "Nested arrays in tokenizers-major, methods-minor order.",
            "source_recovery": "Join base_record_id to frozen BOAT or BIO-BOAT records.",
            "variant_recovery": "Join variant_id to the frozen Stage 2 variant manifest.",
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.candidate_cap < 1:
        raise ValueError("--candidate-cap must be positive")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    validate_heldout_arguments(args)

    input_root = args.input_root.expanduser().resolve()
    variants_path = args.variants.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    validate_heldout_paths(
        args,
        input_root=input_root,
        variants_path=variants_path,
        output_root=output_root,
    )
    _, input_hashes = load_and_validate_freeze(input_root)
    freeze, variants = load_and_validate_variants(variants_path)
    if input_hashes != freeze["input_hashes"]:
        raise ValueError("Frozen input hashes differ from the Stage 2 variant freeze")
    base_records = load_base_records(input_root)
    split_variants = [
        variant for variant in variants.values() if variant["split"] == args.split
    ]
    expected_ids = {variant["variant_id"] for variant in split_variants}
    if len(expected_ids) != freeze["split_rows"][args.split]:
        raise ValueError("Stage 2 split variant IDs are not unique and complete")

    configuration = run_configuration(
        args, input_hashes=input_hashes, variant_hash=sha256_file(variants_path)
    )
    output_dir = output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "multigap_benchmark_records.jsonl"
    config_path = output_dir / "multigap_benchmark_run_config.json"
    metadata_path = output_dir / "multigap_benchmark_run_metadata.json"
    ledger_path = output_dir / "heldout_run_ledger.json"
    if records_path.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"{records_path} already exists; pass --resume or --overwrite"
        )

    if args.split == "heldout_test":
        if ledger_path.exists():
            ledger = read_json(ledger_path)
            if ledger.get("status") == "completed":
                raise ValueError("The canonical held-out run is already complete")
            if ledger.get("status") != "started":
                raise ValueError("Unrecognized held-out ledger state")
            if not args.resume:
                raise ValueError("An interrupted held-out run requires --resume")
            if ledger.get("run_fingerprint") != configuration["run_fingerprint"]:
                raise ValueError("Held-out ledger fingerprint does not match")
        else:
            if args.resume:
                raise ValueError("Held-out resume requires an existing ledger")
            if records_path.exists() or config_path.exists():
                raise ValueError(
                    "Canonical held-out outputs exist without a run ledger"
                )
            write_json(
                ledger_path,
                {
                    "schema_version": "multigap-heldout-ledger-v1",
                    "status": "started",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "run_fingerprint": configuration["run_fingerprint"],
                    "variant_file_sha256": sha256_file(variants_path),
                },
            )

    already_done = strict_task_ids(records_path) if args.resume else set()
    unexpected = already_done - expected_ids
    if unexpected:
        raise ValueError(
            f"Existing output contains {len(unexpected)} task IDs outside the manifest"
        )
    if args.resume:
        if not config_path.exists() or not records_path.exists():
            raise FileNotFoundError(
                "A resumable run requires both the run config and records file"
            )
        if read_json(config_path) != configuration:
            raise ValueError("Resume configuration does not match the frozen run")
    else:
        write_json(config_path, configuration)

    mode = "a" if args.resume else "w"
    probes = {tokenizer: tokenizer_probe(tokenizer) for tokenizer in args.tokenizers}
    started_at = datetime.now(timezone.utc)
    evaluated = 0
    skipped = 0
    errors = 0
    started_clock = time.perf_counter()
    with records_path.open(mode, encoding="utf-8") as output:
        for variant in split_variants:
            task_id = variant["variant_id"]
            if task_id in already_done:
                skipped += 1
                continue
            if args.limit is not None and evaluated >= args.limit:
                break
            task = task_from_variant(variant, base_records)
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
                elapsed = time.perf_counter() - started_clock
                rate = evaluated / elapsed if elapsed else 0.0
                print(
                    f"{args.split}: {evaluated} tasks, {rate:.2f} tasks/s, "
                    f"{errors} method errors",
                    flush=True,
                )

    final_ids = strict_task_ids(records_path)
    complete = args.limit is None and final_ids == expected_ids
    if args.limit is None and not complete:
        missing = len(expected_ids - final_ids)
        extra = len(final_ids - expected_ids)
        raise ValueError(
            f"Completed Stage 2 output does not equal the manifest: "
            f"missing={missing}, extra={extra}"
        )
    ended_at = datetime.now(timezone.utc)
    metadata = {
        "schema_version": "multigap-benchmark-run-metadata-v1",
        "split": args.split,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": (ended_at - started_at).total_seconds(),
        "records_path": path_label(records_path),
        "records_sha256": sha256_file(records_path),
        "run_fingerprint": configuration["run_fingerprint"],
        "tasks_evaluated_this_run": evaluated,
        "tasks_skipped_by_resume": skipped,
        "tasks_in_records_file": len(final_ids),
        "tasks_expected_for_split": len(expected_ids),
        "method_errors_this_run": errors,
        "limit": args.limit,
        "candidate_cap": args.candidate_cap,
        "tokenizers": args.tokenizers,
        "methods": args.methods,
        "variant_file": {
            "path": path_label(variants_path),
            "sha256": sha256_file(variants_path),
        },
        "input_files": input_hashes,
        "stage2_spec": {
            "path": str(STAGE2_SPEC.relative_to(REPO_ROOT)),
            "sha256": sha256_file(STAGE2_SPEC),
        },
        "stage2_freeze": {
            "path": str(STAGE2_FREEZE.relative_to(REPO_ROOT)),
            "sha256": sha256_file(STAGE2_FREEZE),
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
        "complete": complete,
    }
    write_json(metadata_path, metadata)
    if args.split == "heldout_test" and complete:
        ledger = read_json(ledger_path)
        ledger.update(
            {
                "status": "completed",
                "completed_at": ended_at.isoformat(),
                "records_sha256": metadata["records_sha256"],
                "tasks": len(final_ids),
            }
        )
        write_json(ledger_path, ledger)
    return metadata


def main() -> None:
    metadata = run(parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
