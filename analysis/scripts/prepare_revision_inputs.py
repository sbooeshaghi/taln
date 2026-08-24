"""Freeze Stage 0 inputs and document-level splits for the 2026 revision.

This script leaves the legacy datasets untouched. It reconstructs stable
record/document identifiers from the upstream sources and writes auditable
revision artifacts under ``data/revision_2026`` plus tracked manifests under
``analysis/config/revision_2026``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from taln.taln_aln import norm_text
from taln.taln_extract import DEFAULT_PROMPT

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "revision_2026"
DEFAULT_CONFIG_ROOT = REPO_ROOT / "analysis" / "config" / "revision_2026"
DEFAULT_BIOBOAT_ROOT = REPO_ROOT / "tests" / "papers"
DEFAULT_LLMARKERS_ROOT = REPO_ROOT.parent / "llmarkers" / "data"
SCHEMA_VERSION = "revision-2026-stage0-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--bioboat-root", type=Path, default=DEFAULT_BIOBOAT_ROOT)
    parser.add_argument("--llmarkers-root", type=Path, default=DEFAULT_LLMARKERS_ROOT)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}:{sha256_text(payload)[:20]}"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def display_path(path: Path, external_root: Path | None = None) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        if external_root is not None:
            try:
                return "$LLMARKERS_DATA_ROOT/" + str(path.relative_to(external_root.resolve()))
            except ValueError:
                pass
        return str(path)


def source_file_entry(path: Path, external_root: Path | None = None) -> dict[str, Any]:
    return {
        "path": display_path(path, external_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def norm_length_preserved(source: str, target: str) -> bool:
    return len(norm_text(source)) == len(source) and len(norm_text(target)) == len(target)


def shifted_target(source: str, target: str, start: int) -> tuple[str, int]:
    if start > 0 and source[start - 1] == " ":
        return " " + target, start - 1
    return target, start


def split_by_blind_hash(document_ids: list[str], heldout_count: int) -> dict[str, str]:
    ranked = sorted(
        document_ids,
        key=lambda document_id: sha256_text(f"{SCHEMA_VERSION}\x1f{document_id}"),
    )
    heldout = set(ranked[:heldout_count])
    return {
        document_id: "heldout_test" if document_id in heldout else "development"
        for document_id in sorted(document_ids)
    }


def boat_records(path: Path, source_split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = read_json(path)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    counts: defaultdict[str, int] = defaultdict(int)

    for article_index, article in enumerate(raw["data"]):
        title = str(article.get("title", ""))
        document_id = stable_id("boat-document", source_split, article_index, title)
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            source = str(paragraph["context"])
            for qa in paragraph["qas"]:
                if qa.get("is_impossible", False):
                    continue
                question_id = str(qa.get("id", ""))
                for answer_index, answer in enumerate(qa.get("answers", [])):
                    counts["answer_rows"] += 1
                    target = str(answer["text"])
                    start = int(answer["answer_start"])
                    if source.find(target) != start:
                        counts["excluded_not_first_exact_occurrence"] += 1
                        continue
                    counts["first_exact_occurrence"] += 1
                    if not norm_length_preserved(source, target):
                        counts["excluded_normalization_length_change"] += 1
                        continue
                    shifted, shifted_start = shifted_target(source, target, start)
                    if len(shifted.split()) <= 2:
                        counts["excluded_target_at_most_two_whitespace_tokens"] += 1
                        continue

                    key = (document_id, source, shifted)
                    if key not in grouped:
                        grouped[key] = {
                            "dataset": "boat",
                            "source_split": source_split,
                            "document_id": document_id,
                            "article_index": article_index,
                            "article_title": title,
                            "paragraph_index": paragraph_index,
                            "source": source,
                            "source_sha256": sha256_text(source),
                            "target": target,
                            "target_sha256": sha256_text(target),
                            "contextual_target": shifted,
                            "gold_starts": set(),
                            "contextual_gold_starts": set(),
                            "source_question_ids": set(),
                            "source_answer_ids": set(),
                        }
                    row = grouped[key]
                    row["gold_starts"].add(start)
                    row["contextual_gold_starts"].add(shifted_start)
                    row["source_question_ids"].add(question_id)
                    row["source_answer_ids"].add(f"{question_id}:{answer_index}")

    records = []
    for (_, source, _), row in sorted(grouped.items()):
        target = row["target"]
        starts = sorted(row.pop("gold_starts"))
        row["gold_intervals_original"] = [
            {"start": start, "end": start + len(target)} for start in starts
        ]
        contextual_target = row["contextual_target"]
        contextual_starts = sorted(row.pop("contextual_gold_starts"))
        row["contextual_gold_intervals_original"] = [
            {"start": start, "end": start + len(contextual_target)}
            for start in contextual_starts
        ]
        row["source_question_ids"] = sorted(row["source_question_ids"])
        row["source_answer_ids"] = sorted(row["source_answer_ids"])
        row["record_id"] = stable_id("boat-record", row["document_id"], source, target)
        row["split"] = "development" if source_split == "train" else "heldout_test"
        row["target_variant"] = "contiguous"
        row["contextual_target_status"] = (
            "leading_space_added" if contextual_target != target else "unchanged"
        )
        row["generation_method"] = "human_annotated_squad_v2_answer"
        records.append(row)

    counts["final_records"] = len(records)
    counts["documents"] = len({row["document_id"] for row in records})
    return records, dict(sorted(counts.items()))


def parse_bib_field(text: str, field: str) -> str | None:
    match = re.search(
        rf"(?ims)^\s*{re.escape(field)}\s*=\s*[{{\"](.+?)[}}\"]\s*,?\s*$",
        text,
    )
    return " ".join(match.group(1).split()) if match else None


def bioboat_records(
    papers_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paper_dirs = sorted(path.parent for path in papers_root.glob("*/boat.json"))
    if not paper_dirs:
        raise FileNotFoundError(f"No BIO-BOAT paper directories found under {papers_root}")

    split = split_by_blind_hash([path.name for path in paper_dirs], heldout_count=10)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    documents = []
    counts: defaultdict[str, int] = defaultdict(int)

    for paper_dir in paper_dirs:
        paper_id = paper_dir.name
        boat_path = paper_dir / "boat.json"
        paper_path = paper_dir / "paper.md"
        bib_path = paper_dir / "paper.bib"
        bib_text = bib_path.read_text(encoding="utf-8")
        doi = parse_bib_field(bib_text, "doi")
        title = parse_bib_field(bib_text, "title")
        url = parse_bib_field(bib_text, "URL")

        documents.append(
            {
                "dataset": "bioboat",
                "document_id": paper_id,
                "split": split[paper_id],
                "title": title,
                "doi": doi,
                "url": url,
                "license": "CC BY 4.0",
                "license_evidence": "Selection criterion recorded in manuscript Methods; per-paper license evidence is not preserved locally.",
                "paper_markdown": source_file_entry(paper_path),
                "source_records": source_file_entry(boat_path),
                "bibliography": source_file_entry(bib_path),
            }
        )

        seen_expanded = set()
        for raw_index, raw in enumerate(read_json(boat_path)):
            source = str(raw["source"])
            target = str(raw["target"])
            starts = raw["idx_start"]
            if isinstance(starts, int):
                starts = [starts]
            for start_value in starts:
                counts["expanded_rows"] += 1
                start = int(start_value)
                if source.find(target) != start:
                    counts["excluded_not_first_exact_occurrence"] += 1
                    continue
                expanded_key = (paper_id, source, target, start)
                if expanded_key in seen_expanded:
                    counts["excluded_duplicate_expanded_row"] += 1
                    continue
                seen_expanded.add(expanded_key)
                if not norm_length_preserved(source, target):
                    counts["excluded_normalization_length_change"] += 1
                    continue
                shifted, shifted_start = shifted_target(source, target, start)
                if len(shifted.split()) <= 2:
                    counts["excluded_target_at_most_two_whitespace_tokens"] += 1
                    continue

                key = (paper_id, source, shifted)
                if key not in grouped:
                    grouped[key] = {
                        "dataset": "bioboat",
                        "document_id": paper_id,
                        "split": split[paper_id],
                        "source": source,
                        "source_sha256": sha256_text(source),
                        "target": target,
                        "target_sha256": sha256_text(target),
                        "contextual_target": shifted,
                        "gold_starts": set(),
                        "contextual_gold_starts": set(),
                        "source_raw_indices": set(),
                    }
                grouped[key]["gold_starts"].add(start)
                grouped[key]["contextual_gold_starts"].add(shifted_start)
                grouped[key]["source_raw_indices"].add(raw_index)

    records = []
    for (_, source, _), row in sorted(grouped.items()):
        target = row["target"]
        starts = sorted(row.pop("gold_starts"))
        row["gold_intervals_original"] = [
            {"start": start, "end": start + len(target)} for start in starts
        ]
        contextual_target = row["contextual_target"]
        contextual_starts = sorted(row.pop("contextual_gold_starts"))
        row["contextual_gold_intervals_original"] = [
            {"start": start, "end": start + len(contextual_target)}
            for start in contextual_starts
        ]
        row["source_raw_indices"] = sorted(row["source_raw_indices"])
        row["record_id"] = stable_id("bioboat-record", row["document_id"], source, target)
        row["target_variant"] = "contiguous"
        row["contextual_target_status"] = (
            "leading_space_added" if contextual_target != target else "unchanged"
        )
        row["generation_method"] = "llm_selected_exact_contiguous_source_text"
        records.append(row)

    counts["paper_directories"] = len(paper_dirs)
    counts["final_records"] = len(records)
    counts["development_documents"] = sum(
        document["split"] == "development" for document in documents
    )
    counts["heldout_test_documents"] = sum(
        document["split"] == "heldout_test" for document in documents
    )
    return records, documents, dict(sorted(counts.items()))


def llmarkers_records(
    data_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paths = sorted(data_root.glob("*/evidence_llm/extracted_txt.json"))
    if not paths:
        raise FileNotFoundError(f"No LLMarkers evidence files found under {data_root}")

    document_ids = [f"llmarkers:{path.parent.parent.name}" for path in paths]
    split = split_by_blind_hash(document_ids, heldout_count=2)
    records = []
    documents = []
    counts: defaultdict[str, int] = defaultdict(int)

    for path in paths:
        dataset_id = path.parent.parent.name
        document_id = f"llmarkers:{dataset_id}"
        metadata_path = path.parent.parent / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else None
        documents.append(
            {
                "dataset": "llmarkers",
                "document_id": document_id,
                "dataset_id": dataset_id,
                "split": split[document_id],
                "records_file": source_file_entry(path, data_root),
                "metadata_file": source_file_entry(metadata_path, data_root)
                if metadata_path.exists()
                else None,
                "metadata": metadata,
            }
        )

        rows = read_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list in {path}")
        for record_index, raw in enumerate(rows):
            verification = raw.get("_verification") or {}
            source = str(raw.get("source_rationale") or "")
            group_label = str(raw.get("group_label") or "")
            feature_label = str(raw.get("feature_label") or "")
            record = {
                "dataset": "llmarkers",
                "dataset_id": dataset_id,
                "document_id": document_id,
                "split": split[document_id],
                "record_index": record_index,
                "record_id": stable_id(
                    "llmarkers-record",
                    document_id,
                    record_index,
                    source,
                    group_label,
                    feature_label,
                ),
                "source_id": raw.get("source_id"),
                "data_id": raw.get("data_id"),
                "source_rationale": source,
                "source_sha256": sha256_text(source),
                "group_label": group_label,
                "feature_label": feature_label,
                "all_verified": verification.get("all_verified"),
                "verification": verification,
                "inventory_status": "not_independent_gold",
            }
            records.append(record)
            counts["records"] += 1
            counts["all_verified"] += verification.get("all_verified") is True

    counts["documents"] = len(documents)
    counts["development_documents"] = sum(
        document["split"] == "development" for document in documents
    )
    counts["heldout_test_documents"] = sum(
        document["split"] == "heldout_test" for document in documents
    )
    return records, documents, dict(sorted(counts.items()))


def artifact_entry(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    config_root = args.config_root.expanduser().resolve()
    bioboat_root = args.bioboat_root.expanduser().resolve()
    llmarkers_root = args.llmarkers_root.expanduser().resolve()

    train_path = REPO_ROOT / "data" / "train-v2.0.json"
    dev_path = REPO_ROOT / "data" / "dev-v2.0.json"
    boat_train, boat_train_counts = boat_records(train_path, "train")
    boat_dev, boat_dev_counts = boat_records(dev_path, "dev")
    boat = boat_train + boat_dev
    bioboat, bioboat_documents, bioboat_counts = bioboat_records(bioboat_root)
    llmarkers, llmarkers_documents, llmarkers_counts = llmarkers_records(llmarkers_root)

    boat_output = output_root / "boat_records.jsonl"
    bioboat_output = output_root / "bioboat_records.jsonl"
    llmarkers_output = output_root / "llmarkers_inventory.jsonl"
    boat_rows = write_jsonl(boat_output, boat)
    bioboat_rows = write_jsonl(bioboat_output, bioboat)
    llmarkers_rows = write_jsonl(llmarkers_output, llmarkers)

    prompt_path = config_root / "bioboat_extraction_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(DEFAULT_PROMPT.rstrip() + "\n", encoding="utf-8")

    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "freeze_policy": {
            "boat": "Official SQuAD train is development; official SQuAD dev is held-out test.",
            "bioboat": "Exactly 10 of 51 paper IDs selected by ascending SHA-256 of schema-version and document ID; no record contents or outcomes used.",
            "llmarkers": "Exactly 2 of 7 dataset IDs selected by ascending SHA-256 of schema-version and document ID; no record contents or outcomes used.",
        },
        "documents": sorted(
            [
                {
                    "dataset": "boat",
                    "document_id": row["document_id"],
                    "split": row["split"],
                }
                for row in {
                    record["document_id"]: record for record in boat
                }.values()
            ]
            + [
                {
                    "dataset": "bioboat",
                    "document_id": row["document_id"],
                    "split": row["split"],
                }
                for row in bioboat_documents
            ]
            + [
                {
                    "dataset": "llmarkers",
                    "document_id": row["document_id"],
                    "split": row["split"],
                }
                for row in llmarkers_documents
            ],
            key=lambda row: (row["dataset"], row["document_id"]),
        ),
    }

    bioboat_mtimes = [
        (path.parent / "boat.json").stat().st_mtime
        for path in sorted(bioboat_root.glob("*/boat.json"))
    ]
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_collection": {
            "archive": "s3://biorxiv-src-monthly/Current_Content/",
            "archive_month": "2025-01",
            "selection_count": 51,
            "selection_license": "CC BY 4.0",
            "selection_rule_status": "recorded in manuscript Methods; selection script and per-paper license evidence not preserved",
        },
        "conversion": {
            "input": "bioRxiv source XML",
            "output": "paper.md",
            "exact_command": None,
            "status": "output preserved; exact conversion command/version not preserved",
        },
        "llm_extraction": {
            "tool": "taln extract",
            "output_file_mtime_utc_range": [
                datetime.fromtimestamp(min(bioboat_mtimes), tz=timezone.utc).isoformat(),
                datetime.fromtimestamp(max(bioboat_mtimes), tz=timezone.utc).isoformat(),
            ],
            "model": "claude-sonnet-4-20250514",
            "model_status": "confirmed by the author on 2026-08-12; the executed command was not preserved",
            "prompt_snapshot": display_path(prompt_path),
            "prompt_sha256": sha256_file(prompt_path),
            "prompt_status": "use of this default prompt was confirmed by the author on 2026-08-12; the executed command was not preserved",
            "max_output_tokens": 16000,
            "max_output_tokens_status": "inferred default; executed command not preserved",
            "raw_api_responses_preserved": False,
        },
        "verification_and_filtering": {
            "initial_verification": "taln extract retained targets found as exact contiguous substrings, with a whitespace-normalized fallback.",
            "benchmark_filter": "Retain rows where source.find(target) equals idx_start, deduplicate expanded rows, require normalization to preserve source/target lengths, shift a preceding ASCII space into the target, and require more than two whitespace-delimited target tokens.",
            "quality_control": "Deterministic substring verification only; no preserved independent human quality-control labels.",
        },
        "documents": bioboat_documents,
    }

    input_manifest = {
        "schema_version": SCHEMA_VERSION,
        "taln_repository_commit": git_commit(REPO_ROOT),
        "llmarkers_repository_commit": git_commit(llmarkers_root.parent),
        "path_locators": {
            "repository_root": ".",
            "llmarkers_data_root_environment_variable": "LLMARKERS_DATA_ROOT",
            "llmarkers_data_root_default": "../llmarkers/data",
        },
        "datasets": {
            "boat": {
                "provenance": "SQuAD v2.0 human answers; CC BY-SA 4.0.",
                "normalization_and_filtering": "Legacy BOAT cohort reconstructed from raw SQuAD using the documented exact-first-occurrence, length-preserving normalization, contextual-space shift, and target-length filters.",
                "source_files": [source_file_entry(train_path), source_file_entry(dev_path)],
                "counts": {"train": boat_train_counts, "dev": boat_dev_counts},
                "artifact": artifact_entry(boat_output, boat_rows),
            },
            "bioboat": {
                "provenance": "51 bioRxiv papers from the January 2025 source archive; LLM-selected targets deterministically filtered to exact source substrings.",
                "counts": bioboat_counts,
                "artifact": artifact_entry(bioboat_output, bioboat_rows),
                "provenance_manifest": "analysis/config/revision_2026/bioboat_provenance.json",
            },
            "llmarkers": {
                "provenance": "Seven tracked LLMarkers evidence_llm datasets. Existing _verification fields are inventory metadata, not independent gold labels.",
                "counts": llmarkers_counts,
                "source_documents": llmarkers_documents,
                "artifact": artifact_entry(llmarkers_output, llmarkers_rows),
            },
        },
    }

    write_json(config_root / "input_manifest.json", input_manifest)
    write_json(config_root / "split_manifest.json", split_manifest)
    write_json(config_root / "bioboat_provenance.json", provenance)

    package_versions = {}
    for package in ["taln", "tiktoken", "unidecode"]:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    run_metadata_path = output_root / "stage0_run_metadata.json"
    write_json(
        run_metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "run_timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "taln_repository_commit": git_commit(REPO_ROOT),
            "llmarkers_repository_commit": git_commit(llmarkers_root.parent),
            "input_manifest_sha256": sha256_file(config_root / "input_manifest.json"),
            "split_manifest_sha256": sha256_file(config_root / "split_manifest.json"),
            "python": sys.version,
            "packages": package_versions,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "random_seed": None,
        },
    )

    report = {
        "input_manifest": display_path(config_root / "input_manifest.json"),
        "split_manifest": display_path(config_root / "split_manifest.json"),
        "run_metadata": display_path(run_metadata_path),
        "boat_records": boat_rows,
        "bioboat_records": bioboat_rows,
        "llmarkers_records": llmarkers_rows,
        "counts": {
            "boat_train": boat_train_counts,
            "boat_dev": boat_dev_counts,
            "bioboat": bioboat_counts,
            "llmarkers": llmarkers_counts,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
