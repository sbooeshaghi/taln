"""Run the frozen LLM lexical-baseline experiment.

Implements analysis/config/revision_2026/llm_baseline_spec.json. The model
performs the same lexical task as the alignment methods: given the source and
the submitted target, return the verbatim source excerpts that contain the
target's words in order. Returned excerpts are located in the source by
normalized exact search (never by model-reported positions), scored with the
shared evaluator contract, and compared symmetrically against the frozen
alignment-method results on the identical sampled tasks.

Raw API responses are written immediately and are immutable. Scoring is a
deterministic pure function of the raw responses, so it can be repeated
without new API calls via --rescore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from common_evaluator import evaluate_task, tokenize_text
from run_corrected_baseline_matrix import (
    package_version,
    read_jsonl,
    sha256_file,
    write_json,
)
from run_selection_assessment import bootstrap_ci, repo_relative

from taln.taln_aln import norm_text_with_mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "revision_2026"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "llm_baseline"
CONFIG_DIR = REPO_ROOT / "analysis" / "config" / "revision_2026"
SPEC_PATH = CONFIG_DIR / "llm_baseline_spec.json"
PROMPT_PATH = CONFIG_DIR / "llm_baseline_prompt.txt"

MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 1500
TEMPERATURE = 0.0
MAX_TRANSIENT_RETRIES = 3
MAX_EXCERPT_OCCURRENCES = 50
MAX_COMBINATIONS = 1000
BOOTSTRAP_SEED = 20260814
COMPARISON_METHODS = ("exact", "taln", "lcs", "difflib", "semi_global_exact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("development", "heldout_test", "heldout_multigap"),
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--limit", type=int, help="Development only: call at most N tasks."
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-score existing raw responses without any API call.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Call only tasks absent from the existing raw-response file.",
    )
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        from dotenv import dotenv_values

        key = dotenv_values(env_path).get("ANTHROPIC_API_KEY")
        if key:
            return key
    raise RuntimeError("ANTHROPIC_API_KEY not found in environment or .env")


def validate_heldout(args: argparse.Namespace, prompt_sha256: str) -> None:
    if not args.phase.startswith("heldout"):
        return
    if args.limit is not None:
        raise ValueError("Held-out runs cannot use --limit")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    frozen_prompt = spec.get("frozen_prompt_sha256")
    if not frozen_prompt:
        raise ValueError(
            "The spec does not record frozen_prompt_sha256; freeze the prompt "
            "after development before a held-out run"
        )
    if frozen_prompt != prompt_sha256:
        raise ValueError(
            "Prompt file does not match the frozen prompt hash in the spec"
        )


def load_sample(
    output_root: Path, phase: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if phase == "heldout_multigap":
        manifest_path = output_root / "llm_baseline_multigap_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest
    else:
        manifest_path = output_root / "llm_baseline_sample_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest[
            "heldout" if phase == "heldout_test" else "development"
        ]
    tasks_path = output_root / Path(entry["path"]).name
    if sha256_file(tasks_path) != entry["sha256"]:
        raise ValueError(f"Sample file hash mismatch: {tasks_path}")
    return list(read_jsonl(tasks_path)), manifest


def build_prompt(template: str, source: str, target: str) -> str:
    if "<<SOURCE>>" not in template or "<<TARGET>>" not in template:
        raise ValueError("Prompt template is missing <<SOURCE>> or <<TARGET>>")
    return template.replace("<<SOURCE>>", source).replace("<<TARGET>>", target)


def _validate_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    supported = payload.get("supported")
    excerpts = payload.get("excerpts")
    if not isinstance(supported, bool):
        return None
    if not isinstance(excerpts, list) or not all(
        isinstance(item, str) for item in excerpts
    ):
        return None
    return {"supported": supported, "excerpts": excerpts}


def parse_response(text: str) -> dict[str, Any] | None:
    """Parse the model's final JSON object; None means unparseable.

    The prompt allows brief reasoning before the JSON, so parsing targets the
    last JSON object in the reply.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    candidates = [stripped]
    marker = stripped.rfind('"supported"')
    if marker != -1:
        start = stripped.rfind("{", 0, marker)
        if start != -1:
            candidates.insert(0, stripped[start:])
    for candidate in candidates:
        for use_repair in (False, True):
            try:
                if use_repair:
                    from json_repair import repair_json

                    payload = json.loads(repair_json(candidate))
                else:
                    payload = json.loads(candidate)
            except Exception:
                continue
            validated = _validate_payload(payload)
            if validated is not None:
                return validated
    return None


def locate_excerpt(
    normalized_source: str, excerpt: str
) -> list[tuple[int, int]]:
    """All normalized-coordinate occurrences of a verbatim excerpt."""
    normalized_excerpt, _ = norm_text_with_mapping(excerpt)
    if not normalized_excerpt:
        return []
    occurrences = []
    start = normalized_source.find(normalized_excerpt)
    while start != -1 and len(occurrences) < MAX_EXCERPT_OCCURRENCES:
        occurrences.append((start, start + len(normalized_excerpt)))
        start = normalized_source.find(normalized_excerpt, start + 1)
    return occurrences


def ordered_combinations(
    occurrence_lists: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    """Non-overlapping, in-order occurrence assignments, bounded."""
    combinations: list[list[tuple[int, int]]] = []

    def visit(index: int, chosen: list[tuple[int, int]]) -> None:
        if len(combinations) >= MAX_COMBINATIONS:
            return
        if index == len(occurrence_lists):
            combinations.append(list(chosen))
            return
        floor = chosen[-1][1] if chosen else 0
        for occurrence in occurrence_lists[index]:
            if occurrence[0] >= floor:
                chosen.append(occurrence)
                visit(index + 1, chosen)
                chosen.pop()
            if len(combinations) >= MAX_COMBINATIONS:
                return

    visit(0, [])
    return combinations


def region_localization(
    source: str,
    target: str,
    regions: list[tuple[int, int]],
    gold_intervals: list[tuple[int, int]],
) -> bool:
    """Does any complete ordered target alignment inside the regions have a
    gold outer original-source interval?"""
    source_tokenized = tokenize_text(source, "punctuation")
    target_tokenized = tokenize_text(target, "punctuation")
    if not target_tokenized.tokens:
        return False
    allowed = [
        index
        for index, token in enumerate(source_tokenized.tokens)
        if any(
            token.norm_start >= start and token.norm_end <= end
            for start, end in regions
        )
    ]
    allowed_set = set(allowed)
    position_lists = []
    for target_token in target_tokenized.tokens:
        positions = [
            index
            for index, token in enumerate(source_tokenized.tokens)
            if index in allowed_set and token.token_id == target_token.token_id
        ]
        if not positions:
            return False
        position_lists.append(positions)

    gold = set(gold_intervals)
    for first in position_lists[0]:
        for last in position_lists[-1]:
            if len(position_lists) > 1 and last <= first:
                continue
            outer = (
                source_tokenized.tokens[first].start,
                source_tokenized.tokens[last if len(position_lists) > 1 else first].end,
            )
            if outer not in gold:
                continue
            previous = first
            feasible = True
            for positions in position_lists[1:-1]:
                next_position = next(
                    (p for p in positions if previous < p < last), None
                )
                if next_position is None:
                    feasible = False
                    break
                previous = next_position
            if feasible:
                return True
    return False


def score_task(task: dict[str, Any], parsed: dict[str, Any] | None) -> dict[str, Any]:
    """Deterministic scoring of one raw model reply."""
    gold_intervals = [
        (int(interval[0]), int(interval[1]))
        if not isinstance(interval, dict)
        else (int(interval["start"]), int(interval["end"]))
        for interval in task["gold_intervals"] or []
    ]
    outcome: dict[str, Any] = {
        "parse_ok": parsed is not None,
        "declared_supported": None,
        "excerpt_count": None,
        "excerpts_located_in_order": False,
        "llm_reconstruction": False,
        "llm_localization": False if gold_intervals else None,
        "failure_cause": None,
    }
    if parsed is None:
        outcome["failure_cause"] = (
            "model_refusal_or_empty_reply"
            if task.get("_refused")
            else "unparseable_response"
        )
        return outcome
    outcome["declared_supported"] = parsed["supported"]
    outcome["excerpt_count"] = len(parsed["excerpts"])
    if not parsed["supported"]:
        outcome["failure_cause"] = "declared_unsupported"
        return outcome
    excerpts = [excerpt for excerpt in parsed["excerpts"] if excerpt.strip()]
    if not excerpts:
        outcome["failure_cause"] = "no_excerpts"
        return outcome

    normalized_source, _ = norm_text_with_mapping(task["source"])
    occurrence_lists = [
        locate_excerpt(normalized_source, excerpt) for excerpt in excerpts
    ]
    if any(not occurrences for occurrences in occurrence_lists):
        outcome["failure_cause"] = "excerpt_not_verbatim_in_source"
        return outcome
    combinations = ordered_combinations(occurrence_lists)
    if not combinations:
        outcome["failure_cause"] = "excerpts_not_in_source_order"
        return outcome
    outcome["excerpts_located_in_order"] = True

    concatenated = " ".join(excerpts)
    try:
        evidence = evaluate_task(
            task_id=f"llm:{task['task_id']}",
            source=concatenated,
            target=task["target"],
            method="taln",
            tokenizer="punctuation",
            gold_intervals=None,
            candidate_cap=10_000,
            include_candidates=False,
        )
    except ValueError:
        outcome["failure_cause"] = "evidence_evaluation_error"
        return outcome
    if not evidence["full_lexical_support"]:
        outcome["failure_cause"] = "excerpts_do_not_support_target"
        return outcome
    outcome["llm_reconstruction"] = True

    if gold_intervals:
        localized = any(
            region_localization(
                task["source"], task["target"], combination, gold_intervals
            )
            for combination in combinations
        )
        outcome["llm_localization"] = localized
        if not localized:
            outcome["failure_cause"] = "evidence_outside_gold_location"
    return outcome


def load_frozen_method_results(
    input_root: Path, tasks: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Frozen punctuation-tokenizer method outcomes for the sampled tasks."""
    wanted = {task["task_id"] for task in tasks}
    results: dict[str, dict[str, Any]] = {}
    sources = (
        ("stage1", "corrected_baseline_records.jsonl", "corrected_baseline_run_config.json"),
        ("stage2", "multigap_benchmark_records.jsonl", "multigap_benchmark_run_config.json"),
    )
    splits = {task["split"] for task in tasks}
    for split in sorted(splits):
        for stage, records_name, config_name in sources:
            config = json.loads(
                (input_root / stage / split / config_name).read_text(
                    encoding="utf-8"
                )
            )
            tokenizer_index = list(config["tokenizers"]).index("punctuation")
            method_indices = {
                method: list(config["methods"]).index(method)
                for method in COMPARISON_METHODS
                if method in config["methods"]
            }
            for row in read_jsonl(input_root / stage / split / records_name):
                if row["task_id"] not in wanted:
                    continue
                entry: dict[str, Any] = {}
                entry["exact"] = {
                    "full_support": bool(row["exact"][0]),
                    "localization": row["exact"][3],
                }
                for method, method_index in method_indices.items():
                    result = row["sequence"][tokenizer_index][method_index]
                    entry[method] = {
                        "full_support": bool(result[0]),
                        "localization": result[3],
                    }
                results[row["task_id"]] = entry
    return results


def call_model(
    client: Any, prompt: str
) -> tuple[str | None, dict[str, Any]]:
    """One API call with bounded retries on transient errors only."""
    import anthropic

    non_transient = (
        anthropic.BadRequestError,
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
    )
    last_error = None
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "stop_reason": message.stop_reason,
            }
            text_blocks = [
                block.text
                for block in message.content
                if getattr(block, "type", None) == "text"
            ]
            if not text_blocks:
                # A refusal (or otherwise empty reply) is a terminal model
                # outcome, not a transport error; it is scored as a failure.
                return None, usage
            return "".join(text_blocks), usage
        except non_transient as exc:
            return None, {"error": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - retry transient API errors
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2**attempt * 2, 30))
    return None, {"error": last_error}


def summarize(
    tasks: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    method_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        key = f"{task['stage']}:{task['dataset']}:{task['condition']}"
        strata.setdefault(key, []).append(task)

    def rate_block(
        members: list[dict[str, Any]], metric: dict[str, bool]
    ) -> dict[str, Any]:
        by_document: dict[str, tuple[int, int]] = {}
        hits = 0
        for task in members:
            hit = int(bool(metric.get(task["task_id"], False)))
            hits += hit
            doc_hits, doc_total = by_document.get(task["document_id"], (0, 0))
            by_document[task["document_id"]] = (doc_hits + hit, doc_total + 1)
        n = len(members)
        low, high = bootstrap_ci(by_document, rng)
        return {
            "hits": hits,
            "n": n,
            "rate": hits / n if n else None,
            "ci95": [low, high],
        }

    payload: dict[str, Any] = {}
    for key, members in sorted(strata.items()):
        llm_reconstruction = {
            task["task_id"]: outcomes[task["task_id"]]["llm_reconstruction"]
            for task in members
        }
        llm_localization = {
            task["task_id"]: bool(outcomes[task["task_id"]]["llm_localization"])
            for task in members
        }
        failure_causes: dict[str, int] = {}
        for task in members:
            cause = outcomes[task["task_id"]]["failure_cause"]
            if cause:
                failure_causes[cause] = failure_causes.get(cause, 0) + 1
        block = {
            "tasks": len(members),
            "llm": {
                "reconstruction": rate_block(members, llm_reconstruction),
                "localization": rate_block(members, llm_localization),
                "failure_causes": dict(sorted(failure_causes.items())),
            },
            "methods": {},
        }
        for method in COMPARISON_METHODS:
            support_metric = {
                task["task_id"]: method_results.get(task["task_id"], {})
                .get(method, {})
                .get("full_support", False)
                for task in members
            }
            localization_metric = {
                task["task_id"]: bool(
                    method_results.get(task["task_id"], {})
                    .get(method, {})
                    .get("localization")
                )
                for task in members
            }
            block["methods"][method] = {
                "reconstruction": rate_block(members, support_metric),
                "localization": rate_block(members, localization_metric),
            }
        payload[key] = block
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha256 = sha256_text(prompt_template)
    validate_heldout(args, prompt_sha256)

    tasks, manifest = load_sample(output_root, args.phase)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    output_dir = output_root / args.phase
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{args.phase}_raw_responses.jsonl"
    records_path = output_dir / "llm_baseline_records.jsonl"
    summary_path = output_dir / "llm_baseline_summary.json"
    metadata_path = output_dir / "llm_baseline_run_metadata.json"

    existing_raw: dict[str, dict[str, Any]] = {}
    if raw_path.exists():
        for row in read_jsonl(raw_path):
            existing_raw[row["task_id"]] = row

    def has_current_response(task_id: str) -> bool:
        row = existing_raw.get(task_id)
        if row is None or row.get("prompt_sha256") != prompt_sha256:
            return False
        if row.get("response_text") is not None:
            return True
        # Model refusals and empty replies are terminal outcomes, not gaps.
        return row.get("usage", {}).get("stop_reason") is not None

    if args.rescore:
        pending: list[dict[str, Any]] = []
    else:
        if raw_path.exists() and not args.resume:
            raise FileExistsError(
                f"{raw_path} already exists; pass --resume or --rescore"
            )
        pending = [
            task for task in tasks if not has_current_response(task["task_id"])
        ]

    estimated_input_tokens = sum(
        (len(prompt_template) + len(task["source"]) + len(task["target"])) // 4
        for task in pending
    )
    estimated_cost_usd = round(
        estimated_input_tokens / 1e6 * 3.0 + len(pending) * 200 / 1e6 * 15.0, 2
    )
    print(
        f"phase={args.phase} tasks={len(tasks)} pending_calls={len(pending)} "
        f"estimated_input_tokens={estimated_input_tokens} "
        f"estimated_cost_usd={estimated_cost_usd}",
        flush=True,
    )

    started_at = datetime.now(timezone.utc)
    usage_totals = {"input_tokens": 0, "output_tokens": 0}
    if pending:
        import anthropic

        client = anthropic.Anthropic(api_key=load_api_key())
        lock_write = raw_path.open("a", encoding="utf-8")

        def process(task: dict[str, Any]) -> dict[str, Any]:
            prompt = build_prompt(
                prompt_template, task["source"], task["target"]
            )
            text, usage = call_model(client, prompt)
            return {
                "schema_version": "llm-baseline-raw-response-v1",
                "task_id": task["task_id"],
                "phase": args.phase,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "prompt_sha256": prompt_sha256,
                "requested_at": datetime.now(timezone.utc).isoformat(),
                "response_text": text,
                "usage": usage,
            }

        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(process, pending):
                lock_write.write(
                    json.dumps(row, separators=(",", ":")) + "\n"
                )
                lock_write.flush()
                existing_raw[row["task_id"]] = row
                for field in ("input_tokens", "output_tokens"):
                    usage_totals[field] += row["usage"].get(field, 0)
                completed += 1
                if completed % 100 == 0:
                    print(
                        f"{completed}/{len(pending)} calls", flush=True
                    )
        lock_write.close()

    missing = [
        task["task_id"]
        for task in tasks
        if not has_current_response(task["task_id"])
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} sampled tasks have no successful raw response "
            "under the current prompt; resume the run before scoring"
        )

    outcomes: dict[str, dict[str, Any]] = {}
    with records_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            raw = existing_raw[task["task_id"]]
            parsed = (
                parse_response(raw["response_text"])
                if raw["response_text"] is not None
                else None
            )
            task["_refused"] = raw["response_text"] is None
            outcome = score_task(task, parsed)
            outcomes[task["task_id"]] = outcome
            handle.write(
                json.dumps(
                    {
                        "schema_version": "llm-baseline-record-v1",
                        "task_id": task["task_id"],
                        "stage": task["stage"],
                        "dataset": task["dataset"],
                        "condition": task["condition"],
                        "split": task["split"],
                        "document_id": task["document_id"],
                        **outcome,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

    method_results = load_frozen_method_results(input_root, tasks)
    summary = {
        "schema_version": "llm-baseline-summary-v1",
        "phase": args.phase,
        "model": MODEL,
        "prompt_sha256": prompt_sha256,
        "boundary": (
            "All sampled tasks have complete lexical support by construction; "
            "this benchmark measures evidence production and localization, "
            "not rejection of unsupported text."
        ),
        "strata": summarize(tasks, outcomes, method_results),
    }
    write_json(summary_path, summary)

    ended_at = datetime.now(timezone.utc)
    metadata = {
        "schema_version": "llm-baseline-run-metadata-v1",
        "phase": args.phase,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "prompt_path": repo_relative(PROMPT_PATH),
        "prompt_sha256": prompt_sha256,
        "sample_manifest_sha256": sha256_file(
            output_root
            / (
                "llm_baseline_multigap_manifest.json"
                if args.phase == "heldout_multigap"
                else "llm_baseline_sample_manifest.json"
            )
        ),
        "tasks": len(tasks),
        "api_calls_this_run": len(pending),
        "estimated_cost_usd_before_run": estimated_cost_usd,
        "usage_totals_this_run": usage_totals,
        "raw_path": repo_relative(raw_path),
        "raw_sha256": sha256_file(raw_path) if raw_path.exists() else None,
        "records_path": repo_relative(records_path),
        "records_sha256": sha256_file(records_path),
        "summary_path": repo_relative(summary_path),
        "limit": args.limit,
        "rescore": args.rescore,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "anthropic": package_version("anthropic"),
            "taln": package_version("taln"),
        },
        "command": sys.argv,
    }
    write_json(metadata_path, metadata)
    return metadata


def main() -> None:
    metadata = run(parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
