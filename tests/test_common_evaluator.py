"""Tests for the isolated Stage 0 common evaluator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "scripts"))

from common_evaluator import _count_ordered_paths, evaluate_task


@pytest.fixture(scope="module")
def cl100k_available():
    try:
        evaluate_task(
            task_id="encoding-probe",
            source="probe FOX",
            target="FOX",
            method="taln",
            tokenizer="cl100k_base",
        )
    except Exception as exc:
        pytest.skip(f"cl100k_base is unavailable: {exc}")


@pytest.mark.usefixtures("cl100k_available")
@pytest.mark.parametrize(
    "method", ["taln", "lcs", "difflib", "semi_global_exact"]
)
def test_contextual_leading_space_is_symmetric_for_all_sequence_methods(method):
    result = evaluate_task(
        task_id=f"context-{method}",
        source="The marker is FOX.",
        target="FOX",
        method=method,
        tokenizer="cl100k_base",
    )
    assert [variant["name"] for variant in result["target_variants"]] == [
        "standalone",
        "prefixed_space",
    ]
    assert result["full_lexical_support"]
    assert any(
        candidate["target_variant"] == "prefixed_space"
        and candidate["full_lexical_support"]
        for candidate in result["candidates"]
    )


def test_partial_overlap_is_not_complete_alignment():
    result = evaluate_task(
        task_id="partial-only",
        source="alpha gamma",
        target="alpha beta",
        method="taln",
        tokenizer="whitespace",
    )
    assert result["partial_token_overlap"]
    assert not result["complete_token_alignment"]
    assert not result["full_lexical_support"]
    assert not result["localization_evaluable"]
    assert result["localization"] is None
    assert result["candidates"][0]["target_indices"] == (0,)


def test_repeated_adjacent_tokens_are_not_collapsed():
    result = evaluate_task(
        task_id="repeated-tokens",
        source="gene gene marker",
        target="gene gene",
        method="taln",
        tokenizer="whitespace",
    )
    assert result["complete_token_alignment"]
    assert result["full_lexical_support"]
    assert any(
        candidate["target_indices"] == (0, 1)
        and candidate["source_indices"] == (0, 1)
        for candidate in result["candidates"]
    )


def test_whitespace_noncontiguous_reconstruction_restores_separator():
    result = evaluate_task(
        task_id="noncontiguous-whitespace",
        source="alpha unrelated beta",
        target="alpha beta",
        method="taln",
        tokenizer="whitespace",
    )
    assert result["complete_token_alignment"]
    assert result["full_lexical_support"]
    assert any(candidate["reconstruction"] == "alpha beta" for candidate in result["candidates"])


def test_boundary_stripped_whitespace_removes_only_outer_punctuation():
    source = "marker (PDGFRA), cell-type"
    stripped = evaluate_task(
        task_id="boundary-stripped",
        source=source,
        target="PDGFRA",
        method="taln",
        tokenizer="boundary_stripped_whitespace",
    )
    raw = evaluate_task(
        task_id="boundary-raw",
        source=source,
        target="PDGFRA",
        method="taln",
        tokenizer="whitespace",
    )
    assert stripped["full_lexical_support"]
    assert not raw["full_lexical_support"]

    phrase = evaluate_task(
        task_id="boundary-stripped-phrase",
        source="at St. Mary Magdalene's Church today",
        target="St. Mary Magdalene's Church",
        method="taln",
        tokenizer="boundary_stripped_whitespace",
    )
    assert phrase["full_lexical_support"]
    assert phrase["candidates"][0]["reconstruction"] == (
        "St Mary Magdalene's Church"
    )
    assert phrase["candidates"][0]["target_reconstruction"] == (
        "St Mary Magdalene's Church"
    )


def test_semi_global_exact_accepts_a_long_negative_score_gap():
    source = "alpha " + " ".join(f"gap{i}" for i in range(8)) + " beta"
    result = evaluate_task(
        task_id="semi-global-long-gap",
        source=source,
        target="alpha beta",
        method="semi_global_exact",
        tokenizer="whitespace",
    )
    assert result["full_lexical_support"]
    assert result["candidates"][0]["alignment_score"] < 0


def test_semi_global_exact_rejects_reversed_target():
    result = evaluate_task(
        task_id="semi-global-reversed",
        source="alpha beta",
        target="beta alpha",
        method="semi_global_exact",
        tokenizer="whitespace",
    )
    assert not result["full_lexical_support"]


@pytest.mark.parametrize("method", ["taln", "lcs", "semi_global_exact"])
def test_token_sequence_support_is_not_path_whitespace_dependent(method):
    result = evaluate_task(
        task_id=f"punctuation-whitespace-{method}",
        source="alpha,beta and alpha , beta",
        target="alpha,beta",
        method=method,
        tokenizer="punctuation",
    )
    assert result["full_lexical_support"]


def test_compact_taln_scoring_counts_paths_and_checks_every_gold_location():
    source = "alpha beta filler alpha x beta"
    second_start = source.rindex("alpha")
    result = evaluate_task(
        task_id="compact-all-candidates",
        source=source,
        target="alpha beta",
        method="taln",
        tokenizer="whitespace",
        gold_intervals=[(second_start, len(source))],
        include_candidates=False,
    )
    assert result["candidate_count"] == 3
    assert result["materialized_candidate_count"] == 2
    assert result["full_lexical_support"]
    assert result["localization"]
    assert any(
        candidate["localization"] is True for candidate in result["candidates"]
    )
    assert result["unique_source_interval_count"] == 3


def test_path_count_truncation_requires_an_over_cap_complete_path():
    # Three paths reach source position 100 at the second target token, but
    # that saturated branch cannot reach the final token at position 50.
    position_lists = [[0, 10, 20], [5, 100], [50]]
    assert _count_ordered_paths(position_lists, cap=2) == (1, False)
    assert _count_ordered_paths([[0, 10, 20], [5, 100], [200]], cap=2) == (
        2,
        True,
    )


def test_localization_accepts_any_original_text_gold_interval():
    source = "alpha beta and alpha beta"
    second = source.rindex("alpha beta")
    result = evaluate_task(
        task_id="multiple-gold",
        source=source,
        target="alpha beta",
        method="exact",
        tokenizer="whitespace",
        gold_intervals=[(0, len("alpha beta")), (second, second + len("alpha beta"))],
    )
    assert result["candidate_count"] == 2
    assert result["localization"]
    assert {candidate["source_interval"] for candidate in result["candidates"]} == {
        (0, len("alpha beta")),
        (second, second + len("alpha beta")),
    }


def test_exact_substring_is_not_restricted_to_token_boundaries():
    source = "scatter"
    start = source.index("cat")
    result = evaluate_task(
        task_id="exact-inside-token",
        source=source,
        target="cat",
        method="exact",
        tokenizer="whitespace",
        gold_intervals=[(start, start + len("cat"))],
    )
    assert result["full_lexical_support"]
    assert result["localization"]


@pytest.mark.usefixtures("cl100k_available")
def test_contextual_prefix_localizes_the_canonical_target():
    source = "The marker is FOX."
    start = source.index("FOX")
    result = evaluate_task(
        task_id="contextual-localization",
        source=source,
        target="FOX",
        method="taln",
        tokenizer="cl100k_base",
        gold_intervals=[(start, start + len("FOX"))],
    )
    assert result["localization"]
    assert any(
        candidate["target_variant"] == "prefixed_space"
        and candidate["source_interval"] == (start, start + len("FOX"))
        for candidate in result["candidates"]
    )


@pytest.mark.parametrize("tokenizer", ["scibert", "pubmedbert"])
def test_pinned_biomedical_tokenizers_keep_original_offsets(tokenizer):
    source = "The marker is PDGFRA."
    start = source.index("PDGFRA")
    result = evaluate_task(
        task_id=f"biomedical-offset-{tokenizer}",
        source=source,
        target="PDGFRA",
        method="taln",
        tokenizer=tokenizer,
        gold_intervals=[(start, start + len("PDGFRA"))],
    )
    assert result["full_lexical_support"]
    assert result["localization"]
    assert [variant["name"] for variant in result["target_variants"]] == [
        "standalone"
    ]


@pytest.mark.parametrize("tokenizer", ["scibert", "pubmedbert"])
def test_uncased_biomedical_token_ids_do_not_erase_verbatim_case(tokenizer):
    result = evaluate_task(
        task_id=f"biomedical-case-{tokenizer}",
        source="PDGFRA",
        target="pdgfra",
        method="taln",
        tokenizer=tokenizer,
    )
    assert not result["complete_token_alignment"]
    assert not result["full_lexical_support"]


def test_offsets_are_in_original_text_after_normalization():
    source = "alpha\n\nbeta   gamma"
    start = source.index("beta")
    end = source.index("gamma") + len("gamma")
    result = evaluate_task(
        task_id="original-offsets",
        source=source,
        target="beta gamma",
        method="taln",
        tokenizer="whitespace",
        gold_intervals=[(start, end)],
    )
    candidate = next(
        candidate
        for candidate in result["candidates"]
        if candidate["full_lexical_support"]
    )
    assert candidate["source_offsets"] == ((start, start + 4), (end - 5, end))
    assert candidate["source_interval"] == (start, end)
    assert result["localization"]


def test_decomposed_unicode_offsets_are_in_original_text():
    source = "Cafe\u0301 alpha"
    result = evaluate_task(
        task_id="nfd-original-offsets",
        source=source,
        target="Cafe",
        method="taln",
        tokenizer="whitespace",
        gold_intervals=[(0, 5)],
    )
    candidate = next(
        candidate
        for candidate in result["candidates"]
        if candidate["full_lexical_support"]
    )

    assert candidate["source_offsets"] == ((0, 5),)
    assert candidate["source_interval"] == (0, 5)
    assert source[slice(*candidate["source_offsets"][0])] == "Cafe\u0301"
    assert result["localization"]


@pytest.mark.usefixtures("cl100k_available")
def test_transliteration_whitespace_keeps_auditable_original_offsets():
    source = "猫 alpha"
    result = evaluate_task(
        task_id="transliteration-original-offsets",
        source=source,
        target="alpha",
        method="taln",
        tokenizer="cl100k_base",
        gold_intervals=[(2, 7)],
    )
    candidate = next(
        candidate
        for candidate in result["candidates"]
        if candidate["full_lexical_support"] and candidate["localization"]
    )

    assert candidate["source_offsets"] == ((1, 7),)
    assert source[slice(*candidate["source_offsets"][0])] == " alpha"
    assert candidate["source_interval"] == (2, 7)
