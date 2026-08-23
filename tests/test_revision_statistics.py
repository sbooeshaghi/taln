"""Focused tests for clustered revision statistics."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "scripts"))

from revision_statistics import (
    document_clustered_binary_ci,
    mcnemar_descriptive,
    paired_document_clustered_difference_ci,
    paired_document_sign_flip_test,
)


def test_document_clustered_binary_ci_is_deterministic_and_record_weighted():
    document_ids = ["short", "long", "long", "long"]
    outcomes = [True, False, False, False]
    first = document_clustered_binary_ci(document_ids, outcomes, n_resamples=1_000, seed=7)
    second = document_clustered_binary_ci(document_ids, outcomes, n_resamples=1_000, seed=7)

    assert first == second
    assert first["estimate"] == pytest.approx(0.25)
    assert first["n_documents"] == 2
    assert first["n_records"] == 4
    assert 0.0 <= first["ci_lower"] <= first["ci_upper"] <= 1.0


def test_paired_clustered_difference_ci_resamples_documents_together():
    document_ids = ["a", "a", "b", "b"]
    reference = [False, False, True, True]
    comparison = [True, True, True, True]
    result = paired_document_clustered_difference_ci(
        document_ids, reference, comparison, n_resamples=1_000, seed=4
    )

    assert result["contrast"] == "comparison_minus_reference"
    assert result["estimate"] == pytest.approx(0.5)
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]
    assert result["n_documents"] == 2


def test_sign_flip_uses_exact_enumeration_when_feasible():
    document_ids = ["a", "b"]
    reference = [False, False]
    comparison = [True, True]
    result = paired_document_sign_flip_test(document_ids, reference, comparison)

    assert result["method"] == "exact"
    assert result["n_randomizations"] == 4
    assert result["estimate"] == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(0.5)


def test_sign_flip_uses_same_record_weighted_estimand_as_paired_ci():
    document_ids = ["small", "large", "large", "large"]
    reference = [False, True, True, True]
    comparison = [True, False, True, True]
    interval = paired_document_clustered_difference_ci(
        document_ids, reference, comparison, n_resamples=100, seed=3
    )
    test = paired_document_sign_flip_test(
        document_ids, reference, comparison
    )
    assert interval["estimate"] == pytest.approx(0.0)
    assert test["estimate"] == pytest.approx(interval["estimate"])
    assert test["statistic"] == "record_weighted_accuracy_difference"


def test_sign_flip_monte_carlo_fallback_is_deterministic():
    document_ids = [f"doc-{index}" for index in range(4)]
    reference = [False] * 4
    comparison = [True] * 4
    first = paired_document_sign_flip_test(
        document_ids,
        reference,
        comparison,
        max_exact_documents=2,
        monte_carlo_samples=1_000,
        seed=11,
    )
    second = paired_document_sign_flip_test(
        document_ids,
        reference,
        comparison,
        max_exact_documents=2,
        monte_carlo_samples=1_000,
        seed=11,
    )

    assert first == second
    assert first["method"] == "monte_carlo"
    assert first["n_randomizations"] == 1_000
    assert 0.0 < first["p_value"] <= 1.0


def test_mcnemar_is_explicitly_descriptive_and_uses_exact_binomial_test():
    reference = [False, False, False, True]
    comparison = [True, True, True, True]
    result = mcnemar_descriptive(reference, comparison)

    assert result["analysis"] == "record_level_mcnemar_descriptive"
    assert not result["primary_analysis"]
    assert result["comparison_only_successes"] == 3
    assert result["reference_only_successes"] == 0
    assert result["discordant_records"] == 3
    assert result["exact_binomial_p_value"] == pytest.approx(0.25)


def test_mcnemar_exact_binomial_is_stable_for_large_samples():
    reference = [True] * 20_000
    comparison = [False] * 20_000
    result = mcnemar_descriptive(reference, comparison)
    assert result["discordant_records"] == 20_000
    assert 0.0 <= result["exact_binomial_p_value"] < 1e-100


def test_statistics_reject_misaligned_or_nonbinary_inputs():
    with pytest.raises(ValueError, match="equal"):
        paired_document_clustered_difference_ci(["a"], [True], [True, False])
    with pytest.raises(ValueError, match="binary"):
        document_clustered_binary_ci(["a"], [2])
