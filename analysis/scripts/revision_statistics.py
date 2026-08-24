"""Deterministic clustered inference helpers for the revision experiments.

The primary analyses resample documents, preserving every record from a sampled
document.  Record-level McNemar summaries are provided separately because they
do not account for within-document dependence and are descriptive only.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_SEED = 20_260_812
EXACT_SIGN_FLIP_MAX_DOCUMENTS = 16


def document_clustered_binary_ci(
    document_ids: Sequence[Hashable],
    outcomes: Sequence[bool | int],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Return a percentile CI for a binary record-level metric clustered by document.

    Each bootstrap draw samples documents with replacement and retains all of
    each selected document's records.  The reported estimate is the ordinary
    record-level fraction in the observed data.
    """
    document_successes, document_counts = _aggregate_binary_by_document(
        document_ids, outcomes, "outcomes"
    )
    estimate = float(document_successes.sum() / document_counts.sum())
    samples = _clustered_proportion_samples(
        document_successes,
        document_counts,
        n_resamples=n_resamples,
        seed=seed,
    )
    lower, upper = _percentile_interval(samples, confidence_level)
    return {
        "analysis": "document_clustered_bootstrap",
        "metric": "binary_record_fraction",
        "estimate": estimate,
        "confidence_level": confidence_level,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_documents": int(len(document_counts)),
        "n_records": int(document_counts.sum()),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def paired_document_clustered_difference_ci(
    document_ids: Sequence[Hashable],
    reference_outcomes: Sequence[bool | int],
    comparison_outcomes: Sequence[bool | int],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Return a paired clustered CI for ``comparison - reference``.

    Bootstrap samples draw the same documents for both methods, so the
    difference retains their pairing.  Inputs must be record-aligned.
    """
    _validate_paired_inputs(document_ids, reference_outcomes, comparison_outcomes)
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    reference_successes, document_counts = _aggregate_binary_by_document(
        document_ids, reference_outcomes, "reference_outcomes"
    )
    comparison_successes, comparison_counts = _aggregate_binary_by_document(
        document_ids, comparison_outcomes, "comparison_outcomes"
    )
    if not np.array_equal(document_counts, comparison_counts):
        raise AssertionError("paired inputs must produce matching document counts")

    differences = comparison_successes - reference_successes
    estimate = float(differences.sum() / document_counts.sum())
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(
        0, len(document_counts), size=(n_resamples, len(document_counts))
    )
    sampled_numerators = differences[draw_indices].sum(axis=1)
    sampled_denominators = document_counts[draw_indices].sum(axis=1)
    samples = sampled_numerators / sampled_denominators
    lower, upper = _percentile_interval(samples, confidence_level)
    return {
        "analysis": "paired_document_clustered_bootstrap",
        "metric": "binary_record_fraction_difference",
        "contrast": "comparison_minus_reference",
        "estimate": estimate,
        "confidence_level": confidence_level,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_documents": int(len(document_counts)),
        "n_records": int(document_counts.sum()),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def paired_document_sign_flip_test(
    document_ids: Sequence[Hashable],
    reference_outcomes: Sequence[bool | int],
    comparison_outcomes: Sequence[bool | int],
    *,
    seed: int = DEFAULT_SEED,
    max_exact_documents: int = EXACT_SIGN_FLIP_MAX_DOCUMENTS,
    monte_carlo_samples: int = 100_000,
) -> dict[str, Any]:
    """Test paired method differences using document-level sign randomization.

    Each document contributes its count of paired record-level differences.
    Their signed sum, divided by the fixed total record count, is the same
    record-weighted accuracy-difference estimand used by the clustered
    bootstrap interval. Exhaustive enumeration is used through
    ``max_exact_documents`` non-zero document differences; otherwise, a
    deterministic Monte Carlo approximation is used.
    """
    _validate_paired_inputs(document_ids, reference_outcomes, comparison_outcomes)
    reference_successes, document_counts = _aggregate_binary_by_document(
        document_ids, reference_outcomes, "reference_outcomes"
    )
    comparison_successes, comparison_counts = _aggregate_binary_by_document(
        document_ids, comparison_outcomes, "comparison_outcomes"
    )
    if not np.array_equal(document_counts, comparison_counts):
        raise AssertionError("paired inputs must produce matching document counts")

    document_differences = comparison_successes - reference_successes
    nonzero_differences = document_differences[document_differences != 0.0]
    observed = float(document_differences.sum() / document_counts.sum())
    observed_abs_sum = abs(float(nonzero_differences.sum()))
    n_nonzero = len(nonzero_differences)

    if n_nonzero == 0:
        return {
            "analysis": "paired_document_sign_flip",
            "unit": "document",
            "contrast": "comparison_minus_reference",
            "statistic": "record_weighted_accuracy_difference",
            "estimate": observed,
            "p_value": 1.0,
            "method": "exact",
            "n_documents": int(len(document_counts)),
            "n_nonzero_document_differences": 0,
            "n_randomizations": 1,
            "seed": seed,
        }

    if n_nonzero <= max_exact_documents:
        n_randomizations = 1 << n_nonzero
        exceedances = 0
        for signs in range(n_randomizations):
            signed_sum = 0.0
            for index, difference in enumerate(nonzero_differences):
                signed_sum += difference if (signs >> index) & 1 else -difference
            if abs(signed_sum) >= observed_abs_sum - 1e-15:
                exceedances += 1
        p_value = exceedances / n_randomizations
        method = "exact"
    else:
        if monte_carlo_samples <= 0:
            raise ValueError("monte_carlo_samples must be positive")
        rng = np.random.default_rng(seed)
        signs = rng.choice(
            np.array([-1.0, 1.0]), size=(monte_carlo_samples, n_nonzero)
        )
        signed_sums = signs @ nonzero_differences
        exceedances = int(np.count_nonzero(np.abs(signed_sums) >= observed_abs_sum - 1e-15))
        # The add-one estimate keeps a randomization p-value non-zero.
        p_value = (exceedances + 1) / (monte_carlo_samples + 1)
        n_randomizations = monte_carlo_samples
        method = "monte_carlo"

    return {
        "analysis": "paired_document_sign_flip",
        "unit": "document",
        "contrast": "comparison_minus_reference",
        "statistic": "record_weighted_accuracy_difference",
        "estimate": observed,
        "p_value": float(p_value),
        "method": method,
        "n_documents": int(len(document_counts)),
        "n_nonzero_document_differences": int(n_nonzero),
        "n_randomizations": n_randomizations,
        "seed": seed,
    }


def mcnemar_descriptive(
    reference_outcomes: Sequence[bool | int],
    comparison_outcomes: Sequence[bool | int],
) -> dict[str, Any]:
    """Return record-level McNemar discordances and an exact binomial p-value.

    This result is explicitly descriptive: records from the same source
    document are dependent, so document-clustered inference above is primary.
    """
    _validate_binary_values(reference_outcomes, "reference_outcomes")
    _validate_binary_values(comparison_outcomes, "comparison_outcomes")
    if len(reference_outcomes) != len(comparison_outcomes):
        raise ValueError("reference_outcomes and comparison_outcomes must have equal length")

    reference = np.asarray(reference_outcomes, dtype=np.int8)
    comparison = np.asarray(comparison_outcomes, dtype=np.int8)
    reference_only = int(np.count_nonzero((reference == 1) & (comparison == 0)))
    comparison_only = int(np.count_nonzero((reference == 0) & (comparison == 1)))
    discordant = reference_only + comparison_only
    p_value = _two_sided_exact_binomial(min(reference_only, comparison_only), discordant)
    return {
        "analysis": "record_level_mcnemar_descriptive",
        "primary_analysis": False,
        "contrast": "comparison_minus_reference",
        "reference_only_successes": reference_only,
        "comparison_only_successes": comparison_only,
        "discordant_records": discordant,
        "exact_binomial_p_value": p_value,
        "n_records": len(reference),
    }


def _aggregate_binary_by_document(
    document_ids: Sequence[Hashable], outcomes: Sequence[bool | int], name: str
) -> tuple[np.ndarray, np.ndarray]:
    _validate_binary_values(outcomes, name)
    if len(document_ids) != len(outcomes):
        raise ValueError(f"document_ids and {name} must have equal length")
    if len(document_ids) == 0:
        raise ValueError("at least one record is required")

    counts: dict[Hashable, int] = {}
    successes: dict[Hashable, int] = {}
    for document_id, outcome in zip(document_ids, outcomes, strict=True):
        if document_id not in counts:
            counts[document_id] = 0
            successes[document_id] = 0
        counts[document_id] += 1
        successes[document_id] += int(outcome)
    return (
        np.asarray(list(successes.values()), dtype=np.float64),
        np.asarray(list(counts.values()), dtype=np.float64),
    )


def _clustered_proportion_samples(
    document_successes: np.ndarray,
    document_counts: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(
        0, len(document_counts), size=(n_resamples, len(document_counts))
    )
    return document_successes[draw_indices].sum(axis=1) / document_counts[draw_indices].sum(axis=1)


def _percentile_interval(samples: np.ndarray, confidence_level: float) -> tuple[float, float]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha))
    return float(lower), float(upper)


def _validate_binary_values(values: Sequence[bool | int], name: str) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one record")
    if any(value not in (False, True, 0, 1) for value in values):
        raise ValueError(f"{name} must contain only binary values")


def _validate_paired_inputs(
    document_ids: Sequence[Hashable],
    reference_outcomes: Sequence[bool | int],
    comparison_outcomes: Sequence[bool | int],
) -> None:
    _validate_binary_values(reference_outcomes, "reference_outcomes")
    _validate_binary_values(comparison_outcomes, "comparison_outcomes")
    if len(document_ids) != len(reference_outcomes) or len(document_ids) != len(comparison_outcomes):
        raise ValueError("document_ids and paired outcomes must have equal lengths")


def _two_sided_exact_binomial(smaller_count: int, total: int) -> float:
    if total == 0:
        return 1.0
    if not 0 <= smaller_count <= total // 2:
        raise ValueError("smaller_count must be in the lower half of the distribution")

    log_probability_at_k = (
        math.lgamma(total + 1)
        - math.lgamma(smaller_count + 1)
        - math.lgamma(total - smaller_count + 1)
        - total * math.log(2.0)
    )
    scaled_tail = 1.0
    scaled_term = 1.0
    for count in range(smaller_count, 0, -1):
        scaled_term *= count / (total - count + 1)
        scaled_tail += scaled_term
    log_two_sided = math.log(2.0) + log_probability_at_k + math.log(scaled_tail)
    if log_two_sided >= 0.0:
        return 1.0
    return float(math.exp(log_two_sided))
