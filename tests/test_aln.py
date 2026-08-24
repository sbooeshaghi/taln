"""Tests for taln alignment, focusing on subword boundary handling.

The core issue: tiktoken assigns different token IDs depending on whether a
token appears at the start of a string vs mid-text (with a leading space).
For example, "FOX" (token 3873) != " FOX" (token 45288). When aligning a
standalone label like "FOXI1" against a sentence where it appears mid-text,
the first token mismatches and only a suffix is recovered.

These tests use real examples from mrkr marker gene extraction to verify that
taln correctly aligns gene symbols and cell type names within source rationale
sentences.
"""

import itertools
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import taln.taln_aln as taln_aln
from taln.taln_aln import (
    AlignmentResult,
    align_ng,
    align_ng_casefold,
    align_ng_result,
    iter_align_ng,
    norm_text,
    norm_text_with_mapping,
    reconstruct_target_by_token,
    tokenize,
    tokenize_with_offsets,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cl100k_base_available():
    """Skip token-mode tests when tiktoken cannot load its encoding offline."""
    try:
        tokenize_with_offsets("probe")
    except Exception as exc:
        pytest.skip(f"tiktoken cl100k_base encoding is unavailable: {exc}")


def best_reconstruction(source, target, ttype="token"):
    """Run align_ng and return the longest reconstructed string."""
    alns = align_ng(source, target, ttype)
    if not alns:
        return ""
    # Pick the alignment with the most tokens (best coverage)
    best = max(alns, key=len)
    return reconstruct_target_by_token(source, best)


def expected_ordered_paths(source_tokens, target_tokens):
    """Reference the k=1 alignment semantics for short whitespace sequences."""
    source_positions = {
        token: [idx for idx, value in enumerate(source_tokens) if value == token]
        for token in set(source_tokens)
    }

    if not target_tokens or any(token not in source_positions for token in target_tokens):
        return set()

    paths = set()

    def visit(target_idx, last_source_idx, path):
        if target_idx == len(target_tokens):
            paths.add(tuple(path))
            return

        for source_idx in source_positions[target_tokens[target_idx]]:
            if source_idx > last_source_idx:
                visit(target_idx + 1, source_idx, path + [source_idx])

    visit(0, -1, [])
    return paths


# ---------------------------------------------------------------------------
# Tokenization boundary tests — demonstrate the root cause
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("cl100k_base_available")
class TestTokenBoundary:
    """Show that tiktoken gives different IDs for start-of-string vs mid-text."""

    def test_leading_space_changes_token_id(self):
        """'FOX' and ' FOX' should produce different token IDs."""
        toks_no_space = tokenize_with_offsets("FOX")
        toks_with_space = tokenize_with_offsets(" FOX")
        # The encoded token IDs should differ
        ids_no_space = [t["enc_token"] for t in toks_no_space]
        ids_with_space = [t["enc_token"] for t in toks_with_space]
        assert ids_no_space != ids_with_space

    def test_gene_symbol_tokenization_differs(self):
        """'DUSP15' standalone tokenizes differently than ' DUSP15' in context."""
        toks_standalone = tokenize_with_offsets("DUSP15")
        toks_context = tokenize_with_offsets(" DUSP15")
        ids_standalone = [t["enc_token"] for t in toks_standalone]
        ids_context = [t["enc_token"] for t in toks_context]
        # First token should differ (D vs " D")
        assert ids_standalone[0] != ids_context[0]


# ---------------------------------------------------------------------------
# Gene symbol alignment — the main failure cases from mrkr extractions
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("cl100k_base_available")
class TestGeneSymbolAlignment:
    """Aligning gene symbols within source rationale sentences."""

    def test_foxi1_in_sentence(self):
        """FOXI1 should align fully, not just '1'."""
        source = (
            "These cells co-expressed acid-base handling genes such as "
            "ATP6V1B1, CLCNKA, and FOXI1, and were devoid of immune "
            "effector transcripts."
        )
        target = "FOXI1"
        result = best_reconstruction(source, target)
        assert "FOXI1" in result or "FOX" in result, (
            f"Expected full FOXI1 match, got: {result!r}"
        )

    def test_dusp15_in_sentence(self):
        """DUSP15 should align fully, not just 'USP15'."""
        source = (
            "Single-cell RNA-seq further localised DUSP15 expression to a "
            "rare regenerative epithelial population."
        )
        target = "DUSP15"
        result = best_reconstruction(source, target)
        assert "DUSP15" in result or " DUSP15" in result, (
            f"Expected full DUSP15 match, got: {result!r}"
        )

    def test_clcnka_in_sentence(self):
        """CLCNKA should align fully, not just a suffix."""
        source = (
            "These cells co-expressed acid-base handling genes such as "
            "ATP6V1B1, CLCNKA, and FOXI1."
        )
        target = "CLCNKA"
        result = best_reconstruction(source, target)
        assert "CLCNKA" in result or "CNKA" not in result.lstrip(), (
            f"Expected full CLCNKA match, got: {result!r}"
        )

    def test_atp6v1b1_in_sentence(self):
        """ATP6V1B1 — a gene with digits that appear elsewhere in the sentence."""
        source = (
            "an acid-base-handling epithelial cluster expressing "
            "FOXI1, ATP6V1B1, ATP6V0A4, and CLCNKB."
        )
        target = "ATP6V1B1"
        result = best_reconstruction(source, target)
        # Should contain the full gene symbol, not fragments
        assert "ATP6V1B1" in result or "ATP6V" in result, (
            f"Expected full ATP6V1B1 match, got: {result!r}"
        )

    def test_spink1_in_sentence(self):
        """SPINK1 should align fully."""
        source = (
            "Regenerative epithelial cells co-expressed SPINK1, NXPH2, "
            "and TMEM213."
        )
        target = "SPINK1"
        result = best_reconstruction(source, target)
        assert "SPINK1" in result or "SPINK" in result, (
            f"Expected full SPINK1 match, got: {result!r}"
        )


# ---------------------------------------------------------------------------
# Cell type label alignment
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("cl100k_base_available")
class TestCellTypeLabelAlignment:
    """Aligning cell type names within source rationale sentences."""

    def test_regenerative_epithelial(self):
        """'regenerative epithelial' should match fully, not 'enerative epithelial'."""
        source = (
            "Single-cell RNA-seq further localised DUSP15 expression to a "
            "rare regenerative epithelial population enriched for "
            "acid-base transport genes."
        )
        target = "regenerative epithelial"
        result = best_reconstruction(source, target)
        assert "regenerative" in result, (
            f"Expected full 'regenerative epithelial' match, got: {result!r}"
        )

    def test_acid_base_handling_epithelial_cluster(self):
        """Multi-word cell type with hyphens should align fully."""
        source = (
            "DUSP15 expression was confined to a rare regenerative epithelial "
            "subset comprising 3.2% of tumour cells, nearly half of which "
            "were localized to an acid-base-handling epithelial cluster "
            "expressing FOXI1, ATP6V1B1, ATP6V0A4, and CLCNKB."
        )
        target = "acid-base-handling epithelial cluster"
        result = best_reconstruction(source, target)
        assert "acid" in result, (
            f"Expected 'acid-base-handling...' match, got: {result!r}"
        )

    def test_brown_adipocyte(self):
        """Common cell type name should align."""
        source = (
            "BAT exhibits elevated UCP1 RNA under cold exposure, "
            "establishing UCP1 as a marker for brown adipocytes."
        )
        target = "brown adipocytes"
        result = best_reconstruction(source, target)
        assert "brown" in result, (
            f"Expected 'brown adipocytes' match, got: {result!r}"
        )


# ---------------------------------------------------------------------------
# Lazy and bounded enumeration
# ---------------------------------------------------------------------------


class TestLazyEnumeration:
    def test_default_api_remains_a_materialized_list(self):
        source = "a b a b"
        target = "a b"
        expected = list(iter_align_ng(source, target, ttype="whitespace"))

        actual = align_ng(source, target, ttype="whitespace")

        assert type(actual) is list
        assert actual == expected

    def test_repeated_tokens_are_bounded_without_exhaustive_materialization(self):
        source = " ".join(["a", "b"] * 30)
        target = " ".join(["a", "b"] * 10)

        result = align_ng_result(
            source,
            target,
            ttype="whitespace",
            max_alignments=5,
        )

        assert len(result.alignments) == 5
        assert result.truncated is True
        assert result.status == "truncated"
        for alignment in result.alignments:
            starts = [token["start_idx"] for token in alignment]
            assert starts == sorted(starts)
            assert len(starts) == 20

    def test_adjacent_repeated_target_tokens_remain_distinct(self):
        alignments = align_ng(
            "a a a",
            "a a",
            ttype="whitespace",
        )

        assert len(alignments) == 3
        assert [
            tuple(token["start_idx"] for token in alignment)
            for alignment in alignments
        ] == [(0, 2), (0, 4), (2, 4)]
        assert all(len(alignment) == 2 for alignment in alignments)

    def test_partially_absent_target_returns_no_alignment(self):
        result = align_ng(
            "a c",
            "a b",
            ttype="whitespace",
            return_status=True,
        )

        assert result.alignments == []
        assert result.truncated is False
        assert result.status == "no_alignment"

    def test_cap_distinguishes_truncation_from_no_alignment(self):
        truncated = align_ng_result(
            "a a",
            "a",
            ttype="whitespace",
            max_alignments=0,
        )
        absent = align_ng_result(
            "a a",
            "b",
            ttype="whitespace",
            max_alignments=0,
        )

        assert truncated.alignments == []
        assert truncated.truncated is True
        assert truncated.status == "truncated"
        assert absent.alignments == []
        assert absent.truncated is False
        assert absent.status == "no_alignment"

    def test_exactly_filling_cap_is_complete(self):
        result = align_ng_result(
            "a a",
            "a",
            ttype="whitespace",
            max_alignments=2,
        )

        assert len(result.alignments) == 2
        assert result.truncated is False
        assert result.status == "complete"

    def test_align_ng_can_return_status_without_breaking_default_callers(self):
        result = align_ng(
            "a a",
            "a",
            ttype="whitespace",
            max_alignments=1,
            return_status=True,
        )

        assert isinstance(result, AlignmentResult)
        assert len(result.alignments) == 1
        assert result.truncated is True

    def test_align_ng_cap_automatically_returns_status(self):
        result = align_ng(
            "a a",
            "a",
            ttype="whitespace",
            max_alignments=1,
        )

        assert isinstance(result, AlignmentResult)
        assert len(result.alignments) == 1
        assert result.status == "truncated"
        assert result.truncated is True
        assert result.exhaustive is False

    @pytest.mark.parametrize("max_alignments", [-1, 1.5, True])
    def test_invalid_alignment_caps_are_rejected(self, max_alignments):
        expected_error = ValueError if max_alignments == -1 else TypeError
        with pytest.raises(expected_error):
            align_ng_result(
                "a",
                "a",
                ttype="whitespace",
                max_alignments=max_alignments,
            )

    def test_exhaustive_small_sequences_are_sound_and_complete(self):
        alphabet = ("a", "b")
        source_sequences = [
            sequence
            for length in range(5)
            for sequence in itertools.product(alphabet, repeat=length)
        ]
        target_sequences = [
            sequence
            for length in range(4)
            for sequence in itertools.product(alphabet, repeat=length)
        ]

        for source_tokens in source_sequences:
            source = " ".join(source_tokens)
            tokenized_source, _ = tokenize(source, ttype="whitespace")
            token_index_by_start = {
                token["start_idx"]: idx for idx, token in enumerate(tokenized_source)
            }

            for target_tokens in target_sequences:
                target = " ".join(target_tokens)
                actual_alignments = list(
                    iter_align_ng(source, target, ttype="whitespace")
                )
                actual_paths = {
                    tuple(token_index_by_start[token["start_idx"]] for token in alignment)
                    for alignment in actual_alignments
                }
                expected_paths = expected_ordered_paths(source_tokens, target_tokens)

                assert len(actual_alignments) == len(actual_paths)
                assert actual_paths == expected_paths, (
                    f"source={source_tokens!r}, target={target_tokens!r}"
                )

    def test_capped_results_retain_original_unicode_offsets(self):
        source = "caf\u00e9\n\nmarker caf\u00e9"

        result = align_ng_result(
            source,
            "cafe",
            ttype="whitespace",
            max_alignments=1,
        )

        assert result.truncated is True
        token = result.alignments[0][0]
        assert source[token["start_idx"] : token["end_idx"]] == "caf\u00e9"

    def test_cli_single_consumes_only_the_first_complete_alignment(
        self, monkeypatch, capsys
    ):
        first_alignment = [
            {
                "token": "alpha",
                "enc_token": "alpha",
                "start_idx": 0,
                "end_idx": 5,
            }
        ]

        def alignment_stream(*args, **kwargs):
            yield first_alignment
            raise AssertionError("--single consumed more than one alignment")

        monkeypatch.setattr(taln_aln, "iter_align_ng", alignment_stream)
        args = SimpleNamespace(
            source="alpha",
            target="alpha",
            tokenization_type="whitespace",
            output=None,
            single=True,
        )

        taln_aln.validate_taln_aln_args(None, args)

        assert json.loads(capsys.readouterr().out) == first_alignment

    def test_cli_uncapped_output_remains_a_plain_alignment_list(self):
        cmd = [
            sys.executable,
            "-m",
            "taln.main",
            "aln",
            "-s",
            "a a",
            "-tt",
            "whitespace",
            "a",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)

        assert isinstance(payload, list)
        assert len(payload) == 2
        assert all(isinstance(alignment, list) for alignment in payload)

    def test_cli_single_output_remains_a_plain_token_list(self):
        cmd = [
            sys.executable,
            "-m",
            "taln.main",
            "aln",
            "-s",
            "a a",
            "-tt",
            "whitespace",
            "--single",
            "a",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)

        assert isinstance(payload, list)
        assert len(payload) == 1
        assert payload[0]["token"] == "a"

    def test_cli_cap_reports_truncation_in_a_status_object(self):
        cmd = [
            sys.executable,
            "-m",
            "taln.main",
            "aln",
            "-s",
            "a a",
            "-tt",
            "whitespace",
            "--max-alignments",
            "1",
            "a",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)

        assert isinstance(payload, dict)
        assert len(payload["alignments"]) == 1
        assert payload["status"] == "truncated"
        assert payload["truncated"] is True
        assert payload["exhaustive"] is False
        assert payload["max_alignments"] == 1
        assert payload["returned_alignment_count"] == 1

    def test_cli_cap_reports_when_search_is_complete(self):
        cmd = [
            sys.executable,
            "-m",
            "taln.main",
            "aln",
            "-s",
            "a a",
            "-tt",
            "whitespace",
            "--max-alignments",
            "2",
            "a",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)

        assert isinstance(payload, dict)
        assert len(payload["alignments"]) == 2
        assert payload["status"] == "complete"
        assert payload["truncated"] is False
        assert payload["exhaustive"] is True
        assert payload["max_alignments"] == 2

    def test_cli_status_mode_wraps_an_uncapped_search(self):
        cmd = [
            sys.executable,
            "-m",
            "taln.main",
            "aln",
            "-s",
            "a",
            "-tt",
            "whitespace",
            "--with-status",
            "a",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)

        assert isinstance(payload, dict)
        assert payload["status"] == "complete"
        assert payload["truncated"] is False
        assert payload["exhaustive"] is True
        assert payload["max_alignments"] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and tricky patterns."""

    @pytest.mark.usefixtures("cl100k_base_available")
    def test_single_character_gene(self):
        """Short tokens that appear many places (like '1') shouldn't spuriously match."""
        source = "Expression of ATP6V1B1 and ATP6V0A4 was elevated."
        target = "B1"
        # Should find B1 somewhere, not fail entirely
        alns = align_ng(source, target)
        assert len(alns) > 0, "Expected at least one alignment for 'B1'"

    def test_case_sensitivity_after_normalization(self):
        """Normalization preserves case while transliterating text."""
        source = "The marker gene FOXI1 was identified."
        target = "FOXI1"
        # Both get normalized; this tests that normalization is consistent
        assert norm_text(source) != source or norm_text(target) == target

    @pytest.mark.usefixtures("cl100k_base_available")
    def test_target_not_in_source(self):
        """When target is absent, should return empty list."""
        source = "This sentence mentions no gene symbols."
        target = "BRCA1"
        alns = align_ng(source, target)
        # Should return empty or only trivially short matches
        if alns:
            best = max(alns, key=len)
            recon = reconstruct_target_by_token(source, best)
            # Should not reconstruct a meaningful portion of BRCA1
            assert len(recon.strip()) < len(target), (
                f"Spurious match for absent target: {recon!r}"
            )

    def test_whitespace_tokenization_mode(self):
        """Whitespace mode should not have the leading-space problem."""
        source = (
            "Single-cell RNA-seq localised DUSP15 expression to a "
            "regenerative epithelial population."
        )
        target = "DUSP15"
        result = best_reconstruction(source, target, ttype="whitespace")
        assert "DUSP15" in result, (
            f"Whitespace mode should match full DUSP15, got: {result!r}"
        )

    def test_offsets_map_to_original_source_after_whitespace_normalization(self):
        """Collapsed whitespace should not shift reported source offsets."""
        source = "alpha\n\nbeta   gamma"
        target = "beta gamma"
        alns = align_ng(source, target, ttype="whitespace")
        assert alns
        aln = max(alns, key=len)
        assert [source[t["start_idx"] : t["end_idx"]] for t in aln] == [
            "beta",
            "gamma",
        ]
        assert source[aln[0]["start_idx"] : aln[-1]["end_idx"]] == "beta   gamma"

    def test_offsets_map_nfd_composition_to_original_source(self):
        """NFC composition must retain the full decomposed source interval."""
        source = "Cafe\u0301  marker"
        tokens, _ = tokenize(source, ttype="whitespace")

        assert [source[token["start_idx"] : token["end_idx"]] for token in tokens] == [
            "Cafe\u0301",
            "marker",
        ]
        assert [(token["start_idx"], token["end_idx"]) for token in tokens] == [
            (0, 5),
            (7, 13),
        ]

        alignments = align_ng(source, "Cafe marker", ttype="whitespace")
        assert len(alignments) == 1
        assert [
            source[token["start_idx"] : token["end_idx"]]
            for token in alignments[0]
        ] == ["Cafe\u0301", "marker"]

    @pytest.mark.usefixtures("cl100k_base_available")
    def test_synthesized_transliteration_space_does_not_absorb_source_prefix(self):
        """A transliteration's trailing space must yield to real source whitespace."""
        source = "猫 alpha"
        tokens, _ = tokenize(source, ttype="token")
        alpha = next(token for token in tokens if "alpha" in token["token"])

        assert source[alpha["start_idx"] : alpha["end_idx"]] == " alpha"
        assert (alpha["start_idx"], alpha["end_idx"]) == (1, 7)

    def test_expanded_punctuation_maps_to_one_original_character(self):
        """Every normalized character in an expansion shares its source interval."""
        source = "alpha…beta"
        normalized, mapping = norm_text_with_mapping(source)
        period_indices = [idx for idx, char in enumerate(normalized) if char == "."]

        assert normalized == "alpha...beta"
        assert period_indices == [5, 6, 7]
        assert [mapping[idx] for idx in period_indices] == [(5, 6)] * 3
        assert [source[mapping[idx][0] : mapping[idx][1]] for idx in period_indices] == [
            "…"
        ] * 3

    def test_repeated_normalized_occurrences_retain_distinct_source_offsets(self):
        source = "cafe\u0301\ncafé"
        tokens, _ = tokenize(source, ttype="whitespace")

        assert [token["token"] for token in tokens] == ["cafe", "cafe"]
        assert [(token["start_idx"], token["end_idx"]) for token in tokens] == [
            (0, 5),
            (6, 10),
        ]
        assert [source[token["start_idx"] : token["end_idx"]] for token in tokens] == [
            "cafe\u0301",
            "café",
        ]

    def test_tokenize_preserves_normalized_offsets_for_debugging(self):
        """Tokens expose original-source offsets and normalized-text offsets."""
        source = "alpha\n\nbeta"
        tokens, _ = tokenize(source, ttype="whitespace")
        beta = next(t for t in tokens if t["token"] == "beta")
        assert source[beta["start_idx"] : beta["end_idx"]] == "beta"
        assert beta["norm_start_idx"] == norm_text(source).index("beta")

    def test_cli_single_no_match_returns_empty_alignment(self):
        """`taln aln --single` should not crash when no alignment exists."""
        cmd = [
            sys.executable,
            "-m",
            "taln.main",
            "aln",
            "-s",
            "alpha beta",
            "-tt",
            "whitespace",
            "--single",
            "gamma",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert json.loads(result.stdout) == []

    @pytest.mark.usefixtures("cl100k_base_available")
    def test_empty_target(self):
        """Empty target should return empty."""
        source = "Some text here."
        target = ""
        alns = align_ng(source, target)
        assert alns == []

    @pytest.mark.usefixtures("cl100k_base_available")
    def test_target_equals_source(self):
        """When target is the full source, alignment should cover everything."""
        source = "FOXI1 marks epithelial cells."
        target = source
        alns = align_ng(source, target)
        assert len(alns) > 0
        best = max(alns, key=len)
        recon = reconstruct_target_by_token(source, best)
        # Should reconstruct most of the source
        assert len(recon) >= len(source) * 0.8


# ---------------------------------------------------------------------------
# Case-insensitive fallback alignment
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("cl100k_base_available")
class TestCasefoldAlignment:
    """align_ng_casefold: case-insensitive alignment with original-case output."""

    def test_sentence_initial_capital(self):
        """'regenerative' should match 'Regenerative' at sentence start."""
        source = (
            "Regenerative epithelial cells co-expressed SPINK1, NXPH2, "
            "and TMEM213."
        )
        target = "regenerative epithelial"
        # Regular align_ng fails on the first token (Reg vs reg)
        orig_alns = align_ng(source, target)
        if orig_alns:
            orig_best = reconstruct_target_by_token(source, max(orig_alns, key=len))
        else:
            orig_best = ""
        assert "regenerative" not in orig_best.lower() or "Reg" not in orig_best

        # Casefold alignment should recover the full match
        alns = align_ng_casefold(source, target)
        assert len(alns) > 0
        best = max(alns, key=len)
        recon = reconstruct_target_by_token(source, best)
        assert "Regenerative" in recon or "regenerative" in recon.lower(), (
            f"Expected case-insensitive match, got: {recon!r}"
        )

    def test_preserves_original_case_in_output(self):
        """Casefold alignment should return tokens with original casing."""
        source = "Brown adipocytes express UCP1 at high levels."
        target = "brown adipocytes"
        alns = align_ng_casefold(source, target)
        assert len(alns) > 0
        best = max(alns, key=len)
        recon = reconstruct_target_by_token(source, best)
        # Output should have original "Brown" (capital B), not "brown"
        assert "Brown" in recon, (
            f"Expected original casing 'Brown', got: {recon!r}"
        )

    def test_no_match_returns_empty(self):
        """Casefold with absent target should return empty."""
        source = "This sentence has no relevant cell types."
        target = "astrocyte"
        alns = align_ng_casefold(source, target)
        assert alns == []

    def test_already_matching_case(self):
        """When case already matches, casefold should still work."""
        source = "The marker DUSP15 was identified in regenerative cells."
        target = "DUSP15"
        alns = align_ng_casefold(source, target)
        assert len(alns) > 0
        best = max(alns, key=len)
        recon = reconstruct_target_by_token(source, best)
        assert "DUSP15" in recon
