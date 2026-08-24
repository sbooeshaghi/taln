# Stage 1 Corrected Baseline Matrix

This document freezes the interpretation of the Stage 1 experiment before any
manuscript restructuring. It reports lexical support and source localization;
it does not evaluate semantic or relational correctness.

## Run Record

The development run evaluated 70,198 tasks in 289.53 seconds. The held-out run
evaluated 11,326 tasks in 38.44 seconds. Both runs completed with zero method
errors and the exact expected record counts.

| Split | Tasks | Records SHA-256 | Run fingerprint |
| --- | ---: | --- | --- |
| Development | 70,198 | `1f7b5864eccc9054ee05e575058b4e7513d89ec9e7dfdb57db673735f44d8ded` | `4cb280f107cff4f11f96bfe2d79182daa663a9b16e26d58f31765d1b7afa0e1e` |
| Held-out test | 11,326 | `aeed5c3859d4f34f1132bb3a1ef0a8c52e22558ce275dbba5526760a8317aad1` | `25142c9a4000488d1170cc9f4a843832596cd6e1640f627281fcf7a6a99a9778` |

The matrix used exact substring matching and four token-alignment methods:
`taln`, LCS, `difflib.SequenceMatcher`, and an exact source-semi-global,
target-global comparator. Tokenization variants were raw whitespace,
boundary-stripped whitespace, punctuation-aware lexical tokens, `cl100k_base`,
SciBERT, and PubMedBERT. The two biomedical tokenizer revisions are pinned in
`analysis/config/revision_2026/stage1_spec.json`.

The 196 BOAT deletions that remained contiguous after token deletion were
reported as `one_gap_still_contiguous_control`, not as primary one-gap tasks.

## Held-Out Results

The primary one-gap reconstruction rates for `taln` were:

| Tokenizer | BOAT, n=4,284 | BIO-BOAT, n=1,259 |
| --- | ---: | ---: |
| Whitespace | 52.99% | 53.85% |
| Boundary-stripped whitespace | 99.07% | 97.06% |
| Punctuation-aware | 99.79% | 99.68% |
| `cl100k_base` | 98.48% | 97.22% |
| SciBERT | 99.81% | 99.92% |
| PubMedBERT | 99.81% | 99.92% |

For BOAT, punctuation-aware tokenization improved over raw whitespace by 46.80
percentage points (document-clustered 95% CI, 44.11 to 49.33). For BIO-BOAT,
the improvement was 45.83 points (95% CI, 36.64 to 56.42). This is a large and
replicated token-boundary effect.

Biomedical tokenization did not produce a comparably strong held-out gain over
punctuation-aware tokenization. SciBERT improved BIO-BOAT by 0.24 percentage
points (95% CI, 0.00 to 0.54; document sign-flip p=0.25), corresponding to three
additional tasks. On BOAT, SciBERT and PubMedBERT each improved by 0.02 points.
`cl100k_base` was worse than punctuation-aware tokenization by 1.31 points on
BOAT and 2.46 points on BIO-BOAT.

## LLMarkers Strata

The pooled LLMarkers result combines two materially different tasks. Marker
genes (`feature_label`) are usually literal lexical targets. Cell types
(`group_label`) often require aliases, abbreviations, or semantic resolution.

| Held-out tokenizer | Marker genes, n=91 | Cell types, n=91 |
| --- | ---: | ---: |
| Exact substring | 97.80% | 80.22% |
| Whitespace plus `taln` | 26.37% | 69.23% |
| Boundary-stripped plus `taln` | 91.21% | 74.73% |
| Punctuation-aware plus `taln` | 97.80% | 74.73% |
| `cl100k_base` plus `taln` | 100.00% | 80.22% |
| SciBERT plus `taln` | 98.90% | 78.02% |
| PubMedBERT plus `taln` | 97.80% | 78.02% |

All 38 development and 18 held-out LLMarkers tasks missed by every method were
cell-type labels. Manual inspection found examples whose passage used only an
abbreviation, referred to a cell type indirectly, or omitted the submitted
label. The existing `_verification.all_verified` field is therefore not an
independent full-label lexical gold standard. It must not be used to estimate
lexical-verification specificity or to claim that every pooled label should
align verbatim.

The marker-gene stratum is the cleaner lexical validation. Across development
and held-out records, `cl100k_base` recovered all 366 marker-gene labels, while
punctuation-aware alignment and exact substring matching recovered 363. The
three additional cases involve compact or non-contiguous gene notation; this
is a narrow gain, not evidence of broad superiority for biomedical tokenizers.

## Alignment-Method Finding

Under identical tokenization, preprocessing, and full-target scoring, LCS and
the semi-global exact comparator agreed with `taln` on lexical support for every
task in every dataset and split. The earlier 14.1% LCS/`difflib` comparison was
therefore a scoring and preprocessing artifact and must be replaced.

`difflib` never recovered a task that `taln` missed, but it missed valid ordered
alignments because `SequenceMatcher` is a matching-block heuristic rather than
an exact full-coverage subsequence algorithm. The distinctive capability of
`taln` in this matrix is enumeration of possible source localizations, not
higher reconstruction accuracy than exact ordered comparators.

## Audit And Limitations

Every LLMarkers disagreement was manually inspected: 234 development rows and
77 held-out rows. We also inspected deterministic, stratified BOAT and
BIO-BOAT samples and source-backed comparisons. No evaluator or source-join
error was found. Most whitespace failures were caused by attached punctuation;
remaining cases involved hyphens, slashes, citation digits, plural prefixes, or
subword notation.

Exact substring matching intentionally permits character-prefix matches, such
as a singular label inside a plural source form. Conversely, subword alignment
can accept compact notation such as `COL1A2` from `COL1A1/2`. These behaviors
must be described as surface-form choices, not semantic verification.

Seventeen held-out `taln` results from three base records reached an enumeration
cap. Their lexical-support outcomes were unaffected, but candidate counts for
those records are lower bounds. Candidate counts include explored ordered paths
over matched target positions and should not be described only as counts of
complete alignments.

## Evidence Gate

Stage 1 supports the following conclusions for later manuscript discussion:

1. Token-boundary handling has a large, reproducible effect relative to raw
   whitespace tokenization.
2. A good punctuation-aware lexical tokenizer nearly matches biomedical
   tokenizers on held-out BIO-BOAT; broad domain-tokenizer superiority is not
   supported.
3. `taln` should be presented as an enumerative localization method, not as more
   reconstruction-accurate than exact full-coverage LCS or semi-global
   alignment.
4. Current pooled LLMarkers labels are not independent lexical gold. Marker
   genes can support a descriptive lexical analysis, while cell-type and
   relation correctness require the independent annotations planned in Stage 3.

No manuscript text was changed as part of this stage.
