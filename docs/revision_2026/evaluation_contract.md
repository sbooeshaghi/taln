# Alignment Evaluation Contract

Version: `revision-2026-stage0-v2`

This contract applies to every corrected baseline and follow-up experiment. A
result produced under a different contract must be labeled separately.

## Active Tasks

1. **Lexical support:** determine whether every target token has an equal source
   token in the same order under a declared normalizer and tokenizer.
2. **Source localization:** identify which original-source token occurrences
   provide that lexical support.
`taln`, exact matching, LCS, `difflib`, and classical sequence-alignment
baselines are lexical methods. They are evaluated only on the two tasks above.
Semantic entity equivalence, assertion-level correctness, negation, organism
scope, and biological truth are outside this revision. Those questions require
a separate annotation contract and experimental design, and no semantic outcome
is inferred from a lexical match or mismatch.

## Text And Offsets

- **Original text** is the input string before normalization. Public source
  offsets are zero-based, half-open indices into this string.
- **Normalized text** is the result of the declared normalization procedure.
  Normalized offsets are recorded separately and must never be presented as
  original offsets.
- Each normalized source token carries both normalized coordinates and a mapped
  original-source interval.
- Slicing the original source at a returned token's original interval must
  recover the documented original token text or an explicitly documented
  normalization expansion.
- Gold locations are one or more zero-based, half-open original-source
  intervals. Repeated equivalent occurrences are all valid gold locations.

## Tokenization And Context

- Source and target use the same normalizer and tokenizer for a comparison.
- Whitespace tokenization is a relevant first-class baseline.
- Context-sensitive tokenizers may give a standalone target a different first
  token than the same text inside a passage. Every alignment method receives
  the same predeclared target representations, including a contextual
  leading-space representation when applicable.
- Target representations are preprocessing alternatives, not additional
  observations. A task succeeds if any predeclared representation succeeds.
- Adjacent repeated target tokens remain distinct and must each be aligned.

## Candidate Contract

An alignment candidate records:

- the target-token indices it covers;
- the corresponding source-token indices;
- normalized and original source offsets; and
- whether every target token is covered exactly once in increasing source order.

Partial candidates may be retained for diagnostics. They are not successful
lexical verification.

## Metrics

- **Partial token overlap:** at least one target token is aligned, but full target
  coverage is not required. This is diagnostic only.
- **Complete token alignment:** every target token is aligned to an equal source
  token through an injective, order-preserving map.
- **Full lexical reconstruction:** at least one complete candidate reproduces
  the target token sequence under the declared tokenizer and normalization.
  For whitespace tokens, separators are canonicalized rather than deleted.
- **Reconstruction accuracy:** fraction of tasks with full lexical
  reconstruction, independent of which source occurrence supplied it.
- **Localization accuracy:** fraction of tasks for which at least one fully
  reconstructed candidate has the required original-source location. For a
  contiguous target this is an exact match to any gold interval. For a
  controlled non-contiguous target, the predeclared gold criterion is the outer
  boundary of the source evidence from which the variant was derived.
- **Top-k localization:** whether a valid gold location appears among the first
  `k` fully reconstructed, deduplicated source locations under a ranking rule
  frozen on development documents.
- **Zero complete alignment:** no complete candidate exists. A partial LCS,
  matching block, or shared token does not prevent this outcome.

## Candidate Selection

All candidates available within the declared cap are scored for reconstruction.
A method is not penalized because an arbitrary longest or first candidate was
selected before scoring. Ranking and top-k localization are separate analyses.
Truncation, timeouts, and malformed outputs remain in denominators and are
reported explicitly.

## Statistical Unit

Source documents are the split and resampling unit. Development and held-out
test sets are separated by source document. Accuracy intervals use a
document-clustered bootstrap, and method contrasts use paired document-level
comparisons unless a different analysis is predeclared.
