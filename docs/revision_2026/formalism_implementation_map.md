# Formalism And Implementation Map

## Claim Boundary

The Lean development proves relationships among mathematical constraint
families. It does not verify the Python implementation, tokenizer,
normalization, offset mapping, enumeration cap, ranking, runtime, or memory
behavior. Those implementation properties are checked separately with Python
tests.

The precise shared object is a complete exact ordered token alignment:

1. every target-token position has a source-token position;
2. matched source and target token values are equal;
3. source positions strictly increase with target position; and
4. therefore no source position is reused.

Partial token overlap is diagnostic only and is not an instance of this formal
object.

## Definition Map

| Mathematical object | Lean definition | Python realization |
| --- | --- | --- |
| Map from every target position to a source position | `AlignmentMap` | One source-position choice from each target position's candidate list |
| Exact token equality | `Matches` | `align_target` indexes equal encoded token values |
| Strictly increasing source positions | `Ordered` | `iter_group_ngrams` accepts a candidate only when its position is greater than the previous one |
| Complete exact ordered map | `ValidMap Ordered` | `iter_align_ng` yields only paths covering every target token |
| Adjacent source positions | `Adjacent` | A returned path has consecutive integer source-token positions |
| Complete exact contiguous map | `ValidMap Contiguous` | Exact/contiguous evaluation checks full coverage plus adjacent source positions |
| Constraint implication | `ConstraintLE` | Conceptual comparison of candidate sets, not a runtime function |

Adjacent repeated target tokens remain separate target positions. If any target
token has no equal source token, no complete ordered path is returned.

## Theorem Map

The supplementary hierarchy is supported by the following Lean results:

- `contiguous_le_ordered`: every contiguous map is ordered;
- `ordered_le_partial_permutation`: strict order implies injectivity;
- `partial_permutation_le_rearrangement`: every injective map is admitted by
  the unconstrained rearrangement class; and
- `constraint_le_valid_map_inclusion`: implication between constraint families
  induces inclusion between their exact token-matching map sets.

Together these establish the corrected chain:

```text
contiguous <= ordered <= partial permutation <= rearrangement
```

The familiar bijective permutation class is a full-coverage special case, not
the chain node above ordered alignment when the target may be shorter than the
source. The Lean counterexamples make that distinction explicit.

## Implementation Checks

The Python test suite compares `iter_align_ng` with an independent brute-force
enumeration over every binary source sequence of length at most four and every
binary target sequence of length at most three. This checks soundness and
completeness for small sequences, including repeated adjacent target tokens.

Separate regressions check:

- absence of a complete path when any target token is missing;
- injective increasing positions for every returned path;
- original-source slices after Unicode and whitespace normalization;
- lazy behavior for the first result;
- explicit truncation under an alignment cap; and
- distinction between a truncated search and a search with no alignment.

These tests are evidence about the implementation, not machine-checked proofs
of arbitrary Python execution.

## Outside The Formal Model

The following practical choices are deliberately outside the constraint-family
theorem:

- how text is normalized and tokenized;
- how normalized token coordinates map to original-source offsets;
- contextual target tokenization variants;
- lazy traversal and finite candidate caps;
- duplicate-location collapsing;
- ranking or selection of one source occurrence; and
- semantic, relational, or biological correctness.

The manuscript should use the formalism to define the alignment object and
explain the hierarchy, not as a claim that `taln` introduces a new sequence
alignment theorem or that the complete software stack is formally verified.
