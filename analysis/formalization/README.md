# Lean formalization of the supplementary note

This directory contains a Lean 4 formalization of the constraint-family view of
text alignment from the supplementary note.

It formalizes:

- alignment maps between finite target and source positions;
- exact token matching against a source and target sequence;
- contiguous, ordered, partial permutation, bijective permutation, and rearrangement
  constraint families;
- the partial-order relation induced by implication between constraint
  families; and
- the valid inclusions in the corrected hierarchy.

Build with:

```sh
lake build
```

The corrected hierarchy uses partial permutation alignment as the permutation-like
class above ordered alignment:

```text
contiguous <= ordered <= partial permutation <= rearrangement
```

Partial permutation means injective, unordered alignment: each target position
maps to a distinct source position, but the source does not need to be fully
covered. The familiar bijective permutation case is retained as the full-coverage
special case, and the Lean file records why it should not be used as the chain
node for extracted-text alignment.
