# Selection Assessment Results

Status: final. Development and held-out runs completed 2026-08-14 under
`analysis/config/revision_2026/selection_assessment_spec.json` (frozen
2026-08-14, with three pre-held-out amendments recorded in the spec).

Generated outputs:

- `data/revision_2026/selection/development/` (records, summary, metadata)
- `data/revision_2026/selection/heldout_test/` (records, summary, metadata)

Runner: `analysis/scripts/run_selection_assessment.py`.

## Question

`taln` enumerates every complete exact ordered alignment. Enumeration does not
choose among them. This experiment measures how well five frozen,
parameter-free selection rules pick the true source location, and what
ambiguity remains when they cannot.

Selection here is lexical occurrence selection. It does not decide which
occurrence is contextually or semantically correct.

## Design

- Population: all frozen Stage 1 one-gap tasks and all frozen Stage 2 variants
  (BOAT and BIO-BOAT), re-enumerated with `taln` under the punctuation-aware
  tokenizer (primary) and `cl100k_base` (sensitivity), candidate cap 100,000.
- Rules (frozen list, no parameters): `input_order`, `shortest_interval`,
  `fewest_gap_characters`, `matched_token_density`, `earliest_interval`. All
  ties break deterministically by earliest start, then shortest interval, then
  enumeration order.
- Rules rank unique original-source intervals and never see gold intervals.
- Procedure: all rules ran on the development split; the recommended rule was
  selected by the predeclared criterion (highest development top-1
  localization on the pooled primary non-contiguous strata, task-weighted,
  ties broken by frozen-list order) and recorded in the development summary
  before the single held-out run.
- Uncertainty: document-clustered bootstrap, 10,000 resamples, seed 20260814.
- Integrity: re-enumerated support outcomes were checked against every frozen
  Stage 1 and Stage 2 record (28,584 held-out checks, zero disagreements).
  Stage 2 candidate counts matched exactly. Two development Stage 1 tasks
  showed count drift attributable to the evaluator's repeated-adjacent-token
  revision (documented in the spec amendment); their support outcomes agree.

## Development selection

On 78,662 fully supported development primary non-contiguous tasks,
`shortest_interval`, `fewest_gap_characters`, and `matched_token_density` each
reached 99.71% top-1 localization; `input_order` and `earliest_interval`
reached 75.47%. The three interval-compactness rules are monotone-equivalent
whenever matched-character counts are constant across candidates, and they
agreed on every development task. By the frozen tie-break,
**`shortest_interval`** was recorded as the selected rule.

## Held-out confirmation

Pooled primary non-contiguous strata, punctuation-aware tokenizer, 13,391
tasks, 13,351 (99.70%) with full lexical support, zero capped tasks:

| Rule | Top-1 localization | 95% CI | Top-5 | MRR |
| --- | ---: | --- | ---: | ---: |
| `shortest_interval` (selected) | 13,316/13,351 (99.74%) | 99.53-99.87% | 99.90% | 0.998 |
| `fewest_gap_characters` | 13,316/13,351 (99.74%) | same | 99.90% | 0.998 |
| `matched_token_density` | 13,316/13,351 (99.74%) | same | 99.90% | 0.998 |
| `input_order` | 10,067/13,351 (75.40%) | 70.28-80.34% | 93.93% | 0.832 |
| `earliest_interval` | 10,067/13,351 (75.40%) | 70.26-80.32% | 93.93% | 0.832 |

The three compactness rules again produced identical selections on every
held-out task. They are one effective rule; the paper should present one
(shortest interval) and note the equivalence.

Per-stratum selected-rule top-1: Stage 1 one-gap BOAT 4,257/4,275 (99.58%),
BIO-BOAT 1,253/1,255 (99.84%); Stage 2 three-gaps-width-three BOAT 332/334
(99.40%), BIO-BOAT 530/535 (99.07%).

Sensitivity tokenizer: under `cl100k_base`, 13,030 of 13,391 tasks had full
support and the selected rule reached 12,920/13,030 (99.16%, 95% CI
98.66-99.59%).

End-to-end joint rate (full support and correct top-1 selection, over all
13,391 primary non-contiguous held-out tasks): 13,316/13,391 (99.44%).

## Failure categorization

The selected rule missed 35 of 13,351 localizable tasks:

- 22 distinguishable rule errors: the gold interval was enumerated but ranked
  second or third behind a shorter non-gold interval;
- 12 tasks where no enumerated candidate interval matched a gold interval
  (any-rank misses; the same 12 tasks bound every rule);
- 1 task flagged lexically indistinguishable: identical normalized source text
  at gold and non-gold locations.

75 of 13,351 tasks (0.56%) carried the indistinguishability flag overall; the
deterministic tie-break resolved most of them in favor of the gold location.
The flag marks ambiguity a text-only comparison cannot resolve; position-based
tie-breaks may still succeed, so it is not a strict ceiling.

## Candidate burden and cap behavior

Pooled held-out primary non-contiguous tasks: median 1 raw candidate path
(p95 22, p99 156, max 29,468) and median 1 unique source interval (p95 8,
p99 20, max 176). 5,735 of 13,351 supported tasks (43.0%) had more than one
unique source interval, so selection is a routine step, not a corner case.

Cap sweep (retained selected-rule top-1 over 13,351 localizable tasks):

| Cap | Any-gold retained | Selected-rule top-1 retained |
| ---: | ---: | ---: |
| 1 | 10,067 (75.40%) | 10,067 (75.40%) |
| 10 | 12,568 (94.14%) | 12,555 (94.04%) |
| 100 | 13,206 (98.91%) | 13,184 (98.75%) |
| 1,000 | 13,333 (99.87%) | 13,310 (99.69%) |
| 10,000 | 13,338 (99.90%) | 13,315 (99.73%) |

A cap of 1 reproduces enumeration-order selection. Complete or near-complete
enumeration is required before the compactness rule can reach its full
accuracy. This is the operational argument for complete enumeration with an
explicit cap and truncation status.

## Bounded conclusion

On these controlled benchmarks, a frozen shortest-interval rule selects the
constructed source location for 99.74% (99.53-99.87%) of held-out
non-contiguous tasks once complete enumeration is available, while
enumeration-order selection reaches 75.40%. These datasets place the true
location at a known construction site; natural text with repeated passages can
be harder, and 0.56% of tasks here were already lexically indistinguishable.
The rule ranks lexical occurrences only. Whether a selected occurrence is the
contextually correct one remains outside this experiment.
