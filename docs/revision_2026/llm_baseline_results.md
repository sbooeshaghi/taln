# LLM Lexical Baseline Results

Status: final. Development and held-out runs completed 2026-08-14 under
`analysis/config/revision_2026/llm_baseline_spec.json` (frozen 2026-08-14;
model substitution and prompt freeze recorded in the spec).

Generated outputs:

- `data/revision_2026/llm_baseline/llm_baseline_sample_manifest.json`
- `data/revision_2026/llm_baseline/{development,heldout_test}/` (records,
  summary, metadata)
- `data/revision_2026/llm_baseline/raw/` (immutable raw API responses,
  including all prompt-development iterations, keyed by prompt hash)

Scripts: `analysis/scripts/prepare_llm_baseline_sample.py`,
`analysis/scripts/run_llm_baseline.py`.

## Question

A natural question is how exact alignment compares with simply asking an LLM.
Semantic judging is a different task and remains out of scope. This experiment
instead asks the model to perform the same lexical task the alignment methods
perform: given
the source and the submitted text, return the verbatim source excerpts that
contain the submitted words in order. The model's evidence is scored with the
same evaluator, the same normalization, and the same set-valued localization
rule as every alignment method, on the identical sampled tasks.

Boundary: every sampled task has complete lexical support by construction.
This measures whether the model can produce and locate the supporting
evidence. It does not measure rejection of unsupported text.

## Design

- Model: `claude-sonnet-4-5-20250929`, temperature 0, one recorded run. The
  originally pinned extraction-model snapshot (`claude-sonnet-4-20250514`) was
  retired from the API before any successful call; the substitution and the
  extraction-provenance overlap are recorded in the spec.
- Sample: document-clustered, seed 20260814, frozen before any call. Held-out:
  2,153 tasks (BOAT 533 one-gap, 363 contiguous, 20 severe multi-gap;
  BIO-BOAT 624 one-gap, 298 contiguous, 315 severe multi-gap).
- Prompt: frozen after two iterations on a disjoint 200-task development
  sample. Iteration 1 stated the word-subsequence rule; the model instead
  answered whether the text appears as a phrase and declared 32 of 104
  supported one-gap targets unsupported. Iteration 2 added an explicit
  subsequence definition, one worked example, and brief reasoning before the
  final JSON, reaching 199 of 200. The development sample contained only BOAT
  documents, so BIO-BOAT performance is untouched by prompt tuning.
- Excerpts are located by normalized exact search in the source; the model is
  never asked for character positions. Refusals, unparseable replies, and
  non-verbatim excerpts count as failures and remain in the denominator.

## Held-out results

Pooled over all 2,153 held-out tasks, the model produced complete verbatim
lexical evidence for 2,083 (96.75%) and gold-consistent source locations for
2,067 (96.01%). On the same tasks, `taln` reconstructed 2,151 (99.91%) and
localized 2,147 (99.72%).

By stratum (reconstruction; document-clustered 95% CI for the model):

| Stratum | n | LLM | `taln` | exact substring | `difflib` |
| --- | ---: | ---: | ---: | ---: | ---: |
| BOAT contiguous | 363 | 99.72% (99.35-100) | 100% | 100% | 100% |
| BOAT one-gap | 533 | 98.87% (98.24-99.71) | 100% | 0% | 96.06% |
| BOAT 3 gaps x width 3 | 20 | 100% | 100% | 0% | 90.00% |
| BIO-BOAT contiguous | 298 | 94.97% (76.19-100) | 100% | 100% | 100% |
| BIO-BOAT one-gap | 624 | 95.99% (89.05-99.30) | 99.68% | 0% | 99.68% |
| BIO-BOAT 3 gaps x width 3 | 315 | 92.70% (83.33-98.04) | 100% | 0% | 94.60% |

Full method-by-stratum tables, localization rates, and intervals are in
`data/revision_2026/llm_baseline/heldout_test/llm_baseline_summary.json`.

## Failure taxonomy

All 70 model reconstruction failures, by cause:

| Cause | Tasks |
| --- | ---: |
| Model refusal or empty reply | 46 |
| Evidence located outside every gold location | 16 |
| Declared the supported target unsupported | 8 |
| Unparseable reply | 7 |
| Returned excerpts that do not support the target | 6 |
| Excerpt not verbatim in the source | 3 |

(The last three rows overlap the reconstruction count differently: 70 tasks
failed reconstruction; the 16 wrong-location tasks failed localization only.)

Three observations:

1. **All 46 refusals occurred on BIO-BOAT passages.** The model's safety layer
   declined some biomedical texts outright, returning no content. The same 46
   tasks refused deterministically on a second attempt. A deterministic
   alignment method has no refusal behavior.
2. **The model can reason correctly and still fail the lexical contract.** In
   a representative development failure it identified the correct source
   excerpt in its reasoning ("defense and religious ritual") and then returned
   the submitted text ("defense and ritual") instead of the source text.
   Verbatim copying is exactly the step exact alignment does not delegate to
   generation.
3. **Prompt sensitivity is first-order.** Under the initial prompt the model
   missed 31% of supported one-gap targets by answering a different question.
   The corrected result is conditional on the frozen prompt; the alignment
   methods have no analogous degree of freedom.

## Bounded conclusion

A current LLM, given its best frozen prompt, performs the lexical
evidence-production task well but not exactly: 96.75% reconstruction against
99.91% for exact ordered alignment on identical tasks, with a biomedical
refusal mode, occasional non-verbatim copies, wrong locations, and false
negative declarations. These results support one narrow claim: exact lexical
support and source localization do not need to be delegated to a generative
model, and delegating them introduces failure modes that exact enumeration
does not have. The results do not measure semantic judgment, rejection of
unsupported text, or any capability beyond this task, and they are conditional
on one model snapshot and one recorded run.
