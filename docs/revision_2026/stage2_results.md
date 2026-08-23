# Stage 2 Controlled Multi-Gap Stress Test

## Purpose And Interpretation

Stage 2 measures lexical alignment under deterministic interior deletions. It
is a controlled ablation, not a model of natural LLM errors and not an
evaluation of semantic correctness.

The construction has an important consequence: every shortened target remains
an ordered subsequence of its unchanged source. Therefore, an exact ordered
method should not lose full-target support as the number or width of deletions
increases, provided the tokenizer represents every target token in the source.
The experiment tests this invariant, measures localization and enumeration
burden, and separately tests whether the heuristic used by
`difflib.SequenceMatcher` preserves full-target support.

## Frozen Design

- The common cohort contains targets with at least 13 whitespace chunks.
- Each of 5,863 base records contributes one contiguous control and nine
  noncontiguous variants: one to three gaps crossed with widths of one to three
  removed chunks per gap.
- Gap positions use the deterministic `balanced_retained_segments_v1` rule;
  every removed index is stored with the variant.
- The 58,630 variants cover 300 source documents. Document-level splits contain
  49,910 development variants and 8,720 held-out variants.
- All 52,767 shortened variants are noncontiguous under character-exact
  matching.
- Methods are character-exact matching, `taln`, full-coverage LCS, and
  `difflib.SequenceMatcher`. Sequence methods use punctuation, `cl100k_base`,
  and pinned PubMedBERT tokenization. Candidate enumeration is capped at
  100,000 paths.
- Accuracy intervals use 10,000 document-clustered bootstrap resamples with
  seed 20260812. Predeclared paired contrasts also resample by document.

The frozen variant file has SHA-256
`985e539fc03c6c9247d980c05aa97fb96a7f6962da3c9f77205e1c392a326782`.

## Held-Out Results

The held-out run evaluated all 8,720 frozen variants once, with zero method
errors. Its records have SHA-256
`6c12b6a63e35291a8f2d59fc971d76540245eae2009fddb387424e188ef0f0c5`.

### Full-Target Lexical Support

- Character-exact matching recovered 0 noncontiguous variants.
- `taln` and full-coverage LCS agreed on every tokenizer-level task.
- With punctuation tokenization, `taln` support was invariant across all nine
  deletion conditions: 334/337 BOAT records (99.1%) and 535/535 BIO-BOAT
  records (100.0%). The same records failed or succeeded in the contiguous
  controls, so the deletions introduced no ordered-support failures.
- At three gaps of width three, `difflib` full-target support was 292/337 for
  BOAT (86.6%, document-clustered 95% CI 82.1%-89.8%) and 496/535 for BIO-BOAT
  (92.7%, 95% CI 89.0%-95.4%). The corresponding `taln` estimates remained
  99.1% and 100.0%.
- Across all held-out tokenizers and deletion conditions, there were 373
  `taln`-only `difflib` disagreements and no `difflib`-only disagreements.

Independent audits checked all 172 BOAT and 201 BIO-BOAT disagreements. Every
case contained a complete ordered token path and was a genuine
`SequenceMatcher` heuristic miss, usually after its greedy block selection
committed to repeated or common tokens. Source and target hashes, regenerated
deletions, base-record joins, and scoring were consistent. These results show a
limitation of this `difflib` heuristic under sparse, repeated-token targets; they
do not show that `taln` uniquely solves ordered-subsequence matching.

### Tokenization And Localization

At three gaps of width three, held-out `taln` full-target support was:

| Dataset | Punctuation | PubMedBERT | `cl100k_base` |
| --- | ---: | ---: | ---: |
| BOAT | 99.1% | 99.1% | 96.7% |
| BIO-BOAT | 100.0% | 100.0% | 96.6% |

These rates are invariant over deletion conditions. The lower `cl100k_base`
rate reflects contextual token-boundary behavior already present in the
controls, not increasing gap severity.

For punctuation-tokenized `taln`, any enumerated candidate covered an annotated
gold interval in 99.1% of BOAT and 99.8% of BIO-BOAT records at three gaps of
width three. This any-candidate measure is intentionally not treated as a fair
top-1 comparison with LCS or `difflib`, which return one selected path. Ranking
enumerated candidates remains a separate problem.

### Candidate Burden

For the same severe condition, the median number of enumerated candidates was
2 in both datasets and the 95th percentile was 28 for BOAT and 30 for BIO-BOAT.
Multiple source locations occurred in 46.1% and 43.2% of full-support tasks,
respectively. Candidate burden varied across conditions but did not increase
monotonically with gap count or width. No held-out task reached the 100,000-path
cap. Two development PubMedBERT tasks reached the cap and remain explicitly
flagged; counts for those tasks are lower bounds.

Runtime values are retained per record and summarized by dataset, condition,
tokenizer, and method. They are diagnostic rather than benchmark claims because
the serial method order and tokenizer caching were not controlled as a timing
experiment. The aggregate field formerly described as a zero-alignment rate is
reported as `no_matched_token_path`, which accurately distinguishes absence of
any matched path from failure of full-target reconstruction.

## Conclusion

The controlled stress test gives a narrow but clear result. Exact ordered
alignment remains invariant as one gap becomes several gaps, while
`SequenceMatcher` increasingly fails to retain a complete path in the most
fragmented conditions. Enumeration usually remains small but has a repeated-
text tail and often yields more than one source location. These findings should
be used to motivate explicit full-coverage scoring and candidate ranking, not
as a simulation of general LLM behavior or as an algorithmic superiority claim
over exact full-coverage LCS.

## Reproducible Artifacts

Generated data under `data/revision_2026/stage2/` are ignored by git but covered
by frozen manifests and checksums:

- `multigap_variants.jsonl` and `multigap_variant_manifest.json`
- `development/multigap_benchmark_records.jsonl` and its summary/table
- `heldout_test/multigap_benchmark_records.jsonl`, run metadata, completion
  ledger, summary, and table
- combined `multigap_benchmark_summary.json` and table

The tracked figure builder is
`analysis/scripts/make_multigap_figure.py`. It reads the held-out aggregate
summary and writes
`docs/revision_2026/figures/stage2_multigap_stress_test.tex` without notebook
state; compiling that standalone source produces the corresponding PDF.
