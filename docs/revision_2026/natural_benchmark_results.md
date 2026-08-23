# Natural Marker Records Results

Status: final. Run completed 2026-08-16 with
`analysis/scripts/run_natural_benchmark.py`.

Generated outputs: `data/revision_2026/natural/` (per-pair records, summary,
run metadata with source-file checksums).

## Question

The BOAT and BIO-BOAT benchmarks construct non-contiguous targets by deleting
interior tokens. This benchmark asks whether the same tokenizer and alignment
behavior holds for labels that arose naturally, without construction.

## Population

A hand-curated marker corpus over seven single-cell studies (Emont 2022,
Hildreth 2021, He 2021, Gautam 2021, Adams 2020, Wagner 2020, Shamis 2020).
For each reported marker association the curator recorded the cell-type label,
the marker-gene label, and the manuscript passage reporting it, copying the
labels verbatim from the passage.

- 2,528 curated records in total.
- 877 records whose evidence is manuscript prose. The remaining 1,651 are
  curated from figures, carry no passage text, and are excluded.
- Each record contributes two label--passage pairs, giving 1,754 pairs and 512
  unique pairs (140 cell-type, 372 marker-gene).
- 1,210 labels carry leading or trailing whitespace from manual selection;
  outer whitespace is stripped before evaluation.
- Lexical support is a property of the curation procedure, not an annotation
  added afterwards.

Limits: the corpus asserts only real marker associations, so it contains no
curator-marked negatives and supports no precision or false-acceptance claim.
Character offsets were not recorded during curation, so this benchmark reports
recovery rather than localization. Curation predates `taln` and was performed
without exposure to alignment output.

## Recovery by tokenizer

`taln`, unique label--passage pairs (n=512), with the full pair set (n=1,754)
in parentheses:

| Tokenizer | Recovery |
| --- | ---: |
| Word-level (whitespace) | 216/512, 42.2% (52.4%) |
| Punctuation-aware | 506/512, 98.8% (99.6%) |
| `cl100k_base` | 511/512, 99.8% (99.9%) |
| SciBERT | 511/512, 99.8% (99.9%) |
| PubMedBERT | 510/512, 99.6% (99.9%) |
| Exact substring matching | 510/512, 99.6% (99.9%) |

LCS, semi-global alignment, and `difflib` agreed with `taln` on every pair at
every tokenizer, consistent with Stage 1.

By label type under punctuation-aware tokenization: marker-gene 371/372
(99.7%), cell-type 135/140 (96.4%). Under `cl100k_base` both strata reach
99.7% or above.

## Mechanism of the residual failures

Subword tokenization exceeds punctuation-aware tokenization on this corpus,
reversing the ordering seen on BOAT. Six unique pairs fail under
punctuation-aware tokenization. Four are citation fusions: text conversion
attaches citation numbers directly to words, so the passage reads `human
regulatory T cells17`. A punctuation-aware tokenizer keeps `cells17` as a
single token and the label's `cells` cannot match. A subword tokenizer splits
at the letter--digit boundary and recovers the label. The affected labels are
`human regulatory T cells`, `PVMs`, `differentiating corneal epithelial
cells`, and `corneal epithelial cells`. The remaining two failures are `GAS2`
(compact notation, recovered by `cl100k_base` and SciBERT but not PubMedBERT)
and `WIF1-high fibroblast` (inflection, recovered by no tokenizer).

This converges with the BIO-BOAT finding, where biomedical tokenizers
recovered exactly three targets, all citation-number fusions of the same
shape.

## Named examples

- `Iris stroma Schwann cells` is non-contiguous in its passage ("Iris stroma
  is also populated by Schwann cells that help myelination..."). Exact
  matching fails and ordered alignment recovers it. This is a naturally
  occurring instance of the constructed non-contiguous case.
- `GAS2` was curated from the compact notation `GAS1/2`. Subword alignment
  recovers it; word-level alignment cannot.
- `WIF1-high fibroblast` was recovered by no tokenizer. The passage writes
  `WIF1-high fibroblasts`. The difference is inflection, which is a semantic
  judgment rather than a lexical one.

## Bounded conclusion

The tokenizer effect measured on constructed benchmarks holds on naturally
curated biomedical labels, at nearly the same magnitude: word-level
tokenization loses more than half of real labels, while punctuation-aware and
subword tokenization recover almost all of them. Non-contiguous labels are
rare in this corpus (510 of 512 pairs are contiguous), so it demonstrates
audit behavior rather than a large rescue effect. The failures that survive
correct tokenization are differences of meaning, not of matching.
