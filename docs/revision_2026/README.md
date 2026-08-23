# Corrected Evaluation

This directory documents the frozen evaluation behind the manuscript. Each
experiment has a specification under `analysis/config/revision_2026/` that
fixes its design before held-out evaluation, a runner under
`analysis/scripts/`, tracked summaries under `data/revision_2026/`, and a
results document here.

## Stage 0 Artifacts

Tracked configuration is stored in `analysis/config/revision_2026/`:

- `input_manifest.json` records input paths, checksums, record counts,
  provenance, and the exact ignored artifacts generated for later stages.
- `split_manifest.json` freezes development and held-out source documents before
  tokenizer, ranking, threshold, or candidate-cap selection.
- `bioboat_provenance.json` records what is known about BIO-BOAT construction and
  explicitly marks execution details that were not preserved.
- `bioboat_extraction_prompt.txt` snapshots the extraction prompt associated with
  the first tracked `taln extract` implementation. Its use in the original run
  is inferred, not independently logged.

The generated record files live in ignored `data/revision_2026/`. Regenerate
them from the repository root with:

```bash
uv run python analysis/scripts/prepare_revision_inputs.py
```

Set `LLMARKERS_DATA_ROOT` or pass `--llmarkers-root` when the sibling LLMarkers
repository is not at its default location.

## Data Handling

Legacy files under `data/` and `tests/papers/` are inputs and are not modified.
The preparation script reconstructs stable document and record identifiers,
restores BIO-BOAT paper membership, preserves all existing LLMarkers records,
and writes new revision-specific artifacts.

Held-out document identities are frozen in the tracked split manifest. They
must not be used to choose tokenizers, ranking rules, thresholds, or candidate
caps. Any future split change requires a new schema version and an explicit
reason; the existing split must not be overwritten after test outcomes are
examined.

## Evaluation Boundary

The evaluation contract is defined in `docs/revision_2026/evaluation_contract.md`.
The active experiments evaluate lexical support and source localization.
Semantic or relational assertion validation is deferred as separate work and
is not an outcome of this revision.

## Stage 1 Artifacts

The corrected baseline matrix, held-out findings, and manual discrepancy audit
are summarized in `docs/revision_2026/stage1_results.md`. Generated per-record
outputs and aggregate tables are under ignored `data/revision_2026/stage1/`.

Install the pinned biomedical-analysis dependencies in the project environment
before reproducing Stages 1 and 2:

```bash
uv sync --dev
uv pip install --python .venv/bin/python \
  -r analysis/config/revision_2026/stage1_requirements.txt
```

Regenerate the frozen development and held-out matrices and their summaries:

```bash
uv run python analysis/scripts/run_corrected_baseline_matrix.py \
  --split development --output-root data/revision_2026/stage1
uv run python analysis/scripts/run_corrected_baseline_matrix.py \
  --split heldout_test --output-root data/revision_2026/stage1

uv run python analysis/scripts/summarize_corrected_baseline_matrix.py \
  --records data/revision_2026/stage1/development/corrected_baseline_records.jsonl \
  --output-dir data/revision_2026/stage1/development
uv run python analysis/scripts/summarize_corrected_baseline_matrix.py \
  --records data/revision_2026/stage1/heldout_test/corrected_baseline_records.jsonl \
  --output-dir data/revision_2026/stage1/heldout_test
uv run python analysis/scripts/summarize_corrected_baseline_matrix.py \
  --records \
    data/revision_2026/stage1/development/corrected_baseline_records.jsonl \
    data/revision_2026/stage1/heldout_test/corrected_baseline_records.jsonl \
  --output-dir data/revision_2026/stage1
```

## Stage 2 Artifacts

The frozen multi-gap design, held-out findings, candidate-burden analysis, and
complete disagreement audit are summarized in
`docs/revision_2026/stage2_results.md`. Generated variants, per-record outputs,
run metadata, completion ledger, and aggregate tables are under ignored
`data/revision_2026/stage2/`.

The held-out diagnostic figure is generated directly from the aggregate JSON:

```bash
uv run python analysis/scripts/prepare_multigap_variants.py
uv run python analysis/scripts/run_multigap_benchmark.py \
  --split development
uv run python analysis/scripts/run_multigap_benchmark.py \
  --split heldout_test
uv run python analysis/scripts/summarize_multigap_benchmark.py \
  --records \
    data/revision_2026/stage2/development/multigap_benchmark_records.jsonl \
    data/revision_2026/stage2/heldout_test/multigap_benchmark_records.jsonl \
  --output-dir data/revision_2026/stage2
uv run python analysis/scripts/make_multigap_figure.py
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory docs/revision_2026/figures \
  docs/revision_2026/figures/stage2_multigap_stress_test.tex
```

## Selection Assessment

Specification: `analysis/config/revision_2026/selection_assessment_spec.json`.
Results: `selection_assessment_results.md`. Five parameter-free rules rank the
unique source intervals produced by complete enumeration; the recommended rule
is chosen on the development split and evaluated once on held-out data.

```bash
uv run python analysis/scripts/run_selection_assessment.py --split development
uv run python analysis/scripts/run_selection_assessment.py --split heldout_test
```

## LLM Lexical Baseline

Specification: `analysis/config/revision_2026/llm_baseline_spec.json`, with
the frozen prompt in `llm_baseline_prompt.txt`. Results:
`llm_baseline_results.md`. A single model performs the same lexical task as
the alignment methods on a document-clustered held-out sample and is scored
with the shared evaluator. Raw responses are immutable and are the source of
record; re-running the model will not reproduce them exactly.

```bash
uv run python analysis/scripts/prepare_llm_baseline_sample.py
uv run python analysis/scripts/prepare_llm_multigap_extension.py
uv run python analysis/scripts/run_llm_baseline.py --phase heldout_test
uv run python analysis/scripts/run_llm_baseline.py --phase heldout_multigap
```

## Natural Marker Records

Results: `natural_benchmark_results.md`. Alignment methods are evaluated on
hand-curated marker records from seven single-cell studies, read from the
sibling `llmarkers` repository, where each label was copied verbatim from the
passage recorded with it.

```bash
uv run python analysis/scripts/run_natural_benchmark.py
```

## Focused Release Check

The revision is checked with:

```bash
uv lock --locked
uv run --locked ruff check taln tests
uv run --locked pytest -q
uv build
(cd analysis/formalization && lake build)
```

The frozen Stage 1 and Stage 2 record hashes are stored in their run metadata.
The generated data are ignored by git because the complete working directory is
large; the input records, per-record outputs, summaries, metadata, and pinned
tokenizer snapshots must be included in the submission archive. BIO-BOAT can be
rerun from the preserved hashed records, but its original raw-source conversion
cannot be reconstructed because those execution details were not retained.
