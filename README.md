# taln

**T**ext **ALN**ignment: A tool for aligning target strings/spans within source text, supporting both contiguous and non-contiguous matches at token or word level.

## Overview

`taln` finds and highlights complete ordered occurrences of target tokens within
source text, including occurrences interrupted by other source tokens. It
enumerates the valid order-preserving token maps and reports original-source
character offsets for each matched token.

This is lexical grounding: every submitted target token must be present in the
source in order. It does not determine whether a paraphrase, entity
relationship, negated statement, or biological assertion is semantically
correct.

## Features

- **Token-level alignment**: Uses tiktoken (GPT tokenization) for subword-level matching
- **Whitespace alignment**: Word-level matching for cleaner text processing
- **Multiple alignment enumeration**: Finds all complete ordered alignments, not just the first or longest
- **Bounded operation**: Streams candidates lazily and supports explicit alignment caps and truncation metadata
- **Text highlighting**: Visual display of aligned spans with terminal colors
- **Robust text normalization**: Handles Unicode, special characters, and whitespace
- **File or string input**: Accepts direct text strings or paths to `.txt` files

## Installation

```bash
python -m pip install git+https://github.com/sbooeshaghi/taln.git
```

To use LLM-based extraction:

```bash
python -m pip install "taln[extract] @ git+https://github.com/sbooeshaghi/taln.git"
```

For development:

```bash
git clone https://github.com/sbooeshaghi/taln.git
cd taln
python -m pip install -e ".[dev]"
python -m pytest tests
```

**Requirements:**
- Python >= 3.12
- tiktoken
- unidecode
- numpy

Optional for `taln extract`:

- anthropic
- python-dotenv
- json-repair

## Repository Layout

- `taln/`: Python package and command-line entry points.
- `tests/`: unit tests and small fixtures.
- `analysis/scripts/`: reproducible analysis scripts for manuscript results.
- `analysis/notebooks/`: exploratory and figure-generation notebooks.
- `analysis/formalization/`: Lean formalization supporting the supplementary note.
- `docs/submissions/`: manuscript, cover-letter, poster, and talk workspaces.
- `docs/shared/`: shared manuscript figures and bibliography.
- `data/`: benchmark outputs and derived analysis data.

## Usage

### Command Line Interface

`taln` provides two main commands:

#### 1. `aln` - Alignment Extraction

Find and return alignment positions as JSON:

```bash
# Basic usage with strings
taln aln -s "source text here" "target"

# Using file inputs
taln aln -s context.txt query.txt

# Token-level alignment (default)
taln aln -s "Single-cell RNAseq identified Sox2-positive neural progenitor cells." "Sox2 progenitor cells" -tt token

# Whitespace/word-level alignment
taln aln -s "Single-cell RNAseq identified Sox2-positive neural progenitor cells." "Sox2 progenitor cells" -tt whitespace

# Return only first alignment
taln aln -s context.txt query.txt --single

# Return at most 100 alignments with explicit completion/truncation metadata
taln aln -s context.txt query.txt --max-alignments 100

# Request status metadata for an uncapped search
taln aln -s context.txt query.txt --with-status

# Save output to file
taln aln -s context.txt query.txt -o alignments.json
```

**Output format:**
```json
[
  [
    {
      "token": " Sox",
      "enc_token": 39645,
      "start_idx": 29,
      "end_idx": 33
    },
    {
      "token": "2",
      "enc_token": 17,
      "start_idx": 33,
      "end_idx": 34
    },
    ...
  ]
]
```

Each alignment contains:
- `token`: The normalized token surface used for matching
- `enc_token`: The encoded token ID (or the token itself for whitespace mode)
- `start_idx`: Zero-based character index where evidence starts in the original
  source
- `end_idx`: Half-open character index where evidence ends in the original
  source

Tokens also retain `norm_start_idx` and `norm_end_idx` for debugging against
the normalized string. Normalization expansions may map several normalized
characters to the same original-source interval; original offsets always refer
to the input string supplied by the caller.

A capped call returns a JSON object containing `alignments`, `status`,
`truncated`, `exhaustive`, `max_alignments`, and
`returned_alignment_count`. An uncapped call retains the original JSON-list
format. `--single` stops after the first complete alignment and also retains its
original token-list format.

#### 2. `light` - Text Highlighting

Highlight aligned spans in the source text with terminal colors:

```bash
# Basic highlighting
taln light -s "source text here" "target"

# Highlight and save to file
taln light -s context.txt query.txt -o highlighted.txt

# Token-level highlighting
taln light -s context.txt query.txt -tt token

# Word-level highlighting
taln light -s context.txt query.txt -tt whitespace
```

### Python API

```python
from taln.taln_aln import align_ng, tokenize, reconstruct_target_by_token

# Align target within source
source = "Single-cell RNAseq identified Sox2-positive neural progenitor cells."
target = "Sox2 progenitor cells"

# Get all alignments (token-level)
alignments = align_ng(source, target, ttype="token")
print(f"Found {len(alignments)} alignment(s)")

# Get all alignments (whitespace-level)
alignments_ws = align_ng(source, target, ttype="whitespace")

# Bound enumeration; capped calls always return status metadata
result = align_ng(source, target, ttype="token", max_alignments=100)
print(result.status, result.truncated, len(result.alignments))

# Reconstruct target from alignment
if alignments:
    reconstructed = reconstruct_target_by_token(source, alignments[0])
    print(f"Reconstructed: {reconstructed}")
```

## How It Works

1. **Text Normalization**: Source and target texts are normalized to handle Unicode characters, special symbols, and whitespace consistently.

2. **Tokenization**: Text is tokenized either:
   - Token-level: Using tiktoken (GPT's tokenizer) for subword tokenization
   - Whitespace-level: Splitting on whitespace boundaries

3. **N-gram Indexing**: The source text is indexed by building n-grams (default n=1) and mapping them to their positions.

4. **Target Alignment**: Every n-gram in the target is matched against the source index. If any target n-gram is absent, there is no complete alignment.

5. **Grouping**: Valid paths are streamed with depth-first search, ensuring strictly increasing positions (no reuse, overlaps, or backwards matches). Adjacent repeated target tokens remain distinct.

6. **Result**: Returns complete alignments with original-source character indices for each matched token. A caller may request a finite cap and inspect explicit truncation metadata.

## Evaluation

The corrected evaluation compares exact substring matching, `taln`, LCS,
`difflib`, and an exact semi-global baseline under shared tokenization and
full-target coverage rules. It shows that simple whitespace tokenization is
sensitive to attached punctuation, while a punctuation-aware lexical tokenizer
removes most of that deficit. Under the same exact tokenization, `taln`, exact
LCS, and exact semi-global alignment recover the same lexical support. The
distinct software behavior provided by `taln` is complete enumeration with
auditable offsets and an explicit bounded mode.

The frozen designs, per-stage results, and limitations are documented in
[`docs/revision_2026/`](docs/revision_2026/). Historical notebook figures and
point estimates are not the source of record for the revised analysis.

Use exact substring matching when targets are guaranteed to be contiguous. Use
ordered alignment when complete target tokens may be separated in the source
and every valid source occurrence is needed. Tokenizer choice changes which
lexical units can match; determinism does not make the result invariant to that
choice.

## Reproducing the manuscript results

Every number in the manuscript is produced by a script in `analysis/scripts/`
from inputs whose checksums are recorded in `analysis/config/revision_2026/`.
Each experiment has a frozen specification file that fixes its design, splits,
seeds, and candidate cap before held-out evaluation. Summary tables, run
configurations, run metadata, and sample manifests for every experiment are
tracked under `data/revision_2026/`; the per-record outputs they summarize are
regenerated by the commands below.

```bash
uv sync --dev
uv pip install --python .venv/bin/python -r analysis/config/revision_2026/stage1_requirements.txt

# Frozen inputs (BOAT, BIO-BOAT, splits). Set LLMARKERS_DATA_ROOT or pass
# --llmarkers-root if the llmarkers repository is not a sibling directory.
uv run python analysis/scripts/prepare_revision_inputs.py

# Tokenizer-by-method matrix (Figure 2, Supplementary Figures S1-S2)
uv run python analysis/scripts/run_corrected_baseline_matrix.py --split development --output-root data/revision_2026/stage1
uv run python analysis/scripts/run_corrected_baseline_matrix.py --split heldout_test --output-root data/revision_2026/stage1
uv run python analysis/scripts/summarize_corrected_baseline_matrix.py --records data/revision_2026/stage1/heldout_test/corrected_baseline_records.jsonl --output-dir data/revision_2026/stage1/heldout_test

# Multi-gap stress test (Figure 3)
uv run python analysis/scripts/prepare_multigap_variants.py
uv run python analysis/scripts/run_multigap_benchmark.py --split development
uv run python analysis/scripts/run_multigap_benchmark.py --split heldout_test
uv run python analysis/scripts/summarize_multigap_benchmark.py --records data/revision_2026/stage2/development/multigap_benchmark_records.jsonl data/revision_2026/stage2/heldout_test/multigap_benchmark_records.jsonl --output-dir data/revision_2026/stage2

# Selection assessment (Figure 4); development must run before held-out
uv run python analysis/scripts/run_selection_assessment.py --split development
uv run python analysis/scripts/run_selection_assessment.py --split heldout_test

# LLM comparison (Figure 3); requires ANTHROPIC_API_KEY
uv run python analysis/scripts/prepare_llm_baseline_sample.py
uv run python analysis/scripts/prepare_llm_multigap_extension.py
uv run python analysis/scripts/run_llm_baseline.py --phase heldout_test
uv run python analysis/scripts/run_llm_baseline.py --phase heldout_multigap

# Natural marker records; reads the sibling llmarkers repository
uv run python analysis/scripts/run_natural_benchmark.py

# Figures
uv run --with matplotlib python analysis/scripts/make_tokenizer_figure.py
uv run --with matplotlib python analysis/scripts/make_multigap_matplotlib_figure.py
uv run --with matplotlib python analysis/scripts/make_selection_figure.py
uv run --with matplotlib python analysis/scripts/make_supplementary_grids.py
```

The LLM comparison re-runs the model and will not reproduce the recorded
responses exactly; the recorded raw responses and their scored outputs are the
source of record for the reported numbers.

## Limitations

- A lexical match establishes token presence and order, not semantic entity
  equivalence or relation correctness.
- Enumerating all paths can grow combinatorially on repetitive input. Use
  `--single` or `--max-alignments` for bounded operation.
- A truncated candidate set is not evidence that no alignment exists. Inspect
  the returned status metadata.
- Selecting the contextually or biologically relevant occurrence from several
  valid lexical locations is a separate downstream task.

## Use Cases

- **Question Answering**: Aligning answer spans to source documents
- **Information Extraction**: Grounding explicitly extracted labels in source text
- **Data Labeling**: Mapping annotations to tokenized text
- **Text Comparison**: Identifying ordered shared lexical content between documents
- **Search & Retrieval**: Locating complete ordered token sequences

## License

`taln` is released under the MIT License; see [`LICENSE`](LICENSE).

## Author

Sina Booeshaghi

## Citation

If you use `taln` in your research, please cite:

```bibtex
@software{taln,
  author = {Booeshaghi, A. Sina},
  title = {{taln}: Text alignment for non-contiguous spans},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/sbooeshaghi/taln}
}
```
