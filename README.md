# taln

**T**ext **ALN**ignment: A tool for aligning target strings/spans within source text, supporting both contiguous and non-contiguous matches at token or word level.

## Overview

`taln` finds and highlights occurrences of target text within source text, handling cases where the target may appear as non-contiguous tokens. Unlike traditional string matching or greedy subsequence methods (LCS, difflib), `taln` enumerates all possible alignments using n-gram indexing, capturing matches that other methods miss.

## Features

- **Token-level alignment**: Uses tiktoken (GPT tokenization) for subword-level matching
- **Whitespace alignment**: Word-level matching for cleaner text processing
- **Multiple alignment enumeration**: Finds all possible alignments, not just the first or longest
- **Text highlighting**: Visual display of aligned spans with terminal colors
- **Robust text normalization**: Handles Unicode, special characters, and whitespace
- **File or string input**: Accepts direct text strings or paths to `.txt` files

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/taln.git
cd taln

# Install with pip
pip install .

# Or install in development mode
pip install -e .
```

**Requirements:**
- Python >= 3.12
- tiktoken
- unidecode
- numpy

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
- `token`: The actual text token
- `enc_token`: The encoded token ID (or the token itself for whitespace mode)
- `start_idx`: Character index where token starts in source
- `end_idx`: Character index where token ends in source

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

4. **Target Alignment**: Each n-gram in the target is matched against the source index.

5. **Grouping**: All valid alignment paths are enumerated using depth-first search, ensuring monotonically increasing positions (no overlaps or backwards matches).

6. **Result**: Returns all possible alignments with character-level indices for each matched token.

## Benchmarks

We evaluated `taln` against standard alignment methods (LCS, difflib, naive substring matching) on the **BOAT dataset** (Berkeley Ordered Alignment of Text), derived from SQuAD v2.0.

### BOAT Dataset

| Subset | Samples | Unalignable | Mean Alignments | Source (chars/WS/tok) | Target (chars/WS/tok) |
|--------|---------|-------------|-----------------|----------------------|----------------------|
| Contiguous | 35,080 | 159 (0.4%) | 8.4 | 764 / 121 / 158 | 38 / 6 / 8 |
| Non-contiguous | 35,080 | 198 (0.6%) | 7.0 | 764 / 121 / 158 | 33 / 5 / 7 |

WS = whitespace (word-level) tokens; tok = subword tokens.

### Contiguous Alignment Accuracy

| Method | Subword | Word-level |
|--------|---------|------------|
| Naive | **100.0%** | **100.0%** |
| difflib | 99.9% | 95.3% |
| **taln** | 99.4% | 92.3% |
| LCS | 96.3% | 93.3% |

### Non-contiguous Alignment Accuracy

| Method | Subword | Word-level |
|--------|---------|------------|
| **taln** | **98.3%** | 51.6% |
| LCS | 98.0% | 52.0% |
| difflib | 95.0% | 50.0% |
| Naive | 0.5% | 0.6% |

Subword tokenization improves non-contiguous alignment accuracy by ~47% compared to word-level tokenization.

### Runtime Scaling

Runtime scales as a power law with source length (in tokens):

| Method | Scaling |
|--------|---------|
| Naive | ~n^0.19 |
| difflib | ~n^0.89 |
| **taln** | ~n^0.96 |
| LCS | ~n^1.00 |

At 1,000 tokens: difflib is ~5×10² slower than naive, taln is ~7×10² slower, and LCS is ~10³ slower.

### Key Findings

**taln advantages:**
- Enumerates all valid alignments (not just the longest or first)
- Handles non-contiguous spans as well as LCS/difflib under subword tokenization
- Recovers 98.3% of non-contiguous alignments with subword tokenization

**When to use naive alignment:**
- When you only need contiguous exact matches
- When speed is critical and spans are guaranteed contiguous

**When subword tokenization helps:**
- Scientific text with punctuation, symbols, or alphanumeric identifiers (e.g., gene names like "CD96+")
- Text with parenthetical notes or abbreviations that interrupt phrases

## Use Cases

- **Question Answering**: Aligning answer spans to source documents
- **Information Extraction**: Finding entity mentions across paraphrased text
- **Data Labeling**: Mapping annotations to tokenized text
- **Text Comparison**: Identifying shared content between documents
- **Search & Retrieval**: Finding fuzzy matches with tokenization robustness

## License

MIT License

## Author

Sina Booeshaghi

## Citation

If you use `taln` in your research, please cite:

```bibtex
@software{taln,
  author = {Booeshaghi, Sina},
  title = {taln: Text Alignment for Non-contiguous Spans},
  year = {2025},
  url = {https://github.com/sbooeshaghi/taln}
}
```
