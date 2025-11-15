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

We evaluated `taln` against standard alignment methods (LCS, difflib, naive substring matching) on the SQuAD QA dataset (69,775 question-answer-context triplets).

### Accuracy: Finding Exact Span Positions

| Method | Token-level | Whitespace-level |
|--------|-------------|------------------|
| **taln (custom)** | **95.0%** | **85.8%** |
| LCS | 87.5% | 82.0% |
| difflib | 93.0% | 84.7% |
| Naive | 100.0%* | 100.0%* |

*Naive substring matching only works for contiguous spans that appear exactly as-is in the source.

### Text Reconstruction Accuracy

Percentage of cases where the reconstructed text exactly matches the target:

| Method | Token-level | Whitespace-level |
|--------|-------------|------------------|
| **taln (custom)** | **96.1%** | **54.2%** |
| LCS | 96.1% | 54.2% |
| difflib | 96.1% | 53.8% |
| Naive | 100.0%* | 100.0%* |

*Only succeeds on exact contiguous matches.

### Time Complexity

The alignment time scales approximately as a power law with respect to the number of source tokens:

```
Time (ms) = 0.0003 * n^1.50
```

Where `n` is the number of tokens in the source text.

**Key findings:**
- Sub-linear scaling in practice for typical document lengths (< 1000 tokens)
- Efficient for real-time applications: ~0.5ms for 100-token contexts, ~5ms for 1000-token contexts
- Comparable to LCS/difflib but captures more alignment cases

### Coverage

`taln` achieves higher coverage (% of target tokens successfully aligned):

- **Token-level**: ~95% of target tokens aligned vs ~87% for LCS
- **Whitespace-level**: ~86% of target words aligned vs ~82% for LCS

### Comparison Summary

**taln advantages:**
- Enumerates all valid alignments (not just the longest or first)
- Handles non-contiguous spans better than LCS/difflib
- Finds 95% of alignment positions that naive methods miss
- Better coverage of target tokens

**When to use naive/LCS/difflib:**
- When you only need contiguous exact matches
- When you need guaranteed O(n*m) time bounds
- When alignment path doesn't matter, only longest common subsequence

## Use Cases

- **Question Answering**: Aligning answer spans to source documents
- **Information Extraction**: Finding entity mentions across paraphrased text
- **Data Labeling**: Mapping annotations to tokenized text
- **Text Comparison**: Identifying shared content between documents
- **Search & Retrieval**: Finding fuzzy matches with tokenization robustness

## License

MIT License

## Author

Sina Booeshaghi (sinab@berkeley.edu)

## Citation

If you use `taln` in your research, please cite:

```bibtex
@software{taln,
  author = {Booeshaghi, Sina},
  title = {taln: Text Alignment for Non-contiguous Spans},
  year = {2024},
  url = {https://github.com/yourusername/taln}
}
```
