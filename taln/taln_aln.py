from collections import defaultdict
import difflib
import tiktoken
import unicodedata
from unidecode import unidecode
import re
import numpy as np

import json

import logging

logger = logging.getLogger(__name__)


def setup_taln_aln_args(parser):
    # take as input the source -s and target (positional)
    subparser = parser.add_parser(
        "aln",
        help="Align non-contiguous token spans",
    )
    # and the type of tokenization
    subparser.add_argument(
        "-s",
        "--source",
        type=str,
        help="Source text to align",
        required=True,
    )
    subparser.add_argument(
        "--single",
        action="store_true",
        help="Return only first alignment",
    )
    subparser.add_argument(
        "-tt",
        "--tokenization-type",
        type=str,
        default="token",
        choices=["token", "whitespace"],
        help="Type of tokenization to use (default: token)",
    )
    subparser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file to save the alignment results",
    )
    # target is positional
    subparser.add_argument(
        "target",
        type=str,
        help="Target text to align",
    )
    return subparser


def validate_taln_aln_args(parser, args):
    src = args.source
    tgt = args.target
    ttype = args.tokenization_type
    output = args.output
    sng = args.single

    alns = align_ng(src, tgt, ttype)
    logging.info(f"{len(alns)} alignments found")
    if sng:
        alns = alns[0]

    if output:
        with open(output, "w") as f:
            json.dump(alns, f, indent=4)
    else:
        print(json.dumps(alns, indent=4))
    return


def norm_text(text):
    """
    Normalize text to ensure UTF-8 compatibility for NLP processing.

    This function:
    1. Normalizes Unicode to the NFC form
    2. Replaces problematic characters with ASCII equivalents
    3. Handles common special characters that cause issues

    Args:
        text (str): Input text to normalize

    Returns:
        str: Normalized text safe for UTF-8 processing
    """
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return ""

    # Step 1: Unicode normalization to NFC form (composed form)
    # This combines characters and diacritics when possible
    text = unicodedata.normalize("NFC", text)
    text = unidecode(text)

    # Step 2: Map specific problematic characters to ASCII equivalents

    char_map = {
        "–": "-",  # en dash
        "—": "--",  # em dash
        "‘": "'",  # left single quote
        "’": "'",  # right single quote
        "“": '"',  # left double quote
        "”": '"',  # right double quote
        "…": "...",  # ellipsis
        "•": "*",  # bullet
        "·": ".",  # middle dot
        "×": "x",  # multiplication sign
        "÷": "/",  # division sign
        "≤": "<=",  # less than or equal
        "≥": ">=",  # greater than or equal
        "≠": "!=",  # not equal
        "≈": "~",  # approximately equal
        "∞": "inf",  # infinity
        "∂": "d",  # partial differential
        "∫": "integral",  # integral
        "∑": "sum",  # sum
        "∏": "product",  # product
        "√": "sqrt",  # square root
        "∝": "prop to",  # proportional to
        "∠": "angle",  # angle
        "△": "triangle",  # triangle
        "□": "square",  # square
        "∈": "in",  # element of
        "∉": "not in",  # not an element of
        "⊂": "subset",  # subset
        "⊃": "superset",  # superset
        "∪": "union",  # union
        "∩": "intersect",  # intersection
        "⊆": "subseteq",  # subset or equal
        "⊇": "superseteq",  # superset or equal
    }

    for char, replacement in char_map.items():
        text = text.replace(char, replacement)
    # Step 3: Remove any remaining non-ASCII characters (optional)
    # Uncomment if you want to remove ALL non-ASCII characters
    # text = re.sub(r'[^\x00-\x7F]+', '', text)

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)  # .strip()

    return text


def tokenize_with_offsets(text, encoding="cl100k_base"):
    """Tokenizes text and returns tokens with their character start positions."""
    enc = tiktoken.get_encoding(encoding)
    tokens = []

    etks = enc.encode(text)
    dtks, ptks = enc.decode_with_offsets(etks)

    assert len(etks) == len(ptks)

    for t, p in zip(etks, ptks):
        token = enc.decode_single_token_bytes(t).decode("utf-8")
        tobj = {
            "token": token,
            "enc_token": t,
            "start_idx": p,
            "end_idx": p + len(token),
        }
        tokens.append(tobj)
    return tokens


def tokenize_whitespace_with_offsets(text):
    """Tokenizes text on whitespace and returns tokens with their character start and end positions."""
    tokens = []

    for word in re.finditer(r"\S+", text):  # word non-whitespace sequences
        token = word.group()
        start_idx = word.start()
        end_idx = word.end()

        tokens.append(
            {
                "token": token,
                "enc_token": token,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
        )

    return tokens


def tokenize(text, ttype="token"):
    TOKENIZER = {
        "token": tokenize_with_offsets,
        "whitespace": tokenize_whitespace_with_offsets,
    }
    nt = norm_text(text)
    tks = TOKENIZER[ttype](nt)
    t2w = defaultdict(list)
    for tk in tks:
        t2w[tk["enc_token"]].append(tk)
    return (tks, t2w)


def build_index(text, k=1, ttype="token"):
    tokens, t2w = tokenize(text, ttype)
    ngram_to_id = {}  # Maps n-grams to unique IDs
    id_to_ngram = {}  # Map unique IDs to n-grams
    ngram_id_to_pos = defaultdict(list)  # Positions of each n-gram id
    ngrams = []  # actual list of ngrams (and corresponding tokens)
    counter = 0

    enc_tokens = [i["enc_token"] for i in tokens]

    for i in range(len(enc_tokens) - k + 1):
        ngram = tuple(enc_tokens[i : i + k])  # encoded tuple of tokens

        # get the ngram id or add it
        if ngram not in ngram_to_id:
            ngram_to_id[ngram] = counter
            id_to_ngram[counter] = ngram
            counter += 1
        ngram_id = ngram_to_id[ngram]

        ngram_id_to_pos[ngram_id].append(i)

        ngrams.append({"ngram_id": ngram_id, "tks": tokens[i : i + k]})

    return (ngram_to_id, id_to_ngram, ngram_id_to_pos, ngrams)


def align_target(target, ngram_to_id, ngram_id_to_pos, k=1, ttype="token"):
    tokens, t2w = tokenize(target, ttype)

    enc_tokens = [i["enc_token"] for i in tokens]

    aln = []

    prev_id = None
    for i in range(len(enc_tokens) - k + 1):
        target_ngram = tuple(enc_tokens[i : i + k])  # build the ngram from the target

        ngram_id = ngram_to_id.get(
            tuple(target_ngram), None
        )  # align the target ngram to the ngrams built from source

        if ngram_id is not None:
            if prev_id is not None and prev_id == ngram_id:
                continue  # Skip duplicate adjacent n-grams
            aln.append(
                {
                    target_ngram: [
                        {"ngram_id": ngram_id, "pos": j}
                        for j in ngram_id_to_pos[ngram_id]
                    ]
                }
            )
            prev_id = ngram_id

    return aln


def group_ngrams(aln):
    position_lists = [list(entry.values())[0] for entry in aln]

    def dfs(idx, path):
        if idx == len(position_lists):
            yield path
            return
        last_pos = path[-1]["pos"] if path else -1
        for candidate in position_lists[idx]:
            if candidate["pos"] > last_pos:
                yield from dfs(idx + 1, path + [candidate])

    grp = list(dfs(0, []))
    return grp


def group_tokens(grp, ngrams):
    # group tokens for each combination of ngrams
    tk_aln = []
    for aln_ngrams in grp:
        pos = aln_ngrams[0]["pos"]
        nid = aln_ngrams[0]["ngram_id"]

        tks = []
        tks += ngrams[pos]["tks"]

        for ng in aln_ngrams[1:]:
            pos = ng["pos"]
            nid = ng["ngram_id"]
            tks += [ngrams[pos]["tks"][-1]]
        tk_aln.append(tks)
    return tk_aln


def build_graph(ngrams, id_to_ngram):
    graph = defaultdict(set)
    for ng in ngrams:
        ngram_id = ng["ngram_id"]
        ngram = id_to_ngram[ngram_id]
        for existing_ngram_id, existing_ngram in id_to_ngram.items():
            if existing_ngram_id == ngram_id:
                continue  # Skip self

            # Check left adjacency (ngram[:-1] matches existing_ngram[1:])
            if ngram[:-1] == existing_ngram[1:]:
                graph[existing_ngram_id].add(ngram_id)

            # Check right adjacency (ngram[1:] matches existing_ngram[:-1])
            if ngram[1:] == existing_ngram[:-1]:
                graph[ngram_id].add(existing_ngram_id)
    return graph


def align_ng(source, target, ttype="token"):
    k = 1
    # if verbose:
    #     print("source", source)
    #     print("target", target)
    #     print("-" * 80)
    ngram_to_id, id_to_ngram, ngram_id_to_pos, ngrams = build_index(source, k, ttype)
    aln = align_target(target, ngram_to_id, ngram_id_to_pos, k, ttype)

    if len(aln) > 0:
        grp = group_ngrams(aln)
        tks = group_tokens(grp, ngrams)
        return tks
    return []


def reconstruct_target_by_token(source, pos, sep=""):
    # reconstruct the target by joining the tokens
    return sep.join([i["token"] for i in pos])


def reconstruct_target_by_idx(source, pos):
    # reconstruct the target by joining the tokens
    return "".join([source[i["start_idx"] : i["end_idx"]] for i in pos])


def align_difflib(source, target, ttype="token"):
    source_tokens, source_t2w = tokenize(source, ttype)
    target_tokens, target_t2w = tokenize(target, ttype)

    source_enc_tokens = [t["enc_token"] for t in source_tokens]
    target_enc_tokens = [t["enc_token"] for t in target_tokens]

    matcher = difflib.SequenceMatcher(None, source_enc_tokens, target_enc_tokens)

    alignments = []
    for block in matcher.get_matching_blocks():
        if block.size > 0:
            alignment = source_tokens[block.a : block.a + block.size]
            alignments.append(alignment)
    # flatten the list of lists
    alignments = [item for sublist in alignments for item in sublist]

    return [alignments]


def align_lcs(source, target, ttype="token"):
    source_tokens, source_t2w = tokenize(source, ttype)
    target_tokens, target_t2w = tokenize(target, ttype)

    source_enc_tokens = [t["enc_token"] for t in source_tokens]
    target_enc_tokens = [t["enc_token"] for t in target_tokens]

    m, n = len(source_enc_tokens), len(target_enc_tokens)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # DP computation
    for i in range(m):
        for j in range(n):
            if source_enc_tokens[i] == target_enc_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    # Backtracking to get LCS alignment
    i, j = m, n
    alignment = []
    while i > 0 and j > 0:
        if source_enc_tokens[i - 1] == target_enc_tokens[j - 1]:
            alignment.append(source_tokens[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    alignment.reverse()
    return [alignment]  # wrapped in a list to match your desired structure
