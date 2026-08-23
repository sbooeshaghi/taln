import argparse
import difflib
import json
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice

import numpy as np
import tiktoken
from unidecode import unidecode

from taln.utils import load_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentResult:
    """Materialized alignments and the completion state of their search."""

    alignments: list
    truncated: bool
    max_alignments: int | None

    @property
    def status(self):
        if self.truncated:
            return "truncated"
        if not self.alignments:
            return "no_alignment"
        return "complete"

    @property
    def exhaustive(self):
        return not self.truncated

    def to_dict(self):
        """Return the JSON-ready status envelope used by the CLI."""
        return {
            "alignments": self.alignments,
            "status": self.status,
            "truncated": self.truncated,
            "exhaustive": self.exhaustive,
            "max_alignments": self.max_alignments,
            "returned_alignment_count": len(self.alignments),
        }


def _parse_nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


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
        help="Source text to align as str or .txt",
        required=True,
    )
    limit_group = subparser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--single",
        action="store_true",
        help="Return only first alignment",
    )
    limit_group.add_argument(
        "--max-alignments",
        type=_parse_nonnegative_int,
        help="Return at most this many alignments with explicit status metadata",
    )
    subparser.add_argument(
        "--with-status",
        "--status",
        dest="with_status",
        action="store_true",
        help="Return alignments in a status-bearing JSON object",
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
        help="Target text to align as str or .txt",
    )
    return subparser


def validate_taln_aln_args(parser, args):
    src = load_text(args.source)
    tgt = load_text(args.target)
    ttype = args.tokenization_type
    output = args.output
    sng = args.single
    max_alignments = getattr(args, "max_alignments", None)
    with_status = getattr(args, "with_status", False)

    if sng and with_status:
        parser.error("--with-status cannot be used with --single")

    if sng:
        aln = next(iter_align_ng(src, tgt, ttype), None)
        payload = aln if aln is not None else []
        logging.info("%d alignment returned (--single)", int(aln is not None))
    elif max_alignments is not None or with_status:
        result = align_ng_result(
            src,
            tgt,
            ttype,
            max_alignments=max_alignments,
        )
        payload = result.to_dict()
        logging.info(
            "%d alignments returned (%s)",
            len(result.alignments),
            result.status,
        )
    else:
        payload = align_ng(src, tgt, ttype)
        logging.info("%d alignments found", len(payload))

    if output:
        with open(output, "w") as f:
            json.dump(payload, f, indent=4)
    else:
        print(json.dumps(payload, indent=4))
    pass


def _coerce_text(text):
    if isinstance(text, str):
        return text
    try:
        return str(text)
    except Exception:
        return ""


_NORMALIZATION_CHAR_MAP = {
    "–": "-",
    "—": "--",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "•": "*",
    "·": ".",
    "×": "x",
    "÷": "/",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "≈": "~",
    "∞": "inf",
    "∂": "d",
    "∫": "integral",
    "∑": "sum",
    "∏": "product",
    "√": "sqrt",
    "∝": "prop to",
    "∠": "angle",
    "△": "triangle",
    "□": "square",
    "∈": "in",
    "∉": "not in",
    "⊂": "subset",
    "⊃": "superset",
    "∪": "union",
    "∩": "intersect",
    "⊆": "subseteq",
    "⊇": "superseteq",
}
_NORMALIZATION_TRANSLATION = str.maketrans(_NORMALIZATION_CHAR_MAP)


def _nfc_units_with_original_spans(text):
    """Yield NFC characters with intervals into the unnormalized input.

    A starter and its following Unicode marks are normalized together so a
    decomposed sequence such as ``e`` plus an acute accent retains the full
    original interval after composition.
    """
    start = 0
    while start < len(text):
        end = start + 1
        while end < len(text):
            next_char = text[end]
            combined = unicodedata.normalize("NFC", text[start : end + 1])
            separate = (
                unicodedata.normalize("NFC", text[start:end])
                + unicodedata.normalize("NFC", next_char)
            )
            if not (
                unicodedata.category(next_char).startswith("M")
                or combined != separate
            ):
                break
            end += 1
        for char in unicodedata.normalize("NFC", text[start:end]):
            yield char, (start, end)
        start = end


def _normalize_text_and_mapping(text):
    text = _coerce_text(text)

    expanded = []
    spans = []
    original_whitespace = []
    for char, original_span in _nfc_units_with_original_spans(text):
        replacement = unidecode(char).translate(_NORMALIZATION_TRANSLATION)
        replacement = replacement.replace("\n", " ")
        source_fragment = text[original_span[0] : original_span[1]]
        source_is_whitespace = bool(source_fragment) and source_fragment.isspace()
        for out_char in replacement:
            expanded.append(out_char)
            spans.append(original_span)
            original_whitespace.append(source_is_whitespace and out_char.isspace())

    normalized = []
    mapping = []
    idx = 0
    while idx < len(expanded):
        if expanded[idx].isspace():
            start = spans[idx][0]
            end = spans[idx][1]
            direct_whitespace_spans = []
            if original_whitespace[idx]:
                direct_whitespace_spans.append(spans[idx])
            idx += 1
            while idx < len(expanded) and expanded[idx].isspace():
                end = spans[idx][1]
                if original_whitespace[idx]:
                    direct_whitespace_spans.append(spans[idx])
                idx += 1
            if direct_whitespace_spans:
                start = direct_whitespace_spans[0][0]
                end = direct_whitespace_spans[-1][1]
            normalized.append(" ")
            mapping.append((start, end))
        else:
            normalized.append(expanded[idx])
            mapping.append(spans[idx])
            idx += 1

    return "".join(normalized), mapping


def norm_text(text):
    """Return the normalized text used by the alignment tokenizers."""
    normalized, _ = _normalize_text_and_mapping(text)
    return normalized


def norm_text_with_mapping(text):
    """Normalize text and map normalized character offsets to original offsets."""
    return _normalize_text_and_mapping(text)


def _map_token_offsets(tokens, mapping):
    mapped = []
    for token in tokens:
        start = token["start_idx"]
        end = token["end_idx"]
        if start < end and start < len(mapping):
            mapped_start = mapping[start][0]
            mapped_end = mapping[min(end - 1, len(mapping) - 1)][1]
        else:
            mapped_start = start
            mapped_end = end

        tk = dict(token)
        tk["norm_start_idx"] = start
        tk["norm_end_idx"] = end
        tk["start_idx"] = mapped_start
        tk["end_idx"] = mapped_end
        mapped.append(tk)
    return mapped


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
    nt, mapping = norm_text_with_mapping(text)
    tks = TOKENIZER[ttype](nt)
    tks = _map_token_offsets(tks, mapping)
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
    """Build one source-position candidate list for every target n-gram.

    An empty result means that the target has no n-grams or at least one
    target n-gram is absent from the source. Repeated target n-grams remain
    distinct entries so downstream grouping can enforce an injective ordered
    map for every target position.
    """
    tokens, t2w = tokenize(target, ttype)

    enc_tokens = [i["enc_token"] for i in tokens]

    aln = []

    for i in range(len(enc_tokens) - k + 1):
        target_ngram = tuple(enc_tokens[i : i + k])  # build the ngram from the target

        ngram_id = ngram_to_id.get(
            tuple(target_ngram), None
        )  # align the target ngram to the ngrams built from source

        if ngram_id is None:
            return []

        aln.append(
            {
                target_ngram: [
                    {"ngram_id": ngram_id, "pos": j}
                    for j in ngram_id_to_pos[ngram_id]
                ]
            }
        )

    return aln


def iter_group_ngrams(aln):
    """Yield each increasing n-gram position path without materializing all paths."""
    position_lists = [list(entry.values())[0] for entry in aln]

    def dfs(idx, path):
        if idx == len(position_lists):
            yield list(path)
            return
        last_pos = path[-1]["pos"] if path else -1
        for candidate in position_lists[idx]:
            if candidate["pos"] > last_pos:
                path.append(candidate)
                yield from dfs(idx + 1, path)
                path.pop()

    yield from dfs(0, [])


def group_ngrams(aln):
    """Return all increasing n-gram position paths.

    This list-returning wrapper is retained for callers that use the original
    helper directly. New code should use :func:`iter_group_ngrams`.
    """
    return list(iter_group_ngrams(aln))


def _tokens_for_ngram_group(aln_ngrams, ngrams):
    pos = aln_ngrams[0]["pos"]
    tokens = list(ngrams[pos]["tks"])

    for ngram in aln_ngrams[1:]:
        pos = ngram["pos"]
        tokens.append(ngrams[pos]["tks"][-1])

    return tokens


def group_tokens(grp, ngrams):
    # group tokens for each combination of ngrams
    return [_tokens_for_ngram_group(aln_ngrams, ngrams) for aln_ngrams in grp]


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


def iter_align_ng(source, target, ttype="token"):
    """Yield complete injective order-preserving maps for all target tokens."""
    k = 1
    ngram_to_id, id_to_ngram, ngram_id_to_pos, ngrams = build_index(source, k, ttype)

    aln = align_target(target, ngram_to_id, ngram_id_to_pos, k, ttype)
    if aln:
        for ngram_group in iter_group_ngrams(aln):
            yield _tokens_for_ngram_group(ngram_group, ngrams)

    # Tiktoken assigns different token IDs to the same characters at a word
    # boundary: standalone "FOX" != mid-text " FOX".  Prepending a space to
    # the target produces the mid-text tokenization, letting the first token
    # match.  We run both and combine the results so the caller can pick the
    # alignment with the best coverage.
    if ttype == "token" and target and not target.startswith(" "):
        aln_sp = align_target(" " + target, ngram_to_id, ngram_id_to_pos, k, ttype)
        if aln_sp:
            for ngram_group in iter_group_ngrams(aln_sp):
                yield _tokens_for_ngram_group(ngram_group, ngrams)


def _validate_max_alignments(max_alignments):
    if max_alignments is None:
        return
    if isinstance(max_alignments, bool) or not isinstance(max_alignments, int):
        raise TypeError("max_alignments must be an integer or None")
    if max_alignments < 0:
        raise ValueError("max_alignments must be non-negative")


def align_ng_result(source, target, ttype="token", max_alignments=None):
    """Materialize alignments with explicit completion metadata.

    At most one alignment beyond ``max_alignments`` is consumed to determine
    whether additional results exist. A zero cap therefore distinguishes an
    empty search from a non-empty search whose results were all withheld.
    """
    _validate_max_alignments(max_alignments)
    alignment_iter = iter_align_ng(source, target, ttype)

    if max_alignments is None:
        alignments = list(alignment_iter)
        truncated = False
    else:
        alignments = list(islice(alignment_iter, max_alignments + 1))
        truncated = len(alignments) > max_alignments
        if truncated:
            alignments.pop()

    return AlignmentResult(
        alignments=alignments,
        truncated=truncated,
        max_alignments=max_alignments,
    )


def align_ng(
    source,
    target,
    ttype="token",
    max_alignments=None,
    *,
    return_status=False,
):
    """Return ordered token alignments, optionally bounded and status-bearing.

    An uncapped call retains the original exhaustive list return type unless
    ``return_status=True`` is requested. Any call with ``max_alignments``
    returns an :class:`AlignmentResult` so a bounded result cannot be mistaken
    for an exhaustive list.
    """
    result = align_ng_result(source, target, ttype, max_alignments)

    if return_status or max_alignments is not None:
        return result
    return result.alignments


def align_ng_casefold(source, target, ttype="token"):
    """Case-insensitive alignment: lowercases source and target, aligns, then
    maps the matched character range back to the original (cased) source tokens.

    Lowercasing can change token boundaries (e.g. "DUSP15" → D|US|P|15 but
    "dusp15" → d|usp|15), so we map by character span rather than by token
    start position."""
    source_lower = source.lower()
    target_lower = target.lower()

    lower_alns = align_ng(source_lower, target_lower, ttype)
    if not lower_alns:
        return []

    orig_tokens, _ = tokenize(source, ttype)

    orig_alns = []
    for aln in lower_alns:
        start = aln[0]["start_idx"]
        end = aln[-1]["end_idx"]
        mapped = [t for t in orig_tokens if t["end_idx"] > start and t["start_idx"] < end]
        if mapped:
            orig_alns.append(mapped)

    return orig_alns


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

    # If nothing matched, return no alignments (not `[[]]`).
    if len(alignments) == 0:
        return []

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

    # If nothing matched, return no alignments (not `[[]]`).
    if len(alignment) == 0:
        return []

    return [alignment]  # wrapped in a list to match your desired structure
