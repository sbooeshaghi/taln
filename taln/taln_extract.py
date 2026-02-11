import json
import logging
import os
import re

from taln.utils import load_text

logger = logging.getLogger(__name__)

# Default prompt for LLM extraction
DEFAULT_PROMPT = """**Task**

Transform the full text of a scientific paper into a JSON array of `(source, targets)` pairs.

**Input**

Plain text of a paper, with paragraphs separated by blank lines.

**Output**

A JSON array where each element corresponds to one paragraph:

```json
{
  "source": "<full paragraph text>",
  "targets": ["<contiguous span>", ...]
}
```

**Span rules**

- Each target must be an exact, contiguous substring of its source paragraph—copy verbatim, do not paraphrase or summarize.
- Spans may range from short phrases to full sentences.
- Spans within a paragraph should not heavily overlap.

**What to highlight**

Select informative content: definitions, claims, results, comparisons, numeric values, named entities, causal or explanatory statements. Avoid filler or purely connective text.

**Distribution**

- Every paragraph must have at least 1 target.
- Mean ≈ 2.4 targets per paragraph; median = 2.
- Most paragraphs: 1–4 targets; rare paragraphs may reach 8+.

**Output format**

Return only valid JSON. No explanations or commentary."""


def setup_taln_extract_args(parser):
    subparser = parser.add_parser(
        "extract",
        help="Extract source/target pairs from a document using an LLM",
    )
    subparser.add_argument(
        "input",
        type=str,
        help="Input document (string or .txt file path)",
    )
    subparser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    subparser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Custom prompt file (uses default if not provided)",
    )
    subparser.add_argument(
        "--api-key",
        type=str,
        help="Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)",
    )
    subparser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use (default: claude-sonnet-4-20250514)",
    )
    subparser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="Max tokens for LLM response (default: 16000)",
    )
    return subparser


def validate_taln_extract_args(parser, args):
    # Load input document
    document = load_text(args.input)
    logger.info(f"Loaded document: {len(document)} characters")

    # Load custom prompt if provided
    if args.prompt:
        prompt = load_text(args.prompt)
        logger.info(f"Using custom prompt from {args.prompt}")
    else:
        prompt = DEFAULT_PROMPT
        logger.info("Using default prompt")

    # Get API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try loading from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        except ImportError:
            pass

    if not api_key:
        parser.error(
            "API key required. Set ANTHROPIC_API_KEY environment variable, "
            "use --api-key flag, or create a .env file"
        )

    # Process document
    results = process_document(
        document=document,
        prompt=prompt,
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_tokens,
    )

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {len(results)} pairs to {args.output}")
    print(f"Extracted {len(results)} verified source/target pairs to {args.output}")


def call_llm(document: str, prompt: str, api_key: str, model: str, max_tokens: int) -> str:
    """Call Claude API with the document and prompt."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required. Install with: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)

    logger.info(f"Calling {model} with {len(document)} character document")

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n---\n\n{document}",
            }
        ],
    )

    response_text = message.content[0].text
    logger.info(f"Received response: {len(response_text)} characters")

    return response_text


def parse_llm_response(response: str) -> list[dict]:
    """Parse JSON from LLM response, handling markdown code blocks and malformed JSON."""
    from json_repair import repair_json

    # Strip markdown code blocks if present
    text = response.strip()

    # Remove ```json ... ``` wrapper
    if text.startswith("```"):
        # Find the end of the first line (after ```json or ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        # Remove trailing ```
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # First try standard JSON parsing
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Standard JSON parsing failed: {e}")
        logger.info("Attempting JSON repair...")

        # Try to repair the JSON
        try:
            repaired = repair_json(text, return_objects=True)
            if isinstance(repaired, list):
                data = repaired
                logger.info("JSON repair successful")
            else:
                raise ValueError(f"Expected JSON array after repair, got {type(repaired).__name__}")
        except Exception as repair_error:
            logger.error(f"JSON repair also failed: {repair_error}")
            logger.error(f"Response text (first 500 chars): {text[:500]}...")
            raise ValueError(f"LLM response was not valid JSON: {e}")

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    return data


def find_all_occurrences(source: str, target: str) -> list[int]:
    """Find all start indices where target appears in source (case-sensitive, exact match)."""
    if not target:
        return []

    positions = []
    start = 0
    while True:
        idx = source.find(target, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1  # Allow overlapping matches

    return positions


def verify_and_explode_pairs(llm_pairs: list[dict]) -> list[dict]:
    """
    Take LLM output (source with multiple targets) and explode into
    individual verified pairs with idx_start positions.

    Input format from LLM:
    [{"source": "...", "targets": ["...", "..."]}, ...]

    Output format (BOAT):
    [{"source": "...", "target": "...", "idx_start": [...]}, ...]
    """
    results = []
    stats = {
        "total_targets": 0,
        "verified_targets": 0,
        "failed_targets": 0,
    }

    for pair in llm_pairs:
        source = pair.get("source", "")
        targets = pair.get("targets", [])

        if not source:
            logger.warning("Skipping pair with empty source")
            continue

        if not isinstance(targets, list):
            targets = [targets]

        for target in targets:
            stats["total_targets"] += 1

            if not target or not isinstance(target, str):
                logger.warning(f"Skipping invalid target: {target}")
                stats["failed_targets"] += 1
                continue

            # Find all occurrences of target in source
            positions = find_all_occurrences(source, target)

            if positions:
                results.append(
                    {
                        "source": source,
                        "target": target,
                        "idx_start": positions,
                    }
                )
                stats["verified_targets"] += 1
                logger.debug(
                    f"Verified target '{target[:50]}...' at {len(positions)} position(s)"
                )
            else:
                # Target not found - try some normalization
                # Sometimes LLM may have minor whitespace differences
                normalized_source = re.sub(r"\s+", " ", source)
                normalized_target = re.sub(r"\s+", " ", target)
                positions = find_all_occurrences(normalized_source, normalized_target)

                if positions:
                    # Found with normalization - use original source but note it
                    results.append(
                        {
                            "source": source,
                            "target": target,
                            "idx_start": positions,
                            "_normalized": True,
                        }
                    )
                    stats["verified_targets"] += 1
                    logger.debug(
                        f"Verified target (normalized) '{target[:50]}...' at {len(positions)} position(s)"
                    )
                else:
                    stats["failed_targets"] += 1
                    logger.warning(
                        f"Target not found in source: '{target[:80]}...'"
                    )

    logger.info(
        f"Verification: {stats['verified_targets']}/{stats['total_targets']} targets verified "
        f"({stats['failed_targets']} failed)"
    )

    return results


def process_document(
    document: str,
    prompt: str,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 16000,
) -> list[dict]:
    """
    Main processing pipeline:
    1. Call LLM to extract source/target pairs
    2. Parse JSON response
    3. Verify targets exist in sources
    4. Return BOAT-format results
    """
    # Step 1: Call LLM
    response = call_llm(
        document=document,
        prompt=prompt,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
    )

    # Step 2: Parse response
    llm_pairs = parse_llm_response(response)
    logger.info(f"Parsed {len(llm_pairs)} paragraph(s) from LLM response")

    # Step 3: Verify and explode pairs
    results = verify_and_explode_pairs(llm_pairs)

    return results
