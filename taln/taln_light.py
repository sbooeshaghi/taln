from taln.taln_aln import align_ng


# add functionality to check if the source and target are paths to files
def setup_taln_light_args(parser):
    subparser = parser.add_parser(
        "light",
        help="Highlight TALN module",
    )
    subparser.add_argument(
        "-s",
        "--source",
        type=str,
        help="Source text to align",
        required=True,
    )
    subparser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output file for the highlighted content",
        required=False,
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
        "target",
        type=str,
        help="Target text to align",
    )
    return subparser


def validate_taln_light_args(parser, args):
    src = args.source
    tgt = args.target
    ttype = args.tokenization_type
    alns = align_ng(src, tgt, ttype)
    output = args.output
    if output:
        with open(output, "w") as f:
            for i in alns:
                f.write(highlight_tokens(src, i))
    else:
        for i in alns:
            print(highlight_tokens(src, i))


def highlight_tokens(source, pos, start_hi="\033[42m", end_hi="\033[0m"):
    last_idx = 0
    highlighted = []
    for p in pos:
        highlighted.append(source[last_idx : p["start_idx"]])
        highlighted.append(f"{start_hi}{source[p['start_idx'] : p['end_idx']]}{end_hi}")
        last_idx = p["end_idx"]
    highlighted.append(source[last_idx:])
    return "".join(highlighted)
