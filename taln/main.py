import argparse
import logging
import sys

from taln.taln_aln import setup_taln_aln_args, validate_taln_aln_args
from taln.taln_light import setup_taln_light_args, validate_taln_light_args

from . import __version__


def main():
    # setup parsers
    parser = argparse.ArgumentParser(
        description=f"taln {__version__}: align non-contiguous token spans"
    )
    parser.add_argument(
        "--verbose", help="Print debugging information", action="store_true"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<CMD>",
    )

    # Setup the arguments for all subcommands
    command_to_parser = {
        "aln": setup_taln_aln_args(subparsers),
        "light": setup_taln_light_args(subparsers),
    }

    # Show help when no arguments are given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if len(sys.argv) == 2:
        if sys.argv[1] in command_to_parser:
            command_to_parser[sys.argv[1]].print_help(sys.stderr)
        else:
            parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # setup logging
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    # Setup validator and runner for all subcommands (validate and run if valid)
    COMMAND_TO_FUNCTION = {
        "aln": validate_taln_aln_args,
        "light": validate_taln_light_args,
    }
    COMMAND_TO_FUNCTION[sys.argv[1]](parser, args)


if __name__ == "__main__":
    main()
