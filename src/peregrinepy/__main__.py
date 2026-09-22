"""The peregrine command: a case run, and the tools around one, each a
subcommand; `python -m peregrinepy` is the same."""

import argparse

from .tools import tools


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="peregrine", description="Run a PEREGRINE case, or work on its files."
    )
    subparsers = parser.add_subparsers(dest="tool", required=True, metavar="tool")
    for tool in tools:
        sub = subparsers.add_parser(tool.name, help=tool.help, description=tool.help)
        tool.addArguments(sub)
        sub.set_defaults(main=tool.main)
    args = parser.parse_args(argv)
    args.main(args)


if __name__ == "__main__":
    main()
