"""EasyInfer command-line interface entry point."""

from __future__ import annotations

import argparse
import sys

from easyinfer import __version__


def _cmd_register(_args: argparse.Namespace) -> int:
    """Register EasyInfer plugins with vLLM and related frameworks."""
    from easyinfer.plugins import register

    register()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``easyinfer`` CLI."""
    parser = argparse.ArgumentParser(
        prog="easyinfer",
        description="EasyInfer: Ascend NPU LLM inference deployment toolkit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", title="commands")

    register_parser = subparsers.add_parser(
        "register",
        help="Register EasyInfer plugins",
    )
    register_parser.set_defaults(func=_cmd_register)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``easyinfer`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
