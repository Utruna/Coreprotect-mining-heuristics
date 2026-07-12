from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat

from .config import load_config
from .pipeline import prepare_workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xray-detector",
        description="Base CLI for the CoreProtect x-ray detection pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Create the expected folder structure.")
    subparsers.add_parser("show-config", help="Display the resolved project configuration.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(Path.cwd())

    if args.command == "init":
        created = prepare_workspace(config)
        for directory in created:
            print(directory)
        return 0

    if args.command == "show-config":
        print(pformat(config))
        return 0

    parser.print_help()
    return 0
