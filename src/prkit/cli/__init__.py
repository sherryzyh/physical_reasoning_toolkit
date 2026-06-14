"""Command line interface for dataset-focused PRKit workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from prkit import __version__
from prkit.datasets import DatasetHub


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with all subcommands registered."""
    parser = argparse.ArgumentParser(
        prog="prkit",
        description="Dataset utilities for the Physical Reasoning Toolkit.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed PRKit version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List available datasets.")

    info_parser = subparsers.add_parser("info", help="Show dataset metadata.")
    info_parser.add_argument("dataset", help="Dataset name.")

    download_parser = subparsers.add_parser("download", help="Download a dataset.")
    download_parser.add_argument("dataset", help="Dataset name.")
    download_parser.add_argument("--variant", help="Dataset variant to download.")
    download_parser.add_argument("--split", help="Dataset split to download.")
    download_parser.add_argument(
        "--data-dir",
        type=Path,
        help="Target directory. Defaults to the dataset downloader cache path.",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when data already exists.",
    )

    return parser


def _print_dataset_info(info: dict[str, Any]) -> None:
    """Pretty-print *info* as formatted JSON to stdout."""
    print(json.dumps(info, indent=2, sort_keys=True, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the PRKit CLI.

    Args:
        argv: Optional command arguments excluding the program name. If omitted,
            ``sys.argv[1:]`` is used.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    if args.version:
        print(f"prkit {__version__}")
        return 0

    try:
        if args.command == "list":
            for dataset_name in DatasetHub.list_available():
                print(dataset_name)
            return 0

        if args.command == "info":
            _print_dataset_info(DatasetHub.get_info(args.dataset))
            return 0

        if args.command == "download":
            download_path = DatasetHub.download(
                args.dataset,
                data_dir=args.data_dir,
                variant=args.variant,
                split=args.split,
                force=args.force,
            )
            print(f"Downloaded {args.dataset} to {download_path}")
            return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    _build_parser().print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
