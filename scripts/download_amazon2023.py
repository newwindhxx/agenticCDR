#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_cdr.config import load_config
from agentic_cdr.downloader import download_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the configured Amazon Reviews 2023 files")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--connections", type=int, default=16,
        help="Parallel connections per file (1-16; 1 uses the requests fallback)",
    )
    args = parser.parse_args()
    try:
        for path in download_all(load_config(args.config), connections=args.connections):
            print(path)
    except KeyboardInterrupt:
        print("\nDownload interrupted; rerun the same command to resume.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
