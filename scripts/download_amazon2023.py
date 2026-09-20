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
    args = parser.parse_args()
    for path in download_all(load_config(args.config)):
        print(path)


if __name__ == "__main__":
    main()
