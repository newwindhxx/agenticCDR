#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_cdr.config import load_config
from agentic_cdr.data import prepare_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a deterministic Books-to-Movies dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = prepare_pair(load_config(args.config), force=args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
