#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_cdr.config import load_config
from agentic_cdr.estimate import estimate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate AgentCF++ requests and tokens")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    print(json.dumps(estimate_run(load_config(args.config), args.profile), indent=2))


if __name__ == "__main__":
    main()
