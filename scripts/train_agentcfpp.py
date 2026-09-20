#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_cdr.config import load_config
from agentic_cdr.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train two-domain AgentCF++ memories")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    context = train(
        load_config(args.config),
        args.profile,
        resume=args.resume,
        run_id=args.run_id,
    )
    print(context.run_dir)


if __name__ == "__main__":
    main()
