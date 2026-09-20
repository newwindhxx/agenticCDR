#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_cdr.config import load_config
from agentic_cdr.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen AgentCF++ memories on Movies")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--use-group-memory", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    metrics = evaluate(
        load_config(args.config),
        args.profile,
        args.split,
        use_group_memory=args.use_group_memory,
        run_id=args.run_id,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
