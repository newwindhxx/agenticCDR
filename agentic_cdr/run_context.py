from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import configured_path, public_config
from .io_utils import atomic_write_text, atomic_write_yaml


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path
    profile: str

    @property
    def state_path(self) -> Path:
        return self.run_dir / "memory" / "state.json"

    @property
    def cache_path(self) -> Path:
        return self.run_dir / "llm_cache.jsonl"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "run.log"

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")
        print(message, flush=True)


def _pointer_path(config: dict[str, Any], profile: str) -> Path:
    runs = configured_path(config, "runs_dir")
    return runs / f".latest_{config['experiment_name']}_{profile}"


def latest_run(config: dict[str, Any], profile: str) -> RunContext:
    pointer = _pointer_path(config, profile)
    if not pointer.exists():
        raise FileNotFoundError(f"No run found for profile {profile}; train first")
    run_id = pointer.read_text(encoding="utf-8").strip()
    run_dir = configured_path(config, "runs_dir") / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Latest run directory is missing: {run_dir}")
    return RunContext(run_id, run_dir, profile)


def create_or_resume_run(
    config: dict[str, Any], profile: str, resume: bool, run_id: str | None = None
) -> RunContext:
    if resume and run_id is None and _pointer_path(config, profile).exists():
        return latest_run(config, profile)
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{config['experiment_name']}_{profile}_{stamp}"
    run_dir = configured_path(config, "runs_dir") / run_id
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Run already exists: {run_dir}; use --resume or another --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "memory").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    context = RunContext(run_id, run_dir, profile)
    atomic_write_yaml(run_dir / "config.resolved.yaml", public_config(config))
    manifest = configured_path(config, "processed_dir") / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Processed manifest is missing: {manifest}")
    shutil.copyfile(manifest, run_dir / "data_manifest.json")
    atomic_write_text(_pointer_path(config, profile), run_id + "\n")
    return context
