from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a mapping")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = [
        "experiment_name",
        "source_domain",
        "target_domain",
        "seed",
        "candidate_count",
        "paths",
        "data",
        "llm",
        "profiles",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing configuration keys: {', '.join(missing)}")
    if config["source_domain"] == config["target_domain"]:
        raise ConfigError("source_domain and target_domain must be different")
    if int(config["candidate_count"]) < 2:
        raise ConfigError("candidate_count must be at least 2")
    if "smoke" not in config["profiles"] or "full" not in config["profiles"]:
        raise ConfigError("profiles must define smoke and full")
    if config["llm"].get("provider") != "deepseek":
        raise ConfigError("This implementation currently supports provider=deepseek")


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def configured_path(config: dict[str, Any], name: str) -> Path:
    try:
        value = config["paths"][name]
    except KeyError as exc:
        raise ConfigError(f"Missing paths.{name}") from exc
    return project_path(config, value)


def get_profile(config: dict[str, Any], profile: str) -> dict[str, Any]:
    try:
        result = copy.deepcopy(config["profiles"][profile])
    except KeyError as exc:
        raise ConfigError(f"Unknown profile: {profile}") from exc
    result["name"] = profile
    return result


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a serializable config without internal fields or credentials."""
    result = {key: copy.deepcopy(value) for key, value in config.items() if not key.startswith("_")}
    llm = result.get("llm")
    if isinstance(llm, dict):
        llm.pop("api_key", None)
    return result
