"""Configuration loading and validation helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when required config keys are missing."""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            var = match.group(1)
            return os.environ.get(var, "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    _load_dotenv(cfg_path.resolve().parents[1] / ".env")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Config at {cfg_path} must deserialize to a dict")
    return _expand_env(data)


def require_keys(config: dict[str, Any], required: list[str], context: str = "config") -> None:
    missing = [k for k in required if k not in config]
    if missing:
        raise ConfigError(f"Missing keys in {context}: {missing}")


def get_nested(config: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            joined = ".".join(keys)
            raise ConfigError(f"Missing nested key: {joined}")
        cur = cur[key]
    return cur
