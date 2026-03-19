"""
squish/config.py

Persistent user configuration for Squish.

Stored at ``~/.squish/config.json``.  All keys are optional; the module
provides typed accessors with sensible defaults.

Supported keys
──────────────
  default_model     str   — Model ID used when none is specified (e.g. "qwen3:8b")
  port              int   — Default server port (default: 11435)
  host              str   — Default server bind address (default: "127.0.0.1")
  whatsapp.*        dict  — WhatsApp integration credentials
  signal.*          dict  — Signal bot credentials
  auto_compress     bool  — Auto-compress after pull (default: True)
  api_key           str   — Default API key (default: "squish")

Public API
──────────
  load()                        — Load config from disk; returns dict
  save(cfg)                     — Persist dict to disk
  get(key, default=None)        — Read a dot-notation key from config
  set(key, value)               — Write a dot-notation key to config
  config_path()                 — Return Path to config file
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

_CONFIG_DIR_ENV = "SQUISH_CONFIG_DIR"
_CONFIG_FILENAME = "config.json"

_DEFAULTS: dict[str, Any] = {
    "port": 11435,
    "host": "127.0.0.1",
    "api_key": "squish",
    "auto_compress": True,
    "default_model": None,
    "whatsapp": {
        "verify_token": "",
        "app_secret": "",
        "access_token": "",
        "phone_number_id": "",
    },
    "signal": {
        "account": "",
        "socket": "127.0.0.1:7583",
    },
}


def config_path() -> Path:
    """Return the absolute path to the Squish config file."""
    config_dir_env = os.environ.get(_CONFIG_DIR_ENV, "").strip()
    if config_dir_env:
        base = Path(config_dir_env).expanduser()
    else:
        base = Path.home() / ".squish"
    return base / _CONFIG_FILENAME


def load() -> dict[str, Any]:
    """
    Load and return the Squish config dict.

    If the config file doesn't exist or is unreadable, returns a deep copy
    of the defaults.
    """
    path = config_path()
    if not path.exists():
        return _deep_copy_defaults()
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        # Merge with defaults so new keys always have a value
        merged = _deep_copy_defaults()
        _deep_merge(merged, raw)
        return merged
    except (json.JSONDecodeError, OSError):
        return _deep_copy_defaults()


def save(cfg: dict[str, Any]) -> None:
    """
    Persist ``cfg`` to the Squish config file.

    Creates the parent directory if it doesn't exist.
    """
    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
            fh.write("\n")
    except OSError as exc:
        raise RuntimeError(f"Could not write config to {path}: {exc}") from exc


def get(key: str, default: Any = None) -> Any:
    """
    Read a value from the config using dot-notation.

    Example::

        get("whatsapp.access_token")  # → ""
        get("port")                   # → 11435
    """
    cfg = load()
    return _dot_get(cfg, key, default)


def set(key: str, value: Any) -> None:  # noqa: A001 (shadows builtin intentionally)
    """
    Write a value to the config using dot-notation and persist to disk.

    Example::

        set("default_model", "qwen3:8b")
        set("whatsapp.access_token", "EAABxxx…")
    """
    cfg = load()
    _dot_set(cfg, key, value)
    save(cfg)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _deep_copy_defaults() -> dict[str, Any]:
    """Return a deep copy of _DEFAULTS (so callers can mutate freely)."""
    return json.loads(json.dumps(_DEFAULTS))


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge ``override`` into ``base`` in-place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _dot_get(cfg: dict, key: str, default: Any = None) -> Any:
    """Retrieve a value using dot-notation key."""
    parts = key.split(".", 1)
    if len(parts) == 1:
        return cfg.get(parts[0], default)
    sub = cfg.get(parts[0])
    if not isinstance(sub, dict):
        return default
    return _dot_get(sub, parts[1], default)


def _dot_set(cfg: dict, key: str, value: Any) -> None:
    """Set a value using dot-notation key, creating intermediate dicts as needed."""
    parts = key.split(".", 1)
    if len(parts) == 1:
        cfg[parts[0]] = value
        return
    if not isinstance(cfg.get(parts[0]), dict):
        cfg[parts[0]] = {}
    _dot_set(cfg[parts[0]], parts[1], value)
