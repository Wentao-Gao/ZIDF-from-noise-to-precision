"""
Configuration utilities: load YAML/JSON configs and merge with CLI args.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    ext = os.path.splitext(config_path)[1].lower()
    if ext in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise ImportError("PyYAML required to load YAML configs: pip install pyyaml") from e
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    elif ext == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config extension: {ext}")


def apply_config_to_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    # Flatten one level of sections commonly used: data, model, train, diffusion, output
    flat = {}
    for section in ("data", "model", "train", "diffusion", "output", "eval"):
        if section in cfg and isinstance(cfg[section], dict):
            flat.update(cfg[section])
    # Top-level overrides
    for k, v in cfg.items():
        if not isinstance(v, dict):
            flat[k] = v
    for k, v in flat.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args

