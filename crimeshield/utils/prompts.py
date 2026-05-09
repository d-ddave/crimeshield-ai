"""
CrimeShield AI — Prompt Loader Utility.
"""
from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
import yaml

logger = logging.getLogger(__name__)
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

@lru_cache(maxsize=16)
def load_prompt(name: str) -> Dict[str, Any]:
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML mapping")
    return data
