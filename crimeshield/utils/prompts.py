"""
CrimeShield AI — Prompt Loader Utility.

Loads prompt definitions from YAML files in the ``prompts/`` directory.
All agents use ``load_prompt(name)`` instead of hardcoding prompt strings.
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
    """
    Load a prompt definition from ``prompts/<name>.yaml``.

    Parameters
    ----------
    name : str
        The prompt name, e.g. ``"rag_policy"``, ``"safety_guard"``.
        Must correspond to a YAML file in the ``prompts/`` directory.

    Returns
    -------
    dict
        Parsed YAML with keys such as ``system``, ``human_template``,
        ``decline_message``, and any additional config keys.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. "
            f"Available prompts: {[p.stem for p in _PROMPTS_DIR.glob('*.yaml')]}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Prompt file {path} must contain a YAML mapping, got {type(data)}")

    logger.debug("Loaded prompt '%s' from %s", name, path)
    return data
