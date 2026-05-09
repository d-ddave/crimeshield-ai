"""
CrimeShield AI — SAR Drafting Agent.

Generates Suspicious Activity Report narratives from alert records
using a structured 4-section prompt template.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate

from crimeshield.utils.llm import get_llm
from crimeshield.utils.prompts import load_prompt

logger = logging.getLogger(__name__)

_REQUIRED_SECTIONS = [
    "## Subject Summary",
    "## Suspicious Activity Description",
    "## Supporting Evidence",
    "## Recommended Action",
]


class SARDraftingAgent:
    """Generates structured SAR narratives from alert records."""

    def __init__(self) -> None:
        self._prompt_cfg = load_prompt("sar_drafting")

        self.llm = get_llm(small=False)

        system_text = self._prompt_cfg["system"].rstrip()
        human_text = self._prompt_cfg.get(
            "human_template",
            "ALERT DATA:\n{alert_data}\n\nSAR NARRATIVE:"
        ).rstrip()
        full_template = f"{system_text}\n\n{human_text}"

        self.prompt = PromptTemplate(
            template=full_template,
            input_variables=["alert_data"],
        )

    def invoke(self, alert_record: dict) -> Dict[str, Any]:
        alert_id = alert_record.get("alert_id", "UNKNOWN")
        alert_data_str = json.dumps(alert_record, indent=2, default=str)

        chain = self.prompt | self.llm
        result = chain.invoke({"alert_data": alert_data_str})
        narrative = result.content

        sections_present = [s for s in _REQUIRED_SECTIONS if s in narrative]
        grounded = self._check_grounding(narrative, alert_record)

        return {
            "narrative": narrative,
            "alert_id": alert_id,
            "sections_present": sections_present,
            "grounded": grounded,
        }

    @staticmethod
    def _check_grounding(narrative: str, alert_record: dict) -> bool:
        """Return True if all monetary amounts in the narrative exist in the alert."""
        narrative_amounts: set[str] = set()
        for match in re.finditer(r"[\d,]+(?:\.\d{2})?", narrative):
            raw = match.group().replace(",", "")
            try:
                val = float(raw)
                if val >= 100:
                    narrative_amounts.add(raw)
            except ValueError:
                continue

        if not narrative_amounts:
            return True

        alert_amounts: set[str] = set()
        _extract_amounts(alert_record, alert_amounts)

        for amount in narrative_amounts:
            amount_float = float(amount)
            if not any(abs(amount_float - float(a)) < 0.01 for a in alert_amounts):
                logger.warning("Potentially ungrounded amount: %s not found in alert", amount)
                return False

        return True


def _extract_amounts(obj: Any, amounts: set[str]) -> None:
    """Recursively extract numeric values from a nested structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            _extract_amounts(v, amounts)
    elif isinstance(obj, list):
        for item in obj:
            _extract_amounts(item, amounts)
    elif isinstance(obj, (int, float)):
        amounts.add(str(float(obj)))
    elif isinstance(obj, str):
        try:
            val = float(obj.replace(",", ""))
            if val >= 100:
                amounts.add(str(val))
        except ValueError:
            pass
