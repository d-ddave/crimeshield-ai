"""
CrimeShield AI — PII Detection and Redaction Utility.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class PIIRedactor:
    redact_alerts: bool = False
    _patterns: Dict[str, re.Pattern] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._patterns = {
            "sort_codes": re.compile(r"\b\d{2}-\d{2}-\d{2}\b"),
            "customer_ids": re.compile(r"\bCUST-\d+\b", re.IGNORECASE),
            "alert_ids": re.compile(r"\bALT-[A-Z]?\d+\b", re.IGNORECASE),
            "amounts_gbp": re.compile(r"\u00a3[\d,]+(?:\.\d{2})?|\b[\d,]+(?:\.\d{2})?\s*(?:GBP|gbp)\b"),
            "email_addresses": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
            "uk_postcodes": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE),
            "analyst_names": re.compile(r"\b[a-z]\.[a-z]+\b"),
            "account_numbers": re.compile(r"\b\d{8}\b"),
        }

    def detect(self, text: str) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {k: [] for k in self._patterns}
        results["dates"] = []
        for key, pat in self._patterns.items():
            results[key] = pat.findall(text)
        return results

    def redact(self, text: str, redact_alerts: bool | None = None) -> str:
        should_redact_alerts = redact_alerts if redact_alerts is not None else self.redact_alerts
        result = text
        result = self._patterns["customer_ids"].sub("[CUSTOMER-ID]", result)
        if should_redact_alerts:
            result = self._patterns["alert_ids"].sub("[ALERT-ID]", result)
        result = self._patterns["sort_codes"].sub("[SORT-CODE]", result)
        result = self._patterns["amounts_gbp"].sub("[AMOUNT-GBP]", result)
        result = self._patterns["email_addresses"].sub("[EMAIL]", result)
        result = self._patterns["uk_postcodes"].sub("[POSTCODE]", result)
        result = self._patterns["analyst_names"].sub("[ANALYST-NAME]", result)
        result = self._patterns["account_numbers"].sub("[ACCOUNT-NUMBER]", result)
        return result

    def has_pii(self, text: str) -> bool:
        return any(len(v) > 0 for v in self.detect(text).values())
