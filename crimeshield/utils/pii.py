"""
CrimeShield AI — PII Detection and Redaction Utility.

Regex-based PII detection and redaction.  No spaCy, no external NLP.
Works both as a pre-processor (input masking) and post-processor
(output redaction) — same class, called in two different graph nodes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PIIRedactor:
    """Detect and redact PII from free-text using compiled regex patterns."""

    # Whether to redact alert IDs (typically only in logs, not SAR context)
    redact_alerts: bool = False

    # ── Compiled patterns (order matters for overlapping rules) ─────────
    _patterns: Dict[str, re.Pattern] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Compile all regex patterns once at construction time."""
        self._patterns = {
            # Order: most specific → least specific to avoid false positives
            "sort_codes": re.compile(r"\b\d{2}-\d{2}-\d{2}\b"),
            "customer_ids": re.compile(r"\bCUST-\d+\b", re.IGNORECASE),
            "alert_ids": re.compile(r"\bALT-[A-Z]?\d+\b", re.IGNORECASE),
            "amounts_gbp": re.compile(
                r"£[\d,]+(?:\.\d{2})?|\b[\d,]+(?:\.\d{2})?\s*(?:GBP|gbp)\b"
            ),
            "email_addresses": re.compile(
                r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
            ),
            "uk_postcodes": re.compile(
                r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE
            ),
            "analyst_names": re.compile(
                r"\b[a-z]\.[a-z]+\b"
            ),
            "account_numbers": re.compile(r"\b\d{8}\b"),
        }

    # ── Public API ─────────────────────────────────────────────────────

    def detect(self, text: str) -> Dict[str, List[str]]:
        """Return a dict of detected PII types → list of matched strings."""
        results: Dict[str, List[str]] = {
            "sort_codes": [],
            "account_numbers": [],
            "customer_ids": [],
            "alert_ids": [],
            "amounts_gbp": [],
            "dates": [],
            "analyst_names": [],
            "email_addresses": [],
            "uk_postcodes": [],
        }

        # Track character positions already claimed to avoid double-counting
        claimed: set[tuple[int, int]] = set()

        # Sort codes
        for m in self._patterns["sort_codes"].finditer(text):
            results["sort_codes"].append(m.group())
            claimed.add((m.start(), m.end()))

        # Customer IDs
        for m in self._patterns["customer_ids"].finditer(text):
            results["customer_ids"].append(m.group())
            claimed.add((m.start(), m.end()))

        # Alert IDs
        for m in self._patterns["alert_ids"].finditer(text):
            results["alert_ids"].append(m.group())
            claimed.add((m.start(), m.end()))

        # GBP amounts
        for m in self._patterns["amounts_gbp"].finditer(text):
            results["amounts_gbp"].append(m.group())
            claimed.add((m.start(), m.end()))

        # Emails
        for m in self._patterns["email_addresses"].finditer(text):
            results["email_addresses"].append(m.group())
            claimed.add((m.start(), m.end()))

        # UK postcodes
        for m in self._patterns["uk_postcodes"].finditer(text):
            results["uk_postcodes"].append(m.group())
            claimed.add((m.start(), m.end()))

        # Analyst names — only if not overlapping with already-claimed spans
        for m in self._patterns["analyst_names"].finditer(text):
            span = (m.start(), m.end())
            if not any(
                s <= span[0] < e or s < span[1] <= e
                for s, e in claimed
            ):
                results["analyst_names"].append(m.group())
                claimed.add(span)

        # Account numbers (8-digit standalone) — skip if already claimed
        # (overlaps with company numbers — dedup by position)
        for m in self._patterns["account_numbers"].finditer(text):
            span = (m.start(), m.end())
            if not any(
                s <= span[0] < e or s < span[1] <= e
                for s, e in claimed
            ):
                results["account_numbers"].append(m.group())
                claimed.add(span)

        # Dates — simple pattern for dd/mm/yyyy, yyyy-mm-dd, dd-mm-yyyy
        date_pattern = re.compile(
            r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"
        )
        for m in date_pattern.finditer(text):
            span = (m.start(), m.end())
            if not any(
                s <= span[0] < e or s < span[1] <= e
                for s, e in claimed
            ):
                results["dates"].append(m.group())

        return results

    def redact(self, text: str, redact_alerts: bool | None = None) -> str:
        """
        Replace all detected PII with typed placeholders.

        Parameters
        ----------
        text : str
            The input text to redact.
        redact_alerts : bool | None
            Override instance-level ``redact_alerts`` for this call.
        """
        should_redact_alerts = (
            redact_alerts if redact_alerts is not None else self.redact_alerts
        )

        result = text

        # Order: replace longer / more specific patterns first to prevent
        # partial matches from corrupting the text.

        # Customer IDs
        result = self._patterns["customer_ids"].sub("[CUSTOMER-ID]", result)

        # Alert IDs — conditional
        if should_redact_alerts:
            result = self._patterns["alert_ids"].sub("[ALERT-ID]", result)

        # Sort codes (before generic 8-digit to prevent overlap)
        result = self._patterns["sort_codes"].sub("[SORT-CODE]", result)

        # GBP amounts
        result = self._patterns["amounts_gbp"].sub("[AMOUNT-GBP]", result)

        # Email addresses
        result = self._patterns["email_addresses"].sub("[EMAIL]", result)

        # UK postcodes
        result = self._patterns["uk_postcodes"].sub("[POSTCODE]", result)

        # Analyst names
        result = self._patterns["analyst_names"].sub("[ANALYST-NAME]", result)

        # Account / company numbers (8-digit standalone)
        # Only replace remaining 8-digit sequences not already substituted
        result = self._patterns["account_numbers"].sub("[ACCOUNT-NUMBER]", result)

        return result

    def has_pii(self, text: str) -> bool:
        """Return True if *any* PII pattern matches in *text*."""
        detected = self.detect(text)
        return any(len(v) > 0 for v in detected.values())
