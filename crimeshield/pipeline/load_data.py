"""
CrimeShield AI — Data Loader.

Loads CSV and JSON data files, validates required columns, and exposes
module-level singleton DataFrames for use by agents.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from crimeshield.config import ALERTS_CSV, ALERT_JSON, TYPOLOGY_CSV

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate structured data files for CrimeShield agents."""

    # Required columns per dataset
    _ALERTS_COLUMNS = {
        "alert_id", "customer_id", "alert_type", "risk_band",
        "amount_gbp", "status", "triggered_date", "assigned_analyst",
    }

    _TYPOLOGY_COLUMNS = {
        "typology_code", "typology_name", "threshold_value",
        "threshold_unit", "reporting_obligation",
    }

    def __init__(self) -> None:
        self._alerts_df: pd.DataFrame | None = None
        self._typologies_df: pd.DataFrame | None = None
        self._alert_json: list[dict] | None = None

    # ── Alerts CSV ─────────────────────────────────────────────────────

    def load_alerts(self) -> pd.DataFrame:
        """Load alerts.csv, validate columns, return DataFrame."""
        path = Path(ALERTS_CSV)
        if not path.exists():
            raise FileNotFoundError(f"Alerts CSV not found: {path}")

        df = pd.read_csv(path)
        missing = self._ALERTS_COLUMNS - set(df.columns)
        assert not missing, f"Alerts CSV missing columns: {missing}"

        logger.info("Loaded alerts.csv: %d rows", len(df))
        print(f"  ✓ alerts.csv loaded: {len(df)} rows")
        self._alerts_df = df
        return df

    # ── Typologies CSV ─────────────────────────────────────────────────

    def load_typologies(self) -> pd.DataFrame:
        """Load typology_thresholds.csv, validate columns, return DataFrame."""
        path = Path(TYPOLOGY_CSV)
        if not path.exists():
            raise FileNotFoundError(f"Typology CSV not found: {path}")

        df = pd.read_csv(path)
        missing = self._TYPOLOGY_COLUMNS - set(df.columns)
        assert not missing, f"Typology CSV missing columns: {missing}"

        logger.info("Loaded typology_thresholds.csv: %d rows", len(df))
        print(f"  ✓ typology_thresholds.csv loaded: {len(df)} rows")
        self._typologies_df = df
        return df

    # ── Alert JSON ─────────────────────────────────────────────────────

    def load_alert_json(self) -> list[dict]:
        """Load alert.json, return list of alert dicts."""
        path = Path(ALERT_JSON)
        if not path.exists():
            raise FileNotFoundError(f"Alert JSON not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        logger.info("Loaded alert.json: %d records", len(data))
        print(f"  ✓ alert.json loaded: {len(data)} records")
        self._alert_json = data
        return data


# ── Module-level singletons (lazy-loaded) ──────────────────────────────

_loader = DataLoader()

alerts_df: pd.DataFrame | None = None
typologies_df: pd.DataFrame | None = None


def initialize_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both DataFrames and set module-level singletons."""
    global alerts_df, typologies_df
    alerts_df = _loader.load_alerts()
    typologies_df = _loader.load_typologies()
    return alerts_df, typologies_df
