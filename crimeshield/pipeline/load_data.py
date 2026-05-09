"""
CrimeShield AI — Data Loader.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
import pandas as pd
from crimeshield.config import ALERTS_CSV, ALERT_JSON, TYPOLOGY_CSV

logger = logging.getLogger(__name__)

class DataLoader:
    _ALERTS_COLUMNS = {"alert_id", "customer_id", "alert_type", "risk_band", "amount_gbp", "status", "triggered_date", "assigned_analyst"}
    _TYPOLOGY_COLUMNS = {"typology_code", "typology_name", "threshold_value", "threshold_unit", "reporting_obligation"}

    def __init__(self) -> None:
        self._alerts_df = None
        self._typologies_df = None

    def load_alerts(self) -> pd.DataFrame:
        path = Path(ALERTS_CSV)
        if not path.exists():
            raise FileNotFoundError(f"Alerts CSV not found: {path}")
        df = pd.read_csv(path)
        missing = self._ALERTS_COLUMNS - set(df.columns)
        assert not missing, f"Missing columns: {missing}"
        logger.info("Loaded alerts.csv: %d rows", len(df))
        return df

    def load_typologies(self) -> pd.DataFrame:
        path = Path(TYPOLOGY_CSV)
        if not path.exists():
            raise FileNotFoundError(f"Typology CSV not found: {path}")
        df = pd.read_csv(path)
        missing = self._TYPOLOGY_COLUMNS - set(df.columns)
        assert not missing, f"Missing columns: {missing}"
        logger.info("Loaded typology_thresholds.csv: %d rows", len(df))
        return df

    def load_alert_json(self) -> list[dict]:
        path = Path(ALERT_JSON)
        if not path.exists():
            raise FileNotFoundError(f"Alert JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return data
