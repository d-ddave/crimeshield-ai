"""
CrimeShield AI — Structured Data Agent.

Agent with tool access to query alerts and typology DataFrames.
Uses create_tool_calling_agent with ChatGoogleGenerativeAI (Gemini native tool binding).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import pandas as pd
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from crimeshield.config import GEMINI_API_KEY, LLM_MODEL
from crimeshield.utils.prompts import load_prompt

logger = logging.getLogger(__name__)


class StructuredDataAgent:
    """Agent that queries alerts and typology DataFrames via Gemini tool calls."""

    def __init__(self, alerts_df: pd.DataFrame, typologies_df: pd.DataFrame) -> None:
        self.alerts_df = alerts_df
        self.typologies_df = typologies_df
        self._tool_call_log: list[str] = []
        self._prompt_cfg = load_prompt("structured_data")

        _alerts = self.alerts_df
        _typologies = self.typologies_df
        _log = self._tool_call_log

        @tool
        def query_alerts_table(
            risk_band: str = "",
            status: str = "",
            alert_type: str = "",
        ) -> str:
            """Query the alerts table. Filter by risk_band, status, and/or alert_type.
            Empty string means no filter for that parameter.
            Returns summary statistics as JSON."""
            _log.append(f"query_alerts_table(risk_band={risk_band}, status={status}, alert_type={alert_type})")
            df = _alerts.copy()

            if risk_band:
                df = df[df["risk_band"].str.upper() == risk_band.upper()]
            if status:
                df = df[df["status"].str.upper() == status.upper()]
            if alert_type:
                df = df[df["alert_type"].str.upper() == alert_type.upper()]

            if df.empty:
                return json.dumps({
                    "count": 0,
                    "message": f"No alerts found matching: risk_band={risk_band}, status={status}, alert_type={alert_type}"
                })

            amounts = pd.to_numeric(df["amount_gbp"], errors="coerce")
            result = {
                "count": int(len(df)),
                "total_amount_gbp": float(amounts.sum()),
                "mean_amount_gbp": round(float(amounts.mean()), 2),
                "status_breakdown": df["status"].value_counts().to_dict(),
                "sample_rows": df.head(5).to_dict(orient="records"),
            }
            return json.dumps(result, default=str)

        @tool
        def query_typology_table(
            typology_code: str = "",
            typology_name: str = "",
        ) -> str:
            """Look up a typology by code or name (case-insensitive partial match).
            Returns matching typology details as JSON."""
            _log.append(f"query_typology_table(code={typology_code}, name={typology_name})")
            df = _typologies.copy()

            if typology_code:
                df = df[df["typology_code"].str.upper().str.contains(typology_code.upper(), na=False)]
            if typology_name:
                df = df[df["typology_name"].str.upper().str.contains(typology_name.upper(), na=False)]

            if df.empty:
                return json.dumps({"error": f"No typology found matching: {typology_code or typology_name}"})

            results = df.to_dict(orient="records")
            return json.dumps(results[0] if len(results) == 1 else results, default=str)

        self.tools = [query_alerts_table, query_typology_table]

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0,
            google_api_key=GEMINI_API_KEY,
        )

        system_msg = self._prompt_cfg["system"].strip()
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

    def invoke(self, query: str) -> Dict[str, Any]:
        """Run the structured data agent on a query."""
        self._tool_call_log.clear()
        result = self.executor.invoke({"input": query})
        output = result.get("output", "")

        return {
            "answer": output,
            "tool_calls": list(self._tool_call_log),
            "raw_output": output,
        }
