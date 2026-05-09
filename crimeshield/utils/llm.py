"""
CrimeShield AI — LLM factory for xAI Grok via OpenAI-compatible API.
"""
from langchain_openai import ChatOpenAI

from crimeshield.config import XAI_API_KEY, XAI_BASE_URL, LLM_MODEL, SMALL_LLM_MODEL


def get_llm(small: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=SMALL_LLM_MODEL if small else LLM_MODEL,
        temperature=0.3 if small else 0,
        openai_api_key=XAI_API_KEY,
        openai_api_base=XAI_BASE_URL,
    )
