"""
CrimeShield AI — RAG Policy Agent.

Retrieval-augmented generation agent that answers policy and regulatory
compliance questions using the ChromaDB vector store. Filters chunks
by similarity threshold and cites source sections.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from crimeshield.config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K_RETRIEVAL,
)
from crimeshield.utils.prompts import load_prompt

logger = logging.getLogger(__name__)


class RAGPolicyAgent:
    """Policy retrieval agent backed by ChromaDB and Google Gemini."""

    def __init__(self, vectorstore: Chroma) -> None:
        self.vectorstore = vectorstore
        self._prompt_cfg = load_prompt("rag_policy")

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0,
            google_api_key=GEMINI_API_KEY,
        )
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RETRIEVAL},
        )

        prompt = PromptTemplate(
            template=self._prompt_cfg["system"],
            input_variables=["context", "question"],
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def invoke(self, query: str) -> Dict[str, Any]:
        """Run the RAG pipeline for a policy query."""
        docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=TOP_K_RETRIEVAL
        )

        citations: List[Dict[str, Any]] = []
        filtered_docs = []
        for doc, score in docs_with_scores:
            citation = {
                "source_section": doc.metadata.get("source_section", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "relevance_score": round(score, 4),
            }
            citations.append(citation)
            if score >= SIMILARITY_THRESHOLD:
                filtered_docs.append(doc)

        if docs_with_scores:
            confidence = sum(s for _, s in docs_with_scores) / len(docs_with_scores)
        else:
            confidence = 0.0

        if not filtered_docs:
            decline_msg = self._prompt_cfg.get("decline_message", "").strip()
            if "{threshold}" in decline_msg:
                decline_msg = decline_msg.replace("{threshold}", str(SIMILARITY_THRESHOLD))
            return {
                "answer": decline_msg,
                "citations": citations,
                "confidence": round(confidence, 4),
                "declined": True,
            }

        result = self.chain.invoke({"query": query})

        return {
            "answer": result.get("result", ""),
            "citations": citations,
            "confidence": round(confidence, 4),
            "declined": False,
        }
