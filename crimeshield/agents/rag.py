"""
CrimeShield AI — RAG Policy Agent (Advanced Retrieval Pipeline)

Retrieval pipeline:
  1. HyDE  — small LLM generates a hypothetical answer, embed that for retrieval
  2. Dense search  — ChromaDB cosine similarity (with optional metadata filter)
  3. BM25 sparse search  — keyword matching over the same corpus
  4. RRF  — Reciprocal Rank Fusion merges dense + sparse candidate lists
  5. BERT Reranker  — cross-encoder re-scores top-N RRF candidates
  6. Threshold filter  — decline if best reranker score < threshold
  7. Generation  — Grok answers only from filtered context
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from crimeshield.config import (
    HYDE_ENABLED,
    RERANKER_MODEL,
    RRF_K,
    SIMILARITY_THRESHOLD,
    TOP_K_RETRIEVAL,
    TOP_K_RERANKED,
)
from crimeshield.utils.llm import get_llm
from crimeshield.utils.prompts import load_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
_HYDE_PROMPT = PromptTemplate(
    template=(
        "You are an AML compliance expert. Write a 2-3 sentence passage that would "
        "directly answer this question, as if it appeared in a regulatory document:\n\n"
        "Question: {question}\n\nPassage:"
    ),
    input_variables=["question"],
)

_RAG_PROMPT = PromptTemplate(
    template=(
        "You are an economic crime compliance expert at Lloyds Bank.\n"
        "Answer ONLY using the provided context. Do not use general knowledge.\n"
        "If the context does not contain enough information, say so explicitly.\n"
        "Always cite the source_section of every chunk you used.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)


class RAGPolicyAgent:
    """
    Advanced RAG agent with HyDE + hybrid retrieval (BM25 + dense) + RRF + BERT reranker.
    """

    def __init__(self, vectorstore: Chroma) -> None:
        self.vectorstore = vectorstore
        self._prompt_cfg = load_prompt("rag_policy")

        self.llm = get_llm(small=False)
        self.small_llm = get_llm(small=True)

        # BERT cross-encoder reranker (local, no API call)
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # Build BM25 index from the vectorstore corpus
        self._bm25, self._bm25_docs = self._build_bm25_index()

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------
    def _build_bm25_index(self) -> Tuple[BM25Okapi, List[Any]]:
        """Fetch all docs from ChromaDB and build a BM25 index."""
        collection = self.vectorstore._collection
        result = collection.get(include=["documents", "metadatas"])
        docs_text = result["documents"]
        metadatas = result["metadatas"]

        tokenized = [doc.lower().split() for doc in docs_text]
        bm25 = BM25Okapi(tokenized)

        from langchain_core.documents import Document
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(docs_text, metadatas)
        ]
        return bm25, docs

    # ------------------------------------------------------------------
    # HyDE
    # ------------------------------------------------------------------
    def _hyde_expand(self, query: str) -> str:
        """Generate a hypothetical document passage for the query."""
        try:
            chain = _HYDE_PROMPT | self.small_llm
            response = chain.invoke({"question": query})
            return response.content.strip()
        except Exception as exc:
            logger.warning("HyDE expansion failed: %s — falling back to raw query", exc)
            return query

    # ------------------------------------------------------------------
    # Dense retrieval with optional metadata filter
    # ------------------------------------------------------------------
    def _dense_search(
        self,
        query_text: str,
        k: int,
        section_type: Optional[str] = None,
    ) -> List[Tuple[Any, float]]:
        """ChromaDB cosine similarity search, optionally filtered by section_type."""
        filter_dict = {"section_type": section_type} if section_type else None
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query_text,
            k=k,
            filter=filter_dict,
        )
        return results  # list of (Document, score)

    # ------------------------------------------------------------------
    # BM25 sparse retrieval
    # ------------------------------------------------------------------
    def _bm25_search(self, query: str, k: int) -> List[Tuple[Any, float]]:
        """BM25 keyword search over the corpus."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self._bm25_docs[i], float(scores[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------
    @staticmethod
    def _rrf_fuse(
        ranked_lists: List[List[Tuple[Any, float]]],
        k: int = RRF_K,
    ) -> List[Tuple[Any, float]]:
        """
        Reciprocal Rank Fusion.
        ranked_lists: each is a list of (doc, score) sorted best-first.
        Returns merged list of (doc, rrf_score) sorted best-first.
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Any] = {}

        for ranked in ranked_lists:
            for rank, (doc, _) in enumerate(ranked):
                doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
                doc_map[doc_id] = doc

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [(doc_map[doc_id], rrf_scores[doc_id]) for doc_id in sorted_ids]

    # ------------------------------------------------------------------
    # BERT reranker
    # ------------------------------------------------------------------
    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[Any, float]],
        top_n: int,
    ) -> List[Tuple[Any, float]]:
        """Cross-encoder reranks candidate (doc, score) pairs."""
        if not candidates:
            return []
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip([doc for doc, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(doc, float(score)) for doc, score in ranked[:top_n]]

    # ------------------------------------------------------------------
    # Infer metadata filter from query
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_section_type(query: str) -> Optional[str]:
        q = query.lower()
        if any(kw in q for kw in ["jmlsg", "tbml", "trade-based", "structuring", "app scam", "pep"]):
            return "jmlsg_guidance"
        if any(kw in q for kw in ["nca", "invoice fraud", "money mule", "sanctions evasion", "crypto"]):
            return "nca_typology"
        if any(kw in q for kw in ["lloyds policy", "edd", "cdd", "customer due diligence", "mlro"]):
            return "aml_policy"
        return None

    # ------------------------------------------------------------------
    # Public invoke
    # ------------------------------------------------------------------
    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Full advanced retrieval pipeline:
        HyDE → dense + BM25 → RRF → BERT reranker → LLM generation
        """
        # 1. HyDE expansion
        hyde_query = self._hyde_expand(query) if HYDE_ENABLED else query
        logger.debug("HyDE expanded query: %s", hyde_query[:120])

        # 2. Metadata filter inference
        section_type = self._infer_section_type(query)
        logger.debug("Metadata filter: section_type=%s", section_type)

        # 3. Dense retrieval (on HyDE-expanded query)
        dense_results = self._dense_search(hyde_query, k=TOP_K_RETRIEVAL, section_type=section_type)

        # 4. BM25 sparse retrieval (on original query — keywords matter here)
        bm25_results = self._bm25_search(query, k=TOP_K_RETRIEVAL)

        # 5. RRF fusion
        fused = self._rrf_fuse([dense_results, bm25_results])

        # 6. BERT reranker
        reranked = self._rerank(query, fused, top_n=TOP_K_RERANKED)

        all_citations = [
            {
                "source_section": doc.metadata.get("source_section", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "section_type": doc.metadata.get("section_type", "unknown"),
                "relevance_score": round(score, 4),
                "retrieval_method": "reranked",
            }
            for doc, score in reranked
        ]

        # 7. Threshold filter — decline if best reranker score too low
        best_score = reranked[0][1] if reranked else 0.0
        confidence = round(best_score, 4)

        if not reranked or best_score < SIMILARITY_THRESHOLD:
            decline_msg = (
                f"I cannot find relevant information in the policy corpus for this query. "
                f"Best retrieval score ({best_score:.3f}) did not exceed threshold ({SIMILARITY_THRESHOLD})."
            )
            return {
                "answer": decline_msg,
                "citations": all_citations,
                "confidence": confidence,
                "declined": True,
                "hyde_query": hyde_query,
                "metadata_filter": section_type,
            }

        # 8. Build context from reranked docs
        context_parts = []
        for doc, score in reranked:
            section = doc.metadata.get("source_section", "Unknown")
            context_parts.append(f"[Source: {section}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)

        # 9. Generate answer with primary LLM
        chain = _RAG_PROMPT | self.llm
        result = chain.invoke({"context": context, "question": query})
        answer = result.content

        return {
            "answer": answer,
            "citations": all_citations,
            "confidence": confidence,
            "declined": False,
            "hyde_query": hyde_query,
            "metadata_filter": section_type,
        }
