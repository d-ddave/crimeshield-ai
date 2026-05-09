"""
CrimeShield AI — Vector Store Builder.

Loads ``policy_corpus.txt``, splits by === section headers, chunks with
RecursiveCharacterTextSplitter, embeds via local HuggingFace sentence-transformers,
and persists to ChromaDB.
Idempotent: skips rebuild when the collection already has documents.
"""

from __future__ import annotations

import re
import time
import logging
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from crimeshield.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    POLICY_CORPUS,
)

logger = logging.getLogger(__name__)

_SECTION_TYPE_MAP = {
    0: "aml_policy",
    1: "jmlsg_guidance",
    2: "nca_typology",
}


def _slugify(text: str) -> str:
    """Convert a section header to a filesystem-safe slug."""
    text = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
    return text.strip("_")[:80]


def _parse_sections(corpus_path: str) -> list[dict]:
    """
    Split the policy corpus into sections using the ``===`` header pattern.

    Each section dict contains:
        - ``header``: the cleaned section header text
        - ``body``:   the full text of the section (between headers)
    """
    raw_bytes = Path(corpus_path).read_bytes()
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw = raw_bytes.decode("latin-1")
        logger.debug("Corpus decoded with latin-1 fallback (non-UTF-8 bytes detected)")

    separator = re.compile(r"^={3,}.*$", re.MULTILINE)
    sep_matches = list(separator.finditer(raw))

    sections: list[dict] = []
    i = 0
    while i < len(sep_matches) - 1:
        header_start = sep_matches[i].end()
        header_end = sep_matches[i + 1].start()
        header_text = raw[header_start:header_end].strip()

        body_start = sep_matches[i + 1].end()
        if i + 2 < len(sep_matches):
            body_end = sep_matches[i + 2].start()
        else:
            body_end = len(raw)

        body_text = raw[body_start:body_end].strip()

        if header_text and body_text:
            sections.append({"header": header_text, "body": body_text})

        i += 2

    if not sections:
        sections.append({"header": "Full Document", "body": raw})

    return sections


def build_vector_store() -> Chroma:
    """
    Build (or reuse) the ChromaDB vector store for the policy corpus.

    Returns
    -------
    Chroma
        The LangChain ChromaDB wrapper ready for retrieval.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    existing_count = vectorstore._collection.count()
    if existing_count > 0:
        print(f"✓ ChromaDB collection '{CHROMA_COLLECTION}' already has "
              f"{existing_count} documents — skipping rebuild.")
        return vectorstore

    print(f"Building vector store from {POLICY_CORPUS} ...")
    sections = _parse_sections(POLICY_CORPUS)
    print(f"  Sections detected: {len(sections)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_texts: list[str] = []
    all_metadatas: list[dict] = []

    for doc_index, section in enumerate(sections):
        slug = _slugify(section["header"])
        section_type = _SECTION_TYPE_MAP.get(doc_index, f"section_{doc_index}")
        chunks = splitter.split_text(section["body"])
        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append({
                "source_section": section["header"],
                "chunk_id": f"{slug}_{idx}",
                "document": "policy_corpus.txt",
                "doc_index": doc_index,
                "section_type": section_type,
            })

    print(f"  Total chunks: {len(all_texts)}")

    t0 = time.perf_counter()

    vectorstore = Chroma.from_texts(
        texts=all_texts,
        metadatas=all_metadatas,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    elapsed = time.perf_counter() - t0
    print(f"  Embedding time: {elapsed:.2f}s")
    print(f"✓ Vector store built and persisted to {CHROMA_PERSIST_DIR}")

    return vectorstore
