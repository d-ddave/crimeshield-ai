"""
CrimeShield AI — Vector Store Builder.
Embeds policy corpus via Gemini and persists to ChromaDB.
"""
from __future__ import annotations
import re
import time
import logging
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crimeshield.config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, GEMINI_API_KEY, POLICY_CORPUS

logger = logging.getLogger(__name__)

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower().strip()).strip("_")[:80]

def _parse_sections(corpus_path: str) -> list[dict]:
    raw_bytes = Path(corpus_path).read_bytes()
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw = raw_bytes.decode("latin-1")
    separator = re.compile(r"^={3,}.*$", re.MULTILINE)
    sep_matches = list(separator.finditer(raw))
    sections: list[dict] = []
    i = 0
    while i < len(sep_matches) - 1:
        header_text = raw[sep_matches[i].end():sep_matches[i+1].start()].strip()
        body_end = sep_matches[i+2].start() if i+2 < len(sep_matches) else len(raw)
        body_text = raw[sep_matches[i+1].end():body_end].strip()
        if header_text and body_text:
            sections.append({"header": header_text, "body": body_text})
        i += 2
    if not sections:
        sections.append({"header": "Full Document", "body": raw})
    return sections

def build_vector_store() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
    vectorstore = Chroma(collection_name=CHROMA_COLLECTION, embedding_function=embeddings, persist_directory=CHROMA_PERSIST_DIR)
    if vectorstore._collection.count() > 0:
        print(f"✓ ChromaDB already has {vectorstore._collection.count()} docs — skipping rebuild.")
        return vectorstore
    print(f"Building vector store from {POLICY_CORPUS} ...")
    sections = _parse_sections(POLICY_CORPUS)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_texts, all_metadatas = [], []
    for section in sections:
        slug = _slugify(section["header"])
        for idx, chunk in enumerate(splitter.split_text(section["body"])):
            all_texts.append(chunk)
            all_metadatas.append({"source_section": section["header"], "chunk_id": f"{slug}_{idx}", "document": "policy_corpus.txt"})
    t0 = time.perf_counter()
    vectorstore = Chroma.from_texts(texts=all_texts, metadatas=all_metadatas, embedding=embeddings, collection_name=CHROMA_COLLECTION, persist_directory=CHROMA_PERSIST_DIR)
    print(f"  {len(all_texts)} chunks embedded in {time.perf_counter()-t0:.2f}s")
    return vectorstore
