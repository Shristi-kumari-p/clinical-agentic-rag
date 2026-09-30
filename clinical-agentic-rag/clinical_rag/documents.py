"""clinical_rag/documents.py

Handles transforming raw source data frames/dicts into standardized documents,
cleaning, chunking, and saving them to disk.
"""

import json
from collections import Counter
from typing import List, Dict, Any
from . import config
from .transformers import (
    transform_clinical_trial,
    transform_fda_event,
    transform_pubmed_article,
)


# ---------------------------------------------------------------------------
# 1. Builders (Delegated to transformers.py)
# ---------------------------------------------------------------------------

def build_clinical_documents(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [transform_clinical_trial(item) for item in items]


def build_fda_documents(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [transform_fda_event(item) for item in items]


def build_pubmed_documents(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [transform_pubmed_article(item) for item in items]


# ---------------------------------------------------------------------------
# 2. Document Processing Helpers
# ---------------------------------------------------------------------------

def count_by_source(items: List[Dict[str, Any]]) -> Counter:
    """Counts items grouped by their 'source' key."""
    return Counter(item.get("source", "unknown") for item in items)


def clean_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates documents by document_id and strips empty texts."""
    seen = set()
    cleaned = []
    for doc in docs:
        doc_id = doc.get("document_id")
        text = doc.get("text", "").strip()

        if not text or len(text) < config.MIN_DOC_CHARS:
            continue
        if doc_id in seen:
            continue

        seen.add(doc_id)
        cleaned.append(doc)
    return cleaned


def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Splits documents into overlapping chunks based on CHUNK_SIZE and CHUNK_OVERLAP."""
    chunks = []
    chunk_size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP

    for doc in docs:
        text = doc["text"]
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if len(chunk_text.strip()) >= config.MIN_DOC_CHARS:
                chunks.append({
                    "document_id": doc["document_id"],
                    "source": doc["source"],
                    "chunk_id": chunk_idx,
                    "text": chunk_text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "title": doc.get("title", ""),
                    }
                })
                chunk_idx += 1

            start += chunk_size - overlap
            if start >= len(text):
                break

    return chunks


# ---------------------------------------------------------------------------
# 3. Persistence Helpers
# ---------------------------------------------------------------------------

def save_json(data: List[Dict[str, Any]], filename: str) -> None:
    """Saves documents/data to JSON in the PROCESSED_DIR."""
    filepath = config.PROCESSED_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_chunks(chunks: List[Dict[str, Any]]) -> None:
    """Saves chunked documents to CHUNKS_FILE (chunks.jsonl)."""
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def load_chunks() -> List[Dict[str, Any]]:
    """Loads chunks back from CHUNKS_FILE (chunks.jsonl) for indexing."""
    if not config.CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found at {config.CHUNKS_FILE}. Run build-docs first.")
    
    chunks = []
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks