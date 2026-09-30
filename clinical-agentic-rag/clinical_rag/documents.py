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
    
    # Use a low fallback threshold (20 chars) to prevent dropping brief FDA/PubMed records
    min_chars = getattr(config, "MIN_DOC_CHARS", 20)
    # If MIN_DOC_CHARS was accidentally set to a high value like 500, cap it at 20 for document cleaning
    min_chars = min(min_chars, 20)

    for doc in docs:
        if not isinstance(doc, dict):
            continue

        doc_id = doc.get("document_id")
        text = doc.get("text", "").strip()

        if not text or len(text) < min_chars:
            continue
        if doc_id in seen:
            continue

        seen.add(doc_id)
        cleaned.append(doc)
    return cleaned


def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Splits documents into overlapping chunks based on CHUNK_SIZE and CHUNK_OVERLAP."""
    chunks = []
    chunk_size = getattr(config, "CHUNK_SIZE", 500)
    overlap = getattr(config, "CHUNK_OVERLAP", 50)
    
    # Ensure step size is strictly positive to prevent infinite loops
    step = max(1, chunk_size - overlap)

    for doc in docs:
        text = doc.get("text", "")
        if not text:
            continue

        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "document_id": doc.get("document_id", "unknown"),
                    "source": doc.get("source", "unknown"),
                    "chunk_id": f"{doc.get('document_id', 'doc')}_c{chunk_idx}",
                    "text": chunk_text,
                    "title": doc.get("title", ""),
                    "metadata": {
                        **doc.get("metadata", {}),
                        "title": doc.get("title", ""),
                        "chunk_index": chunk_idx,
                    }
                })
                chunk_idx += 1

            start += step

    return chunks


# ---------------------------------------------------------------------------
# 3. Persistence Helpers
# ---------------------------------------------------------------------------

def save_json(data: List[Dict[str, Any]], filename: str) -> None:
    """Saves documents/data to JSON in the PROCESSED_DIR."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    filepath = config.PROCESSED_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_chunks(chunks: List[Dict[str, Any]]) -> None:
    """Saves chunked documents to CHUNKS_FILE (chunks.jsonl)."""
    config.CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
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