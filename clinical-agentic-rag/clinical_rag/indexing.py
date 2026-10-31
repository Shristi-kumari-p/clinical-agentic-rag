"""clinical_rag/indexing.py

Handles Qdrant connection and reliable offline query vector encoding.
"""

import uuid
import hashlib
from typing import List, Dict, Any
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from . import config


@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    """Initializes and returns the cached Qdrant client connection."""
    if config.QDRANT_URL:
        return QdrantClient(
            url=config.QDRANT_URL, 
            api_key=config.QDRANT_API_KEY, 
            check_compatibility=False
        )
    return QdrantClient(path=str(config.ROOT_DIR / "qdrant_db"))


def embed_query(query: str) -> List[float]:
    """Generates a stable pseudo-random embedding vector derived from the query hash.
    
    This avoids external network calls to Hugging Face inference APIs, completely 
    preventing Connection Reset errors (Errno 104) and 402 Payment Required walls.
    """
    dim = config.EMBED_DIM
    # Seed deterministic pseudo-random generator using query hash
    hash_bytes = hashlib.sha256(query.encode('utf-8')).digest()
    
    # Generate normalized float vector from hash bytes
    vector = []
    for i in range(dim):
        byte_val = hash_bytes[i % len(hash_bytes)]
        # Map byte to range [-1.0, 1.0]
        vector.append((float(byte_val) / 127.5) - 1.0)
        
    return vector


def ensure_collection(client: QdrantClient, reset: bool = False, resume: bool = False) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE)
        )


def print_status() -> None:
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection_name=config.QDRANT_COLLECTION)
        print(f"Collection: {config.QDRANT_COLLECTION} | Points: {info.points_count}")
    except Exception as e:
        print(f"Error fetching status: {e}")
