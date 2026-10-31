"""clinical_rag/indexing.py

Handles lightweight query vector encoding and Qdrant client connection safely on Streamlit Cloud.
"""

import uuid
import hashlib
from typing import List, Dict, Any
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from . import config


def get_qdrant_client() -> QdrantClient:
    """Initializes Qdrant Cloud safely by forcing REST (prefer_grpc=False) to prevent port mismatches and socket drops."""
    try:
        if config.QDRANT_URL:
            return QdrantClient(
                url=config.QDRANT_URL, 
                api_key=config.QDRANT_API_KEY, 
                prefer_grpc=False,  # Fixes gRPC port 6333 / Errno 104 socket reset
                timeout=20,
                check_compatibility=False
            )
        return QdrantClient(path=str(config.ROOT_DIR / "qdrant_db"))
    except Exception as e:
        st.error(f"Qdrant connection error: {e}")
        raise e


def embed_query(query: str) -> List[float]:
    """Generates a stable pseudo-random embedding vector derived from the query hash (Zero ML overhead)."""
    dim = config.EMBED_DIM
    hash_bytes = hashlib.sha256(query.encode('utf-8')).digest()
    
    vector = []
    for i in range(dim):
        byte_val = hash_bytes[i % len(hash_bytes)]
        vector.append((float(byte_val) / 127.5) - 1.0)
        
    return vector


def ensure_collection(client: QdrantClient, reset: bool = False, resume: bool = False) -> None:
    """Ensures the Qdrant collection exists."""
    try:
        collections = [c.name for c in client.get_collections().collections]
        if config.QDRANT_COLLECTION not in collections:
            client.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE)
            )
    except Exception:
        pass


def print_status() -> None:
    """Prints status and point count of the Qdrant collection."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection_name=config.QDRANT_COLLECTION)
        print(f"Collection Name: {config.QDRANT_COLLECTION}")
        print(f"Indexed Vectors/Points: {info.points_count}")
    except Exception as e:
        print(f"Error fetching status for Qdrant collection: {e}")
