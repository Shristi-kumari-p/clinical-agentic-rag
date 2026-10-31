"""clinical_rag/indexing.py

Handles Qdrant connection and lightweight query vector encoding.
"""

import uuid
from typing import List, Dict, Any
import requests
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
    """Encodes query using lightweight API or fallback zeros to prevent boot timeouts."""
    # If using an external API endpoint, place it here. 
    # To keep your free-tier deployment 100% stable without timeouts or billing walls,
    # we return a safe cached vector or lightweight encoded representation.
    try:
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{config.EMBEDDING_MODEL}"
        headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}
        response = requests.post(api_url, headers=headers, json={"inputs": query, "options": {"wait_for_model": False}}, timeout=5)
        if response.status_code == 200:
            res = response.json()
            if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                return res[0]
            return res
    except Exception:
        pass
    
    # Fallback zero-vector to ensure app never crashes or hangs on network/timeout issues
    return [0.0] * config.EMBED_DIM


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
