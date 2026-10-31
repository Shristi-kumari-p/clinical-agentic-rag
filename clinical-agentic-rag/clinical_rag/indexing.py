"""clinical_rag/indexing.py

Handles query vector encoding via Hugging Face InferenceClient and connecting to Qdrant Vector Database.
"""

import uuid
from typing import List, Dict, Any
import streamlit as st
from huggingface_hub import InferenceClient
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


@st.cache_resource
def get_hf_client() -> InferenceClient:
    """Initializes and caches the Hugging Face Inference Client."""
    if not config.HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set in secrets/environment variables.")
    return InferenceClient(token=config.HF_TOKEN)


def embed_query(query: str) -> List[float]:
    """Embeds a single query string using Hugging Face Hub client."""
    client = get_hf_client()
    
    # Use the official feature-extraction client method
    embedding = client.feature_extraction(
        query, 
        model=config.EMBEDDING_MODEL
    )
    
    # Handle numpy array / list formatting returned by the client
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
        
    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        return embedding[0]
        
    return embedding


def ensure_collection(client: QdrantClient, reset: bool = False, resume: bool = False) -> None:
    """Creates the Qdrant collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    
    if reset and not resume and config.QDRANT_COLLECTION in collections:
        client.delete_collection(collection_name=config.QDRANT_COLLECTION)
        collections.remove(config.QDRANT_COLLECTION)

    if config.QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=config.EMBED_DIM,
                distance=Distance.COSINE
            )
        )


def build_index(chunks: List[Dict[str, Any]], reset: bool = False, resume: bool = False) -> None:
    """Embeds document chunks locally for offline ingestion if needed."""
    if not chunks:
        return

    from sentence_transformers import SentenceTransformer

    client = get_qdrant_client()
    ensure_collection(client, reset=reset, resume=resume)

    existing_count = 0
    if resume:
        try:
            existing_count = client.get_collection(collection_name=config.QDRANT_COLLECTION).points_count
        except Exception:
            existing_count = 0

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    batch_size = config.UPLOAD_BATCH_SIZE

    for i in range(0, len(chunks), batch_size):
        if resume and (i + batch_size <= existing_count):
            continue

        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, batch_size=config.EMBED_BATCH_SIZE, show_progress_bar=False)

        points = []
        for idx, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk['document_id']}_{chunk['chunk_id']}"))
            payload = {
                "document_id": chunk["document_id"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                **chunk.get("metadata", {})
            }
            points.append(PointStruct(id=point_id, vector=embedding.tolist(), payload=payload))

        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)


def print_status() -> None:
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection_name=config.QDRANT_COLLECTION)
        print(f"Collection: {config.QDRANT_COLLECTION} | Points: {info.points_count}")
    except Exception as e:
        print(f"Error fetching status: {e}")
