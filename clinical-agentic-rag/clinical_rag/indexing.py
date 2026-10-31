"""clinical_rag/indexing.py

Handles query vector encoding via Hugging Face API and connecting to Qdrant Vector Database.
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
            check_compatibility=False  # Disables strict server version check to avoid timeouts
        )
    # Default to local persistent/in-memory instance if no URL is provided
    return QdrantClient(path=str(config.ROOT_DIR / "qdrant_db"))


def embed_query(query: str) -> List[float]:
    """Embeds a single query string using Hugging Face Serverless Inference API (Zero local RAM overhead)."""
    if not config.HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set in secrets/environment variables.")

    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{config.EMBEDDING_MODEL}"
    headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}
    
    response = requests.post(
        api_url, 
        headers=headers, 
        json={"inputs": query, "options": {"wait_for_model": True}}
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face Embedding API failed ({response.status_code}): {response.text}")
    
    result = response.json()
    # Handle nested list response format from feature-extraction pipeline
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        return result[0]
    return result


def ensure_collection(client: QdrantClient, reset: bool = False, resume: bool = False) -> None:
    """Creates the Qdrant collection if it doesn't exist, or recreates it if reset=True (and not resuming)."""
    collections = [c.name for c in client.get_collections().collections]
    
    if reset and not resume and config.QDRANT_COLLECTION in collections:
        print(f"Resetting collection: {config.QDRANT_COLLECTION}")
        client.delete_collection(collection_name=config.QDRANT_COLLECTION)
        collections.remove(config.QDRANT_COLLECTION)

    if config.QDRANT_COLLECTION not in collections:
        print(f"Creating collection '{config.QDRANT_COLLECTION}' with dimension {config.EMBED_DIM}...")
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=config.EMBED_DIM,
                distance=Distance.COSINE
            )
        )


def build_index(chunks: List[Dict[str, Any]], reset: bool = False, resume: bool = False) -> None:
    """Embeds document chunks and upserts them into Qdrant in batches (used locally/offline for data ingestion)."""
    if not chunks:
        print("No chunks provided to index.")
        return

    from sentence_transformers import SentenceTransformer

    client = get_qdrant_client()
    ensure_collection(client, reset=reset, resume=resume)

    existing_count = 0
    if resume:
        try:
            existing_count = client.get_collection(collection_name=config.QDRANT_COLLECTION).points_count
            print(f"Resuming indexing. Found {existing_count} existing points in collection.")
        except Exception:
            existing_count = 0

    print(f"Loading local embedding model for offline build: {config.EMBEDDING_MODEL}...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print(f"Indexing {len(chunks)} chunks into Qdrant...")
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

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            )

        client.upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=points
        )
        print(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks.")

    print(f"Indexing complete! Collection '{config.QDRANT_COLLECTION}' is ready.")


def print_status() -> None:
    """Prints status and point count of the Qdrant collection."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection_name=config.QDRANT_COLLECTION)
        print(f"Collection Name: {config.QDRANT_COLLECTION}")
        print(f"Indexed Vectors/Points: {info.points_count}")
        print(f"Status: {info.status}")
    except Exception as e:
        print(f"Error fetching status for Qdrant collection: {e}")
