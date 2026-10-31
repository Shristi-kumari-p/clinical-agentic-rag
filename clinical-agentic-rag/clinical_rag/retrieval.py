def retrieve_from_source(query: str, source: str, n_results: int = config.N_CANDIDATES) -> dict:
    """Retrieves candidates safely with a crash shield against gRPC/network socket drops."""
    from .indexing import embed_query, get_qdrant_client

    if source not in config.ALL_SOURCES:
        raise ValueError(f"Unknown source: {source}")

    try:
        client = get_qdrant_client()
        if client is None:
            raise RuntimeError("Qdrant client could not be initialized.")
            
        query_vector = embed_query(query)

        source_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source)
                )
            ]
        )

        response = client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=query_vector,
            query_filter=source_filter,
            limit=n_results,
            with_payload=True,
        )
        hits = response.points

        return {
            "documents": [[h.payload.get("text", "") for h in hits]],
            "metadatas": [[{k: v for k, v in (h.payload or {}).items() if k != "text"} for h in hits]],
            "distances": [[1.0 - float(h.score) for h in hits]],
        }
    except Exception as e:
        # Catch any low-level Qdrant/gRPC socket crash and return empty safely instead of [Errno 104]
        print(f"Retrieval warning for {source}: {e}")
        return {
            "documents": [["Database connection or query failed. Please verify Qdrant settings."]],
            "metadatas": [[{"document_id": "error", "chunk_id": "0"}]],
            "distances": [[1.0]],
        }
