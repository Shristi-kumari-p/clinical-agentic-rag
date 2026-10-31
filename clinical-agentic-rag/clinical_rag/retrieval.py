"""Step 4 - agentic routing, vector search and cross-encoder reranking."""
from functools import lru_cache
from qdrant_client.models import Filter, FieldCondition, MatchValue

from . import config
from .config import SOURCE_CLINICAL, SOURCE_FDA, SOURCE_PUBMED

COMPLETE_KEYWORDS = [
    "complete evidence", "complete summary", "complete evidence summary",
    "overall evidence", "overall summary", "full evidence", "full summary",
    "comprehensive evidence", "comprehensive summary", "all evidence",
    "all sources", "complete analysis", "overall analysis",
]
FDA_KEYWORDS = [
    "side effect", "side effects", "adverse effect", "adverse effects",
    "adverse event", "adverse events", "safety", "reaction", "reactions",
    "drug reaction", "drug reactions", "drug safety", "toxicity",
]
CLINICAL_KEYWORDS = [
    "clinical trial", "clinical trials", "trial", "phase 1", "phase 2",
    "phase 3", "phase 4", "enrollment", "intervention", "recruiting",
    "completed", "terminated", "clinical study",
]
PUBMED_KEYWORDS = [
    "research", "research paper", "published research", "publication",
    "publications", "literature", "article", "journal", "published study",
    "published studies",
]


def route_query(query: str) -> list[str]:
    """Rule-based router: decide which knowledge bases should answer the query."""
    q = query.lower()

    if any(k in q for k in COMPLETE_KEYWORDS):
        return list(config.ALL_SOURCES)

    selected = []
    if any(k in q for k in FDA_KEYWORDS):
        selected.append(SOURCE_FDA)
    if any(k in q for k in CLINICAL_KEYWORDS):
        selected.append(SOURCE_CLINICAL)
    if any(k in q for k in PUBMED_KEYWORDS):
        selected.append(SOURCE_PUBMED)

    return selected or list(config.ALL_SOURCES)


def retrieve_from_source(query: str, source: str, n_results: int = config.N_CANDIDATES) -> dict:
    """Retrieves candidates filtered by source from Qdrant using query_points API."""
    from .indexing import embed_query, get_qdrant_client

    if source not in config.ALL_SOURCES:
        raise ValueError(f"Unknown source: {source}")

    client = get_qdrant_client()
    query_vector = embed_query(query)

    # Filter points in the unified Qdrant collection by source
    source_filter = Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=source)
            )
        ]
    )

    # Retrieve points using Qdrant's query_points API
    response = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        query_filter=source_filter,
        limit=n_results,
        with_payload=True,
    )
    hits = response.points

    # Format output to maintain strict backwards compatibility with rerank_results()
    return {
        "documents": [[h.payload.get("text", "") for h in hits]],
        "metadatas": [[{k: v for k, v in (h.payload or {}).items() if k != "text"} for h in hits]],
        "distances": [[1.0 - float(h.score) for h in hits]],  # Convert Cosine similarity to distance
    }


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANKER_MODEL)


def rerank_results(query: str, results: dict, top_k: int = config.TOP_K) -> list[dict]:
    documents = results["documents"][0]
    if not documents:
        return []

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    scores = get_reranker().predict([[query, d] for d in documents])

    ranked = [
        {
            "document": doc,
            "metadata": meta,
            "distance": dist,
            "rerank_score": float(score),
        }
        for doc, meta, dist, score in zip(documents, metadatas, distances, scores)
    ]
    ranked.sort(key=lambda r: r["rerank_score"], reverse=True)
    return ranked[:top_k]


def agentic_retrieve(query: str, top_k: int = config.TOP_K, verbose: bool = True) -> dict:
    sources = route_query(query)
    if verbose:
        print(f"[agent] question: {query}")
        print(f"[agent] selected sources: {sources}")

    final = {}
    for source in sources:
        if verbose:
            print(f"[agent] searching {source} ...")
        results = retrieve_from_source(query, source)
        final[source] = rerank_results(query, results, top_k=top_k)
    return final