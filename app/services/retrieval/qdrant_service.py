
import logfire

from qdrant_client import QdrantClient

from app.config import settings
from app.services.retrieval.embedding import embed_query


client = QdrantClient(
    url=settings.QDRANT_CLUSTER_ENDPOINT,
    api_key=settings.QDRANT_API_KEY,
    timeout=120
)


def search_enterprise_knowledge(
    query: str,
    limit: int = 15
) -> list[dict]:

    """
    Performs semantic vector search in KubeAssist.

    Returns document content, score and original metadata.
    """

    if not query or not query.strip():
        logfire.warning("[Qdrant] Empty query received.")
        return []

    if limit <= 0:
        return []

    try:
        with logfire.span("Qdrant Semantic Search"):

            query_vector = embed_query(query)

            response = client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            results = []

            for point in response.points:

                payload = point.payload or {}

                results.append({
                    "id": str(point.id),
                    "content": payload.get("text", ""),
                    "source": payload.get("source", "Unknown"),
                    "score": point.score,
                    "metadata": {
                        "source_type": payload.get("source_type"),
                        "document_id": payload.get("document_id"),
                        "chunk_index": payload.get("chunk_index"),
                        "version": payload.get("version"),
                        "is_latest": payload.get("is_latest"),
                        "embedding_model": payload.get("embedding_model")
                    }
                })

            logfire.info(
                f"[Qdrant] Retrieved {len(results)} candidates."
            )

            return results

    except Exception:
        logfire.exception("[Qdrant] Semantic search failed.")
        raise
