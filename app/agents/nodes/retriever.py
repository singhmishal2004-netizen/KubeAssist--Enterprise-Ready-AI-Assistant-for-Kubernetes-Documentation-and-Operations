
import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge
)
from app.services.retrieval.reranking_service import (
    rerank_documents
)


def retrieve_node(state: AgentState):
    """
    Retrieves relevant Kubernetes knowledge from Qdrant
    and reranks the results using Jina Reranker.
    """

    query = state.get("current_query", "").strip()

    if not query:
        logfire.warning("[Retriever] Empty search query.")

        return {
            "documents": [],
            "status": "No search query provided.",
            "plan": state.get("plan", []) + ["Retrieval skipped"]
        }

    with logfire.span("Knowledge Retrieval"):

        # STEP 1: Retrieve candidates from Qdrant
        logfire.info(f"[Retriever] Searching for: {query}")

        raw_results = search_enterprise_knowledge(
            query=query,
            limit=15
        )

        logfire.info(
            f"[Retriever] Retrieved {len(raw_results)} candidates."
        )

        if not raw_results:
            logfire.warning("[Retriever] No documents found.")

            return {
                "documents": [],
                "status": "No relevant documents found.",
                "plan": state.get("plan", []) + [
                    "Retrieval completed: No results"
                ]
            }

        # STEP 2: Extract document content
        doc_contents = [
            doc["content"]
            for doc in raw_results
            if doc.get("content")
        ]

        if not doc_contents:
            return {
                "documents": [],
                "status": "Retrieved documents contained no text.",
                "plan": state.get("plan", []) + [
                    "Retrieval completed: Empty content"
                ]
            }

        # STEP 3: Rerank candidates using Jina
        with logfire.span("Semantic Reranking"):

            reranked_contents = rerank_documents(
                query=query,
                documents=doc_contents,
                top_n=5
            )

            logfire.info(
                f"[Retriever] Reranking complete. "
                f"Selected {len(reranked_contents)} documents."
            )

        # STEP 4: Prepare context for the responder
        formatted_docs = [
            f"CONTENT:\n{doc}"
            for doc in reranked_contents
        ]

    return {
        "documents": formatted_docs,
        "status": "Technical context retrieved successfully.",
        "plan": state.get("plan", []) + [
            f"Qdrant: {len(raw_results)} candidates retrieved",
            f"Reranking: {len(reranked_contents)} documents selected"
        ]
    }
