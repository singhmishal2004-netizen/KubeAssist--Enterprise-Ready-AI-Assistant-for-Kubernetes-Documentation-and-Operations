
import time
import requests
import logfire

from app.config import settings


JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
RERANKER_MODEL = "jina-reranker-v3.5"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


def rerank_documents(
    query: str,
    documents: list[str],
    top_n: int = 5
) -> list[str]:
    """
    Rerank Qdrant-retrieved documents using Jina AI.

    Returns the top_n documents sorted by relevance.
    Falls back to original Qdrant order if reranking fails.
    """

    if not documents or top_n <= 0:
        return []

    if not query or not query.strip():
        logfire.warning("[Reranker] Empty query. Skipping reranking.")
        return documents[:top_n]

    if not settings.JINA_API_KEY:
        logfire.error("[Reranker] JINA_API_KEY is missing.")
        return documents[:top_n]

    # Preserve original indices when filtering empty documents.
    valid_documents = [
        (index, doc)
        for index, doc in enumerate(documents)
        if isinstance(doc, str) and doc.strip()
    ]

    if not valid_documents:
        return []

    passages = [doc for _, doc in valid_documents]

    payload = {
        "model": RERANKER_MODEL,
        "query": query,
        "top_n": min(top_n, len(passages)),
        "documents": passages,
        "return_documents": False
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.JINA_API_KEY}"
    }

    start_time = time.time()

    logfire.info(
        f"[Reranker] Sending {len(passages)} candidates "
        f"to {RERANKER_MODEL}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                JINA_RERANK_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            results = response.json().get("results", [])

            if not results:
                logfire.warning(
                    "[Reranker] No results returned. "
                    "Using original Qdrant order."
                )
                return documents[:top_n]

            # Results contain indices because return_documents=False.
            reranked_documents = []

            for result in results:
                result_index = result.get("index")

                if (
                    isinstance(result_index, int)
                    and 0 <= result_index < len(passages)
                ):
                    reranked_documents.append(passages[result_index])

            duration = time.time() - start_time

            top_score = results[0].get(
                "relevance_score", "N/A"
            )

            logfire.info(
                f"[Reranker] Completed in {duration:.2f}s. "
                f"Returned {len(reranked_documents)} documents. "
                f"Top relevance score: {top_score}"
            )

            return reranked_documents[:top_n]

        except requests.exceptions.HTTPError as e:
            status_code = (
                e.response.status_code
                if e.response is not None
                else None
            )

            # Authentication and validation errors should not be retried.
            if (
                status_code is not None
                and status_code < 500
                and status_code != 429
            ):
                logfire.error(
                    f"[Reranker] HTTP error: {status_code}"
                )
                break

            logfire.warning(
                f"[Reranker] Temporary HTTP error "
                f"({attempt + 1}/{MAX_RETRIES}): {e}"
            )

        except requests.exceptions.RequestException as e:
            logfire.warning(
                f"[Reranker] Request failed "
                f"({attempt + 1}/{MAX_RETRIES}): {e}"
            )

        except (ValueError, TypeError, KeyError) as e:
            logfire.error(f"[Reranker] Invalid API response: {e}")
            break

        except Exception as e:
            logfire.error(f"[Reranker] Unexpected error: {e}")
            break

        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)

    logfire.error(
        "[Reranker] Reranking failed. "
        "Falling back to Qdrant results."
    )

    return documents[:top_n]
