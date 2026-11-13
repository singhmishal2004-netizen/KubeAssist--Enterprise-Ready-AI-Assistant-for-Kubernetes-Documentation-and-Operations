
import math
import time
import threading

from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests
import logfire

from app.config import settings


# ── Jina API Configuration ─────────────────────────────────────

JINA_API_URL = "https://api.jina.ai/v1/embeddings"

# Model configuration
_active_model = settings.JINA_EMBEDDING_MODEL
EMBEDDING_DIM = 1024

# API limits shown in your Jina dashboard
API_RPM_LIMIT = 100
API_TPM_LIMIT = 100_000

# Safety margins to avoid hitting provider limits
SAFE_RPM_LIMIT = 90
SAFE_TPM_LIMIT = 90_000

# Batch configuration
BATCH_SIZE = 32
MAX_TOKENS_PER_REQUEST = 20_000

# Request configuration
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60
RATE_LIMIT_WINDOW = 60


# ── Token Estimation ───────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Estimate token count without installing a tokenizer.

    Uses a conservative character-based approximation.
    This is not an exact tokenizer.
    """

    return max(1, math.ceil(len(text) / 3))


# ── Rate Limiter ───────────────────────────────────────────────

class JinaRateLimiter:
    """
    Thread-safe rolling-window rate limiter.

    Tracks:
        - Requests per minute
        - Estimated tokens per minute

    Every API attempt, including retries, is counted.
    """

    def __init__(
        self,
        requests_per_minute: int = SAFE_RPM_LIMIT,
        tokens_per_minute: int = SAFE_TPM_LIMIT,
        window_seconds: int = RATE_LIMIT_WINDOW
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.window_seconds = window_seconds

        self.request_history = deque()

        self.lock = threading.Lock()

    def _remove_expired(self, current_time: float):
        """Remove requests outside the rolling time window."""

        while self.request_history:
            timestamp, _ = self.request_history[0]

            if current_time - timestamp >= self.window_seconds:
                self.request_history.popleft()
            else:
                break

    def acquire(self, estimated_tokens: int):
        """
        Reserve request and token capacity.

        Waits automatically until both limits allow
        another API request.
        """

        if estimated_tokens > self.tokens_per_minute:
            raise ValueError(
                "A single request exceeds the configured "
                "tokens-per-minute budget."
            )

        while True:

            with self.lock:

                current_time = time.monotonic()

                self._remove_expired(current_time)

                current_requests = len(self.request_history)

                current_tokens = sum(
                    tokens
                    for _, tokens in self.request_history
                )

                requests_available = (
                    current_requests < self.requests_per_minute
                )

                tokens_available = (
                    current_tokens + estimated_tokens
                    <= self.tokens_per_minute
                )

                if requests_available and tokens_available:

                    self.request_history.append(
                        (current_time, estimated_tokens)
                    )

                    return

                # Wait until the oldest request expires.
                oldest_timestamp, _ = self.request_history[0]

                wait_seconds = max(
                    0.1,
                    self.window_seconds
                    - (current_time - oldest_timestamp)
                )

            logfire.info(
                "Jina rate limit reached. Waiting before next request.",
                wait_seconds=round(wait_seconds, 2),
                estimated_tokens=estimated_tokens
            )

            time.sleep(wait_seconds)


# One shared limiter for this Python process.
rate_limiter = JinaRateLimiter()


# ── Retry Helper ───────────────────────────────────────────────

def _get_retry_delay(
    retry_after: str | None,
    attempt: int
) -> float:
    """
    Calculate retry delay.

    Supports numeric Retry-After values and HTTP dates.
    Falls back to exponential backoff.
    """

    if retry_after:

        try:
            return min(max(float(retry_after), 0), 60)

        except ValueError:

            try:
                retry_date = parsedate_to_datetime(retry_after)

                if retry_date.tzinfo is None:
                    retry_date = retry_date.replace(
                        tzinfo=timezone.utc
                    )

                delay = (
                    retry_date - datetime.now(timezone.utc)
                ).total_seconds()

                return min(max(delay, 0), 60)

            except (TypeError, ValueError, OverflowError):
                pass

    return min(2 ** attempt, 60)


# ── API Request ─────────────────────────────────────────────────

def _jina_call(
    texts: list[str],
    task: str
) -> list[list[float]]:
    """
    Generate embeddings using Jina AI.

    task:
        retrieval.passage -> document chunks
        retrieval.query   -> user queries
    """

    if not settings.JINA_API_KEY:
        raise ValueError("JINA_API_KEY is not configured.")

    if not texts:
        return []

    estimated_tokens = sum(
        estimate_tokens(text)
        for text in texts
    )

    if estimated_tokens > MAX_TOKENS_PER_REQUEST:
        raise ValueError(
            f"Request exceeds the configured batch token budget "
            f"of {MAX_TOKENS_PER_REQUEST} estimated tokens."
        )

    payload = {
        "model": _active_model,
        "task": task,
        "normalized": True,
        "input": [
            {"text": text}
            for text in texts
        ]
    }

    headers = {
        "Authorization": f"Bearer {settings.JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(MAX_RETRIES):

        try:

            # Every attempt consumes request capacity.
            rate_limiter.acquire(estimated_tokens)

            response = requests.post(
                JINA_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            result = response.json()

            # Preserve original input order.
            data = sorted(
                result["data"],
                key=lambda item: item["index"]
            )

            embeddings = [
                item["embedding"]
                for item in data
            ]

            # Validate response count.
            if len(embeddings) != len(texts):
                raise ValueError(
                    "Jina returned an unexpected number of embeddings."
                )

            # Validate vector dimensions.
            invalid_dimensions = [
                len(vector)
                for vector in embeddings
                if len(vector) != EMBEDDING_DIM
            ]

            if invalid_dimensions:
                raise ValueError(
                    f"Expected embedding dimension {EMBEDDING_DIM}, "
                    f"but received dimensions: {invalid_dimensions}"
                )

            # Capture actual usage when supplied by the API.
            usage = result.get("usage", {})
            actual_tokens = usage.get("total_tokens")

            logfire.info(
                "Jina embeddings generated",
                model=_active_model,
                task=task,
                input_count=len(texts),
                estimated_tokens=estimated_tokens,
                actual_tokens=actual_tokens,
                vector_dimension=EMBEDDING_DIM
            )

            return embeddings

        except requests.RequestException as error:

            status_code = (
                error.response.status_code
                if error.response is not None
                else None
            )

            retryable = (
                status_code == 429
                or status_code in (500, 502, 503, 504)
                or status_code is None
            )

            if not retryable or attempt == MAX_RETRIES - 1:

                logfire.exception(
                    "Jina embedding request failed",
                    status_code=status_code
                )

                raise

            retry_after = (
                error.response.headers.get("Retry-After")
                if error.response is not None
                else None
            )

            wait_seconds = _get_retry_delay(
                retry_after,
                attempt
            )

            logfire.warning(
                "Jina API retry scheduled",
                attempt=attempt + 1,
                wait_seconds=wait_seconds,
                status_code=status_code
            )

            time.sleep(wait_seconds)

        except Exception:

            logfire.exception(
                "Jina embedding processing failed"
            )

            raise

    raise RuntimeError("Jina embedding retries exhausted.")


# ── Dynamic Batch Processing ───────────────────────────────────

def _batches(texts: list[str]):
    """
    Create batches using both text-count and token limits.

    A batch stops growing when either limit is reached.
    """

    batch = []
    batch_tokens = 0

    for text in texts:

        text_tokens = estimate_tokens(text)

        if text_tokens > MAX_TOKENS_PER_REQUEST:
            raise ValueError(
                "A single text exceeds the configured "
                "batch token budget. Split the text before embedding."
            )

        should_flush = (
            len(batch) >= BATCH_SIZE
            or batch_tokens + text_tokens > MAX_TOKENS_PER_REQUEST
        )

        if batch and should_flush:

            yield batch

            batch = []
            batch_tokens = 0

        batch.append(text)
        batch_tokens += text_tokens

    if batch:
        yield batch


def _embed_batch(
    batch: list[str],
    task: str
) -> list[list[float]]:
    """Generate embeddings for one batch."""

    return _jina_call(batch, task)


# ── Public API ──────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """
    Return the expected embedding dimension.

    This is a configured value, not a live API lookup.
    """

    return EMBEDDING_DIM


def embed_query(query: str) -> list[float]:
    """
    Generate an embedding for a user search query.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Query must be a non-empty string."
        )

    with logfire.span(
        "jina.embed_query",
        model=_active_model
    ):

        embeddings = _jina_call(
            [query],
            task="retrieval.query"
        )

        return embeddings[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for document chunks.

    Uses dynamic batching and rate limiting.
    """

    if not texts:
        return []

    if any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "Input texts must contain only non-empty strings."
        )

    all_embeddings = []

    with logfire.span(
        "jina.embed_documents",
        model=_active_model,
        total_texts=len(texts)
    ):

        for batch_number, batch in enumerate(
            _batches(texts),
            start=1
        ):

            with logfire.span(
                "jina.embed_batch",
                batch_number=batch_number,
                batch_size=len(batch),
                estimated_tokens=sum(
                    estimate_tokens(text)
                    for text in batch
                )
            ):

                embeddings = _embed_batch(
                    batch,
                    task="retrieval.passage"
                )

                all_embeddings.extend(embeddings)

    return all_embeddings
