
from functools import lru_cache
from typing import Any

from portkey_ai import (
    Portkey,
    createHeaders,
    PORTKEY_GATEWAY_URL,
)

from langchain_openai import ChatOpenAI

from app.config import settings


# ==================================================
# Gateway Configuration
# ==================================================

SUPPORTED_FEATURES = {
    "planner",
    "responder",
    "guardrails",
    "rag",
}


def validate_gateway_settings() -> None:
    """Validate required Portkey and Groq configuration."""

    if not settings.PORTKEY_API_KEY:
        raise RuntimeError(
            "PORTKEY_API_KEY is missing from environment."
        )

    if not settings.GROQ_SLUG:
        raise RuntimeError(
            "GROQ_SLUG is missing. Add the exact saved "
            "integration slug from your Portkey Model Catalog."
        )

    if not settings.GROQ_MODEL:
        raise RuntimeError(
            "GROQ_MODEL is missing from environment."
        )


validate_gateway_settings()


# ==================================================
# Model Routing
# ==================================================

# Route through the saved Portkey integration.
#
# Example:
# @my-groq-integration/openai/gpt-oss-120b
#
# GROQ_MODEL must contain the model identifier expected
# by your saved integration.

PRIMARY_MODEL = (
    f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"
)


# ==================================================
# Native Portkey Client
# ==================================================

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    **(
        {"config": settings.PORTKEY_CONFIG_ID}
        if settings.PORTKEY_CONFIG_ID
        else {}
    ),
)


# ==================================================
# LangChain + Portkey Client Factory
# ==================================================

@lru_cache(maxsize=4)
def get_langchain_llm(
    feature: str = "rag",
) -> ChatOpenAI:
    """
    Return a cached Portkey-backed ChatOpenAI instance.

    Architecture:

        LangChain
            |
        Portkey Gateway
            |
        Saved Groq Integration
            |
        Groq Model

    Each feature receives separate Portkey metadata.
    """

    if feature not in SUPPORTED_FEATURES:
        raise ValueError(
            f"Unsupported LLM feature: {feature}. "
            f"Supported features: {sorted(SUPPORTED_FEATURES)}"
        )

    headers = createHeaders(
        api_key=settings.PORTKEY_API_KEY,
        metadata={
            "feature": feature,
            "_user": "kubeassist",
            "environment": settings.ENVIRONMENT,
        },
        **(
            {"config": settings.PORTKEY_CONFIG_ID}
            if settings.PORTKEY_CONFIG_ID
            else {}
        ),
    )

    return ChatOpenAI(
        model=PRIMARY_MODEL,
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        temperature=0,
        timeout=60,
        max_retries=2,
        include_response_headers=True,
    )


# ==================================================
# Portkey Cache Metadata
# ==================================================

def extract_cache_status(response: Any) -> str:
    """
    Extract x-portkey-cache-status from a LangChain
    AIMessage or a raw HTTP response object.
    """

    candidate_headers = []

    # LangChain AIMessage metadata
    response_metadata = getattr(
        response,
        "response_metadata",
        None,
    )

    if isinstance(response_metadata, dict):
        candidate_headers.append(
            response_metadata.get("headers", {})
        )

    # Raw response fallbacks
    for attribute in (
        "_raw_response",
        "_response",
        "_http_response",
    ):
        raw_response = getattr(
            response,
            attribute,
            None,
        )

        if raw_response is not None:
            candidate_headers.append(
                getattr(raw_response, "headers", {})
            )

    for headers in candidate_headers:

        if not hasattr(headers, "items"):
            continue

        normalized_headers = {
            str(key).lower(): value
            for key, value in headers.items()
        }

        status = normalized_headers.get(
            "x-portkey-cache-status"
        )

        if status:
            return str(status).upper()

    return "UNKNOWN"
