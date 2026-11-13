
import os
from pathlib import Path

from dotenv import load_dotenv


# ==================================================
# Environment Configuration
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def get_bool(name: str, default: bool = False) -> bool:
    """Convert environment variable into a boolean."""

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "true",
        "1",
        "yes",
        "on",
    }


class Settings:

    # ==================================================
    # Application
    # ==================================================

    APP_NAME = "KubeAssist"

    ENVIRONMENT = os.getenv(
        "ENVIRONMENT",
        "development",
    )

    DEBUG = get_bool("DEBUG", False)

    # ==================================================
    # Groq LLM
    # ==================================================

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    GROQ_FALLBACK_API_KEY = os.getenv(
        "GROQ_FALLBACK_API_KEY"
    )

    GROQ_MODEL = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b",
    )

    # Exact saved integration slug from Portkey Model Catalog
    GROQ_SLUG = os.getenv("GROQ_SLUG")

    # Evaluation Judge
    JUDGE_GROQ = os.getenv("JUDGE_GROQ")

    # ==================================================
    # Portkey LLM Gateway
    # ==================================================

    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")

    PORTKEY_CONFIG_ID = (
        os.getenv("PORTKEY_CONFIG_ID") or None
    )

    # ==================================================
    # Qdrant Vector Database
    # ==================================================

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    QDRANT_CLUSTER_ENDPOINT = os.getenv(
        "QDRANT_CLUSTER_ENDPOINT"
    )

    QDRANT_COLLECTION_NAME = os.getenv(
        "QDRANT_COLLECTION_NAME",
        "kubeassist_knowledge",
    )

    @property
    def QDRANT_COLLECTION(self):
        return self.QDRANT_COLLECTION_NAME

    # ==================================================
    # Embedding Providers
    # ==================================================

    # Primary: Jina AI
    JINA_API_KEY = os.getenv("JINA_API_KEY")

    JINA_EMBEDDING_MODEL = os.getenv(
        "JINA_EMBEDDING_MODEL",
        "jina-embeddings-v5-omni-small",
    )

    JINA_RERANKER_MODEL = os.getenv(
        "JINA_RERANKER_MODEL",
        "jina-reranker-v3.5",
    )

    # Fallback: Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # ==================================================
    # Observability - Logfire
    # ==================================================

    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

    # ==================================================
    # Observability - LangSmith
    # ==================================================

    LANGSMITH_TRACING = get_bool(
        "LANGSMITH_TRACING",
        False,
    )

    LANGSMITH_ENDPOINT = os.getenv(
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com",
    )

    LANGSMITH_API_KEY = os.getenv(
        "LANGSMITH_API_KEY"
    )

    LANGSMITH_PROJECT = os.getenv(
        "LANGSMITH_PROJECT",
        "KubeAssist",
    )

    # ==================================================
    # Frontend - Streamlit to FastAPI
    # ==================================================

    BACKEND_URL = os.getenv(
        "BACKEND_URL",
        "http://localhost:8000",
    )


settings = Settings()
