
"""Central configuration for the KubeAssist RAG evaluation workflow."""
import os

# Dataset
GOLDEN_DATASET_PATH = os.getenv(
    "GOLDEN_DATASET_PATH",
    "Evals/goldens.json"
)

# KubeAssist API
API_BASE_URL = os.getenv(
    "KUBEASSIST_API_URL",
    "http://localhost:8000"
)
QUERY_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/query"

# API rate limits and retries
REQUESTS_PER_MINUTE = int(
    os.getenv("EVAL_REQUESTS_PER_MINUTE", "8")
)
REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("EVAL_REQUEST_TIMEOUT_SECONDS", "120")
)
MAX_RETRIES = int(
    os.getenv("EVAL_MAX_RETRIES", "5")
)

# Evaluation output paths
CHECKPOINT_PATH = os.getenv(
    "EVAL_CHECKPOINT_PATH",
    "Evals/evaluation/results/raw_responses.jsonl"
)
RAGAS_INPUT_PATH = os.getenv(
    "RAGAS_INPUT_PATH",
    "Evals/evaluation/results/ragas_dataset.json"
)
REPORT_PATH = os.getenv(
    "EVAL_REPORT_PATH",
    "Evals/evaluation/results/ragas_scores.csv"
)

# RAGAS batch settings
RAGAS_BATCH_SIZE = int(
    os.getenv("RAGAS_BATCH_SIZE", "5")
)
RAGAS_BATCH_PAUSE_SECONDS = float(
    os.getenv("RAGAS_BATCH_PAUSE_SECONDS", "15")
)
