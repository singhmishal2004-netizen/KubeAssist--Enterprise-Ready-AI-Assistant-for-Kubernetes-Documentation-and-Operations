
"""
Run golden questions against the local or deployed KubeAssist API.

Features:
- Sequential API requests
- Unique thread ID per benchmark question
- Checkpoint after every successful response
- Resume interrupted runs
- Exponential backoff for transient failures
- Retry-After header support
- Golden dataset validation
- RAGAS dataset export in original order
"""

import json
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from requests.exceptions import RequestException

from .config import (
    QUERY_ENDPOINT,
    GOLDEN_DATASET_PATH,
    REQUESTS_PER_MINUTE,
    REQUEST_TIMEOUT_SECONDS,
    MAX_RETRIES,
    CHECKPOINT_PATH,
    RAGAS_INPUT_PATH,
)


def load_json(path):
    """Load JSON data from a file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_golden_dataset(golden):
    """Validate the golden dataset before making API calls."""

    if not isinstance(golden, list):
        raise ValueError("Golden dataset must contain a JSON array.")

    if not golden:
        raise ValueError("Golden dataset is empty.")

    required_fields = ["id", "user_input", "reference"]
    ids = []

    for index, item in enumerate(golden, start=1):

        if not isinstance(item, dict):
            raise ValueError(
                f"Dataset item {index} must be a JSON object."
            )

        missing = [
            field for field in required_fields
            if field not in item
        ]

        if missing:
            raise ValueError(
                f"Dataset item {index} is missing fields: {missing}"
            )

        if not str(item["id"]).strip():
            raise ValueError(f"Dataset item {index} has an empty ID.")

        if not isinstance(item["user_input"], str) or not item["user_input"].strip():
            raise ValueError(
                f"Dataset item {index} has an invalid user_input."
            )

        if not isinstance(item["reference"], str) or not item["reference"].strip():
            raise ValueError(
                f"Dataset item {index} has an invalid reference."
            )

        ids.append(str(item["id"]))

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs found in golden dataset.")

    return golden


def load_completed_ids(path):
    """Read IDs already saved in the checkpoint file."""

    completed = set()
    checkpoint = Path(path)

    if not checkpoint.exists():
        return completed

    with checkpoint.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            try:
                record = json.loads(line)
                completed.add(str(record["id"]))

            except (json.JSONDecodeError, KeyError, TypeError):
                print(
                    f"Warning: Ignoring invalid checkpoint "
                    f"line {line_number}."
                )

    return completed


def append_jsonl(path, record):
    """Append one record to a JSONL checkpoint file."""

    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    with checkpoint.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def get_retry_after(response, attempt):
    """
    Calculate retry delay.

    Supports:
    - Retry-After in seconds
    - Retry-After as an HTTP date
    - Exponential backoff fallback
    """

    retry_after = response.headers.get("Retry-After")

    if retry_after:

        try:
            return max(0.0, float(retry_after))

        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)

                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(
                        tzinfo=timezone.utc
                    )

                delay = (
                    retry_time - datetime.now(timezone.utc)
                ).total_seconds()

                return max(0.0, delay)

            except (TypeError, ValueError, OverflowError):
                pass

    return min(2 ** attempt, 60)


def call_kubeassist(question, sample_id):
    """
    Send one question to KubeAssist.

    Retries transient network errors, HTTP 429 and HTTP 5xx.
    Other HTTP errors are raised immediately.
    """

    payload = {
        "q": question,
        "thread_id": (
            f"ragas-{sample_id}-{uuid.uuid4().hex[:8]}"
        ),
    }

    for attempt in range(MAX_RETRIES):

        try:
            response = requests.post(
                QUERY_ENDPOINT,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        except RequestException:

            if attempt == MAX_RETRIES - 1:
                raise

            wait = min(2 ** attempt, 60)

            print(
                f"  Network error. Retrying in {wait} seconds..."
            )

            time.sleep(wait)
            continue

        if response.status_code == 429 or response.status_code >= 500:

            if attempt == MAX_RETRIES - 1:
                response.raise_for_status()

            wait = get_retry_after(response, attempt)

            print(
                f"  HTTP {response.status_code}. "
                f"Retrying in {wait:.1f} seconds..."
            )

            time.sleep(wait)
            continue

        # Non-retryable HTTP errors fail immediately.
        response.raise_for_status()

        return response.json()

    raise RuntimeError(
        "KubeAssist API request failed after retries."
    )


def normalize_contexts(contexts):
    """Normalize retrieved contexts into a list of strings."""

    if contexts is None:
        return []

    if not isinstance(contexts, list):
        contexts = [contexts]

    return [
        str(context)
        for context in contexts
        if context is not None and str(context).strip()
    ]


def main():

    if REQUESTS_PER_MINUTE < 1:
        raise ValueError(
            "REQUESTS_PER_MINUTE must be at least 1."
        )

    if MAX_RETRIES < 1:
        raise ValueError(
            "MAX_RETRIES must be at least 1."
        )

    # Load and validate dataset.
    golden = validate_golden_dataset(
        load_json(GOLDEN_DATASET_PATH)
    )

    print(f"Loaded {len(golden)} golden questions.")

    # Prepare checkpoint directory.
    Path(CHECKPOINT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = load_completed_ids(CHECKPOINT_PATH)

    print(
        f"Already completed: "
        f"{len(completed.intersection(str(x['id']) for x in golden))}"
        f"/{len(golden)}"
    )

    # Minimum interval between new benchmark questions.
    interval = 60.0 / REQUESTS_PER_MINUTE
    last_request_at = 0.0

    for index, item in enumerate(golden, start=1):

        sample_id = str(item["id"])

        if sample_id in completed:
            print(
                f"[{index}/{len(golden)}] "
                f"Skipping {sample_id}: already checkpointed."
            )
            continue

        # Enforce sequential request spacing.
        elapsed = time.monotonic() - last_request_at

        if last_request_at and elapsed < interval:
            time.sleep(interval - elapsed)

        print(
            f"\n[{index}/{len(golden)}] "
            f"Evaluating {sample_id}: {item['user_input']}"
        )

        result = call_kubeassist(
            item["user_input"],
            sample_id,
        )

        last_request_at = time.monotonic()

        if not isinstance(result, dict):
            raise ValueError(
                f"Invalid API response for sample {sample_id}."
            )

        answer = result.get("answer", "")

        if not isinstance(answer, str):
            answer = str(answer) if answer is not None else ""

        contexts = normalize_contexts(
            result.get("retrieved_context", [])
        )

        record = {
            "id": sample_id,
            "user_input": item["user_input"],
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": item["reference"],
            "reference_contexts": item.get(
                "reference_contexts", []
            ),
            "metadata": item.get("metadata", {}),
            "api_status": result.get("status"),
            "thought_process": result.get(
                "thought_process", []
            ),
        }

        # Save immediately after successful API response.
        append_jsonl(CHECKPOINT_PATH, record)

        completed.add(sample_id)

        print(
            f"  Checkpoint saved. "
            f"Retrieved contexts: {len(contexts)}"
        )

    # Read checkpoint records.
    by_id = {}

    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as file:

        for line in file:

            try:
                row = json.loads(line)
                by_id[str(row["id"])] = row

            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    # Export in original golden dataset order.
    ordered = [
        by_id[str(item["id"])]
        for item in golden
        if str(item["id"]) in by_id
    ]

    Path(RAGAS_INPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(RAGAS_INPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            ordered,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 55)
    print("KUBEASSIST DATA COLLECTION SUMMARY")
    print("=" * 55)

    print(f"Golden questions: {len(golden)}")
    print(f"Collected responses: {len(ordered)}")
    print(f"Remaining: {len(golden) - len(ordered)}")

    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"RAGAS input: {RAGAS_INPUT_PATH}")


if __name__ == "__main__":
    main()
