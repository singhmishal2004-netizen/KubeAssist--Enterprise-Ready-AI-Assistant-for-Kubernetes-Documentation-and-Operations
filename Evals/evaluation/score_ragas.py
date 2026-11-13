
"""
Score collected KubeAssist responses using RAGAS 0.2.15.

Judge LLM:
    Existing KubeAssist Portkey/Groq gateway.

Embeddings:
    Existing hosted Jina embedding integration.

Features:
- Sequential RAGAS execution
- Small evaluation batches
- Configurable pauses between batches
- Existing Jina embedding rate limiter
- Incremental CSV report generation
- Handles individual metric failures without stopping
"""

import json
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from .config import (
    RAGAS_INPUT_PATH,
    REPORT_PATH,
    RAGAS_BATCH_SIZE,
    RAGAS_BATCH_PAUSE_SECONDS,
)


def load_rows():
    """Load and validate collected RAGAS evaluation records."""

    input_path = Path(RAGAS_INPUT_PATH)

    if not input_path.exists():
        raise FileNotFoundError(
            f"RAGAS input file not found: {input_path}"
        )

    with input_path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    if not isinstance(rows, list):
        raise ValueError(
            "RAGAS input must contain a JSON array."
        )

    clean = []

    for row in rows:

        if not row.get("response"):
            print(
                f"Skipping {row.get('id')}: empty response."
            )
            continue

        if not row.get("user_input") or not row.get("reference"):
            print(
                f"Skipping {row.get('id')}: "
                f"missing question or reference."
            )
            continue

        contexts = row.get("retrieved_contexts", [])

        if not isinstance(contexts, list):
            contexts = [str(contexts)] if contexts else []

        clean.append({
            "user_input": row["user_input"],
            "response": row["response"],
            "retrieved_contexts": [
                str(context)
                for context in contexts
                if context is not None and str(context).strip()
            ],
            "reference": row["reference"],
        })

    return clean


def build_jina_embeddings():
    """
    Build a LangChain-compatible adapter for the existing
    Jina embedding service.

    embed_texts() uses document-side embedding behavior.
    embed_query() uses query-side embedding behavior.

    Both retain the project's existing rate limiter.
    """

    from langchain_core.embeddings import Embeddings

    from app.services.retrieval.embedding import (
        embed_query,
        embed_texts,
    )

    class JinaQueryEmbeddings(Embeddings):

        def embed_documents(self, texts):
            return embed_texts(texts)

        def embed_query(self, text):
            return embed_query(text)

    return LangchainEmbeddingsWrapper(
        JinaQueryEmbeddings()
    )


def build_judge_llm():
    """Initialize the judge LLM through the existing Portkey gateway."""

    from app.gateways.client import get_langchain_llm

    llm = get_langchain_llm(
        feature="ragas_evaluation"
    )

    return LangchainLLMWrapper(llm)


def main():

    if RAGAS_BATCH_SIZE < 1:
        raise ValueError(
            "RAGAS_BATCH_SIZE must be at least 1."
        )

    if RAGAS_BATCH_PAUSE_SECONDS < 0:
        raise ValueError(
            "RAGAS_BATCH_PAUSE_SECONDS cannot be negative."
        )

    rows = load_rows()

    if not rows:
        raise ValueError(
            "No usable rows found in the RAGAS input file."
        )

    print("=" * 55)
    print("KUBEASSIST RAGAS EVALUATION")
    print("=" * 55)

    print(f"Usable evaluation samples: {len(rows)}")
    print(f"Batch size: {RAGAS_BATCH_SIZE}")
    print("Maximum RAGAS workers: 1")

    # Initialize hosted evaluation services.
    print("\nInitializing judge LLM...")
    judge_llm = build_judge_llm()

    print("Initializing Jina embeddings...")
    judge_embeddings = build_jina_embeddings()

    # RAGAS evaluation metrics.
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    # Explicitly limit concurrent metric execution.
    run_config = RunConfig(
        max_workers=1,
    )

    results = []

    total_batches = (
        len(rows) + RAGAS_BATCH_SIZE - 1
    ) // RAGAS_BATCH_SIZE

    for batch_index, start in enumerate(
        range(0, len(rows), RAGAS_BATCH_SIZE),
        start=1,
    ):

        batch = rows[
            start:start + RAGAS_BATCH_SIZE
        ]

        end = start + len(batch)

        print(
            f"\nScoring batch {batch_index}/{total_batches}"
        )

        print(
            f"Samples: {start + 1}-{end} "
            f"of {len(rows)}"
        )

        dataset = Dataset.from_list(batch)

        scored = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config,
            raise_exceptions=False,
        )

        frame = scored.to_pandas()

        results.append(frame)

        # Save results after every completed batch.
        combined = pd.concat(
            results,
            ignore_index=True,
        )

        Path(REPORT_PATH).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        combined.to_csv(
            REPORT_PATH,
            index=False,
        )

        print(
            f"Saved partial report: {REPORT_PATH}"
        )

        if batch_index < total_batches:
            print(
                f"Waiting {RAGAS_BATCH_PAUSE_SECONDS} "
                f"seconds before next batch..."
            )

            time.sleep(
                RAGAS_BATCH_PAUSE_SECONDS
            )

    final = pd.concat(
        results,
        ignore_index=True,
    )

    print("\n" + "=" * 55)
    print("KUBEASSIST EVALUATION RESULTS")
    print("=" * 55)

    print("\nMean metric scores:")

    print(
        final.mean(numeric_only=True)
        .round(4)
        .to_string()
    )

    print(f"\nTotal evaluated samples: {len(final)}")
    print(f"Detailed report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
