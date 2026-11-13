# KubeAssist RAG evaluation

This folder contains a two-stage evaluation workflow.

## Files

- `../goldens.json`: source-grounded questions, references and reference contexts.
- `run_pipeline.py`: calls the KubeAssist `/query` API sequentially and checkpoints each result.
- `score_ragas.py`: scores collected responses with RAGAS.
- `config.py`: rate limits, paths and batch sizes.

## Important separation

The pipeline runner calls KubeAssist, so KubeAssist may call Portkey/Groq, Jina and Qdrant.
The scoring script calls the Portkey-backed judge and the Jina embedding endpoint.
No model/API calls are made when merely loading or validating `goldens.json`.

## Setup

Run from the repository root in the project's virtual environment:

```powershell
pip install -r Evals/evaluation/requirements-evaluation.txt
```

Ensure the FastAPI backend is running and `.env` contains the existing project credentials.

## Stage 1: collect application outputs

```powershell
$env:KUBEASSIST_API_URL="http://localhost:8000"
$env:EVAL_REQUESTS_PER_MINUTE="8"
$env:PYTHONPATH="Evals"
python -m evaluation.run_pipeline
```

The script loads `Evals/goldens.json` by default, uses a unique `thread_id` per test case, and saves each completed sample to `Evals/evaluation/results/raw_responses.jsonl`. It then writes `Evals/evaluation/results/ragas_dataset.json`.

If interrupted, rerun the command. Already checkpointed IDs are skipped.

## Stage 2: score with RAGAS

```powershell
$env:RAGAS_BATCH_SIZE="5"
$env:RAGAS_BATCH_PAUSE_SECONDS="15"
$env:PYTHONPATH="Evals"
python -m evaluation.score_ragas
```

The judge uses the existing `get_langchain_llm(feature="ragas_evaluation")` gateway integration. The embedding adapter uses the existing Jina `embed_query` function. Evaluation batches are intentionally small and sequential.

## Metrics

- Faithfulness
- Answer relevancy
- Context precision
- Context recall
- Answer correctness

The RAGAS 0.2.15 API is pinned in the evaluation-only requirements file. Confirm compatibility with the project's installed LangChain dependency set before installing into the main virtual environment; preferably use a separate evaluation virtual environment if dependency resolution would downgrade production packages.

## Rate-limit notes

- Default collection limit: 8 API requests/minute.
- Default RAGAS batch size: 5.
- Default pause between RAGAS batches: 15 seconds.
- Retries use exponential backoff and honor numeric `Retry-After`.
- The existing Jina embedding service has its own process-local limiter. Do not run multiple evaluation processes concurrently.
- Groq/Portkey provider limits vary by account/model. If 429 responses occur, lower `EVAL_REQUESTS_PER_MINUTE`, `RAGAS_BATCH_SIZE`, or increase `RAGAS_BATCH_PAUSE_SECONDS`.

## Important interpretation notes

1. This is a retrieval benchmark, not a conversational-memory benchmark. Each sample gets a fresh thread ID.
2. `goldens.json` includes Databricks examples as a separate category. If KubeAssist is intended to answer only Kubernetes, either remove those four records from the benchmark or treat them as expected off-topic/guardrail cases instead of scoring them as supported RAG questions.
3. Some uploaded source material is historical/version-specific (the architecture deck identifies Kubernetes v1.8). Scores measure grounding against the supplied corpus, not whether every statement is current Kubernetes best practice.
4. Review the raw responses and retrieved contexts before trusting aggregate scores. RAGAS is an evaluator, not a substitute for human review.
