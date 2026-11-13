
import os
from pathlib import Path
from contextlib import asynccontextmanager

import logfire
from dotenv import load_dotenv

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
# ============================================================
# LOGFIRE CONFIGURATION
# ============================================================

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    send_to_logfire="if-token-present"
)

# ============================================================
# APPLICATION IMPORTS
# ============================================================

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from app.agents.graph import rag_agent
from app.guardrails.rails import initialize_rails, guard


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logfire.info("Initializing KubeAssist services.")

    try:
        initialize_rails()
        logfire.info("Guardrails classifier initialized successfully.")

    except Exception:
        logfire.exception("Guardrails initialization failed.")
        raise

    yield

    logfire.info("KubeAssist application shutting down.")


# ============================================================
# INITIALIZE APPLICATION
# ============================================================

app = FastAPI(
    title="KubeAssist - Agentic RAG API",
    description="Kubernetes Documentation and Operations Assistant",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class QueryRequest(BaseModel):

    q: str = Field(
        min_length=1,
        max_length=10000,
        description="User's question"
    )

    thread_id: str = Field(
        min_length=1,
        max_length=256,
        description="Unique conversation thread identifier"
    )


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "KubeAssist Agentic RAG API is live.",
        "status": "healthy"
    }


# ============================================================
# GRAPH VISUALIZATION
# ============================================================

@app.get("/graph")
def get_graph_image():

    """Return the Mermaid visualization of the LangGraph workflow."""

    try:

        png_bytes = rag_agent.get_graph().draw_mermaid_png()

        return Response(
            content=png_bytes,
            media_type="image/png"
        )

    except Exception:

        logfire.exception("Graph visualization failed.")

        raise HTTPException(
            status_code=500,
            detail="Could not generate graph visualization."
        )


# ============================================================
# QUERY ENDPOINT
# ============================================================

@app.post("/query")
def query(request: QueryRequest):

    """
    Executes the KubeAssist Agentic RAG pipeline.

    Flow:
    Guardrails → LangGraph Planner → Retrieval →
    Jina Reranker → Response Generation
    """

    q = request.q.strip()
    thread_id = request.thread_id.strip()

    if not q:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    if not thread_id:

        raise HTTPException(
            status_code=400,
            detail="Thread ID cannot be empty."
        )

    logfire.info(
        "Received user query.",
        thread_id=thread_id
    )

    try:

        # ====================================================
        # GATE 1: GUARDRAILS
        # ====================================================

        rail_fired, rail_response = guard(q)

        if rail_fired:

            logfire.info(
                "Guardrails blocked or handled the request.",
                thread_id=thread_id
            )

            return {
                "question": q,
                "answer": rail_response or "Request could not be processed.",
                "thought_process": [
                    "Guardrails: Request handled",
                    "Retrieval: Skipped"
                ],
                "status": "blocked",
                "retrieved_context": [],
                "sources": []
            }

        # ====================================================
        # GATE 2: INITIAL LANGGRAPH STATE
        # ====================================================

        initial_state = {

            "messages": [
                {
                    "role": "user",
                    "content": q
                }
            ],

            "current_query": q,

            "documents": [],

            "plan": ["Start"],

            "status": "Initializing graph.",

            "final_answer": ""
        }

        # ====================================================
        # GATE 3: CHECKPOINT CONFIGURATION
        # ====================================================

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # ====================================================
        # GATE 4: EXECUTE LANGGRAPH
        # ====================================================

        logfire.info(
            "Executing LangGraph workflow.",
            thread_id=thread_id
        )

        final_output = rag_agent.invoke(
            initial_state,
            config=config
        )

        # ====================================================
        # GATE 5: EXTRACT FINAL OUTPUT
        # ====================================================

        answer = (
            final_output.get("final_answer")
            or "KubeAssist could not generate a response."
        )

        execution_steps = final_output.get("plan") or []

        retrieved_context = final_output.get("documents") or []

        workflow_status = final_output.get(
            "status",
            "Completed"
        )

        logfire.info(
            "Query completed successfully.",
            thread_id=thread_id
        )

        # ====================================================
        # GATE 6: API RESPONSE
        # ====================================================

        return {

            "question": q,

            "answer": answer,

            "thought_process": execution_steps,

            "status": "completed",

            "workflow_status": workflow_status,

            "retrieved_context": retrieved_context,

            "sources": []
        }

    except HTTPException:
        raise

    except Exception:

        logfire.exception(
            "Backend execution failed.",
            thread_id=thread_id
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred while "
                "processing your request."
            )
        )
