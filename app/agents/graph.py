
# ============================================================
# KubeAssist - LangGraph Agent Workflow
# ============================================================
#
# Workflow:
#
# START
#   |
#   v
# Planner
#   |
#   v
# Conditional Router
#   |
#   +----------------------+
#   |                      |
#   v                      v
# Retrieve               Respond
#   |                      |
#   v                      |
# Respond                  |
#   |                      |
#   +----------+-----------+
#              |
#              v
#             END
#
# Guardrails:
#   Handled before invoking this graph.
#   Only approved queries should reach LangGraph.
#
# ============================================================

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.state import AgentState

from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.retriever import retrieve_node


# ============================================================
# 1. Conditional Routing After Planner
# ============================================================

def route_after_planning(
    state: AgentState,
) -> Literal["retrieve", "respond"]:
    """
    Decide whether the agent should retrieve documents
    or directly generate a conversational response.

    Routing rules:

    1. CONVERSATIONAL -> Respond directly.
    2. Empty query -> Respond directly.
    3. Technical query -> Retrieve relevant documents.

    Args:
        state: Current LangGraph AgentState.

    Returns:
        "retrieve" or "respond".
    """

    query = state.get("current_query") or ""

    # Validate query type.
    if not isinstance(query, str):
        return "respond"

    query = query.strip()

    # Direct conversational response.
    if query.upper() == "CONVERSATIONAL":
        return "respond"

    # Avoid retrieval for empty queries.
    if not query:
        return "respond"

    # Continue with RAG retrieval.
    return "retrieve"


# ============================================================
# 2. Initialize StateGraph
# ============================================================

workflow = StateGraph(AgentState)


# ============================================================
# 3. Register Agent Nodes
# ============================================================

workflow.add_node(
    "planner",
    planner_node,
)

workflow.add_node(
    "retrieve",
    retrieve_node,
)

workflow.add_node(
    "respond",
    generate_node,
)


# ============================================================
# 4. Define Workflow Entry Point
# ============================================================

workflow.add_edge(
    START,
    "planner",
)


# ============================================================
# 5. Define Conditional Routing
# ============================================================

workflow.add_conditional_edges(
    "planner",
    route_after_planning,
    {
        "retrieve": "retrieve",
        "respond": "respond",
    },
)


# ============================================================
# 6. Define Node Transitions
# ============================================================

# Retrieved context is passed to the responder.
workflow.add_edge(
    "retrieve",
    "respond",
)

# Finish after generating the response.
workflow.add_edge(
    "respond",
    END,
)


# ============================================================
# 7. Compile LangGraph
# ============================================================

# MemorySaver provides in-memory checkpointing.
# This is suitable for development and testing.
# It is not durable across application restarts.

checkpointer = MemorySaver()

rag_agent = workflow.compile(
    checkpointer=checkpointer,
)


# ============================================================
# 8. Exported Graph
# ============================================================

__all__ = [
    "rag_agent",
    "workflow",
    "route_after_planning",
]
