from typing import Annotated, TypedDict, NotRequired
import operator


class AgentState(TypedDict):
    # Conversation history: append messages from each node.
    messages: Annotated[list[dict], operator.add]

    # Planner output.
    current_query: NotRequired[str]

    # Retrieved document chunks.
    documents: NotRequired[list[str]]

    # Execution steps and observations.
    plan: NotRequired[list[str]]

    # Current workflow status.
    status: NotRequired[str]

    # Final generated response.
    final_answer: NotRequired[str]