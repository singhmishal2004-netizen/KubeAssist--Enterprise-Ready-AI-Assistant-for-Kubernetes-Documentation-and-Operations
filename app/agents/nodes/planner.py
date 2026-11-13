
import re

import logfire

from app.agents.state import AgentState
from app.gateways.client import get_langchain_llm


# Portkey-backed LLM
llm = get_langchain_llm(feature="planner")


def normalize_question(text: str) -> str:
    """
    Normalize questions so that differences in capitalization,
    punctuation, and whitespace do not prevent exact matching.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def has_previous_answer(messages: list, latest_question: str) -> bool:
    """
    Check whether the same question was asked previously
    and received an assistant response.

    The assistant response must occur before the next user
    message, so an unrelated answer is not mistaken for one.
    """
    normalized_latest = normalize_question(latest_question)

    if not normalized_latest:
        return False

    # Exclude the latest user message.
    previous_messages = messages[:-1]

    for index, msg in enumerate(previous_messages):
        if not isinstance(msg, dict):
            continue

        role = str(msg.get("role", "")).lower()
        content = msg.get("content", "")

        if role != "user" or not isinstance(content, str):
            continue

        if normalize_question(content) != normalized_latest:
            continue

        # Check whether an assistant answered this question
        # before another user message appeared.
        for next_msg in previous_messages[index + 1:]:
            if not isinstance(next_msg, dict):
                continue

            next_role = str(next_msg.get("role", "")).lower()

            if next_role == "user":
                break

            if next_role == "assistant":
                return True

    return False


def planner_node(state: AgentState):
    """
    Analyze conversation history and decide whether
    document retrieval is required.
    """

    messages = state.get("messages", [])

    if not messages:
        return {
            "current_query": "",
            "status": "No user message received.",
            "plan": ["Intent: Invalid/Empty"],
        }

    latest_message = messages[-1]

    if not isinstance(latest_message, dict):
        raise ValueError("Latest message must be a dictionary.")

    user_message = latest_message.get("content", "")

    if not isinstance(user_message, str) or not user_message.strip():
        return {
            "current_query": "",
            "status": "Empty user message.",
            "plan": ["Intent: Invalid/Empty"],
        }

    user_message = user_message.strip()

    # --------------------------------------------------
    # STEP 1: Deterministic repeated-question detection
    # --------------------------------------------------

    if has_previous_answer(messages, user_message):

        logfire.info(
            "Exact repeated question detected. "
            "Using conversation history instead of retrieval."
        )

        return {
            "current_query": "CONVERSATIONAL",
            "status": "Repeated question answered using conversation history.",
            "plan": [
                "Intent: Repeated Question",
                "Memory: Previous assistant response available",
                "Retrieval: Skipped",
            ],
        }

    # --------------------------------------------------
    # STEP 2: Build conversation history
    # --------------------------------------------------

    history = []

    for msg in messages[:-1]:

        if not isinstance(msg, dict):
            continue

        role = (
            "User"
            if msg.get("role") == "user"
            else "Assistant"
        )

        content = msg.get("content", "")

        history.append(f"{role}: {content}")

    conversation_history = "\n".join(history)

    # --------------------------------------------------
    # STEP 3: Existing LLM-based planning
    # --------------------------------------------------

    prompt = f"""
You are the planning agent for KubeAssist,
a Kubernetes documentation and operations assistant.

Analyze the conversation history and latest user message.

CONVERSATION HISTORY:
{conversation_history}

LATEST MESSAGE:
{user_message}

TASK:

1. If the latest message is a greeting or can be answered
   using only the conversation history, output CONVERSATIONAL.

2. If the latest message requires Kubernetes documentation,
   troubleshooting knowledge, or technical information,
   generate a clear and refined search query.

3. Consider previous conversation context when interpreting
   follow-up questions.

4. Do not answer the user's question.

OUTPUT RULES:
- Output exactly CONVERSATIONAL for conversational requests.
- Otherwise, output only the refined search query.
"""

    with logfire.span("Planner Decision"):

        response = llm.invoke(prompt)

        decision = str(response.content).strip()
        decision = decision.strip('"').strip("'").strip()

        logfire.info(f"Intent identified: {decision}")

    # --------------------------------------------------
    # STEP 4: Route based on LLM decision
    # --------------------------------------------------

    if decision.upper() == "CONVERSATIONAL":

        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally.",
            "plan": [
                "Intent: Conversational/Memory",
                "Retrieval: Skipped",
            ],
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed: {decision}",
        "plan": [
            "Intent: Technical",
            f"Search Term: {decision}",
        ],
    }
