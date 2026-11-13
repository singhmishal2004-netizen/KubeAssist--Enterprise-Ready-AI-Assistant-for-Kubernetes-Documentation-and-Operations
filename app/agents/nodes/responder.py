
import logfire

from app.agents.state import AgentState
from app.gateways.client import (
    PRIMARY_MODEL,
    portkey_client,
    extract_cache_status,
)


def generate_node(state: AgentState):
    """
    Generates the final response using conversation history
    or retrieved technical documentation.

    Uses the native Portkey client to capture cache metadata.
    """

    messages = state.get("messages", [])

    if not messages:
        raise ValueError("Cannot generate response without messages.")

    query = state.get("current_query", "")
    documents = state.get("documents", [])
    plan = state.get("plan", [])

    # Build conversation history
    history_parts = []

    for msg in messages[:-1]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history_parts.append(
            f"{role}: {msg.get('content', '')}"
        )

    history_str = "\n".join(history_parts)

    user_msg = messages[-1].get("content", "")

    # --------------------------------------------------
    # CONVERSATIONAL RESPONSE
    # --------------------------------------------------

    if query == "CONVERSATIONAL":

        logfire.info(
            "Generating conversational response using memory."
        )

        prompt = f"""
You are KubeAssist, a friendly and helpful assistant.

Answer the user's latest message using the conversation
history when relevant.

Do not invent personal information that is not present
in the conversation.

CONVERSATION HISTORY:
{history_str}

LATEST USER MESSAGE:
{user_msg}
"""

    # --------------------------------------------------
    # TECHNICAL RAG RESPONSE
    # --------------------------------------------------

    else:

        logfire.info("Generating technical RAG response.")

        max_context_chars = 25000
        context_parts = []
        current_context_length = 0

        for doc in documents:

            if not isinstance(doc, str):
                continue

            if current_context_length + len(doc) > max_context_chars:
                logfire.warning(
                    "Technical context truncated."
                )
                break

            context_parts.append(doc)
            current_context_length += len(doc)

        full_context = "\n\n".join(context_parts)

        prompt = f"""
You are KubeAssist, a Kubernetes technical assistant.

Answer the user's question using the supplied technical
documentation and relevant conversation history.

RULES:
1. Prioritize the supplied documentation.
2. Do not invent technical facts.
3. If the documentation does not contain enough
   information, clearly state that.
4. Use conversation history to understand follow-up
   questions.
5. Provide clear, structured technical explanations.

TECHNICAL CONTEXT:
{full_context if full_context else "No relevant documents were retrieved."}

CONVERSATION HISTORY:
{history_str}

USER QUESTION:
{user_msg}
"""

    # --------------------------------------------------
    # LLM GENERATION
    # --------------------------------------------------

    with logfire.span("LLM Synthesis"):

        try:

            response = portkey_client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.1,
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("LLM returned an empty response.")

            cache_status = extract_cache_status(response)

            is_cache_hit = (
                str(cache_status).strip().upper() == "HIT"
            )

            if is_cache_hit:

                logfire.info("Portkey cache hit.")

                plan_update = plan + ["Cache: Hit"]
                status = "Cache hit — response served by gateway."

            else:

                logfire.info("Response generated through LLM.")

                plan_update = plan
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [
                    {
                        "role": "assistant",
                        "content": content,
                    }
                ],
            }

        except Exception:

            logfire.exception("LLM generation failed.")
            raise
