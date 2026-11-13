
import os
import time
import uuid

import logfire
import requests
import streamlit as st

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

# Load .env from the project root.
env_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".env")
)

load_dotenv(dotenv_path=env_path)


# ============================================================
# LOGFIRE CONFIGURATION
# ============================================================

try:
    token = os.getenv("LOGFIRE_TOKEN")

    logfire.configure(
        token=token,
        send_to_logfire="if-token-present"
    )

    LOGFIRE_STATUS = (
        "Connected & Tracing"
        if token
        else "Local / No Cloud Token"
    )

except Exception as e:

    print(f"Logfire initialization error: {e}")
    LOGFIRE_STATUS = "Standby"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KubeAssist",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

AI_AVATAR = "🤖"
USER_AVATAR = "👤"

# Render private networking provides the backend host and port.
# BACKEND_URL remains supported for local development or overrides.
BACKEND_HOSTPORT = os.getenv("BACKEND_HOSTPORT")

BACKEND_URL = (
    os.getenv("BACKEND_URL")
    or (
        f"http://{BACKEND_HOSTPORT}"
        if BACKEND_HOSTPORT
        else "http://localhost:8000"
    )
).strip().rstrip("/")

REQUEST_TIMEOUT = 120


# ============================================================
# SESSION MANAGEMENT
# ============================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4())

    logfire.info(
        f"New conversation created: "
        f"{st.session_state.session_id}"
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 KubeAssist")

    st.markdown("---")

    st.success(f"Logfire: {LOGFIRE_STATUS}")

    st.info(
        f"Memory ID: {st.session_state.session_id[:8]}"
    )

    st.markdown("---")

    if st.button(
        "🗑️ Clear History & Memory",
        width="stretch",
        type="primary"
    ):

        logfire.warning(
            f"Conversation reset: "
            f"{st.session_state.session_id}"
        )

        st.session_state.messages = []

        # Generate a new LangGraph thread ID.
        st.session_state.session_id = str(uuid.uuid4())

        st.rerun()


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================

st.title("🤖 KubeAssist")

st.caption(
    "Agentic RAG Assistant for Kubernetes Documentation "
    "and Operations."
)


# ============================================================
# DISPLAY CONVERSATION HISTORY
# ============================================================

for message in st.session_state.messages:

    avatar = (
        AI_AVATAR
        if message["role"] == "assistant"
        else USER_AVATAR
    )

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(message["content"])


# ============================================================
# DISPLAY RETRIEVED CONTEXT
# ============================================================

def display_retrieved_context(sources: list):

    """
    Displays retrieved document chunks returned by FastAPI.

    Supports both string-based context and structured
    document records containing content and metadata.
    """

    if not sources:
        return

    with st.expander("📄 View Retrieved Context"):

        for index, source in enumerate(sources):

            if isinstance(source, dict):

                content = source.get("content", "")

                metadata = source.get("metadata", {})

                title = source.get(
                    "source",
                    f"Chunk {index + 1}"
                )

                score = source.get("score")

            else:

                content = str(source)

                title = f"Chunk {index + 1}"

                metadata = {}

                score = None

            with st.expander(
                f"{index + 1}. {title}"
            ):

                if score is not None:
                    st.caption(
                        f"Qdrant similarity score: {score}"
                    )

                st.markdown(content)

                if metadata:
                    st.caption("Document metadata")

                    st.json(metadata)


# ============================================================
# CHAT INPUT
# ============================================================

if prompt := st.chat_input(
    "Ask about Kubernetes documentation..."
):

    with logfire.span(
        "User Chat Interaction",
        user_query=prompt,
        session_id=st.session_state.session_id
    ):

        # ----------------------------------------------------
        # STORE USER MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message(
            "user",
            avatar=USER_AVATAR
        ):

            st.markdown(prompt)

        # ----------------------------------------------------
        # ASSISTANT RESPONSE
        # ----------------------------------------------------

        with st.chat_message(
            "assistant",
            avatar=AI_AVATAR
        ):

            with st.status(
                "🔍 KubeAssist is thinking...",
                expanded=True
            ) as status:

                try:

                    # ----------------------------------------
                    # CALL FASTAPI BACKEND
                    # ----------------------------------------

                    url = f"{BACKEND_URL}/query"

                    payload = {
                        "q": prompt,
                        "thread_id": st.session_state.session_id
                    }

                    with logfire.span(
                        "Calling KubeAssist Backend"
                    ):

                        response = requests.post(
                            url,
                            json=payload,
                            timeout=REQUEST_TIMEOUT
                        )

                        response.raise_for_status()

                        data = response.json()

                    # ----------------------------------------
                    # HANDLE BACKEND ERROR RESPONSE
                    # ----------------------------------------

                    if data.get("status") == "error":

                        raise RuntimeError(
                            data.get(
                                "answer",
                                "Backend processing failed."
                            )
                        )

                    # ----------------------------------------
                    # DISPLAY EXECUTION STEPS
                    # ----------------------------------------

                    steps = data.get(
                        "thought_process",
                        []
                    )

                    if steps:

                        st.markdown("**Agent Execution Steps**")

                        for step in steps:
                            st.write(f"⚙️ {step}")

                    # ----------------------------------------
                    # DISPLAY RETRIEVED CONTEXT
                    # ----------------------------------------

                    # Current FastAPI implementation returns
                    # retrieved_context. The fallback supports
                    # older responses using the sources key.

                    sources = data.get(
                        "retrieved_context",
                        data.get("sources", [])
                    )

                    display_retrieved_context(sources)

                    # ----------------------------------------
                    # UPDATE STATUS
                    # ----------------------------------------

                    status.update(
                        label="✅ Answer Generated",
                        state="complete",
                        expanded=False
                    )

                # --------------------------------------------
                # CONNECTION ERROR
                # --------------------------------------------

                except requests.exceptions.ConnectionError as e:

                    logfire.error(
                        f"Backend connection failed: {e}"
                    )

                    status.update(
                        label="❌ Backend Unavailable",
                        state="error"
                    )

                    st.error(
                        "Cannot connect to KubeAssist backend. "
                        "Check whether the FastAPI server is running."
                    )

                    st.stop()

                # --------------------------------------------
                # TIMEOUT ERROR
                # --------------------------------------------

                except requests.exceptions.Timeout:

                    logfire.error(
                        "Backend request timed out."
                    )

                    status.update(
                        label="⏱️ Request Timed Out",
                        state="error"
                    )

                    st.error(
                        "The request took too long. "
                        "Please try again."
                    )

                    st.stop()

                # --------------------------------------------
                # HTTP ERROR
                # --------------------------------------------

                except requests.exceptions.HTTPError as e:

                    logfire.error(
                        f"Backend HTTP error: {e}"
                    )

                    status.update(
                        label="❌ Backend Error",
                        state="error"
                    )

                    try:

                        error_detail = response.json().get(
                            "detail",
                            "Unknown backend error."
                        )

                    except ValueError:

                        error_detail = response.text[:500]

                    st.error(
                        f"Backend error: {error_detail}"
                    )

                    st.stop()

                # --------------------------------------------
                # OTHER ERRORS
                # --------------------------------------------

                except Exception as e:

                    logfire.exception(
                        f"Unexpected frontend error: {e}"
                    )

                    status.update(
                        label="❌ Processing Failed",
                        state="error"
                    )

                    st.error(
                        "An unexpected error occurred "
                        "while processing your request."
                    )

                    st.stop()

            # =================================================
            # DISPLAY FINAL ANSWER
            # =================================================

            full_answer = data.get(
                "answer",
                "No response received from KubeAssist."
            )

            answer_placeholder = st.empty()

            # Simulated streaming effect.
            curr_text = ""

            for char in full_answer:

                curr_text += char

                answer_placeholder.markdown(
                    curr_text + "▌"
                )

                time.sleep(0.005)

            answer_placeholder.markdown(full_answer)

            # ------------------------------------------------
            # STORE ASSISTANT RESPONSE
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_answer
                }
            )

            logfire.info(
                "Chat interaction completed successfully."
            )
