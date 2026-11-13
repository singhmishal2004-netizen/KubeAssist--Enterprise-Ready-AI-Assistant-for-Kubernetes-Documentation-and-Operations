import html
import os
import time
import uuid
from pathlib import Path
from typing import Any

import logfire
import requests
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# ============================================================
# LOGFIRE CONFIGURATION
# ============================================================

LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

try:
    logfire.configure(token=LOGFIRE_TOKEN, send_to_logfire="if-token-present")
    LOGFIRE_ON = bool(LOGFIRE_TOKEN)
except Exception as exc:
    print(f"Logfire initialization error: {exc}")
    LOGFIRE_ON = False


# ============================================================
# PAGE + APP CONFIGURATION
# ============================================================

st.set_page_config(page_title="KubeAssist", page_icon="☸️", layout="wide")

AI_AVATAR = "☸️"
USER_AVATAR = "🧑‍💻"

# Render private networking provides BACKEND_HOSTPORT.
# BACKEND_URL remains supported for local development or overrides.
BACKEND_HOSTPORT = os.getenv("BACKEND_HOSTPORT")
BACKEND_URL = (
    os.getenv("BACKEND_URL")
    or (f"http://{BACKEND_HOSTPORT}" if BACKEND_HOSTPORT else "http://localhost:8000")
).strip().rstrip("/")

REQUEST_TIMEOUT = 120

SUGGESTIONS = [
    ("What is a Pod?", "What is a Pod in Kubernetes?"),
    ("How Services work", "How does a Kubernetes Service work?"),
    ("Debug CrashLoopBackOff", "Explain CrashLoopBackOff and how to debug it."),
    ("Deployments & ReplicaSets", "How do Deployments manage ReplicaSets?"),
]

TICKER_WORDS = [
    "PODS", "SERVICES", "DEPLOYMENTS", "REPLICASETS", "INGRESS", "CONFIGMAPS",
    "SECRETS", "STATEFULSETS", "DAEMONSETS", "NAMESPACES", "VOLUMES", "RBAC",
    "KUBECTL", "NODES", "HELM",
]


# ============================================================
# THEME  (bold, chunky, one system: lime + K8s blue on near-black)
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #08080d;
        --surface: #12121a;
        --surface-2: #1a1a25;
        --line: #2a2a3a;
        --ink: #f6f6f1;
        --muted: #8e8ea3;
        --lime: #c6ff3d;
        --blue: #326ce5;
        --bad: #ff5d6c;
        --warn: #ffb020;
        --radius: 14px;
        --display: 'Archivo Black', 'Arial Black', sans-serif;
    }

    html, body, .stApp, button, input, textarea { font-family: 'Space Grotesk', sans-serif; }
    .stApp {
        background:
            radial-gradient(900px 420px at 85% -10%, rgba(50,108,229,.22), transparent 60%),
            radial-gradient(700px 360px at -5% 0%, rgba(198,255,61,.08), transparent 55%),
            var(--bg);
        color: var(--ink); font-size: 16px;
    }

    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 920px; padding: 1.2rem 1.2rem 7rem; }

    /* ---------- Typography ---------- */
    h1, h2, h3, h4 { font-family: var(--display); color: var(--ink); font-weight: 400; letter-spacing: -.01em; }
    h4 { font-size: 1.5rem; margin: 1.4rem 0 .2rem; text-transform: uppercase; }
    p, li, label { color: var(--ink); }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted); font-weight: 500; }
    hr { margin: 1rem 0; border-color: var(--line); }
    code { color: var(--lime); background: var(--surface-2); border-radius: 6px; padding: 1px 6px; }

    /* ---------- Brand + hero ---------- */
    .ka-brand { font-family: var(--display); font-size: 1.6rem; letter-spacing: -.01em; line-height: 1; text-transform: uppercase; }
    .ka-brand i { color: var(--lime); font-style: normal; }
    .ka-brand b { color: var(--blue); font-weight: 400; }
    .ka-kicker { display: inline-flex; align-items: center; gap: 8px; font-size: .74rem; font-weight: 700;
                 letter-spacing: .16em; text-transform: uppercase; color: var(--lime);
                 border: 2px solid var(--lime); border-radius: 999px; padding: 5px 12px; margin: 1.2rem 0 1rem; }
    .ka-kicker::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--lime);
                         animation: pulse 1.6s ease-in-out infinite; }
    .ka-hero { font-family: var(--display); text-transform: uppercase; line-height: .92; letter-spacing: -.03em;
               font-size: clamp(3rem, 9vw, 6rem); margin: 0 0 1rem; color: var(--ink); }
    .ka-hero em { font-style: normal; color: var(--lime); }
    .ka-hero u { text-decoration: none; color: var(--blue); }
    .ka-sub { color: var(--muted); font-size: 1.1rem; font-weight: 500; max-width: 600px; line-height: 1.5; margin-bottom: 1.4rem; }
    .ka-sub b { color: var(--ink); }

    /* ---------- Ticker ---------- */
    .ka-ticker { overflow: hidden; border-block: 2px solid var(--line); margin: 0 0 1.6rem; padding: 10px 0;
                 mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent); }
    .ka-track { display: inline-flex; gap: 28px; white-space: nowrap; animation: scroll 32s linear infinite; }
    .ka-track span { font-family: var(--display); font-size: 1.05rem; color: var(--muted); letter-spacing: .04em; }
    .ka-track span:nth-child(3n) { color: var(--lime); }
    .ka-track span:nth-child(3n+1) { color: var(--blue); }
    @keyframes scroll { to { transform: translateX(-50%); } }
    @keyframes pulse { 50% { opacity: .25; transform: scale(.7); } }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] { background: var(--surface); border-right: 2px solid var(--line); }
    [data-testid="stSidebar"] .block-container { padding: 1.4rem 1.1rem; }
    .ka-tag { color: var(--muted); font-size: .7rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin: 8px 0 0; }
    .ka-label { font-family: var(--display); font-size: .78rem; letter-spacing: .08em; text-transform: uppercase;
                color: var(--lime); margin: 20px 0 6px; }
    .ka-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; margin: 6px 0;
              background: var(--surface-2); border: 2px solid var(--line); border-radius: 10px;
              font-size: .86rem; font-weight: 700; }
    .ka-row small { color: var(--muted); font-weight: 600; font-size: .78rem; }
    .ka-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; }
    .ka-dot.on  { background: var(--lime); box-shadow: 0 0 10px var(--lime); }
    .ka-dot.off { background: var(--bad);  box-shadow: 0 0 10px var(--bad); }

    /* ---------- Buttons (hard-shadow, press effect) ---------- */
    div.stButton > button {
        background: var(--surface); color: var(--ink); border: 2px solid var(--line);
        border-radius: var(--radius); font-weight: 700; font-size: .95rem; min-height: 46px; padding: 8px 16px;
        box-shadow: 4px 4px 0 var(--line); transition: transform .1s, box-shadow .1s, border-color .1s, background .1s;
    }
    div.stButton > button:hover { border-color: var(--lime); color: var(--ink); background: var(--surface-2);
                                  transform: translate(-2px, -2px); box-shadow: 6px 6px 0 var(--lime); }
    div.stButton > button:active { transform: translate(3px, 3px); box-shadow: 1px 1px 0 var(--lime); }
    div.stButton > button[kind="primary"] { background: var(--lime); color: #08080d; border-color: var(--lime);
                                            box-shadow: 4px 4px 0 var(--blue); font-weight: 800; }
    div.stButton > button[kind="primary"]:hover { background: var(--lime); box-shadow: 6px 6px 0 var(--blue); }
    div.stButton > button[kind="primary"] p { color: #08080d; }

    /* Big starter cards */
    [data-testid="stMain"] div.stButton > button { min-height: 92px; justify-content: flex-start; text-align: left;
                                                    font-family: var(--display); font-weight: 400; font-size: 1.12rem;
                                                    padding: 16px 20px; }
    [data-testid="stMain"] div.stButton > button p::before { content: "→ "; color: var(--lime); }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] { background: var(--surface); border: 2px solid var(--line); border-radius: var(--radius);
                                    padding: .9rem 1.1rem; margin-bottom: .9rem; gap: .8rem;
                                    box-shadow: 5px 5px 0 rgba(50,108,229,.55); }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: var(--lime); border-color: var(--lime); box-shadow: 5px 5px 0 var(--blue); }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span { color: #08080d; font-weight: 700; font-size: 1.05rem; }
    [data-testid="stChatMessage"] p { line-height: 1.65; }

    [data-testid="stChatInput"] { background: var(--surface); border: 2px solid var(--line); border-radius: var(--radius); box-shadow: 4px 4px 0 var(--line); }
    [data-testid="stChatInput"]:focus-within { border-color: var(--lime); box-shadow: 4px 4px 0 var(--lime); }
    [data-testid="stChatInput"] textarea { color: var(--ink); font-weight: 600; font-size: 1rem; }

    /* ---------- Expanders / status ---------- */
    [data-testid="stExpander"], [data-testid="stStatus"] { background: var(--surface-2); border: 2px solid var(--line); border-radius: 12px; }
    [data-testid="stExpander"] summary, [data-testid="stStatus"] summary { padding: .55rem .9rem; font-weight: 700; font-size: .9rem; }
    [data-testid="stExpanderDetails"] { padding: .3rem .9rem .8rem; }
    [data-testid="stAlert"] { border-radius: 12px; border: 2px solid var(--line); }

    /* ---------- Agent step rows ---------- */
    .ka-step { --t: var(--muted); display: flex; align-items: center; gap: 12px; margin: 8px 0; padding: 10px 14px;
               background: var(--surface); border: 2px solid var(--line); border-left: 6px solid var(--t);
               border-radius: 12px; }
    .ka-step.lime { --t: var(--lime); }
    .ka-step.blue { --t: var(--blue); }
    .ka-step.warn { --t: var(--warn); }
    .ka-step.danger { --t: var(--bad); background: rgba(255,93,108,.09); border-color: rgba(255,93,108,.45); border-left-color: var(--bad); }
    .ka-ico { flex: 0 0 38px; height: 38px; display: grid; place-items: center; font-size: 1.15rem; border-radius: 10px;
              background: color-mix(in srgb, var(--t) 20%, transparent); border: 2px solid var(--t); }
    .ka-st { flex: 1; min-width: 0; }
    .ka-sl { font-family: var(--display); font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: var(--t); }
    .ka-sv { font-size: .95rem; font-weight: 600; color: var(--ink); line-height: 1.4; word-break: break-word; }
    .ka-no { font-family: var(--display); font-size: .8rem; color: var(--muted); opacity: .7; }

    /* ---------- Guardrail block ---------- */
    .ka-block-banner { display: flex; align-items: center; gap: 16px; padding: 14px 18px; margin: 2px 0 14px;
                       background: repeating-linear-gradient(135deg, rgba(255,93,108,.14) 0 12px, rgba(255,93,108,.07) 12px 24px);
                       border: 2px solid var(--bad); border-radius: 12px; box-shadow: 4px 4px 0 var(--bad); }
    .ka-shield { flex: 0 0 54px; height: 54px; display: grid; place-items: center; font-size: 1.7rem; border-radius: 12px;
                 background: var(--bad); animation: alarm 1.8s ease-in-out infinite; }
    .ka-bt { font-family: var(--display); font-size: 1.15rem; letter-spacing: .02em; text-transform: uppercase; color: var(--bad); line-height: 1.1; }
    .ka-bs { font-size: .9rem; font-weight: 600; color: var(--ink); margin-top: 4px; }
    @keyframes alarm { 50% { box-shadow: 0 0 0 7px rgba(255,93,108,.25); } }
    [data-testid="stChatMessage"]:has(.ka-block-banner) { border-color: var(--bad); box-shadow: 5px 5px 0 rgba(255,93,108,.55); }

    @media (max-width: 760px) { .block-container { padding: 1rem .8rem 7rem; } .ka-brand { font-size: 1.3rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"New conversation created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================
# HELPERS
# ============================================================

def text_value(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return value if isinstance(value, str) else str(value)


@st.cache_data(ttl=15, show_spinner=False)
def backend_is_up(url: str) -> bool:
    try:
        return requests.get(f"{url}/", timeout=3).ok
    except requests.exceptions.RequestException:
        return False


def reset_conversation() -> None:
    logfire.warning(f"Conversation reset: {st.session_state.session_id}")
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    # New LangGraph thread ID.
    st.session_state.session_id = str(uuid.uuid4())


def queue_prompt(prompt: str) -> None:
    st.session_state.pending_prompt = prompt


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    up = backend_is_up(BACKEND_URL)

    with st.sidebar:
        st.markdown(
            '<div class="ka-brand">Kube<i>Assist</i></div>'
            '<div class="ka-tag">Kubernetes AI Copilot</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="ka-label">Status</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ka-row"><div><span class="ka-dot {"on" if up else "off"}"></span>Backend</div>'
            f'<small>{"Online" if up else "Offline"}</small></div>'
            f'<div class="ka-row"><div><span class="ka-dot {"on" if LOGFIRE_ON else "off"}"></span>Logfire</div>'
            f'<small>{"Tracing" if LOGFIRE_ON else "Local only"}</small></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="ka-label">Session</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ka-row"><div>Thread</div><small><code>{st.session_state.session_id[:8]}</code></small></div>'
            f'<div class="ka-row"><div>Messages</div><small>{len(st.session_state.messages)}</small></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.button(
            "New conversation",
            type="primary",
            width="stretch",
            on_click=reset_conversation,
        )
        st.caption("Starts a fresh thread and clears memory.")


# ============================================================
# RENDERERS
# ============================================================

# (keyword in step label, icon, tone). First match wins.
STEP_STYLES = [
    ("guardrail", "🛡️", "danger"),
    ("blocked", "⛔", "danger"),
    ("intent", "🎯", "blue"),
    ("search", "🔎", "lime"),
    ("qdrant", "🗄️", "blue"),
    ("rerank", "📊", "lime"),
    ("retriev", "📚", "blue"),
    ("generat", "✍️", "lime"),
    ("respon", "✍️", "lime"),
    ("plan", "🧭", "blue"),
    ("start", "🚀", "lime"),
]


def step_style(label: str, value: str) -> tuple[str, str]:
    """Pick an icon and colour tone for an agent step."""
    key, text = label.lower(), value.lower()
    if "skipped" in text:
        return "⏭️", "warn"
    if key.startswith("intent") and "technical" not in text.replace("non-technical", ""):
        return "🚫", "warn"
    for needle, icon, tone in STEP_STYLES:
        if needle in key:
            return icon, tone
    return "⚙️", "muted"


def render_trace(steps: list) -> None:
    rows = []
    for index, step in enumerate(steps, start=1):
        label, sep, value = text_value(step).partition(":")
        if not sep:
            label, value = "Step", label
        label, value = label.strip(), value.strip()
        icon, tone = step_style(label, value)
        rows.append(
            f'<div class="ka-step {tone}"><div class="ka-ico">{icon}</div>'
            f'<div class="ka-st"><div class="ka-sl">{html.escape(label)}</div>'
            f'<div class="ka-sv">{html.escape(value)}</div></div>'
            f'<div class="ka-no">{index:02d}</div></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_blocked_banner() -> None:
    st.markdown(
        '<div class="ka-block-banner"><div class="ka-shield">🛡️</div><div>'
        '<div class="ka-bt">Blocked by guardrails</div>'
        '<div class="ka-bs">This request was stopped before retrieval. No documents were searched.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_sources(sources: list) -> None:
    """Supports string-based context and structured document records."""
    for index, source in enumerate(sources, start=1):
        if isinstance(source, dict):
            content = text_value(source.get("content"), "(No text content)")
            metadata = source.get("metadata") or {}
            title = text_value(
                source.get("source") or source.get("title") or source.get("file_name"),
                f"Chunk {index}",
            )
            score = source.get("score")
        else:
            content, metadata, score = text_value(source), {}, None
            title = f"Chunk {index}"

        with st.expander(f"{index}. {title}"):
            if score is not None:
                st.caption(f"Similarity score: {score}")
            st.markdown(content)
            if metadata:
                st.json(metadata, expanded=False)


def render_extras(details: dict, include_trace: bool = True) -> None:
    """Execution trace and sources under an answer."""
    steps = details.get("steps") or []
    sources = details.get("sources") or []

    if include_trace and steps:
        with st.expander(f"Agent steps · {len(steps)}"):
            render_trace(steps)
    if sources:
        with st.expander(f"Sources · {len(sources)}"):
            render_sources(sources)


def render_history() -> None:
    for message in st.session_state.messages:
        is_ai = message["role"] == "assistant"
        with st.chat_message(message["role"], avatar=AI_AVATAR if is_ai else USER_AVATAR):
            details = message.get("details") or {}
            if is_ai and details.get("status") == "blocked":
                render_blocked_banner()
            st.markdown(message["content"])
            if is_ai:
                render_extras(details)


def render_hero() -> None:
    words = TICKER_WORDS * 2  # duplicated so the loop is seamless
    ticker = "".join(f"<span>{w}</span>" for w in words)
    st.markdown(
        '<div class="ka-kicker">Kubernetes · Agentic RAG</div>'
        '<div class="ka-hero">Ask <em>anything</em><br>about <u>K8s</u>.</div>'
        '<div class="ka-sub"><b>Grounded answers</b> from the official docs, with the full '
        'agent trace and sources one click away.</div>'
        f'<div class="ka-ticker"><div class="ka-track">{ticker}</div></div>',
        unsafe_allow_html=True,
    )


def render_chat_header() -> None:
    st.markdown(
        '<div class="ka-brand" style="font-size:2rem;margin:.2rem 0 1.2rem">Kube<i>Assist</i><b>.</b></div>',
        unsafe_allow_html=True,
    )


def render_starters() -> None:
    st.markdown("#### Start here")
    cols = st.columns(2)
    for i, (label, prompt) in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            st.button(
                label,
                key=f"suggestion_{i}",
                width="stretch",
                on_click=queue_prompt,
                args=(prompt,),
            )


# ============================================================
# BACKEND CALL
# ============================================================

def stream_text(placeholder, text: str) -> None:
    """Simulated typing effect, capped at ~1.5s regardless of length."""
    words = text.split(" ")
    delay = min(0.02, 1.5 / max(len(words), 1))
    shown = ""
    for word in words:
        shown += word + " "
        placeholder.markdown(shown + "▌")
        time.sleep(delay)
    placeholder.markdown(text)


def process_prompt(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with logfire.span(
        "User Chat Interaction",
        user_query=prompt,
        session_id=st.session_state.session_id,
    ):
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AI_AVATAR):
            response = None

            with st.status("Thinking…", expanded=True) as status:
                try:
                    with logfire.span("Calling KubeAssist Backend"):
                        response = requests.post(
                            f"{BACKEND_URL}/query",
                            json={"q": prompt, "thread_id": st.session_state.session_id},
                            timeout=REQUEST_TIMEOUT,
                        )
                        response.raise_for_status()
                        data = response.json()

                    if data.get("status") == "error":
                        raise RuntimeError(data.get("answer", "Backend processing failed."))

                    steps = data.get("thought_process") or []
                    sources = data.get("retrieved_context") or data.get("sources") or []
                    blocked = data.get("status") == "blocked"

                    if steps:
                        render_trace(steps)

                    status.update(
                        label="Blocked by guardrails" if blocked else f"Done · {len(steps)} steps",
                        state="error" if blocked else "complete",
                        expanded=False,
                    )

                except requests.exceptions.ConnectionError as exc:
                    logfire.error(f"Backend connection failed: {exc}")
                    status.update(label="Backend unavailable", state="error", expanded=False)
                    st.error("Cannot connect to the KubeAssist backend. Check that the FastAPI server is running.")
                    return

                except requests.exceptions.Timeout:
                    logfire.error("Backend request timed out.")
                    status.update(label="Request timed out", state="error", expanded=False)
                    st.error("The request took too long. Please try again.")
                    return

                except requests.exceptions.HTTPError as exc:
                    logfire.error(f"Backend HTTP error: {exc}")
                    status.update(label="Backend error", state="error", expanded=False)
                    try:
                        detail = response.json().get("detail", "Unknown backend error.")
                    except (ValueError, AttributeError):
                        detail = response.text[:500] if response is not None else "Unknown backend error."
                    st.error(f"Backend error: {detail}")
                    return

                except Exception as exc:
                    logfire.exception(f"Unexpected frontend error: {exc}")
                    status.update(label="Processing failed", state="error", expanded=False)
                    st.error("An unexpected error occurred while processing your request.")
                    return

            if blocked:
                render_blocked_banner()

            answer = text_value(data.get("answer"), "No response received from KubeAssist.")
            stream_text(st.empty(), answer)

            details = {
                "status": data.get("status", "completed"),
                "steps": steps,
                "sources": sources,
            }
            # The trace is already shown in the status box above.
            render_extras(details, include_trace=False)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "details": details}
            )
            logfire.info(
                "Chat interaction completed successfully.",
                thread_id=st.session_state.session_id,
            )


# ============================================================
# MAIN
# ============================================================

render_sidebar()

pending = st.session_state.pending_prompt
st.session_state.pending_prompt = None
typed = st.chat_input("Ask about Kubernetes…")
prompt = (pending or typed or "").strip()

if not st.session_state.messages and not prompt:
    render_hero()
    render_starters()
else:
    render_chat_header()
    render_history()
    if prompt:
        process_prompt(prompt)