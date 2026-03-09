# ==============================================================================
# streamlit.py
# ==============================================================================
# PURPOSE:
#   The complete Streamlit frontend for the Healthcare RAG system.
#   Provides two main experiences:
#
#   PATIENT VIEW  — A clean chat interface where users can:
#     - Register and log in
#     - Ask medical questions and receive RAG-powered answers
#     - View source citations (which clinical cases backed the answer)
#     - See their conversation history
#     - Track their token usage budget
#
#   ADMIN VIEW  — A dashboard where admins can:
#     - Monitor system health (DB, FAISS, cache)
#     - Review and approve/reject flagged AI responses
#     - Manage user accounts and token limits
#     - View and clear the query cache
#
# HOW TO RUN:
#   Ensure the FastAPI backend is running first:
#     uvicorn src.api.main:app --reload --port 8000
#
#   Then run Streamlit:
#     streamlit run streamlit.py
#
# ENVIRONMENT VARIABLES (set in .env or Streamlit Cloud secrets):
#   API_BASE_URL — The base URL of the FastAPI backend
#                  Default: http://localhost:8000/api/v1
# ==============================================================================

import os
import time
import uuid

import requests
import streamlit as st

# ==============================================================================
# CONFIG
# ==============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
REQUEST_TIMEOUT = 60  # seconds — generous for LLM generation

st.set_page_config(
    page_title="MediQuery RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# GLOBAL STYLES
# ==============================================================================
# Clean clinical aesthetic:
#   - Deep navy + bright teal accent  (professional, medical-grade)
#   - DM Serif Display for headings   (elegant, distinctive)
#   - DM Sans for body text           (clean, readable)
#   - Tight spacing, generous padding (breathes without wasting space)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── ROOT TOKENS ─────────────────────────────────────────── */
:root {
    --navy:      #0d1b2a;
    --navy-mid:  #1a2f45;
    --teal:      #00c2a8;
    --teal-dim:  #007d6e;
    --white:     #f8fafb;
    --muted:     #8fa3b1;
    --border:    rgba(0,194,168,.18);
    --card:      rgba(26,47,69,.55);
    --danger:    #e05c6a;
    --warn:      #f0a500;
    --success:   #3ecf8e;
    --font-head: 'DM Serif Display', serif;
    --font-body: 'DM Sans', sans-serif;
    --radius:    12px;
    --shadow:    0 4px 24px rgba(0,0,0,.35);
}

/* ── BASE ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--navy);
    color: var(--white);
}
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1100px; }

/* ── HEADINGS ─────────────────────────────────────────────── */
h1, h2, h3 { font-family: var(--font-head); color: var(--white); }

/* ── SIDEBAR ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--navy-mid) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* ── INPUTS ───────────────────────────────────────────────── */
input, textarea, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
    background: var(--navy-mid) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--white) !important;
    font-family: var(--font-body) !important;
}
input:focus, textarea:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 2px rgba(0,194,168,.2) !important;
}
label { color: var(--muted) !important; font-size: .83rem !important; letter-spacing: .04em; }

/* ── BUTTONS ──────────────────────────────────────────────── */
.stButton > button {
    background: var(--teal) !important;
    color: var(--navy) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: .55rem 1.4rem !important;
    transition: all .2s ease !important;
    letter-spacing: .02em;
}
.stButton > button:hover {
    background: #00dfc0 !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,194,168,.35) !important;
}
.stButton > button:active { transform: translateY(0); }

/* Secondary button */
.btn-secondary > button {
    background: transparent !important;
    color: var(--teal) !important;
    border: 1px solid var(--border) !important;
}
.btn-secondary > button:hover {
    background: rgba(0,194,168,.08) !important;
    border-color: var(--teal) !important;
}

/* ── TABS ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--navy-mid);
    border-radius: var(--radius);
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    color: var(--muted) !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    padding: .4rem 1.1rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--teal) !important;
    color: var(--navy) !important;
}

/* ── CARDS ────────────────────────────────────────────────── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(6px);
}
.card-sm {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: .9rem 1.1rem;
    margin-bottom: .6rem;
}

/* ── CHAT BUBBLES ─────────────────────────────────────────── */
.bubble-user {
    background: rgba(0,194,168,.12);
    border: 1px solid rgba(0,194,168,.3);
    border-radius: 18px 18px 4px 18px;
    padding: .85rem 1.2rem;
    margin: .5rem 0 .5rem 15%;
    color: var(--white);
    font-size: .95rem;
    line-height: 1.6;
}
.bubble-ai {
    background: var(--navy-mid);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: .85rem 1.2rem;
    margin: .5rem 15% .5rem 0;
    color: var(--white);
    font-size: .95rem;
    line-height: 1.6;
}
.bubble-label {
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .3rem;
}

/* ── METRIC TILES ─────────────────────────────────────────── */
.metric-tile {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.metric-tile .value {
    font-family: var(--font-head);
    font-size: 2rem;
    color: var(--teal);
    line-height: 1.1;
}
.metric-tile .label {
    font-size: .78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-top: .3rem;
}

/* ── STATUS BADGES ────────────────────────────────────────── */
.badge {
    display: inline-block;
    border-radius: 20px;
    padding: .18rem .65rem;
    font-size: .73rem;
    font-weight: 600;
    letter-spacing: .05em;
    text-transform: uppercase;
}
.badge-ok      { background: rgba(62,207,142,.15); color: var(--success); border: 1px solid rgba(62,207,142,.3); }
.badge-warn    { background: rgba(240,165,0,.15);  color: var(--warn);    border: 1px solid rgba(240,165,0,.3);  }
.badge-danger  { background: rgba(224,92,106,.15); color: var(--danger);  border: 1px solid rgba(224,92,106,.3); }
.badge-teal    { background: rgba(0,194,168,.15);  color: var(--teal);    border: 1px solid rgba(0,194,168,.3);  }

/* ── SOURCE CITATION CHIPS ────────────────────────────────── */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    background: rgba(0,194,168,.08);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .3rem .7rem;
    font-size: .78rem;
    color: var(--teal);
    margin: .2rem .2rem .2rem 0;
}

/* ── EXPANDERS ────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--navy-mid) !important;
    border-radius: 9px !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-size: .85rem !important;
}

/* ── PROGRESS / DIVIDERS ──────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.5rem 0; }
.stProgress > div > div { background: var(--teal) !important; }

/* ── LOGO AREA ────────────────────────────────────────────── */
.logo-area {
    display: flex;
    align-items: center;
    gap: .7rem;
    margin-bottom: 1.8rem;
}
.logo-icon {
    font-size: 2.1rem;
    line-height: 1;
}
.logo-text {
    font-family: var(--font-head);
    font-size: 1.5rem;
    color: var(--white);
    line-height: 1.1;
}
.logo-sub {
    font-size: .72rem;
    color: var(--muted);
    letter-spacing: .08em;
    text-transform: uppercase;
}

/* ── SELECT / RADIO ────────────────────────────────────────── */
[data-baseweb="select"] > div {
    background: var(--navy-mid) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--white) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# SESSION STATE HELPERS
# ==============================================================================


def _init_state():
    """Initialises all session state keys with safe defaults on first load."""
    defaults = {
        "token": None,  # JWT access token
        "user_id": None,
        "email": None,
        "role": None,
        "full_name": None,
        "session_id": str(uuid.uuid4()),  # RAG session for short-term memory
        "messages": [],  # Chat history for display: [{role, content, meta}]
        "auth_mode": "login",  # "login" | "register"
        "query_loading": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _is_logged_in() -> bool:
    return st.session_state.get("token") is not None


def _is_admin() -> bool:
    return st.session_state.get("role") == "admin"


def _logout():
    for k in ["token", "user_id", "email", "role", "full_name", "messages"]:
        st.session_state[k] = None
    st.session_state["messages"] = []
    st.session_state["session_id"] = str(uuid.uuid4())
    st.rerun()


# ==============================================================================
# API HELPERS
# ==============================================================================


def _headers() -> dict:
    """Returns auth headers for protected API calls."""
    return {"Authorization": f"Bearer {st.session_state['token']}"}


def _api(method: str, path: str, **kwargs) -> tuple[bool, dict | list | None, str]:
    """
    Makes an API call and returns (success, data, error_message).
    Centralises error handling so route handlers stay clean.
    """
    url = f"{API_BASE_URL}{path}"
    try:
        resp = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        if resp.status_code in (200, 201):
            return True, resp.json(), ""
        # Try to parse error detail
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return False, None, str(detail)
    except requests.exceptions.ConnectionError:
        return False, None, "Cannot reach the API server. Is the backend running?"
    except requests.exceptions.Timeout:
        return False, None, "Request timed out. The server may be busy."
    except Exception as e:
        return False, None, f"Unexpected error: {e}"


# ==============================================================================
# AUTH SCREENS
# ==============================================================================


def _render_auth_screen():
    """Renders the login / register screen when the user is not logged in."""

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(
            """
        <div style="text-align:center; margin-bottom:2.5rem; padding-top:2rem;">
            <div style="font-size:3.5rem; margin-bottom:.5rem;">🩺</div>
            <div style="font-family:'DM Serif Display',serif; font-size:2.2rem; color:#f8fafb;">
                MediQuery RAG
            </div>
            <div style="font-size:.8rem; color:#8fa3b1; letter-spacing:.1em;
                        text-transform:uppercase; margin-top:.4rem;">
                Clinical Knowledge Assistant
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            email = st.text_input(
                "Email address", key="login_email", placeholder="you@example.com"
            )
            password = st.text_input(
                "Password",
                type="password",
                key="login_password",
                placeholder="••••••••",
            )
            st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

            if st.button("Sign In", key="btn_login", use_container_width=True):
                if not email or not password:
                    st.error("Please enter your email and password.")
                else:
                    with st.spinner("Signing in..."):
                        ok, data, err = _api(
                            "POST",
                            "/auth/login",
                            json={"email": email, "password": password},
                        )
                    if ok:
                        st.session_state.update(
                            {
                                "token": data["access_token"],
                                "user_id": data["user_id"],
                                "email": data["email"],
                                "role": data["role"],
                                "full_name": data["full_name"],
                            }
                        )
                        st.rerun()
                    else:
                        st.error(f"Sign in failed: {err}")

        with tab_register:
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            reg_name = st.text_input(
                "Full name", key="reg_name", placeholder="Dr. Jane Smith"
            )
            reg_email = st.text_input(
                "Email address", key="reg_email", placeholder="you@example.com"
            )
            reg_pass = st.text_input(
                "Password",
                type="password",
                key="reg_pass",
                placeholder="Min. 8 characters",
            )

            if st.button(
                "Create Account", key="btn_register", use_container_width=True
            ):
                if not all([reg_name, reg_email, reg_pass]):
                    st.error("All fields are required.")
                elif len(reg_pass) < 8:
                    st.error("Password must be at least 8 characters.")
                else:
                    with st.spinner("Creating account..."):
                        ok, data, err = _api(
                            "POST",
                            "/auth/register",
                            json={
                                "email": reg_email,
                                "password": reg_pass,
                                "full_name": reg_name,
                                "role": "patient",
                            },
                        )
                    if ok:
                        st.success("Account created! Please sign in.")
                    else:
                        st.error(f"Registration failed: {err}")

        # System health indicator
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        ok, health, _ = _api("GET", "/health")
        if ok:
            status = health.get("status", "unknown")
            badge_cls = "badge-ok" if status == "healthy" else "badge-warn"
            st.markdown(
                f"<div style='text-align:center'>"
                f"<span class='badge {badge_cls}'>API {status}</span>"
                f"&nbsp;<span class='badge badge-teal'>{health.get('faiss_index', '—')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ==============================================================================
# SIDEBAR
# ==============================================================================


def _render_sidebar():
    """Renders the sidebar with branding, user info, and navigation."""
    with st.sidebar:
        # Logo
        st.markdown(
            """
        <div class="logo-area">
            <div class="logo-icon">🩺</div>
            <div>
                <div class="logo-text">MediQuery</div>
                <div class="logo-sub">RAG · Clinical AI</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # User card
        st.markdown(
            f"""
        <div class="card-sm" style="margin-bottom:1.2rem;">
            <div style="font-size:.8rem; color:var(--muted); margin-bottom:.2rem;">Signed in as</div>
            <div style="font-weight:600; font-size:.95rem;">{st.session_state["full_name"]}</div>
            <div style="font-size:.78rem; color:var(--muted);">{st.session_state["email"]}</div>
            <div style="margin-top:.5rem;">
                <span class="badge {"badge-warn" if _is_admin() else "badge-teal"}">
                    {"Admin" if _is_admin() else "Patient"}
                </span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Token budget meter
        ok, me, _ = _api("GET", "/me", headers=_headers())
        if ok:
            used = me.get("tokens_used", 0)
            limit = me.get("token_limit", 100_000)
            pct = min(used / max(limit, 1), 1.0)
            remaining = me.get("tokens_remaining", limit)

            badge_cls = (
                "badge-danger"
                if pct > 0.9
                else "badge-warn"
                if pct > 0.7
                else "badge-ok"
            )
            st.markdown(
                f"""
            <div style="margin-bottom:1.2rem;">
                <div style="display:flex; justify-content:space-between;
                            align-items:center; margin-bottom:.4rem;">
                    <span style="font-size:.78rem; color:var(--muted); text-transform:uppercase;
                                 letter-spacing:.05em;">Token Budget</span>
                    <span class="badge {badge_cls}">{pct * 100:.0f}%</span>
                </div>
            """,
                unsafe_allow_html=True,
            )
            st.progress(pct)
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between;
                            font-size:.75rem; color:var(--muted); margin-top:.3rem;">
                    <span>{used:,} used</span>
                    <span>{remaining:,} remaining</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.divider()

        # New conversation button
        if st.button("＋  New Conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["session_id"] = str(uuid.uuid4())
            st.rerun()

        st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)

        # Sign out
        st.markdown("<div class='btn-secondary'>", unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True):
            _logout()
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Disclaimer
        st.markdown(
            """
        <div style="font-size:.72rem; color:var(--muted); line-height:1.5;">
            ⚠️ This tool is for <strong>educational purposes only</strong>.
            Always consult a qualified healthcare professional for
            medical advice, diagnosis, or treatment.
        </div>
        """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PATIENT CHAT VIEW
# ==============================================================================


def _render_chat():
    """Renders the main patient chat interface."""

    st.markdown(
        """
    <h1 style="margin-bottom:.3rem;">Clinical Knowledge Assistant</h1>
    <p style="color:var(--muted); font-size:.9rem; margin-bottom:2rem;">
        Ask medical questions based on real clinical cases from the MultiCaRe dataset.
    </p>
    """,
        unsafe_allow_html=True,
    )

    chat_tab, history_tab = st.tabs(["💬  Chat", "📋  History"])

    # ── CHAT TAB ──────────────────────────────────────────────────────────────
    with chat_tab:
        # Render existing messages
        if not st.session_state["messages"]:
            st.markdown(
                """
            <div style="text-align:center; padding:3rem 1rem; color:var(--muted);">
                <div style="font-size:2.8rem; margin-bottom:1rem;">🔬</div>
                <div style="font-family:'DM Serif Display',serif; font-size:1.3rem;
                             color:var(--white); margin-bottom:.5rem;">
                    Ask a clinical question
                </div>
                <div style="font-size:.85rem; max-width:440px; margin:0 auto; line-height:1.6;">
                    Powered by 93,816 clinical cases and 130,791 medical image captions
                    from the MultiCaRe dataset.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Example questions
            example_qs = [
                "What are the early symptoms of type 2 diabetes?",
                "How is pneumonia diagnosed in clinical practice?",
                "What imaging findings suggest pulmonary embolism?",
                "What are the MRI findings in multiple sclerosis?",
            ]
            cols = st.columns(2)
            for i, q in enumerate(example_qs):
                with cols[i % 2]:
                    if st.button(q, key=f"eq_{i}", use_container_width=True):
                        st.session_state["_prefill"] = q
                        st.rerun()

        else:
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    st.markdown(
                        f"""
                    <div class="bubble-label" style="text-align:right;">You</div>
                    <div class="bubble-user">{msg["content"]}</div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                    <div class="bubble-label">MediQuery AI</div>
                    <div class="bubble-ai">{msg["content"]}</div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Source citations
                    meta = msg.get("meta", {})
                    chunks = meta.get("retrieved_chunks", [])
                    if chunks:
                        with st.expander(
                            f"📎  {len(chunks)} source{'s' if len(chunks) > 1 else ''} retrieved"
                        ):
                            for i, chunk in enumerate(chunks, 1):
                                source = chunk.get("source", "unknown")
                                score = chunk.get("similarity_score", 0)
                                icon = "🏥" if source == "clinical_case" else "🖼️"
                                label = (
                                    "Clinical Case"
                                    if source == "clinical_case"
                                    else "Image Caption"
                                )

                                if source == "clinical_case":
                                    age = chunk.get("patient_age")
                                    gender = chunk.get("patient_gender", "Unknown")
                                    detail = f"{age}yo {gender}" if age else gender
                                else:
                                    itype = chunk.get("image_type", "")
                                    labels = chunk.get("labels", [])
                                    detail = (
                                        f"{itype} — {', '.join(labels[:3])}"
                                        if labels
                                        else itype
                                    )

                                st.markdown(
                                    f"""
                                <div class="card-sm">
                                    <div style="display:flex; justify-content:space-between;
                                                align-items:center; margin-bottom:.5rem;">
                                        <span class="source-chip">{icon} {label}</span>
                                        <span style="font-size:.75rem; color:var(--muted);">
                                            {score:.2f} similarity
                                        </span>
                                    </div>
                                    <div style="font-size:.78rem; color:var(--muted);
                                                margin-bottom:.4rem;">{detail}</div>
                                    <div style="font-size:.82rem; line-height:1.55;
                                                color:var(--white);">
                                        {chunk.get("chunk_text", "")[:280]}...
                                    </div>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )

                    # Token/cache indicators
                    total_tok = meta.get("total_tokens", 0)
                    cached = meta.get("served_from_cache", False)
                    flagged = meta.get("was_flagged_for_review", False)

                    indicators = []
                    if cached:
                        indicators.append(
                            "<span class='badge badge-teal'>⚡ Cached</span>"
                        )
                    elif total_tok:
                        indicators.append(
                            f"<span class='badge badge-teal'>{total_tok:,} tokens</span>"
                        )
                    if flagged:
                        indicators.append(
                            "<span class='badge badge-warn'>⚠️ Under review</span>"
                        )

                    if indicators:
                        st.markdown(
                            "<div style='margin-top:.3rem;'>"
                            + " ".join(indicators)
                            + "</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

        # ── QUERY INPUT ───────────────────────────────────────────────────────
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        with st.container():
            col_q, col_opts = st.columns([4, 1])

            with col_q:
                prefill = st.session_state.pop("_prefill", "")
                # Dynamic key clears the textarea after each submit
                input_key = f"query_input_{st.session_state.get('_input_version', 0)}"
                query = st.text_area(
                    "Your question",
                    value=prefill,
                    placeholder="Ask a clinical question...",
                    height=80,
                    key=input_key,
                    label_visibility="collapsed",
                )

            with col_opts:
                source_filter = st.selectbox(
                    "Source",
                    options=["Both", "Clinical Cases", "Image Captions"],
                    key="source_filter",
                    label_visibility="visible",
                )
                top_k = st.slider(
                    "Top K", min_value=1, max_value=10, value=5, key="top_k"
                )

            filter_map = {
                "Both": None,
                "Clinical Cases": "clinical_case",
                "Image Captions": "image_caption",
            }

            if st.button("Ask  →", key="btn_ask", use_container_width=False):
                if not query.strip():
                    st.warning("Please enter a question.")
                else:
                    st.session_state["_input_version"] = (
                        st.session_state.get("_input_version", 0) + 1
                    )
                    _submit_query(query.strip(), filter_map[source_filter], top_k)

    # ── HISTORY TAB ───────────────────────────────────────────────────────────
    with history_tab:
        st.markdown("#### Your conversation history")

        ok, history, err = _api("GET", "/history?limit=30", headers=_headers())

        if not ok:
            st.error(f"Could not load history: {err}")
        elif not history:
            st.markdown(
                """
            <div style="color:var(--muted); text-align:center; padding:2rem;">
                No conversations yet. Start asking questions in the Chat tab!
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            for record in history:
                ts = record.get("created_at", "")[:16].replace("T", " ")
                flagged = record.get("was_flagged", False)
                with st.expander(
                    f"🗓 {ts}  —  {record.get('user_message', '')[:70]}..."
                ):
                    st.markdown(
                        f"""
                    <div class="bubble-label">You asked</div>
                    <div class="bubble-user">{record.get("user_message", "")}</div>
                    <div class="bubble-label" style="margin-top:.8rem;">AI Response</div>
                    <div class="bubble-ai">{record.get("ai_response", "")}</div>
                    """,
                        unsafe_allow_html=True,
                    )
                    if flagged:
                        st.markdown(
                            "<span class='badge badge-warn'>⚠️ Flagged for review</span>",
                            unsafe_allow_html=True,
                        )


def _submit_query(query: str, source_filter, top_k: int):
    """Submits the query to the API and appends messages to session state."""

    # Append user message immediately for snappy feel
    st.session_state["messages"].append({"role": "user", "content": query})

    with st.spinner("Searching clinical knowledge base..."):
        payload = {
            "query": query,
            "session_id": st.session_state["session_id"],
            "top_k": top_k,
        }
        if source_filter:
            payload["source_filter"] = source_filter

        ok, data, err = _api("POST", "/query", json=payload, headers=_headers())

    if ok:
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": data["response_text"],
                "meta": data,
            }
        )
    else:
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": f"⚠️ {err}",
                "meta": {},
            }
        )

    st.rerun()


# ==============================================================================
# ADMIN DASHBOARD
# ==============================================================================


def _render_admin():
    """Renders the admin dashboard with 4 tabs."""

    st.markdown(
        """
    <h1 style="margin-bottom:.3rem;">Admin Dashboard</h1>
    <p style="color:var(--muted); font-size:.9rem; margin-bottom:2rem;">
        System monitoring, human review queue, user management, and cache control.
    </p>
    """,
        unsafe_allow_html=True,
    )

    tab_health, tab_review, tab_users, tab_cache = st.tabs(
        [
            "🟢  System Health",
            "🔍  Review Queue",
            "👥  Users",
            "⚡  Cache",
        ]
    )

    # ── SYSTEM HEALTH ─────────────────────────────────────────────────────────
    with tab_health:
        ok, health, err = _api("GET", "/health")

        if not ok:
            st.error(f"Cannot reach API: {err}")
        else:
            status = health.get("status", "unknown")
            st.markdown(
                f"""
            <div class="card" style="margin-bottom:1.5rem; display:flex;
                         align-items:center; gap:1rem;">
                <div style="font-size:2.5rem;">
                    {"✅" if status == "healthy" else "⚠️"}
                </div>
                <div>
                    <div style="font-family:'DM Serif Display',serif; font-size:1.4rem;">
                        System {status.title()}
                    </div>
                    <div style="color:var(--muted); font-size:.85rem;">
                        Environment: <strong style="color:var(--white);">
                        {health.get("app_env", "—")}</strong>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            _metric_tile(
                c1,
                "Database",
                health.get("database", "—"),
                "badge-ok"
                if "connected" in health.get("database", "")
                else "badge-danger",
            )
            _metric_tile(
                c2, "FAISS Index", health.get("faiss_index", "—"), "badge-teal"
            )
            _metric_tile(
                c3, "Active Sessions", str(health.get("active_sessions", 0)), "badge-ok"
            )

    # ── REVIEW QUEUE ──────────────────────────────────────────────────────────
    with tab_review:
        st.markdown("#### Flagged responses awaiting review")

        status_choice = st.radio(
            "Filter by status",
            options=["pending", "approved", "rejected"],
            horizontal=True,
            key="review_status_filter",
        )

        ok, items, err = _api(
            "GET",
            f"/admin/review?status_filter={status_choice}&limit=50",
            headers=_headers(),
        )

        if not ok:
            st.error(f"Could not load review queue: {err}")
        elif not items:
            st.markdown(
                f"""
            <div style="color:var(--muted); text-align:center; padding:2.5rem;">
                No <strong>{status_choice}</strong> items in the queue.
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='color:var(--muted); font-size:.85rem; "
                f"margin-bottom:1rem;'>{len(items)} item(s) found</div>",
                unsafe_allow_html=True,
            )

            for item in items:
                risk = item.get("risk_score", 0)
                badge = (
                    "badge-danger"
                    if risk >= 0.7
                    else "badge-warn"
                    if risk >= 0.4
                    else "badge-ok"
                )
                ts = item.get("created_at", "")[:16].replace("T", " ")

                with st.expander(
                    f"#{item['id']}  ·  Risk {risk:.2f}  ·  {item.get('user_query', '')[:60]}..."
                ):
                    st.markdown(
                        f"""
                    <div style="display:flex; gap:.5rem; margin-bottom:1rem; flex-wrap:wrap;">
                        <span class="badge {badge}">Risk: {risk:.2f}</span>
                        <span class="badge badge-teal">{ts}</span>
                        <span class="badge badge-teal">Session: {item.get("session_id", "")[:8]}…</span>
                    </div>
                    <div class="bubble-label">User Query</div>
                    <div class="bubble-user">{item.get("user_query", "")}</div>
                    <div class="bubble-label" style="margin-top:.8rem;">AI Response</div>
                    <div class="bubble-ai">{item.get("ai_response", "")}</div>
                    <div style="font-size:.78rem; color:var(--muted); margin-top:.8rem;">
                        Flag reason: {item.get("flag_reason", "—")}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    if item.get("status") == "pending":
                        notes = st.text_input(
                            "Reviewer notes (optional)",
                            key=f"notes_{item['id']}",
                            placeholder="e.g. Response is factually correct and appropriately hedged.",
                        )
                        col_a, col_r, _ = st.columns([1, 1, 3])
                        with col_a:
                            if st.button("✅ Approve", key=f"approve_{item['id']}"):
                                _submit_review(item["id"], "approve", notes)
                        with col_r:
                            if st.button("❌ Reject", key=f"reject_{item['id']}"):
                                _submit_review(item["id"], "reject", notes)
                    else:
                        st.markdown(
                            f"""
                        <div style="margin-top:.8rem;">
                            <span class="badge {"badge-ok" if item["status"] == "approved" else "badge-danger"}">
                                {item["status"].title()}
                            </span>
                            {f"<span style='font-size:.8rem; color:var(--muted); margin-left:.5rem;'>{item.get('reviewer_notes', '')}</span>" if item.get("reviewer_notes") else ""}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

    # ── USERS ─────────────────────────────────────────────────────────────────
    with tab_users:
        st.markdown("#### Registered users")

        ok, users, err = _api("GET", "/admin/users?limit=200", headers=_headers())

        if not ok:
            st.error(f"Could not load users: {err}")
        else:
            # Summary row
            total_u = len(users)
            active_u = sum(1 for u in users if u.get("is_active"))
            admin_u = sum(1 for u in users if u.get("role") == "admin")

            c1, c2, c3 = st.columns(3)
            _counter_tile(c1, str(total_u), "Total Users")
            _counter_tile(c2, str(active_u), "Active")
            _counter_tile(c3, str(admin_u), "Admins")

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            for user in users:
                used = user.get("tokens_used", 0)
                limit = user.get("token_limit", 100_000)
                pct = min(used / max(limit, 1), 1.0)
                role = user.get("role", "patient")
                active = user.get("is_active", True)

                badge_role = "badge-warn" if role == "admin" else "badge-teal"
                badge_act = "badge-ok" if active else "badge-danger"

                with st.expander(
                    f"{user.get('full_name', '—')}  ·  {user.get('email', '—')}"
                ):
                    st.markdown(
                        f"""
                    <div style="display:flex; gap:.5rem; margin-bottom:1rem; flex-wrap:wrap;">
                        <span class="badge {badge_role}">{role.title()}</span>
                        <span class="badge {badge_act}">{"Active" if active else "Deactivated"}</span>
                        <span class="badge badge-teal">ID: {user.get("id", "")[:8]}…</span>
                    </div>
                    <div style="font-size:.82rem; color:var(--muted); margin-bottom:.8rem;">
                        Tokens: <strong style="color:var(--white);">{used:,}</strong>
                        / {limit:,} ({pct * 100:.1f}%)
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    st.progress(pct)

                    st.markdown(
                        "<div style='height:.5rem'></div>", unsafe_allow_html=True
                    )

                    col_r, col_l, _ = st.columns([1.2, 1.5, 2])
                    with col_r:
                        if st.button("Reset Tokens", key=f"reset_{user['id']}"):
                            ok2, _, err2 = _api(
                                "POST",
                                f"/admin/users/{user['id']}/reset-tokens",
                                json={"confirm": True},
                                headers=_headers(),
                            )
                            if ok2:
                                st.success("Tokens reset to 0.")
                                time.sleep(0.8)
                                st.rerun()
                            else:
                                st.error(err2)

                    with col_l:
                        new_limit = st.number_input(
                            "Set new token limit",
                            min_value=1000,
                            max_value=10_000_000,
                            value=limit,
                            step=10_000,
                            key=f"newlimit_{user['id']}",
                            label_visibility="collapsed",
                        )
                        if st.button("Update Limit", key=f"updlimit_{user['id']}"):
                            ok3, _, err3 = _api(
                                "POST",
                                f"/admin/users/{user['id']}/token-limit",
                                json={"new_limit": new_limit},
                                headers=_headers(),
                            )
                            if ok3:
                                st.success(f"Limit updated to {new_limit:,}.")
                                time.sleep(0.8)
                                st.rerun()
                            else:
                                st.error(err3)

    # ── CACHE ─────────────────────────────────────────────────────────────────
    with tab_cache:
        st.markdown("#### Query cache statistics")

        ok, stats, err = _api("GET", "/admin/cache/stats", headers=_headers())

        if not ok:
            st.error(f"Could not load cache stats: {err}")
        else:
            c1, c2 = st.columns(2)
            _counter_tile(c1, f"{stats.get('total_entries', 0):,}", "Cached Queries")
            _counter_tile(c2, f"{stats.get('total_hits', 0):,}", "Total Cache Hits")

            most_cached = stats.get("most_cached", [])
            if most_cached:
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                st.markdown("**Top cached queries**")
                for entry in most_cached:
                    st.markdown(
                        f"""
                    <div class="card-sm" style="display:flex; justify-content:space-between;
                                align-items:center;">
                        <div style="font-size:.85rem; flex:1; margin-right:1rem;">
                            {entry.get("query_preview", "")[:80]}
                        </div>
                        <span class="badge badge-teal">{entry.get("hit_count", 0)} hits</span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            """
        <div class="card" style="border-color:rgba(224,92,106,.25);">
            <div style="font-family:'DM Serif Display',serif; font-size:1.1rem;
                        margin-bottom:.5rem;">🗑  Clear All Cache</div>
            <div style="font-size:.85rem; color:var(--muted); margin-bottom:1rem;">
                Use this after rebuilding the FAISS index. Cached responses may be
                based on outdated retrieval results. This action is irreversible.
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Clear Entire Cache", key="btn_clear_cache"):
            ok, result, err = _api("DELETE", "/admin/cache", headers=_headers())
            if ok:
                st.success(f"Cleared {result.get('entries_cleared', 0)} cache entries.")
                time.sleep(0.8)
                st.rerun()
            else:
                st.error(f"Clear failed: {err}")

        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# REVIEW HELPER
# ==============================================================================


def _submit_review(item_id: int, action: str, notes: str):
    ok, _, err = _api(
        "POST",
        f"/admin/review/{item_id}",
        json={"action": action, "reviewer_notes": notes or None},
        headers=_headers(),
    )
    if ok:
        st.success(f"Response {action}d.")
        time.sleep(0.8)
        st.rerun()
    else:
        st.error(f"Review failed: {err}")


# ==============================================================================
# REUSABLE UI COMPONENTS
# ==============================================================================


def _metric_tile(col, label: str, value: str, badge_cls: str):
    with col:
        st.markdown(
            f"""
        <div class="card">
            <div style="font-size:.75rem; color:var(--muted); text-transform:uppercase;
                        letter-spacing:.06em; margin-bottom:.5rem;">{label}</div>
            <span class="badge {badge_cls}">{value}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )


def _counter_tile(col, value: str, label: str):
    with col:
        st.markdown(
            f"""
        <div class="metric-tile">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# MAIN APP ENTRYPOINT
# ==============================================================================


def main():
    _init_state()

    if not _is_logged_in():
        _render_auth_screen()
        return

    _render_sidebar()

    if _is_admin():
        # Admins choose their view
        view = st.radio(
            "View",
            options=["Patient Chat", "Admin Dashboard"],
            horizontal=True,
            key="admin_view_toggle",
            label_visibility="collapsed",
        )
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        if view == "Admin Dashboard":
            _render_admin()
        else:
            _render_chat()
    else:
        _render_chat()


if __name__ == "__main__":
    main()
