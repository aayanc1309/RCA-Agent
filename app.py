"""
Streamlit UI for the Incident Root Cause Analysis Agent.

Run with:  streamlit run app.py

The page preserves its last result in st.session_state so widget interactions
do not cause a full re-run and loss of analysis results.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Incident RCA Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-mode design
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ─────────────────────────────────────── */
:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2128;
    --bg-hover:      #21262d;
    --accent-blue:   #58a6ff;
    --accent-green:  #3fb950;
    --accent-orange: #d29922;
    --accent-red:    #f85149;
    --accent-purple: #bc8cff;
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --border:        #30363d;
    --border-subtle: #21262d;
    --shadow:        0 8px 24px rgba(0,0,0,.4);
}

/* ── Base Reset ─────────────────────────────────────────── */
html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Header bar ─────────────────────────────────────────── */
.rca-header {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.rca-title {
    font-size: 1.75rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.rca-subtitle {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin: 0.2rem 0 0 0;
}
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: linear-gradient(135deg, #1c2128, #21262d);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--accent-blue);
    font-family: 'JetBrains Mono', monospace;
}

/* ── Cards ──────────────────────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    transition: border-color .2s, box-shadow .2s;
}
.metric-card:hover {
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 1px var(--accent-blue)22;
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--text-secondary);
    margin-bottom: .5rem;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

/* ── Category badges ────────────────────────────────────── */
.badge {
    display: inline-block;
    border-radius: 6px;
    padding: .2rem .6rem;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .03em;
    font-family: 'JetBrains Mono', monospace;
}
.badge-red    { background:#f851491a; color:#f85149; border:1px solid #f8514944; }
.badge-orange { background:#d299221a; color:#d29922; border:1px solid #d2992244; }
.badge-yellow { background:#e3b3411a; color:#e3b341; border:1px solid #e3b34144; }
.badge-blue   { background:#58a6ff1a; color:#58a6ff; border:1px solid #58a6ff44; }
.badge-purple { background:#bc8cff1a; color:#bc8cff; border:1px solid #bc8cff44; }
.badge-grey   { background:#8b949e1a; color:#8b949e; border:1px solid #8b949e44; }
.badge-green  { background:#3fb9501a; color:#3fb950; border:1px solid #3fb95044; }

/* ── Fix step cards ─────────────────────────────────────── */
.fix-step {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--accent-blue);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: .75rem;
    transition: border-left-color .2s;
}
.fix-step.immediate { border-left-color: var(--accent-red); }
.fix-step.short_term { border-left-color: var(--accent-orange); }
.fix-step.long_term { border-left-color: var(--accent-green); }
.fix-step-header {
    display: flex;
    align-items: center;
    gap: .75rem;
    margin-bottom: .5rem;
}
.fix-step-num {
    background: var(--accent-blue);
    color: #0d1117;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: .75rem;
    font-weight: 700;
    flex-shrink: 0;
}
.fix-step-action {
    font-weight: 600;
    font-size: .95rem;
    color: var(--text-primary);
}
.fix-step-rationale {
    font-size: .85rem;
    color: var(--text-secondary);
    padding-left: 2.25rem;
    line-height: 1.5;
}

/* ── Confidence bar ─────────────────────────────────────── */
.confidence-bar-bg {
    background: var(--bg-secondary);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin: .5rem 0;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width .6s ease;
}
.conf-low    { background: linear-gradient(90deg, #f85149, #ff7b72); }
.conf-medium { background: linear-gradient(90deg, #d29922, #e3b341); }
.conf-high   { background: linear-gradient(90deg, #3fb950, #56d364); }

/* ── Expander overrides ─────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Text area ──────────────────────────────────────────── */
.stTextArea textarea {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .82rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px #58a6ff22 !important;
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: .6rem 1.5rem !important;
    transition: opacity .2s, transform .1s !important;
}
.stButton > button:hover {
    opacity: .9 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* ── Divider ────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Error / info boxes ─────────────────────────────────── */
.stAlert {
    border-radius: 8px !important;
}

/* ── Spinner ────────────────────────────────────────────── */
.stSpinner > div > div {
    border-top-color: var(--accent-blue) !important;
}

/* ── Table ──────────────────────────────────────────────── */
.stDataFrame { border-radius: 8px !important; overflow: hidden !important; }

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Import agent (deferred so Streamlit page config runs first)
# ---------------------------------------------------------------------------

# Ensure the incident_rca_agent directory is on the path
import os

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CATEGORY_BADGE: dict[str, str] = {
    "code_bug":           "badge-red",
    "config_error":       "badge-orange",
    "dependency_failure": "badge-yellow",
    "infrastructure":     "badge-blue",
    "timeout":            "badge-purple",
    "auth_failure":       "badge-orange",
    "unknown":            "badge-grey",
}

_CATEGORY_EMOJI: dict[str, str] = {
    "code_bug":           "🐛",
    "config_error":       "⚙️",
    "dependency_failure": "📦",
    "infrastructure":     "🏗",
    "timeout":            "⏱",
    "auth_failure":       "🔐",
    "unknown":            "❓",
}

_PRIORITY_BADGE: dict[str, str] = {
    "immediate":  "badge-red",
    "short_term": "badge-orange",
    "long_term":  "badge-green",
}

_CONFIDENCE_COLOUR: dict[str, str] = {
    "low":    "conf-low",
    "medium": "conf-medium",
    "high":   "conf-high",
}


def _category_badge_html(category: str) -> str:
    """Render an HTML badge for a root cause category.

    Args:
        category: Root cause category string.

    Returns:
        HTML string for the badge.
    """
    cls = _CATEGORY_BADGE.get(category, "badge-grey")
    emoji = _CATEGORY_EMOJI.get(category, "❓")
    label = category.replace("_", " ").title()
    return f'<span class="badge {cls}">{emoji} {label}</span>'


def _priority_badge_html(priority: str) -> str:
    """Render an HTML badge for a fix step priority.

    Args:
        priority: Priority string.

    Returns:
        HTML string for the badge.
    """
    cls = _PRIORITY_BADGE.get(priority, "badge-grey")
    label = priority.replace("_", " ").title()
    return f'<span class="badge {cls}">{label}</span>'


def _confidence_bar_html(score: float, label: str) -> str:
    """Render a coloured confidence progress bar.

    Args:
        score: Confidence score 0.0–1.0.
        label: Confidence label string.

    Returns:
        HTML string for the bar.
    """
    pct = int(score * 100)
    colour_cls = _CONFIDENCE_COLOUR.get(label, "conf-low")
    return (
        f'<div class="confidence-bar-bg">'
        f'<div class="confidence-bar-fill {colour_cls}" style="width:{pct}%"></div>'
        f'</div>'
        f'<span style="font-size:.8rem;color:var(--text-secondary);">{pct}% — {label.upper()}</span>'
    )


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_title, col_spacer, col_badge = st.columns([6, 2, 2])
with col_title:
    st.markdown(
        """
<div style="padding:.5rem 0 1rem 0;">
  <p class="rca-title">🔍 Incident RCA Agent</p>
  <p class="rca-subtitle">
    AI-powered root cause analysis powered by Gemini · LangGraph · Streamlit
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        '<div style="padding-top:1rem;text-align:right;">'
        '<span class="model-badge">⚡ gemini-2.0-flash-lite</span>'
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

_PLACEHOLDER = """\
Traceback (most recent call last):
  File "/app/views.py", line 42, in get_profile
    user = User.objects.get(pk=user_id)
  File "/usr/local/lib/python3.11/site-packages/django/db/models/manager.py", line 87, in get
    return self.get_queryset().get(*args, **kwargs)
django.core.exceptions.ObjectDoesNotExist: User matching query does not exist.
"""

st.markdown("### 📋 Input")

uploaded_file = st.file_uploader(
    "Upload a log file (.txt or .log) — or paste below",
    type=["txt", "log"],
    label_visibility="visible",
)

input_text = ""
if uploaded_file is not None:
    try:
        input_text = uploaded_file.read().decode("utf-8", errors="replace")
        st.success(f"✅ Loaded **{uploaded_file.name}** ({len(input_text):,} chars)")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read file: {exc}")

paste_area = st.text_area(
    label="Paste logs, stack trace, or error output",
    value=input_text,
    height=220,
    placeholder=_PLACEHOLDER,
    key="input_text_area",
    help="Supports Python, Java, Node.js, Go, Ruby, and generic log formats.",
)

# Prefer pasted text over uploaded file if user typed something
if paste_area.strip():
    input_text = paste_area

btn_col, clear_col, spacer_col = st.columns([2, 1, 5])

with btn_col:
    analyse_clicked = st.button(
        "🔍 Analyse Incident",
        disabled=st.session_state.is_running,
        use_container_width=True,
        key="btn_analyse",
    )

with clear_col:
    clear_clicked = st.button(
        "🗑 Clear",
        use_container_width=True,
        key="btn_clear",
    )

if clear_clicked:
    st.session_state.result = None
    st.session_state.last_error = None
    st.rerun()

# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------

if analyse_clicked and input_text.strip():
    st.session_state.is_running = True
    st.session_state.last_error = None
    st.session_state.result = None

    with st.spinner("🔎 Analysing incident…"):
        try:
            from agent.graph import run_agent

            result = run_agent(input_text.strip())
            st.session_state.result = result
        except Exception:  # noqa: BLE001
            st.session_state.last_error = traceback.format_exc()
        finally:
            st.session_state.is_running = False

    st.rerun()

elif analyse_clicked and not input_text.strip():
    st.warning("⚠️ Please paste a stack trace or upload a log file first.")

# ---------------------------------------------------------------------------
# Display error if agent crashed
# ---------------------------------------------------------------------------

if st.session_state.last_error:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.error("❌ The agent encountered an error during analysis.")
    with st.expander("🔧 Full Error Traceback", expanded=True):
        st.code(st.session_state.last_error, language="python")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

result = st.session_state.result
if result is not None:
    rca = result.rca
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📊 Analysis Results")

    # ── Summary bar ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Root Cause Category</div>'
            f'<div class="metric-value">{_category_badge_html(rca.root_cause_category)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with m2:
        bar_html = _confidence_bar_html(rca.confidence_score, rca.confidence_label)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Confidence</div>'
            f"<div>{bar_html}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with m3:
        svc_count = len(rca.affected_services)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Services Affected</div>'
            f'<div class="metric-value">{svc_count}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with m4:
        proc_time = f"{result.processing_time_seconds:.2f}s"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Processing Time</div>'
            f'<div class="metric-value">{proc_time}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Expander 1: Root Cause ──────────────────────────────────────────────
    with st.expander("🔍 Root Cause", expanded=True):
        st.markdown(f"**{rca.root_cause}**")
        st.markdown("**Error Signature:**")
        st.code(rca.error_signature, language=None)

    # ── Expander 2: Fix Steps ───────────────────────────────────────────────
    with st.expander("🛠 Fix Steps", expanded=True):
        if rca.fix_steps:
            for step in rca.fix_steps:
                priority_css = step.priority  # "immediate" | "short_term" | "long_term"
                priority_html = _priority_badge_html(step.priority)
                st.markdown(
                    f'<div class="fix-step {priority_css}">'
                    f'<div class="fix-step-header">'
                    f'<div class="fix-step-num">{step.step_number}</div>'
                    f'<span class="fix-step-action">{step.action}</span>'
                    f"&nbsp;&nbsp;{priority_html}"
                    f"</div>"
                    f'<div class="fix-step-rationale">{step.rationale}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No fix steps were generated.")

    # ── Expander 3: Affected Services ───────────────────────────────────────
    with st.expander("🏗 Affected Services", expanded=True):
        if rca.affected_services:
            import pandas as pd

            df = pd.DataFrame(
                [{"Service": s.name, "Impact": s.impact} for s in rca.affected_services]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No affected services identified.")

    # ── Expander 4: Postmortem Draft ────────────────────────────────────────
    with st.expander("📋 Postmortem Draft", expanded=True):
        st.markdown(rca.postmortem_draft)
        st.download_button(
            label="⬇ Download Postmortem (.md)",
            data=rca.postmortem_draft.encode("utf-8"),
            file_name="postmortem.md",
            mime="text/markdown",
        )

    # ── Expander 5: Similar Known Issues ────────────────────────────────────
    with st.expander("🔗 Similar Known Issues", expanded=True):
        if rca.similar_known_issues:
            for issue in rca.similar_known_issues:
                st.markdown(f"- {issue}")
        else:
            st.info("No similar issues identified.")

    # ── Expander 6: Debug Info ───────────────────────────────────────────────
    with st.expander("🔧 Debug Info", expanded=False):
        st.markdown("**Model used:**")
        st.code(result.model_used, language=None)

        st.markdown("**Parsed Stack Trace:**")
        parsed = result.tool_outputs.get("parse_stack_trace", {})
        st.json(parsed)

        st.markdown("**Tool Outputs — Codebase Search:**")
        codebase_out = result.tool_outputs.get("search_codebase", {})
        st.json(codebase_out)

        st.markdown("**Tool Outputs — Git History:**")
        git_out = result.tool_outputs.get("check_git_history", {})
        st.json(git_out)

        st.markdown("**Processing Log:**")
        # Retrieve from the agent result via session for tool_calls_log
        st.caption("Tool call timing is logged to the terminal/stderr during the run.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;font-size:.75rem;color:var(--text-secondary,#8b949e);">'
    "Incident RCA Agent · Powered by Google Gemini 2.0 Flash Lite · Built with LangGraph & Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
