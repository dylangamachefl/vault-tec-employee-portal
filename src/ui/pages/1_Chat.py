import sys
from pathlib import Path

# Ensure src is in python path
src_path = Path(__file__).parent.parent.parent.resolve()
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

import streamlit as st  # noqa: E402

from src.pipelines.retrieval_chain import QueryInput, VaultRetriever  # noqa: E402
from src.ui.components.chat_history import render_chat_history  # noqa: E402
from src.ui.components.knowledge_base import render_sidebar  # noqa: E402
from src.ui.styles.vault_theme import apply_theme, render_footer  # noqa: E402

# Must be the very first st call (even though Streamlit technically complains if called after another, we use a single page config approach or let multipages inherit)  # noqa: E501
# We will just rely on inheritance and basic setting if it allows.
try:
    st.set_page_config(page_title="Vault-Tec | Chat", page_icon="☢️", layout="wide")
except Exception:
    pass

apply_theme()

# Guard: redirect to login if not authenticated
if not st.session_state.get("authenticated", False):
    st.switch_page("app.py")


@st.cache_resource
def get_retriever():
    return VaultRetriever(collection_name="vault_documents")


retriever = get_retriever()

# Layout: Sidebar
render_sidebar()

# Main Area
st.markdown(
    """
<div class="vault-header" style="text-align: left; margin-bottom: 10px;">
    ▶ VAULT-TEC KNOWLEDGE ASSISTANT
</div>
""",
    unsafe_allow_html=True,
)

role = st.session_state.get("role", "general")
username = st.session_state.get("username", "unknown")

st.markdown(
    f"""
<div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">
    <div style="background-color: var(--surface-color); padding: 5px 15px; border: 1px solid var(--primary-accent); border-radius: 2px; font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: var(--primary-accent);">  # noqa: E501
        ACTIVE: {username.upper()} [{role.upper()}]
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Dismissible warning
if "demo_warning_dismissed" not in st.session_state:
    st.session_state["demo_warning_dismissed"] = False

if not st.session_state["demo_warning_dismissed"]:
    warning_col1, warning_col2 = st.columns([10, 1])
    with warning_col1:
        st.warning(
            "⚠️ **DEMO MODE** — No access restrictions active on chat queries. Phase 2 enforces role-based retrieval filtering."  # noqa: E501
        )
    with warning_col2:
        if st.button("✕", key="dismiss_warning"):
            st.session_state["demo_warning_dismissed"] = True
            st.rerun()

render_chat_history()

# Chat Input
query = st.chat_input("Ask the Vault-Tec Knowledge Assistant...")

if query:
    # 1. Append user message
    st.session_state["chat_history"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Show spinner & run query
    with st.spinner("▶ Querying Vault-Tec Knowledge Systems..."):
        try:
            top_k = st.session_state.get("top_k", 5)

            # PHASE 2 HOOK: replace None with st.session_state["role"]
            q_input = QueryInput(
                query=query,
                collection_name="vault_documents",
                top_k=top_k,
                access_level_filter=None,
            )

            result = retriever.query(q_input)

            # Append assistant response
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": result.answer, "sources": result.sources}
            )

            st.rerun()

        except Exception as e:
            st.error(
                f"⚠️ TERMINAL ERROR: Knowledge retrieval failed. Please contact your Floor Liaison Officer. ({str(e)})"  # noqa: E501
            )

st.markdown("<br><br>", unsafe_allow_html=True)
render_footer()
