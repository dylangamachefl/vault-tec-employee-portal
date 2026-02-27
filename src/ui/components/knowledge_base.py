from pathlib import Path

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from src.config import settings


@st.cache_data(ttl=300)
def get_accessible_documents(role: str) -> list[dict]:
    """
    Query Qdrant vault_documents collection for distinct source_document values
    where access_level matches the current role, applying an additive hierarchy.
    """
    try:
        # Since data is in docker, use the REST API config, matching VaultRetriever
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return []

    hierarchy = {
        "general": ["general"],
        "hr": ["hr", "general"],
        "marketing": ["marketing", "general"],
        "admin": [
            "admin",
            "general",
        ],  # Wait, wait, admin should theoretically see all, but Task spec specifically says "admin: access_level IN ['admin', 'general']".  # noqa: E501
    }

    allowed_levels = hierarchy.get(role, ["general"])

    # We query Qdrant to get some payload data to deduplicate docs
    # Qdrant client scroll() gets all points with pagination.
    # To avoid retrieving huge DBs, we'll try pulling a large batch.

    q_filter = Filter(
        must=[
            FieldCondition(
                key="access_level",
                match=MatchAny(any=allowed_levels),
            )
        ]
    )

    try:
        # We only need payloads, not vectors.
        # It's an internal utility, 10,000 limit is fine for a demo corpus.
        scroll_res, _ = client.scroll(
            collection_name="vault_documents",
            scroll_filter=q_filter,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        print(f"Qdrant scroll error: {e}")
        return []

    unique_docs = {}
    for point in scroll_res:
        payload = point.payload or {}
        doc_name = payload.get("source_document")
        if doc_name and doc_name not in unique_docs:
            unique_docs[doc_name] = {
                "source_document": doc_name,
                "access_level": payload.get("access_level", "Unknown"),
                "department": payload.get("department", "Unknown"),
                "doc_status": payload.get("doc_status", "ACTIVE"),
            }

    return list(unique_docs.values())


def render_sidebar():
    with st.sidebar:
        role = st.session_state.get("role", "general")
        username = st.session_state.get("username", "unknown")

        st.markdown(
            "<div style=\"font-family: 'Share Tech Mono', monospace; color: var(--primary-accent); font-size: 1.2rem; font-weight: bold; margin-bottom: 5px;\">▶ KNOWLEDGE BASE</div>",  # noqa: E501
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style=\"font-family: 'Share Tech Mono', monospace; color: var(--text-primary); font-size: 0.9rem; margin-bottom: 5px;\">CLEARANCE: {role.upper()}</div>",  # noqa: E501
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style=\"font-family: 'Share Tech Mono', monospace; color: var(--text-muted); font-size: 0.8rem; margin-bottom: 20px;\">LOGGED IN AS: {username}</div>",  # noqa: E501
            unsafe_allow_html=True,
        )

        if st.button("LOGOUT", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.switch_page("app.py")

        st.markdown("---")

        docs = get_accessible_documents(role)

        # Group by department
        docs_by_dept = {}
        for doc in docs:
            dept = doc.get("department", "Unknown")
            if dept not in docs_by_dept:
                docs_by_dept[dept] = []
            docs_by_dept[dept].append(doc)

        for dept in sorted(docs_by_dept.keys()):
            with st.expander(f"📁 {dept.upper()}"):
                for doc in sorted(docs_by_dept[dept], key=lambda x: x.get("source_document", "")):
                    doc_name = doc.get("source_document", "Unknown")
                    access = doc.get("access_level", "Unknown")
                    status = doc.get("doc_status", "ACTIVE")

                    status_color = (
                        "var(--success-color)"
                        if status.upper() == "ACTIVE"
                        else "var(--text-muted)"
                    )

                    st.markdown(
                        f"""
                    <div style="background-color: var(--surface-color); padding: 8px; border: 1px solid var(--border-color); border-radius: 4px; margin-bottom: 8px; font-family: 'Share Tech Mono', monospace; font-size: 0.85em; word-wrap: break-word;">
                        <div style="font-weight: bold; color: var(--primary-accent); margin-bottom: 4px;">📄 {doc_name}</div>
                        <div style="color: var(--text-muted);">Access: {access.upper()}</div>
                        <div style="color: {status_color};">Status: {status.upper()}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    project_root = Path(__file__).resolve().parent.parent.parent.parent
                    file_path = project_root / "data" / "raw" / doc_name
                    if file_path.exists():
                        file_bytes = file_path.read_bytes()
                        st.download_button(
                            label="📥 Download",
                            data=file_bytes,
                            file_name=doc_name,
                            mime="application/octet-stream",
                            key=f"kb_dl_{doc_name}",
                        )
                    else:
                        st.markdown(
                            "<small style='color: var(--danger-color); font-family: monospace;'>File not found locally</small>",  # noqa: E501
                            unsafe_allow_html=True,
                        )

        st.markdown("---")

        # Add top_k slider to sidebar, bound to session state or returning value
        if "top_k" not in st.session_state:
            st.session_state["top_k"] = 5
        st.session_state["top_k"] = st.slider(
            "▶ SEARCH DEPTH (Top K)",
            min_value=3,
            max_value=10,
            value=st.session_state.get("top_k", 5),
        )
