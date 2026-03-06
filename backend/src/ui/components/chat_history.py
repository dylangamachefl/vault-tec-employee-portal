from pathlib import Path

import streamlit as st


def render_citation_card(source):
    # source is a SourceCitation model (or dict if serialized)

    # Handle both Pydantic model and dict
    if isinstance(source, dict):
        source_doc = source.get("source_document", "Unknown")
        section = source.get("section_title") or "N/A"
        dept = source.get("department", "Unknown")
        access = source.get("access_level", "Unknown")
        status = source.get("doc_status") or "ACTIVE"
        date = source.get("doc_date") or "Unknown"
    else:
        source_doc = getattr(source, "source_document", "Unknown")
        section = getattr(source, "section_title", "N/A")
        dept = getattr(source, "department", "Unknown")
        access = getattr(source, "access_level", "Unknown")
        status = getattr(source, "doc_status", "ACTIVE")
        date = getattr(source, "doc_date", "Unknown")

    status_color = "var(--success-color)" if status == "ACTIVE" else "var(--text-muted)"

    st.markdown(
        f"""
    <div style="background-color: var(--surface-color); padding: 10px; border: 1px solid var(--border-color); border-radius: 4px; margin-bottom: 5px; font-family: 'Share Tech Mono', monospace; font-size: 0.9em;">
        <div style="color: var(--primary-accent); font-weight: bold; margin-bottom: 5px; word-wrap: break-word;">
            📄 {source_doc}
        </div>
        <div style="margin-bottom: 2px;">Section: {section}</div>
        <div style="margin-bottom: 2px;">
            Dept: <span style="color: var(--text-primary);">{dept}</span> | 
            Access: <span style="color: var(--text-primary);">{access}</span>
        </div>
        <div style="margin-bottom: 5px;">
            Status: <span style="color: {status_color};">{status}</span> | 
            Date: {date}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    file_path = project_root / "data" / "raw" / source_doc
    if file_path.exists():
        file_bytes = file_path.read_bytes()
        # Add a unique key so Streamlit doesn't complain about duplicate buttons
        unique_key = f"dl_cit_{source_doc}_{hash(section)}_{hash(date)}"
        st.download_button(
            label="📥 Download Source Document",
            data=file_bytes,
            file_name=source_doc,
            mime="application/octet-stream",
            key=unique_key,
        )
    else:
        st.markdown(
            "<small style='color: var(--danger-color); font-family: monospace;'>File not found locally</small>",  # noqa: E501
            unsafe_allow_html=True,
        )


def render_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            if role == "assistant" and content == "WARNING: INSUFFICIENT INFORMATION":
                st.warning(
                    "⚠️ "
                    + message.get(
                        "error_msg", "This information is not available in the provided documents."
                    )
                )
            elif role == "assistant" and "This information is not available" in content:
                st.warning("⚠️ " + content)
            else:
                st.markdown(content)

            if role == "assistant" and "sources" in message and message["sources"]:
                st.markdown(
                    "<div style='font-family: \"Share Tech Mono\", monospace; color: var(--text-muted); margin-top: 15px; margin-bottom: 5px; font-size: 0.9em;'>CITED SOURCES:</div>",  # noqa: E501
                    unsafe_allow_html=True,
                )
                for source in message["sources"]:
                    render_citation_card(source)
