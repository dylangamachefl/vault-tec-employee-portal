import sys
from pathlib import Path

# Ensure src is in python path
src_path = Path(__file__).parent.parent.parent.resolve()
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

import streamlit as st  # noqa: E402

from src.ui.styles.vault_theme import apply_theme, render_footer  # noqa: E402

# Must be the very first st call
st.set_page_config(page_title="Vault-Tec | Login", page_icon="☢️", layout="centered")

apply_theme()

st.markdown(
    """
<div class="login-card">
    <h1 style="font-size: 2.2em; margin-bottom: 0;">VAULT-TEC CORP.</h1>
    <p style="font-family: 'Share Tech Mono', monospace; color: #888888; font-size: 1.1em; margin-top: 5px; margin-bottom: 20px;">
        EMPLOYEE KNOWLEDGE TERMINAL v2.3.1
    </p>
    <p style="color: #4caf50; font-weight: bold; margin-bottom: 2rem; font-family: 'Share Tech Mono', monospace;">
        STAND READY. KNOWLEDGE IS YOUR GREATEST ASSET.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


def set_credentials(role, username):
    st.session_state["role"] = role
    st.session_state["username"] = username
    st.session_state["authenticated"] = True
    st.switch_page("pages/1_Chat.py")


col1, col2 = st.columns(2)

with col1:
    if st.button("GENERAL EMPLOYEE\nCLEARANCE: GENERAL", use_container_width=True):
        set_credentials("general", "dweller_101")
    st.markdown(
        '<p style="font-size: 0.8em; color: var(--text-muted); text-align: center; margin-top:-10px; margin-bottom: 20px;">Standard residential access. Stay in your lane.</p>',  # noqa: E501
        unsafe_allow_html=True,
    )

    if st.button("MARKETING DIVISION\nCLEARANCE: MARKETING", use_container_width=True):
        set_credentials("marketing", "barnsworth_b")
    st.markdown(
        '<p style="font-size: 0.8em; color: var(--text-muted); text-align: center; margin-top:-10px; margin-bottom: 20px;">Enthusiasm Management Division. Keep smiling.</p>',  # noqa: E501
        unsafe_allow_html=True,
    )

with col2:
    if st.button("HR DIVISION\nCLEARANCE: HR-3", use_container_width=True):
        set_credentials("hr", "hr_associate_gable")
    st.markdown(
        '<p style="font-size: 0.8em; color: var(--text-muted); text-align: center; margin-top:-10px; margin-bottom: 20px;">Human Resources. You didn\'t see anything.</p>',  # noqa: E501
        unsafe_allow_html=True,
    )

    if st.button("ADMIN / IT\nCLEARANCE: SYSTEM", use_container_width=True):
        set_credentials("admin", "it_admin_carmichael")
    st.markdown(
        '<p style="font-size: 0.8em; color: var(--text-muted); text-align: center; margin-top:-10px; margin-bottom: 20px;">Systems access. ZAX is watching.</p>',  # noqa: E501
        unsafe_allow_html=True,
    )

st.write("")
st.write("")
st.info(
    "ℹ️ **DEMO MODE** — Role selection loads preset read-only credentials. "
    "Authentication and access control are enforced in the full portal deployment."
)

render_footer()
