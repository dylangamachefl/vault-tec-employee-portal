import streamlit as st


def apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Share+Tech+Mono&display=swap');

        :root {
            --bg-color: #1a1a1a;
            --surface-color: #2a2a2a;
            --primary-accent: #f5c518;
            --secondary-accent: #ff6b00;
            --text-primary: #e8e8e8;
            --text-muted: #888888;
            --success-color: #4caf50;
            --danger-color: #e53935;
            --border-color: #3a3a3a;
        }

        .stApp {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Roboto', sans-serif;
            background-image: repeating-linear-gradient(
                0deg,
                rgba(0,0,0,0.15),
                rgba(0,0,0,0.15) 1px,
                transparent 1px,
                transparent 2px
            );
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Share Tech Mono', monospace !important;
            color: var(--primary-accent) !important;
        }

        h1::before, h2::before, h3::before {
            content: "▶ ";
            color: var(--primary-accent);
        }

        .stButton > button {
            background-color: var(--primary-accent) !important;
            color: #000000 !important;
            font-family: 'Share Tech Mono', monospace !important;
            border-radius: 2px !important;
            border: 1px solid var(--primary-accent) !important;
            font-weight: bold;
        }
        .stButton > button:hover {
            box-shadow: 0 0 10px var(--primary-accent);
            border-color: var(--primary-accent) !important;
            color: #000000 !important;
        }

        .stTextInput > div > div > input, .stChatInput > div > div > textarea {
            background-color: var(--surface-color) !important;
            color: var(--primary-accent) !important;
            border: 1px solid var(--primary-accent) !important;
            border-radius: 2px !important;
            font-family: 'Share Tech Mono', monospace !important;
        }
        .stTextInput > div > div > input:focus, .stChatInput > div > div > textarea:focus {
            border-color: var(--secondary-accent) !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] {
            background-color: #242424 !important;
            border-right: 2px solid var(--primary-accent) !important;
        }

        .streamlit-expanderHeader {
            background-color: var(--surface-color) !important;
            color: var(--primary-accent) !important;
            font-family: 'Share Tech Mono', monospace !important;
            border: 1px solid var(--border-color);
            border-radius: 2px !important;
        }
        .streamlit-expanderHeader:hover {
            color: var(--secondary-accent) !important;
        }
        .streamlit-expanderContent {
            background-color: var(--bg-color) !important;
            border: 1px solid var(--border-color);
            border-top: none;
        }

        [data-testid="stMetricValue"] {
            color: var(--primary-accent) !important;
            font-family: 'Share Tech Mono', monospace !important;
        }

        .stAlert {
            background-color: var(--surface-color) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            border-radius: 2px !important;
        }
        
        [data-baseweb="notification"] {
            border-radius: 2px !important;
            font-family: 'Share Tech Mono', monospace !important;
        }
        [data-testid="stNotificationSuccess"] {
            background-color: rgba(76, 175, 80, 0.1) !important;
            border-left: 4px solid var(--success-color) !important;
            color: var(--success-color) !important;
        }
        [data-testid="stNotificationWarning"] {
            background-color: rgba(255, 107, 0, 0.1) !important;
            border-left: 4px solid var(--secondary-accent) !important;
            color: var(--primary-accent) !important;
        }
        [data-testid="stNotificationError"] {
            background-color: rgba(229, 57, 53, 0.1) !important;
            border-left: 4px solid var(--danger-color) !important;
            color: var(--danger-color) !important;
        }

        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-color);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--primary-accent);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--secondary-accent);
        }

        .login-card {
            max-width: 480px;
            margin: 0 auto;
            background-color: var(--surface-color);
            padding: 2rem;
            border: 1px solid var(--border-color);
            border-top: 4px solid var(--primary-accent);
            text-align: center;
        }

        .vault-header {
            font-family: 'Share Tech Mono', monospace;
            color: var(--primary-accent);
            font-size: 24px;
            margin-bottom: 2rem;
            text-align: center;
            letter-spacing: 2px;
            text-shadow: 0 0 5px rgba(245, 197, 24, 0.5);
            font-weight: bold;
        }

        .vault-footer {
            margin-top: 4rem;
            border-top: 1px solid var(--border-color);
            padding-top: 1rem;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
            font-family: 'Share Tech Mono', monospace;
        }
        
        /* Main chat typography */
        [data-testid="stChatMessageContent"] {
            font-family: 'Roboto', sans-serif !important;
        }

        /* Adjust chat input border */
        [data-testid="stChatInput"] {
            border-radius: 2px !important;
            border: 1px solid var(--primary-accent) !important;
            background-color: var(--surface-color) !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title="VAULT-TEC CORPORATION — INTERNAL KNOWLEDGE SYSTEM"):
    st.markdown(f'<div class="vault-header">{title}</div>', unsafe_allow_html=True)


def render_footer():
    st.markdown(
        """
        <div class="vault-footer">
            VAULT-TEC CORP. | EST. 2031 | BUILDING A BETTER TOMORROW, TODAY™ | AUTHORIZED PERSONNEL ONLY
        </div>
        """,
        unsafe_allow_html=True,
    )
