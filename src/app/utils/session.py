import streamlit as st
from typing import Any


def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'ingestion_stats' not in st.session_state:
        st.session_state.ingestion_stats = {'success': 0, 'failed': 0}
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = True


def add_message(role: str, content: str, sources=None):
    """Add a message to chat history with optional sources."""
    message = {'role': role, 'content': content}
    if sources is not None:
        message['sources'] = sources
    st.session_state.chat_history.append(message)


def clear_chat_history():
    st.session_state.chat_history = []


def get_session_value(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any):
    st.session_state[key] = value
