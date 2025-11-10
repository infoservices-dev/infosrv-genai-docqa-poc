import streamlit as st


def render_header():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("Document Q&A RAG System - POC")
        st.caption("Enterprise Document Intelligence & AI Assistant")
    
    with col2:
        if st.session_state.get('authenticated', True):
            st.success("Connected")
        else:
            st.error("Disconnected")
