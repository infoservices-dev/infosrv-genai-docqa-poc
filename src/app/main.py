import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import config
from components.sidebar import render_sidebar as render_config_sidebar
from tabs.ingestion_tab import render_ingestion_tab
from tabs.chatbot_tab import render_chatbot_tab
from tabs.metrics_tab import render_metrics_tab

st.set_page_config(
    page_title="Document Q&A RAG System - POC",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        
        .info-card {
            background: white;
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            padding: 1.2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        
        .status-success { color: #10b981; font-weight: 600; }
        .status-warning { color: #f59e0b; font-weight: 600; }
        .status-error { color: #ef4444; font-weight: 600; }
        
        /* Custom styling for API connection status */
        .api-status-connected {
            background: linear-gradient(90deg, #10b981, #059669);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

config.setup_logging()

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'ingestion_stats' not in st.session_state:
        st.session_state.ingestion_stats = {'success': 0, 'failed': 0}
    if 'metrics_loaded' not in st.session_state:
        st.session_state.metrics_loaded = False
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = None

def render_sidebar():
    chunk_size, chunk_overlap = render_config_sidebar()
    
    with st.sidebar:
        st.divider()
        
        st.subheader("🔗 API Status")
        st.markdown("""
        <div class="api-status-connected">
            ✅ Step Functions API Ready
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Endpoint:** `execute-api.us-east-1.amazonaws.com`")
        st.markdown("**Region:** `us-east-1`")
        
        st.divider()
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <p style='font-size: 18px; color: #999; margin: 0;'>Powered by</p>
            <p style='font-size: 24px; font-weight: 600; color: #1f1f1f; margin: 5px 0;'>
                Infoservices
            </p>
            <p style='font-size: 18px; color: #999; margin: 0;'>© 2025 All Rights Reserved</p>
        </div>
        """, unsafe_allow_html=True)
    
    return chunk_size, chunk_overlap

def main():
    init_session_state()
    render_sidebar()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Document Q&A RAG System - POC")
        st.caption("Multi-Agent Orchestration with AWS Step Functions")
    with col2:
        st.success("✅ Connected to API")
    
    tab1, tab2, tab3 = st.tabs([
        "📄 Document Ingestion", 
        "🤖 AI Chatbot", 
        "📊 System Metrics"
    ])
    
    with tab1:
        if st.session_state.current_tab != 'ingestion':
            st.session_state.current_tab = 'ingestion'
        render_ingestion_tab()
    
    with tab2:
        if st.session_state.current_tab != 'chatbot':
            st.session_state.current_tab = 'chatbot'
        render_chatbot_tab()
    
    with tab3:
        if st.session_state.current_tab != 'metrics':
            st.session_state.current_tab = 'metrics'
            # Reset metrics loaded state when switching away and back to metrics tab
            if not st.session_state.get('metrics_tab_visited', False):
                st.session_state.metrics_loaded = False
            st.session_state.metrics_tab_visited = True
        render_metrics_tab(lazy_load=True)

if __name__ == "__main__":
    main()
