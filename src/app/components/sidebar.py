import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.config import config


def render_sidebar():
    with st.sidebar:
        st.sidebar.image("assets/logo-dark.png", width=300)
        st.divider()
        st.markdown("""
            <style>
                .sidebar-header {
                    font-family: 'Verdana', 'Geneva', sans-serif;
                    font-size: 20px;
                    font-weight: 600;
                    margin-bottom: 20px;
                    color: #1f1f1f;
                }
                .config-label {
                    font-family: 'Verdana', 'Geneva', sans-serif;
                    font-size: 4px;
                    font-weight: 200;
                    color: #666;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-bottom: 2px;
                }
                .config-value {
                    font-family: 'Verdana', 'Geneva', sans-serif;
                    font-size: 14px;
                    font-weight: 400;
                    color: #1f1f1f;
                    background-color: #f0f2f6;
                    padding: 8px 12px;
                    border-radius: 6px;
                    margin-bottom: 12px;
                    border-left: 3px solid #0078d4;
                }
                .section-divider {
                    margin: 20px 0;
                    border-top: 1px solid #e0e0e0;
                }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
        
        with st.expander("Embedding Model", expanded=False):
            st.markdown('<p class="config-label">Provider</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="config-value">{config.EMBEDDING_PROVIDER}</div>', unsafe_allow_html=True)
            
            st.markdown('<p class="config-label">Model</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="config-value">{config.EMBEDDING_MODEL}</div>', unsafe_allow_html=True)
            
            st.markdown('<p class="config-label">Dimension</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="config-value">{config.EMBEDDING_DIMENSION}</div>', unsafe_allow_html=True)
        
        with st.expander("Vector Database", expanded=False):
            st.markdown('<p class="config-label">Type</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="config-value">{config.VECTOR_DB_TYPE}</div>', unsafe_allow_html=True)
            
            st.markdown('<p class="config-label">Host</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="config-value">{config.CHROMADB_HOST}:{config.CHROMADB_PORT}</div>', unsafe_allow_html=True)
            
            st.markdown('<p class="config-label">Collection</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="config-value">{config.COLLECTION_NAME}</div>', unsafe_allow_html=True)
        
        with st.expander("Processing Settings", expanded=False):
            st.markdown('<p class="config-label">Chunk Size</p>', unsafe_allow_html=True)
            chunk_size = st.slider(
                "Chunk Size",
                min_value=256,
                max_value=1024,
                value=config.CHUNK_SIZE,
                step=64,
                help="Size of text chunks for processing",
                label_visibility="collapsed"
            )
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            st.markdown('<p class="config-label">Chunk Overlap</p>', unsafe_allow_html=True)
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=200,
                value=config.CHUNK_OVERLAP,
                step=10,
                help="Overlap between consecutive chunks",
                label_visibility="collapsed"
            )
        
        # AWS Region Section
        with st.expander("AWS Configuration", expanded=False):
            st.markdown('<p class="config-label">Region</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="config-value">{config.AWS_REGION}</div>', unsafe_allow_html=True)
                
        return chunk_size, chunk_overlap
