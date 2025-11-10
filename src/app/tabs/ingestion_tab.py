import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.ingestion_helper import ingest_documents


def render_ingestion_tab():
    """Render the document ingestion tab with simplified UI."""
    st.header("Document Ingestion")
    
    st.info("Upload documents to be processed and stored in the vector database. Supported: PDF, TXT, MD")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'txt', 'md'],
        accept_multiple_files=True,
        help="Upload one or multiple documents"
    )
    
    # Display stats
    _render_stats_panel(uploaded_files)
    
    # Process files
    if uploaded_files:
        _render_file_preview(uploaded_files)
        _handle_file_processing(uploaded_files)


def _render_stats_panel(uploaded_files):
    """Render the statistics panel."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Files Selected", len(uploaded_files) if uploaded_files else 0)
    
    with col2:
        stats = st.session_state.get('ingestion_stats', {'success': 0, 'failed': 0})
        st.metric("Successfully Processed", stats['success'])
    
    with col3:
        st.metric("Failed", stats.get('failed', 0))


def _render_file_preview(uploaded_files):
    """Render preview of selected files."""
    with st.expander("📋 File Details", expanded=True):
        for file in uploaded_files:
            col_name, col_size, col_type = st.columns([3, 1, 1])
            with col_name:
                st.text(f"📎 {file.name}")
            with col_size:
                st.text(f"{file.size / 1024:.1f} KB")
            with col_type:
                st.text(file.type or "Unknown")


def _handle_file_processing(uploaded_files):
    """Handle the file processing workflow."""
    if st.button("🚀 Process Documents", type="primary", use_container_width=True):
        
        def update_stats(success_count, failed_count):
            """Callback to update ingestion statistics."""
            if 'ingestion_stats' not in st.session_state:
                st.session_state.ingestion_stats = {'success': 0, 'failed': 0}
            
            st.session_state.ingestion_stats['success'] += success_count
            st.session_state.ingestion_stats['failed'] += failed_count
            
            # Clear metrics cache to force refresh
            if 'metrics_data' in st.session_state:
                st.session_state.metrics_data = None
        
        # Process documents with progress feedback
        with st.spinner("Processing documents..."):
            try:
                ingest_documents(uploaded_files, update_stats_callback=update_stats)
                st.success("Documents processed successfully!")
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
