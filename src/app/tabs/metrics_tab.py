import streamlit as st
import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.ingestion_helper import get_system_metrics, reset_system


def render_metrics_tab(lazy_load=False):
    """Render metrics tab with optional lazy loading"""
    st.header("📊 System Metrics Dashboard")
    
    # Handle lazy loading
    if lazy_load and not st.session_state.get('metrics_loaded', False):
        st.info("📊 System metrics are ready to load when you need them.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Available Metrics:**")
            st.markdown("• Document categories and collections")
            st.markdown("• Vector database statistics") 
            st.markdown("• Processing performance data")
        
        with col2:
            if st.button("📈 Load Metrics", type="primary", use_container_width=True):
                st.session_state.metrics_loaded = True
                st.rerun()
        
        return
    
    _render_control_buttons(lazy_load)
    
    metrics = _load_metrics()
    if metrics:
        _render_metrics_overview(metrics)
        _render_detailed_metrics(metrics)


def _render_control_buttons(lazy_load=False):
    if lazy_load:
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                if 'metrics_data' in st.session_state:
                    del st.session_state.metrics_data
                st.rerun()
        
        with col2:
            if st.button("📤 Unload Metrics", use_container_width=True):
                st.session_state.metrics_loaded = False
                if 'metrics_data' in st.session_state:
                    del st.session_state.metrics_data
                st.rerun()
        
        with col3:
            if st.button("🗑️ Reset", type="secondary", use_container_width=True):
                st.session_state['show_reset_confirm'] = True
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                if 'metrics_data' in st.session_state:
                    del st.session_state.metrics_data
                st.rerun()
        
        with col2:
            if st.button("🗑️ Reset System", type="secondary", use_container_width=True):
                st.session_state['show_reset_confirm'] = True


def _load_metrics():
    if 'metrics_data' not in st.session_state:
        with st.spinner("Loading system metrics..."):
            try:
                st.session_state.metrics_data = get_system_metrics()
            except Exception as e:
                st.error(f"❌ Failed to load metrics: {str(e)}")
                return None
    
    metrics = st.session_state.metrics_data
    
    if metrics is None:
        st.error("❌ System Error: Failed to retrieve metrics data")
        return None
    
    if metrics.get('status') == 'error':
        st.error(f"❌ System Error: {metrics.get('error', 'Unknown error')}")
        return None
    
    return metrics


def _render_metrics_overview(metrics):
    summary = metrics.get('summary', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Categories", summary.get('total_categories', 0))
    with col2:
        st.metric("Collections", summary.get('total_collections', 0))
    with col3:
        st.metric("Documents", summary.get('total_stored_documents', 0))
    with col4:
        st.metric("Session Processed", summary.get('session_documents', 0))


def _render_detailed_metrics(metrics):
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        _render_categories_section(metrics.get('dynamodb', {}))
    
    with col_right:
        _render_collections_section(metrics.get('vector_database', {}))
    
    if st.session_state.get('show_reset_confirm', False):
        _render_reset_confirmation()


def _render_categories_section(dynamodb_info):
    st.subheader("Document Categories")
    
    total_categories = dynamodb_info.get('total_categories', 0)
    if total_categories == 0:
        st.info("No categories found. Upload documents to create categories.")
        return
    
    categories = dynamodb_info.get('categories', [])
    st.write(f"Found {len(categories)} categories:")
    
    for category in categories:
        category_id = category.get('category_id', 'unknown')
        description = category.get('description', 'No description')
        doc_count = str(category.get('document_count', '0'))
        
        if doc_count.startswith("Decimal('"):
            doc_count = doc_count[9:-2]
        
        with st.expander(f"{category_id.replace('_', ' ').title()}", expanded=False):
            st.write(f"**Description:** {description}")
            st.write(f"**Documents:** {doc_count}")
            st.write(f"**Collection:** {category.get('vector_collection_name', 'N/A')}")


def _render_collections_section(vector_info):
    st.subheader("🔍 Vector Collections")
    
    total_collections = vector_info.get('total_collections', 0)
    if total_collections == 0:
        st.info("No collections found. Process documents to create collections.")
        return
    
    collections = vector_info.get('collections', [])
    total_docs = vector_info.get('total_documents', 0)
    
    st.write(f"Active collections storing {total_docs:,} document chunks:")
    
    for collection in collections:
        collection_name = collection.get('collection_name', 'unknown')
        doc_count = collection.get('document_count', 0)
        percentage = (doc_count / total_docs * 100) if total_docs > 0 else 0
        
        st.metric(
            f"{collection_name}",
            f"{doc_count:,} documents",
            f"{percentage:.1f}%"
        )


def _render_reset_confirmation():
    st.warning("**SYSTEM RESET CONFIRMATION**")
    
    st.error("""
    **This will permanently delete:**
    - All categories from DynamoDB
    - All collections and documents from vector database
    - All ingestion statistics
    
    **This cannot be undone!**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ YES, RESET EVERYTHING", type="primary", use_container_width=True):
            _handle_system_reset()
    
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state['show_reset_confirm'] = False
            st.rerun()


def _handle_system_reset():
    with st.spinner("🔄 Resetting system..."):
        result = reset_system(confirm=True)
        
        if result.get('status') == 'success':
            st.success("System reset successfully!")
            st.session_state.ingestion_stats = {'success': 0, 'failed': 0}
            if 'metrics_data' in st.session_state:
                del st.session_state.metrics_data
            st.session_state['show_reset_confirm'] = False
            st.balloons()
            st.rerun()
        else:
            st.error(f"Reset failed: {result.get('error', 'Unknown error')}")
