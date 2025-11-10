import streamlit as st
from pathlib import Path
import tempfile
from typing import List, Dict
import logging
import asyncio

from ingestion.core import create_ingestion_pipeline

logger = logging.getLogger(__name__)


def ingest_documents(uploaded_files: List, update_stats_callback=None):
    """
    Ingest uploaded documents into the vector database.
    
    Args:
        uploaded_files: List of uploaded file objects
        update_stats_callback: Optional callback function to update stats (success_count, failed_count)
    """
    try:
        pipeline = create_ingestion_pipeline()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        failed_count = 0
        
        for idx, uploaded_file in enumerate(uploaded_files):
            original_filename = uploaded_file.name
            status_text.text(f"Processing: {original_filename}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_filename).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            try:
                # Pass original filename as metadata
                metadata = {'original_filename': original_filename}
                result = asyncio.run(pipeline.ingest_document(
                    str(tmp_file_path),
                    metadata=metadata
                ))
                
                if result:
                    success_count += 1
                    category = result.get('category_id', 'unknown')
                    st.success(f"✅ {original_filename} → {category}")
                else:
                    failed_count += 1
                    st.error(f"❌ {original_filename} failed")
                    
            except Exception as e:
                failed_count += 1
                st.error(f"❌ {original_filename}: {str(e)}")
                logger.error(f"Error ingesting {original_filename}: {e}", exc_info=True)
            
            finally:
                Path(tmp_file_path).unlink(missing_ok=True)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("✨ Ingestion complete!")
        
        if update_stats_callback:
            update_stats_callback(success_count, failed_count)
        
        return success_count, failed_count
                
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        st.error(f"🚨 Pipeline error: {str(e)}")
        return 0, len(uploaded_files) if uploaded_files else 0


def get_system_metrics() -> Dict:
    """
    Fetch comprehensive system metrics including DynamoDB and vector store info.
    
    Returns:
        Dictionary with system information
    """
    try:
        pipeline = create_ingestion_pipeline()
        metrics = asyncio.run(pipeline.get_system_info())
        return metrics
    except Exception as e:
        logger.error(f"Failed to fetch system metrics: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }


def reset_system(confirm: bool = False) -> Dict:
    """
    Reset all system data (DynamoDB categories and vector store).
    
    Args:
        confirm: Must be True to execute the reset
    
    Returns:
        Dictionary with reset operation results
    """
    try:
        pipeline = create_ingestion_pipeline()
        result = asyncio.run(pipeline.reset_all_data(confirm=confirm))
        return result
    except Exception as e:
        logger.error(f"Failed to reset system: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }
