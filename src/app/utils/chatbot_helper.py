import logging
from typing import Tuple, List, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.config import config
from ingestion.services.embedder import EmbeddingFactory
from ingestion.services.storage import VectorStoreFactory
from retrieval import DocumentRetriever, BedrockLLM

logger = logging.getLogger(__name__)


def generate_response_with_sources(
    query: str, 
    top_k: int, 
    temperature: float = 0.7, 
    max_tokens: int = 2000
) -> Tuple[str, List[Dict]]:
    """
    Generate an AI response with source documents.
    
    Args:
        query: User's question
        top_k: Number of documents to retrieve
        temperature: LLM temperature (creativity)
        max_tokens: Maximum response length
        
    Returns:
        Tuple of (response_text, source_documents)
    """
    try:
        # Create embedding provider
        embedding_provider = EmbeddingFactory.create(
            config.EMBEDDING_PROVIDER,
            model_name=config.EMBEDDING_MODEL,
            region_name=config.AWS_REGION
        )
        
        # Create vector store
        vector_store = VectorStoreFactory.create(
            config.VECTOR_DB_TYPE,
            collection_name=config.COLLECTION_NAME,
            host=config.CHROMADB_HOST,
            port=config.CHROMADB_PORT,
            embedding_dimension=embedding_provider.get_dimension()
        )
        
        # Retrieve relevant documents
        retriever = DocumentRetriever(embedding_provider, vector_store, top_k)
        documents = retriever.search(query, top_k)
        
        if not documents:
            return (
                "I couldn't find any relevant documents to answer your question. "
                "Please ensure documents are uploaded in the **Document Ingestion** tab.", 
                []
            )
        
        logger.info(f"Retrieved {len(documents)} documents for query: {query}")
        
        # Generate response using LLM
        llm = BedrockLLM()
        response = llm.generate_response(query, documents, temperature=temperature, max_tokens=max_tokens)
        
        return response, documents
        
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return f"⚠️ **Error**: {str(e)}\n\nPlease check your configuration and try again.", []
