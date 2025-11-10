import logging
from ingestion import config
from ingestion.core.pipeline import IngestionPipeline
from ingestion.services import DocumentProcessor, EmbeddingGenerator, EmbeddingFactory, VectorStoreFactory
from ingestion.services.classifier import DocumentClassifier

logger = logging.getLogger(__name__)


def create_ingestion_pipeline() -> IngestionPipeline:
    logger.info("Initializing ingestion pipeline")
    
    embedding_kwargs = {
        'model_name': config.EMBEDDING_MODEL,
        'region_name': config.AWS_REGION
    }
    
    embedding_provider = EmbeddingFactory.create(config.EMBEDDING_PROVIDER, **embedding_kwargs)
    embedder = EmbeddingGenerator(embedding_provider)
    
    vector_store_kwargs = {
        'collection_name': config.COLLECTION_NAME,
        'host': config.CHROMADB_HOST,
        'port': config.CHROMADB_PORT,
        'embedding_dimension': embedder.get_dimension()
    }
    
    processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    classifier = DocumentClassifier()
    
    return IngestionPipeline(
        processor=processor,
        embedder=embedder,
        vector_store=VectorStoreFactory.create(config.VECTOR_DB_TYPE, **vector_store_kwargs),
        classifier=classifier
    )
