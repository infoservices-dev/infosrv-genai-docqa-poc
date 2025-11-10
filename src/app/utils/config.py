from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion import config as ingestion_config


class AppConfig:
    """Application configuration singleton."""
    
    # Embedding Settings
    EMBEDDING_PROVIDER = ingestion_config.EMBEDDING_PROVIDER
    EMBEDDING_MODEL = ingestion_config.EMBEDDING_MODEL
    EMBEDDING_DIMENSION = ingestion_config.EMBEDDING_DIMENSION
    
    # Vector Database Settings
    VECTOR_DB_TYPE = ingestion_config.VECTOR_DB_TYPE
    CHROMADB_HOST = ingestion_config.CHROMADB_HOST
    CHROMADB_PORT = ingestion_config.CHROMADB_PORT
    COLLECTION_NAME = ingestion_config.COLLECTION_NAME
    
    # Processing Settings
    CHUNK_SIZE = ingestion_config.CHUNK_SIZE
    CHUNK_OVERLAP = ingestion_config.CHUNK_OVERLAP
    
    # AWS Settings
    AWS_REGION = ingestion_config.AWS_REGION
    
    @staticmethod
    def setup_logging():
        """Initialize application logging."""
        ingestion_config.setup_logging()
    
    @classmethod
    def get_all_config(cls) -> dict:
        """Get all configuration as a dictionary."""
        return {
            'embedding': {
                'provider': cls.EMBEDDING_PROVIDER,
                'model': cls.EMBEDDING_MODEL,
                'dimension': cls.EMBEDDING_DIMENSION,
            },
            'vector_db': {
                'type': cls.VECTOR_DB_TYPE,
                'host': cls.CHROMADB_HOST,
                'port': cls.CHROMADB_PORT,
                'collection': cls.COLLECTION_NAME,
            },
            'processing': {
                'chunk_size': cls.CHUNK_SIZE,
                'chunk_overlap': cls.CHUNK_OVERLAP,
            },
            'aws': {
                'region': cls.AWS_REGION,
            }
        }


config = AppConfig()
