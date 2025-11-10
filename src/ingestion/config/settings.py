import os
import logging

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'bedrock')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'amazon.titan-embed-text-v1')
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '1536'))
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '512'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
VECTOR_DB_TYPE = os.getenv('VECTOR_DB_TYPE', 'chromadb')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'documents')
CHROMADB_HOST = os.getenv('CHROMADB_HOST', 'localhost')
CHROMADB_PORT = int(os.getenv('CHROMADB_PORT', '8000'))

CLASSIFICATION_TABLE_NAME = os.getenv('CLASSIFICATION_TABLE_NAME',"genai-docqa-poc-dev-data-classification")
CLASSIFICATION_MODEL_ID = os.getenv('CLASSIFICATION_MODEL_ID',"anthropic.claude-3-sonnet-20240229-v1:0")
CLASSIFICATION_MAX_TOKENS = int(os.getenv('CLASSIFICATION_MAX_TOKENS', '1000'))
CLASSIFICATION_TEMPERATURE = float(os.getenv('CLASSIFICATION_TEMPERATURE', '0.1'))
CONTENT_PREVIEW_MAX_CHARS = int(os.getenv('CONTENT_PREVIEW_MAX_CHARS', '500'))


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
