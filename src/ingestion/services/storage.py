import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    
    @abstractmethod
    def add_embedding(self, embedding: List[float], content: str, metadata: Dict[str, Any]):
        pass
    
    @abstractmethod
    def add_embeddings_bulk(self, vectors_data: List[Dict[str, Any]]):
        """Bulk insert multiple embeddings at once"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass


class ChromaDBStore(VectorStore):
    
    def __init__(
        self, 
        collection_name: str = "documents", 
        host: str = "localhost", 
        port: int = 8000,
        embedding_dimension: Optional[int] = None
    ):
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            
            self.default_collection_name = collection_name
            self.current_collection_name = collection_name
            self.embedding_dimension = embedding_dimension
            
            # Initialize the default collection
            self.collection = self._get_or_create_collection(collection_name)
            
            logger.info(f"Connected to ChromaDB at {host}:{port}, default collection: {collection_name}")
            
        except ImportError:
            logger.error("chromadb not installed. Install: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def _get_or_create_collection(self, collection_name: str):
        """Get existing collection or create new one"""
        try:
            existing_collection = self.client.get_collection(name=collection_name)
            
            if self.embedding_dimension:
                test_embedding = [0.0] * self.embedding_dimension
                try:
                    existing_collection.add(
                        ids=["_dimension_test"],
                        embeddings=[test_embedding],
                        documents=["test"]
                    )
                    existing_collection.delete(ids=["_dimension_test"])
                    logger.info(f"Using existing collection: {collection_name}")
                    return existing_collection
                except Exception as e:
                    logger.warning(f"Dimension mismatch detected, recreating collection {collection_name}: {e}")
                    self.client.delete_collection(name=collection_name)
                    return self._create_collection(collection_name)
            else:
                logger.info(f"Using existing collection: {collection_name}")
                return existing_collection
                
        except Exception:
            return self._create_collection(collection_name)
    
    def _create_collection(self, collection_name: str):
        """Create new collection"""
        logger.info(f"Creating new collection: {collection_name}")
        return self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def set_collection(self, collection_name: str):
        """Switch to a different collection"""
        if collection_name != self.current_collection_name:
            logger.info(f"Switching from collection '{self.current_collection_name}' to '{collection_name}'")
            self.collection = self._get_or_create_collection(collection_name)
            self.current_collection_name = collection_name
        else:
            logger.debug(f"Already using collection: {collection_name}")
    
    def use_collection(self, collection_name: str):
        """Alias for set_collection for compatibility"""
        self.set_collection(collection_name)
    
    def get_collection_name(self) -> str:
        """Get current collection name"""
        return self.current_collection_name
    
    def list_collections(self):
        """List all collections"""
        try:
            return self.client.list_collections()
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def get_collections_info(self) -> List[Dict]:
        """Get detailed info about all collections"""
        try:
            collections = self.client.list_collections()
            collections_info = []
            
            for collection in collections:
                try:
                    count = collection.count()
                    collections_info.append({
                        'collection_name': collection.name,
                        'document_count': count,
                        'status': 'active'
                    })
                except Exception as e:
                    logger.warning(f"Error getting count for collection {collection.name}: {e}")
                    collections_info.append({
                        'collection_name': collection.name,
                        'document_count': 0,
                        'status': 'error'
                    })
            
            return collections_info
        except Exception as e:
            logger.error(f"Failed to get collections info: {e}")
            return []

    def add_embedding(self, embedding: List[float], content: str, metadata: Dict[str, Any]):
        try:
            doc_id = str(uuid.uuid4())
            
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )
            
            logger.debug(f"Added embedding {doc_id} with dimension {len(embedding)}")
            
        except Exception as e:
            logger.error(f"Failed to add embedding: {e}")
            raise
    
    def add_embeddings_bulk(self, vectors_data: List[Dict[str, Any]]):
        """
        PERFORMANCE FIX: Bulk insert vectors instead of one-by-one
        Uses the current active collection
        
        Args:
            vectors_data: List of dicts with 'embedding', 'content', 'metadata'
        """
        if not vectors_data:
            return
        
        try:
            # Check if we need to switch collections based on metadata
            collection_name = None
            if vectors_data and 'metadata' in vectors_data[0]:
                collection_name = vectors_data[0]['metadata'].get('vector_collection_name')
                if collection_name and collection_name != self.current_collection_name:
                    self.set_collection(collection_name)
            
            ids = [str(uuid.uuid4()) for _ in vectors_data]
            embeddings = [v['embedding'] for v in vectors_data]
            documents = [v['content'] for v in vectors_data]
            metadatas = [v['metadata'] for v in vectors_data]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Bulk added {len(vectors_data)} embeddings to collection '{self.current_collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to bulk add embeddings to collection '{self.current_collection_name}': {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection.name,
                'total_vectors': count,
                'metadata': self.collection.metadata
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}


class VectorStoreFactory:
    
    @staticmethod
    def create(store_type: str, **kwargs) -> VectorStore:
        if store_type.lower() == 'chromadb':
            collection_name = kwargs.get('collection_name', 'documents')
            host = kwargs.get('host', 'localhost')
            port = kwargs.get('port', 8000)
            embedding_dimension = kwargs.get('embedding_dimension')
            return ChromaDBStore(collection_name, host, port, embedding_dimension)
        
        raise ValueError(f"Unsupported store type: {store_type}")
