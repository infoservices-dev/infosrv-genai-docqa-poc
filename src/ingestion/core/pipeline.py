import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, List
from ingestion.services import DocumentProcessor, EmbeddingGenerator, VectorStore
from ingestion.services.classifier import DocumentClassifier

logger = logging.getLogger(__name__)


class IngestionPipeline:
    
    def __init__(self, processor: DocumentProcessor, embedder: EmbeddingGenerator, vector_store: VectorStore, classifier: DocumentClassifier):
        self.processor = processor
        self.embedder = embedder
        self.vector_store = vector_store
        self.classifier = classifier
        self._stats = {'documents': 0, 'chunks': 0, 'failed': 0}
        self._lock = asyncio.Lock()
    
    async def ingest_document(self, file_path: str, metadata: Optional[Dict] = None, content_preview: Optional[str] = None) -> Optional[Dict]:
        try:
            logger.info(f"Ingesting: {file_path}")
            
            file_path_obj = Path(file_path)
            
            filename = metadata.get('original_filename') if metadata else file_path_obj.name
            if not filename:
                filename = file_path_obj.name
            
            if not content_preview:
                content_preview = await self._extract_preview(file_path)
            
            try:
                classification = await self.classifier.classify_document(filename, content_preview or "")
                logger.info(f"Classification: {classification['category_id']} "
                           f"({'NEW' if classification.get('is_new') else 'EXISTING'})")
            except Exception as e:
                logger.warning(f"Classification failed, using default: {e}")
                classification = {
                    'category_id': 'general_documents',
                    'description': 'Automatically classified document',
                    'vector_collection_name': 'general_documents',
                    'is_new': False
                }
            
            vector_collection_name = classification.get('vector_collection_name', 'general_documents')
            
            chunks = await asyncio.to_thread(self.processor.process_document, file_path)
            if not chunks:
                logger.warning(f"No chunks from {file_path}")
                return None
            
            chunk_contents = [c.content for c in chunks]
            embeddings = await self.embedder.generate_embeddings_async(chunk_contents)
            
            doc_metadata = metadata or {}
            doc_metadata.update({
                'source_file': filename,
                'filename': filename,
                'description': classification['description'],
                'category_id': classification['category_id'],
                'vector_collection_name': classification['vector_collection_name']
            })
            
            vectors_data = [
                {
                    'embedding': embedding,
                    'content': chunk.content,
                    'metadata': {**doc_metadata, **chunk.metadata}
                }
                for chunk, embedding in zip(chunks, embeddings)
            ]
            
            try:
                await asyncio.to_thread(self.vector_store.add_embeddings_bulk, vectors_data)
                logger.info(f"Successfully added {len(vectors_data)} vectors to collection: {vector_collection_name}")
            except Exception as e:
                logger.error(f"Vector store bulk insert failed: {e}")
                success_count = 0
                for i, vector_data in enumerate(vectors_data):
                    try:
                        if hasattr(self.vector_store, 'set_collection'):
                            await asyncio.to_thread(self.vector_store.set_collection, vector_collection_name)
                        
                        if hasattr(self.vector_store, 'add_embedding'):
                            await asyncio.to_thread(
                                self.vector_store.add_embedding,
                                vector_data['embedding'],
                                vector_data['content'],
                                vector_data['metadata']
                            )
                            success_count += 1
                        else:
                            logger.error(f"No suitable method found for adding embeddings to vector store")
                            break
                    except Exception as individual_error:
                        logger.error(f"Failed to add vector {i}: {individual_error}")
                        continue
                
                if success_count > 0:
                    logger.info(f"Successfully added {success_count}/{len(vectors_data)} vectors using fallback method")
                else:
                    raise Exception(f"All vector insertion attempts failed for collection: {vector_collection_name}")
            
            async with self._lock:
                self._stats['documents'] += 1
                self._stats['chunks'] += len(chunks)
            
            logger.info(f"Ingested: {file_path} ({len(chunks)} chunks) to collection: {vector_collection_name}")
            return classification
            
        except Exception as e:
            logger.error(f"Failed: {file_path} - {e}", exc_info=True)
            async with self._lock:
                self._stats['failed'] += 1
            return None
    
    async def _extract_preview(self, file_path: str, max_lines: int = 5) -> str:
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.txt', '.md', '.csv']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [f.readline().strip() for _ in range(max_lines)]
                    return '\n'.join([line for line in lines if line])
            
            elif file_ext == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        if len(reader.pages) > 0:
                            text = reader.pages[0].extract_text()
                            lines = [line.strip() for line in text.split('\n') if line.strip()]
                            return '\n'.join(lines[:max_lines])
                except Exception as e:
                    logger.warning(f"PDF preview extraction failed: {e}")
            
            return ""
        except Exception as e:
            logger.warning(f"Preview extraction failed for {file_path}: {e}")
            return ""
    
    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()
    
    async def get_existing_categories(self) -> List[Dict]:
        try:
            logger.info("Fetching existing categories from DynamoDB")
            categories = await self.classifier._get_all_categories()
            logger.info(f"Found {len(categories)} categories in DynamoDB")
            return categories
        except Exception as e:
            logger.error(f"Failed to fetch categories: {e}", exc_info=True)
            return []
    
    async def get_vector_collections_info(self) -> List[Dict]:
        try:
            logger.info("Fetching vector database collections info")
            
            if hasattr(self.vector_store, 'get_collections_info'):
                collections_info = await asyncio.to_thread(self.vector_store.get_collections_info)
                logger.info(f"Found {len(collections_info)} collections in vector database")
                return collections_info
            elif hasattr(self.vector_store, 'list_collections'):
                collections = await asyncio.to_thread(self.vector_store.list_collections)
                collections_info = []
                
                for collection in collections:
                    try:
                        if hasattr(collection, 'name') and hasattr(collection, 'count'):
                            count = await asyncio.to_thread(collection.count)
                            collections_info.append({
                                'collection_name': collection.name,
                                'document_count': count,
                                'status': 'active'
                            })
                        elif isinstance(collection, str):
                            if hasattr(self.vector_store, 'get_collection'):
                                try:
                                    col_obj = await asyncio.to_thread(self.vector_store.get_collection, collection)
                                    count = await asyncio.to_thread(col_obj.count) if hasattr(col_obj, 'count') else 0
                                    collections_info.append({
                                        'collection_name': collection,
                                        'document_count': count,
                                        'status': 'active'
                                    })
                                except Exception as e:
                                    logger.warning(f"Could not get count for collection {collection}: {e}")
                                    collections_info.append({
                                        'collection_name': collection,
                                        'document_count': 0,
                                        'status': 'unknown'
                                    })
                            else:
                                collections_info.append({
                                    'collection_name': collection,
                                    'document_count': 0,
                                    'status': 'unknown'
                                })
                        else:
                            collections_info.append({
                                'collection_name': str(collection),
                                'document_count': 0,
                                'status': 'unknown'
                            })
                    except Exception as e:
                        logger.warning(f"Error processing collection {collection}: {e}")
                        collections_info.append({
                            'collection_name': str(collection),
                            'document_count': 0,
                            'status': 'error'
                        })
                
                logger.info(f"Found {len(collections_info)} collections (using list_collections method)")
                return collections_info
            elif hasattr(self.vector_store, 'client') and hasattr(self.vector_store.client, 'list_collections'):
                collections = await asyncio.to_thread(self.vector_store.client.list_collections)
                collections_info = []
                
                for collection in collections:
                    try:
                        count = await asyncio.to_thread(collection.count) if hasattr(collection, 'count') else 0
                        collections_info.append({
                            'collection_name': collection.name if hasattr(collection, 'name') else str(collection),
                            'document_count': count,
                            'status': 'active'
                        })
                    except Exception as e:
                        logger.warning(f"Error getting count for collection: {e}")
                        collections_info.append({
                            'collection_name': collection.name if hasattr(collection, 'name') else str(collection),
                            'document_count': 0,
                            'status': 'error'
                        })
                
                logger.info(f"Found {len(collections_info)} collections (using client.list_collections)")
                return collections_info
            else:
                logger.warning("No collection info methods available on vector store")
                return []
                
        except Exception as e:
            logger.error(f"Failed to fetch vector collections info: {e}", exc_info=True)
            return []
    
    async def get_comprehensive_metrics(self) -> Dict:
        try:
            logger.info("Gathering comprehensive system metrics")
            
            categories_task = self.get_existing_categories()
            collections_task = self.get_vector_collections_info()
            
            categories, collections = await asyncio.gather(
                categories_task,
                collections_task,
                return_exceptions=True
            )
            
            if isinstance(categories, Exception):
                logger.error(f"Categories fetch failed: {categories}")
                categories = []
            if isinstance(collections, Exception):
                logger.error(f"Collections fetch failed: {collections}")
                collections = []
            
            total_documents_in_vectors = sum(c.get('document_count', 0) for c in collections)
            
            metrics = {
                'ingestion_stats': self.get_stats(),
                'categories': {
                    'total_count': len(categories),
                    'category_ids': [cat['category_id'] for cat in categories],
                    'details': categories
                },
                'vector_collections': {
                    'total_count': len(collections),
                    'total_documents': total_documents_in_vectors,
                    'collections': collections
                },
                'summary': {
                    'unique_categories': len(categories),
                    'active_collections': len(collections),
                    'total_stored_documents': total_documents_in_vectors,
                    'current_session_documents': self._stats['documents'],
                    'current_session_chunks': self._stats['chunks'],
                    'current_session_failures': self._stats['failed']
                }
            }
            
            logger.info(f"Comprehensive metrics gathered: {metrics['summary']}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to gather comprehensive metrics: {e}", exc_info=True)
            return {
                'error': str(e),
                'ingestion_stats': self.get_stats()
            }
    
    async def get_category_distribution(self) -> Dict[str, int]:
        try:
            logger.info("Calculating category distribution")
            
            if hasattr(self.vector_store, 'get_category_distribution'):
                distribution = await asyncio.to_thread(self.vector_store.get_category_distribution)
                logger.info(f"Category distribution: {distribution}")
                return distribution
            else:
                collections_info = await self.get_vector_collections_info()
                distribution = {}
                for collection in collections_info:
                    if 'collection_name' in collection and 'document_count' in collection:
                        distribution[collection['collection_name']] = collection['document_count']
                logger.info(f"Category distribution (from collections): {distribution}")
                return distribution
                
        except Exception as e:
            logger.error(f"Failed to get category distribution: {e}", exc_info=True)
            return {}
    
    async def get_system_info(self) -> Dict:
        try:
            logger.info("Fetching complete system information")
            
            categories_task = self.get_existing_categories()
            collections_task = self.get_vector_collections_info()
            distribution_task = self.get_category_distribution()
            
            categories, collections, distribution = await asyncio.gather(
                categories_task,
                collections_task,
                distribution_task,
                return_exceptions=True
            )
            
            if isinstance(categories, Exception):
                logger.error(f"Categories fetch failed: {categories}")
                categories = []
            if isinstance(collections, Exception):
                logger.error(f"Collections fetch failed: {collections}")
                collections = []
            if isinstance(distribution, Exception):
                logger.error(f"Distribution fetch failed: {distribution}")
                distribution = {}
            
            total_documents_in_vectors = sum(c.get('document_count', 0) for c in collections)
            
            system_info = {
                'timestamp': asyncio.get_event_loop().time(),
                'status': 'success',
                'dynamodb': {
                    'total_categories': len(categories),
                    'category_ids': [cat.get('category_id', 'unknown') for cat in categories],
                    'categories': categories
                },
                'vector_database': {
                    'total_collections': len(collections),
                    'total_documents': total_documents_in_vectors,
                    'index_names': [c.get('collection_name', 'unknown') for c in collections],
                    'collections': collections,
                    'category_distribution': distribution
                },
                'ingestion_session': {
                    'documents_processed': self._stats['documents'],
                    'chunks_created': self._stats['chunks'],
                    'failures': self._stats['failed']
                },
                'summary': {
                    'total_categories': len(categories),
                    'total_collections': len(collections),
                    'total_stored_documents': total_documents_in_vectors,
                    'session_documents': self._stats['documents'],
                    'session_chunks': self._stats['chunks'],
                    'session_failures': self._stats['failed']
                }
            }
            
            logger.info(f"System info retrieved: {system_info['summary']}")
            return system_info
            
        except Exception as e:
            logger.error(f"Failed to fetch system info: {e}", exc_info=True)
            return {
                'timestamp': asyncio.get_event_loop().time(),
                'status': 'error',
                'error': str(e),
                'ingestion_session': self._stats
            }
    
    async def reset_all_data(self, confirm: bool = False) -> Dict:
        if not confirm:
            logger.warning("Reset operation called without confirmation")
            return {
                'status': 'cancelled',
                'message': 'Reset operation requires explicit confirmation (confirm=True)'
            }
        
        try:
            logger.warning("Starting DESTRUCTIVE reset operation - clearing all data")
            
            results = {
                'status': 'in_progress',
                'dynamodb': {'status': 'pending'},
                'vector_database': {'status': 'pending'},
                'ingestion_stats': {'status': 'pending'}
            }
            
            try:
                logger.info("Resetting DynamoDB categories...")
                
                if hasattr(self.classifier, 'reset_all_categories'):
                    logger.info("Using classifier.reset_all_categories() method")
                    dynamodb_result = await asyncio.to_thread(self.classifier.reset_all_categories)
                    results['dynamodb'] = {
                        'status': 'success',
                        'message': 'All categories deleted from DynamoDB using reset_all_categories',
                        'details': dynamodb_result
                    }
                elif hasattr(self.classifier, '_get_all_categories') and hasattr(self.classifier, '_delete_category'):
                    logger.info("Using fallback: deleting categories individually")
                    categories = await self.classifier._get_all_categories()
                    deleted_count = 0
                    failed_deletions = []
                    
                    for category in categories:
                        try:
                            category_id = category.get('category_id')
                            if category_id:
                                await asyncio.to_thread(self.classifier._delete_category, category_id)
                                deleted_count += 1
                                logger.info(f"Deleted category: {category_id}")
                        except Exception as e:
                            failed_deletions.append({'category_id': category_id, 'error': str(e)})
                            logger.error(f"Failed to delete category {category_id}: {e}")
                    
                    results['dynamodb'] = {
                        'status': 'success' if not failed_deletions else 'partial_success',
                        'message': f'Deleted {deleted_count} categories from DynamoDB',
                        'details': {
                            'deleted_count': deleted_count,
                            'failed_deletions': failed_deletions,
                            'total_found': len(categories)
                        }
                    }
                elif hasattr(self.classifier, 'table'):
                    logger.info("Using direct DynamoDB table operations")
                    try:
                        response = await asyncio.to_thread(self.classifier.table.scan)
                        items = response.get('Items', [])
                        deleted_count = 0
                        
                        for item in items:
                            try:
                                key = {'category_id': item['category_id']}
                                await asyncio.to_thread(self.classifier.table.delete_item, Key=key)
                                deleted_count += 1
                                logger.info(f"Deleted category: {item['category_id']}")
                            except Exception as e:
                                logger.error(f"Failed to delete item {item.get('category_id', 'unknown')}: {e}")
                        
                        results['dynamodb'] = {
                            'status': 'success',
                            'message': f'Deleted {deleted_count} categories using direct table operations',
                            'details': {'deleted_count': deleted_count, 'total_found': len(items)}
                        }
                    except Exception as e:
                        logger.error(f"Direct DynamoDB operations failed: {e}")
                        raise e
                else:
                    logger.warning("No DynamoDB reset methods available on classifier")
                    results['dynamodb'] = {
                        'status': 'skipped',
                        'message': 'No reset methods available on classifier'
                    }
                logger.info("DynamoDB reset completed")
            except Exception as e:
                logger.error(f"DynamoDB reset failed: {e}", exc_info=True)
                results['dynamodb'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            try:
                logger.info("Resetting vector database...")
                
                if hasattr(self.vector_store, 'reset_all_collections'):
                    logger.info("Using vector_store.reset_all_collections() method")
                    vector_result = await asyncio.to_thread(self.vector_store.reset_all_collections)
                    results['vector_database'] = {
                        'status': 'success',
                        'message': 'All collections deleted from vector database',
                        'details': vector_result
                    }
                elif hasattr(self.vector_store, 'delete_all'):
                    logger.info("Using vector_store.delete_all() method")
                    vector_result = await asyncio.to_thread(self.vector_store.delete_all)
                    results['vector_database'] = {
                        'status': 'success',
                        'message': 'All data deleted from vector database (using delete_all)',
                        'details': vector_result
                    }
                elif hasattr(self.vector_store, 'client') and hasattr(self.vector_store.client, 'list_collections'):
                    logger.info("Using ChromaDB client collection deletion")
                    collections = await asyncio.to_thread(self.vector_store.client.list_collections)
                    deleted_collections = []
                    failed_deletions = []
                    
                    for collection in collections:
                        try:
                            collection_name = collection.name if hasattr(collection, 'name') else str(collection)
                            logger.info(f"Attempting to delete collection: {collection_name}")
                            
                            try:
                                await asyncio.to_thread(self.vector_store.client.delete_collection, collection_name)
                                deleted_collections.append(collection_name)
                                logger.info(f"Successfully deleted collection: {collection_name}")
                            except Exception as e:
                                logger.warning(f"Standard deletion failed for {collection_name}, trying alternative: {e}")
                                await asyncio.to_thread(collection.delete)
                                deleted_collections.append(collection_name)
                                logger.info(f"Successfully deleted collection using alternative method: {collection_name}")
                        except Exception as e:
                            failed_deletions.append({'collection_name': collection_name, 'error': str(e)})
                            logger.error(f"Failed to delete collection {collection_name}: {e}")
                    
                    results['vector_database'] = {
                        'status': 'success' if not failed_deletions else 'partial_success',
                        'message': f'ChromaDB deletion completed: {len(deleted_collections)} successful, {len(failed_deletions)} failed',
                        'details': {
                            'deleted_collections': deleted_collections,
                            'failed_deletions': failed_deletions,
                            'total_found': len(collections)
                        }
                    }
                elif hasattr(self.vector_store, 'list_collections'):
                    logger.info("Using vector_store.list_collections() method")
                    collections = await asyncio.to_thread(self.vector_store.list_collections)
                    deleted_collections = []
                    failed_deletions = []
                    
                    for collection in collections:
                        try:
                            collection_name = collection.name if hasattr(collection, 'name') else str(collection)
                            logger.info(f"Attempting to delete collection: {collection_name}")
                            
                            if hasattr(self.vector_store, 'delete_collection'):
                                await asyncio.to_thread(self.vector_store.delete_collection, collection_name)
                                deleted_collections.append(collection_name)
                                logger.info(f"Successfully deleted collection: {collection_name}")
                            else:
                                raise Exception("No delete_collection method available")
                        except Exception as e:
                            failed_deletions.append({'collection_name': collection_name, 'error': str(e)})
                            logger.error(f"Failed to delete collection {collection_name}: {e}")
                    
                    results['vector_database'] = {
                        'status': 'success' if not failed_deletions else 'partial_success',
                        'message': f'Vector store deletion completed: {len(deleted_collections)} successful, {len(failed_deletions)} failed',
                        'details': {
                            'deleted_collections': deleted_collections,
                            'failed_deletions': failed_deletions,
                            'total_found': len(collections)
                        }
                    }
                else:
                    logger.warning("No vector database reset methods available")
                    results['vector_database'] = {
                        'status': 'skipped',
                        'message': 'No reset methods available on vector store'
                    }
                logger.info("Vector database reset completed")
            except Exception as e:
                logger.error(f"Vector database reset failed: {e}", exc_info=True)
                results['vector_database'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            async with self._lock:
                self._stats = {'documents': 0, 'chunks': 0, 'failed': 0}
            results['ingestion_stats'] = {
                'status': 'success',
                'message': 'Session statistics reset'
            }
            
            successful_operations = sum(1 for result in [results['dynamodb'], results['vector_database']] 
                                      if result['status'] in ['success', 'skipped'])
            partial_operations = sum(1 for result in [results['dynamodb'], results['vector_database']] 
                                   if result['status'] == 'partial_success')
            
            if successful_operations == 2:
                results['status'] = 'success'
            elif successful_operations + partial_operations >= 1:
                results['status'] = 'partial_success'
            else:
                results['status'] = 'error'
            
            logger.warning(f"Reset operation completed with status: {results['status']}")
            return results
            
        except Exception as e:
            logger.error(f"Reset operation failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'message': 'Reset operation encountered an unexpected error'
            }
