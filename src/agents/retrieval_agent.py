import os
import json
import boto3
import chromadb
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, config: dict):
        self.config = config

    def invoke(self, context: dict) -> dict:
        raise NotImplementedError()

class BedrockEmbedder:
    def __init__(self, model_name: str = "amazon.titan-embed-text-v1", region_name: str = "us-east-1"):
        self.model_name = model_name
        self.bedrock = boto3.client('bedrock-runtime', region_name=region_name)
        self.dimensions = {
            'amazon.titan-embed-text-v1': 1536,
            'amazon.titan-embed-text-v2:0': 1024,
            'cohere.embed-english-v3': 1024,
            'cohere.embed-multilingual-v3': 1024
        }
        self.dimension = self.dimensions.get(model_name, 1536)

    def embed_text(self, text: str) -> list:
        try:
            logger.info(f"Embedding text with model {self.model_name}: {text[:100]}...")
            
            if self.model_name.startswith('cohere'):
                body = json.dumps({"texts": [text], "input_type": "search_document"})
            else:
                body = json.dumps({"inputText": text})
                
            response = self.bedrock.invoke_model(
                modelId=self.model_name,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            logger.debug(f"Bedrock response: {response_body}")
            
            if self.model_name.startswith('cohere'):
                embedding = response_body.get('embeddings', [[]])[0]
            else:
                embedding = response_body.get('embedding')
                
            if embedding:
                logger.info(f"Successfully generated embedding of dimension {len(embedding)}")
                return embedding
            else:
                logger.warning("No embedding returned from Bedrock, using zero vector")
                return [0.0] * self.dimension
            
        except Exception as e:
            logger.error(f"Bedrock embedding failed: {e}")
            return [0.0] * self.dimension

class ChromaVectorClient:
    def __init__(self, endpoint: str, embedding_model: str = "amazon.titan-embed-text-v1"):
        logger.info(f"Initializing ChromaDB client with endpoint: {endpoint}")
        
        if ":" in endpoint:
            host, port = endpoint.split(":")
            port = int(port)
        else:
            host = endpoint
            port = 8000
        
        try:
            self.client = chromadb.HttpClient(host=host, port=port)
            # Test connection
            self.client.heartbeat()
            logger.info(f"Successfully connected to ChromaDB at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
        
        self.embedder = BedrockEmbedder(
            model_name=embedding_model,
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )

    def search(self, collection_name: str, query: str, top_k: int = 5) -> list:
        try:
            logger.info(f"Searching collection '{collection_name}' for query: {query[:100]}...")
            
            # Check if collection exists
            try:
                collection = self.client.get_collection(name=collection_name)
                logger.info(f"Found collection '{collection_name}'")
            except Exception as e:
                logger.error(f"Collection '{collection_name}' not found: {e}")
                # List available collections for debugging
                try:
                    collections = self.client.list_collections()
                    logger.info(f"Available collections: {[c.name for c in collections]}")
                except Exception as list_e:
                    logger.error(f"Failed to list collections: {list_e}")
                return []
            
            # Get collection info
            try:
                count = collection.count()
                logger.info(f"Collection '{collection_name}' has {count} documents")
                if count == 0:
                    logger.warning(f"Collection '{collection_name}' is empty")
                    return []
            except Exception as e:
                logger.warning(f"Could not get collection count: {e}")
            
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            if not query_embedding or all(x == 0.0 for x in query_embedding):
                logger.error("Query embedding is empty or all zeros")
                return []
            
            logger.info(f"Generated query embedding of dimension {len(query_embedding)}")
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            logger.info(f"ChromaDB query returned: {len(results.get('ids', [[]])[0])} results")
            logger.debug(f"Raw results: {results}")
            
            documents = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i] or {}
                    distance = results['distances'][0][i]
                    relevance_score = 1 - distance
                    
                    logger.debug(f"Result {i}: distance={distance}, relevance={relevance_score}")
                    
                    source_uri = (
                        metadata.get("source_uri") or 
                        metadata.get("source") or 
                        metadata.get("file_path") or 
                        metadata.get("filename") or 
                        metadata.get("document_id") or
                        "N/A"
                    )
                    
                    chunk_id = (
                        metadata.get("chunk_id") or 
                        metadata.get("chunk_index") or 
                        metadata.get("page_number") or 
                        metadata.get("section_id") or
                        results['ids'][0][i]
                    )
                    
                    documents.append({
                        "text": results['documents'][0][i],
                        "source_uri": source_uri,
                        "chunk_id": chunk_id,
                        "relevance_score": relevance_score,
                        "metadata": metadata
                    })
                
                logger.info(f"Processed {len(documents)} documents from search results")
            else:
                logger.warning("No documents found in search results")
            
            return documents
            
        except Exception as e:
            logger.error(f"Chroma search failed: {e}", exc_info=True)
            return []

class RetrievalAgent(BaseAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        logger.info(f"Initializing RetrievalAgent with config: {config}")
        self.vector_client = ChromaVectorClient(config.get("vector_endpoint"))

    def invoke(self, context: dict) -> dict:
        logger.info(f"RetrievalAgent invoked with context: {context}")
        
        query = context.get("query")
        context_data_raw = context.get("context_data", "{}")
        
        # Handle both string and dict context_data
        if isinstance(context_data_raw, str):
            try:
                context_data = json.loads(context_data_raw)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse context_data: {context_data_raw}")
                context_data = {}
        else:
            context_data = context_data_raw or {}
            
        search_scopes = context_data.get("search_scope", [])
        routing_category = context_data.get("routing_category")
        category_config = context_data.get("category_config", {})

        logger.info(f"Query: {query}")
        logger.info(f"Search scopes: {search_scopes}")
        logger.info(f"Routing category: {routing_category}")

        if not query or not search_scopes:
            error_msg = "Missing query or search_scope."
            logger.error(error_msg)
            return {"status": "ERROR", "message": error_msg, "context_data": {}}

        collection_name = search_scopes[0]
        logger.info(f"Searching in collection: {collection_name}")
        
        raw_results = self.vector_client.search(collection_name, query)
        logger.info(f"Retrieved {len(raw_results)} raw results")

        # Lower relevance threshold for debugging
        relevance_threshold = 0.3  # Lowered from 0.7 to see more results
        logger.info(f"Applying relevance threshold: {relevance_threshold}")
        
        filtered_results = [r for r in raw_results if r['relevance_score'] >= relevance_threshold]
        logger.info(f"After filtering: {len(filtered_results)} results remain")
        
        # Log relevance scores for debugging
        for i, result in enumerate(raw_results[:5]):  # Show first 5
            logger.info(f"Result {i}: relevance_score={result['relevance_score']:.3f}")

        retrieved_texts = [r["text"] for r in filtered_results]
        source_metadata = [
            {
                "source_uri": r["source_uri"],
                "chunk_id": r["chunk_id"],
                "score": r["relevance_score"]
            }
            for r in filtered_results
        ]

        logger.info(f"Returning {len(retrieved_texts)} texts and {len(source_metadata)} metadata entries")

        # Return context_data as a dictionary, not JSON string
        return {
            "status": "CONTINUE",
            "next_agent": "SynthesisAgent",
            "query": query,
            "context_data": {
                "routing_category": routing_category,
                "retrieved_texts": retrieved_texts,
                "source_metadata": source_metadata,
                "category_config": category_config,
                "search_scope": search_scopes
            }
        }

def lambda_handler(event, context):
    logger.info("Lambda handler started")
    
    agent_config = {
        "vector_endpoint": os.environ.get("CHROMADB_ENDPOINT")
    }
    
    logger.info(f"Agent config: {agent_config}")
    
    try:
        agent = RetrievalAgent(agent_config)
        result = agent.invoke(event)
        logger.info(f"Agent returned: {result}")
        return result
    except Exception as e:
        logger.error(f"Lambda handler error: {e}", exc_info=True)
        return {"status": "FATAL_ERROR", "message": str(e), "input": event}

if __name__ == "__main__":
    test_event = {'status': 'CONTINUE', 'next_agent': 'RetrievalAgent', 'query': 'Hello, may i know key care policy details', 'context_data': '{"routing_category": "insurance_keycare_policy", "search_scope": ["insurance_keycare_policy"], "domain": "General", "category_config": {"collection_name": "insurance_keycare_policy", "description": "Documents related to key insurance policies, including coverage for lost/stolen keys, key replacement, locksmith services, and related terms and conditions", "document_count": 1, "keywords": ["key insurance", "lost keys", "stolen keys", "key replacement", "locksmith", "terms", "conditions"], "domain": "General"}}'}
    test_event = {'status': 'CONTINUE', 'next_agent': 'RetrievalAgent', 'query': 'what is policy about lost car key', 'context_data': {'routing_category': 'insurance_keycare_policy', 'search_scope': ['insurance_keycare_policy'], 'domain': 'General', 'category_config': {'collection_name': 'insurance_keycare_policy', 'description': 'Documents related to key insurance policies, including coverage for lost/stolen keys, key replacement, locksmith services, and related terms and conditions', 'document_count': 1, 'keywords': ['key insurance', 'lost keys', 'stolen keys', 'key replacement', 'locksmith', 'terms', 'conditions'], 'domain': 'General'}}}
    print(lambda_handler(test_event, None))
