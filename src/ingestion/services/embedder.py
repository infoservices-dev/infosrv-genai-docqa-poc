import asyncio
import json
import boto3
import logging
from abc import ABC, abstractmethod
from typing import List
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        pass


class BedrockEmbedding(EmbeddingProvider):
    
    DIMENSIONS = {
        'amazon.titan-embed-text-v1': 1536,
        'amazon.titan-embed-text-v2:0': 1024,
        'cohere.embed-english-v3': 1024,
        'cohere.embed-multilingual-v3': 1024
    }
    
    def __init__(self, model_name: str, region_name: str = "us-east-1", batch_size: int = 10, max_workers: int = 5):
        self.model_name = model_name
        self.bedrock = boto3.client('bedrock-runtime', region_name=region_name)
        self.dimension = self.DIMENSIONS.get(model_name, 1536)
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"Initialized Bedrock: {model_name} in {region_name} (batch_size={batch_size}, workers={max_workers})")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]
    
    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        if len(texts) <= self.batch_size:
            tasks = [asyncio.to_thread(self._embed_text, text) for text in texts]
            return await asyncio.gather(*tasks)
        
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            tasks = [asyncio.to_thread(self._embed_text, text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks)
            all_embeddings.extend(batch_embeddings)
            logger.debug(f"Processed batch {i//self.batch_size + 1}: {len(batch)} embeddings")
        
        return all_embeddings
    
    def get_dimension(self) -> int:
        return self.dimension
    
    def _embed_text(self, text: str) -> List[float]:
        try:
            body = self._build_request_body(text)
            response = self.bedrock.invoke_model(
                modelId=self.model_name,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            return self._extract_embedding(json.loads(response['body'].read()))
            
        except Exception as e:
            logger.error(f"Bedrock embedding failed: {e}")
            return [0.0] * self.dimension
    
    def _build_request_body(self, text: str) -> str:
        if self.model_name.startswith('cohere'):
            return json.dumps({"texts": [text], "input_type": "search_document"})
        return json.dumps({"inputText": text})
    
    def _extract_embedding(self, response: dict) -> List[float]:
        if self.model_name.startswith('cohere'):
            embedding = response.get('embeddings', [[]])[0]
        else:
            embedding = response.get('embedding')
        
        if not embedding:
            raise ValueError(f"No embedding in response from {self.model_name}")
        
        return embedding

class EmbeddingFactory:
    
    @staticmethod
    def create(provider: str, **kwargs) -> EmbeddingProvider:
        provider_lower = provider.lower()
        
        if provider_lower == 'bedrock':
            model_name = kwargs.get('model_name', 'amazon.titan-embed-text-v1')
            region_name = kwargs.get('region_name', 'us-east-1')
            batch_size = kwargs.get('batch_size', 10)
            max_workers = kwargs.get('max_workers', 5)
            return BedrockEmbedding(model_name, region_name, batch_size, max_workers)
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")


class EmbeddingGenerator:
    
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.provider.generate_embeddings(texts)
    
    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        return await self.provider.generate_embeddings_async(texts)
    
    def get_dimension(self) -> int:
        return self.provider.get_dimension()
