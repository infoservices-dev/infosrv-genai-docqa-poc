import logging
from pathlib import Path
from typing import List
from ingestion.models import DocumentChunk
from ingestion.services.loaders import DocumentLoaderFactory

logger = logging.getLogger(__name__)


class DocumentProcessor:
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.loader_factory = DocumentLoaderFactory()
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        loader = self.loader_factory.get_loader(path)
        content = loader.load(path)
        
        if not content:
            return []
        
        chunks = self._split_text(content)
        return [
            DocumentChunk(
                content=chunk,
                metadata={'chunk_index': i, 'chunk_size': len(chunk)}
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def _split_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            if end >= len(text):
                break
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
