import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    
    @abstractmethod
    def load(self, file_path: Path) -> str:
        pass
    
    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        pass


class TxtLoader(DocumentLoader):
    
    def load(self, file_path: Path) -> str:
        return file_path.read_text(encoding='utf-8').strip()
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == '.txt'


class MarkdownLoader(DocumentLoader):
    
    def load(self, file_path: Path) -> str:
        return file_path.read_text(encoding='utf-8').strip()
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.md', '.markdown']


class PdfLoader(DocumentLoader):
    
    def load(self, file_path: Path) -> str:
        try:
            import PyPDF2
            text = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text).strip()
        except ImportError:
            logger.error("PyPDF2 not installed. Install: pip install PyPDF2")
            raise
        except Exception as e:
            logger.error(f"PDF loading failed: {e}")
            raise
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == '.pdf'


class DocumentLoaderFactory:
    
    def __init__(self):
        self.loaders = [
            TxtLoader(),
            MarkdownLoader(),
            PdfLoader()
        ]
    
    def get_loader(self, file_path: Path) -> DocumentLoader:
        extension = file_path.suffix
        
        for loader in self.loaders:
            if loader.supports(extension):
                return loader
        
        raise ValueError(f"No loader found for file type: {extension}")
    
    def register_loader(self, loader: DocumentLoader):
        self.loaders.append(loader)
