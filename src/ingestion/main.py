import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.config.settings import setup_logging, CONTENT_PREVIEW_MAX_CHARS
from ingestion.core import create_ingestion_pipeline

logger = logging.getLogger(__name__)


async def ingest_files(file_paths: list[str]):
    pipeline = create_ingestion_pipeline()
    
    tasks = [
        ingest_single_file(pipeline, file_path)
        for file_path in file_paths
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for file_path, result in zip(file_paths, results):
        if isinstance(result, Exception):
            logger.error(f"✗ Failed: {file_path} - {result}")
        elif result:
            logger.info(f"✓ Completed: {file_path} -> {result.get('category_id')}")
        else:
            logger.error(f"✗ Failed: {file_path}")
    
    logger.info(f"Stats: {pipeline.get_stats()}")


async def ingest_single_file(pipeline, file_path: str):
    logger.info(f"Ingesting: {file_path}")
    
    # Extract content preview
    content_preview = extract_document_preview(file_path)
    
    # Ingest to vector store (classification happens inside pipeline)
    result = await pipeline.ingest_document(file_path, content_preview=content_preview)
    
    return result


def extract_document_preview(file_path: str, max_chars: int = None) -> str:
    if max_chars is None:
        max_chars = CONTENT_PREVIEW_MAX_CHARS
    
    file_path_obj = Path(file_path)
    file_extension = file_path_obj.suffix.lower()
    
    try:
        if file_extension in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_chars)
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                preview = ' '.join(lines[:3]) 
                return preview[:200] if len(preview) > 200 else preview
        
        elif file_extension == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    if len(reader.pages) > 0:
                        text = reader.pages[0].extract_text()
                        lines = [line.strip() for line in text.split('\n') if line.strip()]
                        preview = ' '.join(lines[:3])
                        return preview[:200] if len(preview) > 200 else preview
            except Exception as e:
                logger.warning(f"Could not extract PDF preview: {e}")
                return ""
        
        return ""
    
    except Exception as e:
        logger.warning(f"Could not extract preview from {file_path}: {e}")
        return ""


def main():
    setup_logging()
    
    if len(sys.argv) < 2:
        logger.error("Usage: python main.py <file1> <file2> ...")
        return 1
    
    file_paths = sys.argv[1:]
    
    try:
        asyncio.run(ingest_files(file_paths))
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
