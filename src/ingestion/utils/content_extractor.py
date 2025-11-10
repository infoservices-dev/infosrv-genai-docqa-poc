import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_document_preview(file_path: str, max_lines: int = 5) -> str:
    """
    Extract first N lines from document for classification.
    
    Args:
        file_path: Path to document
        max_lines: Maximum number of lines to extract
        
    Returns:
        String containing first N lines of content
    """
    file_path_obj = Path(file_path)
    file_extension = file_path_obj.suffix.lower()
    
    try:
        if file_extension in ['.txt', '.md', '.csv']:
            return _extract_text_preview(file_path, max_lines)
        elif file_extension == '.pdf':
            return _extract_pdf_preview(file_path, max_lines)
        elif file_extension in ['.doc', '.docx']:
            return _extract_docx_preview(file_path, max_lines)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return ""
    except Exception as e:
        logger.warning(f"Could not extract preview from {file_path}: {e}")
        return ""


def _extract_text_preview(file_path: str, max_lines: int) -> str:
    """Extract preview from text files"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    lines.append(line)
            return '\n'.join(lines)
    except Exception as e:
        logger.error(f"Error extracting text preview: {e}")
        return ""


def _extract_pdf_preview(file_path: str, max_lines: int) -> str:
    """Extract preview from PDF files"""
    try:
        import PyPDF2
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            if len(reader.pages) > 0:
                text = reader.pages[0].extract_text()
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                return '\n'.join(lines[:max_lines])
        return ""
    except ImportError:
        logger.warning("PyPDF2 not installed, cannot extract PDF preview")
        return ""
    except Exception as e:
        logger.error(f"Error extracting PDF preview: {e}")
        return ""


def _extract_docx_preview(file_path: str, max_lines: int) -> str:
    """Extract preview from DOCX files"""
    try:
        from docx import Document
        doc = Document(file_path)
        lines = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                lines.append(text)
                if len(lines) >= max_lines:
                    break
        return '\n'.join(lines)
    except ImportError:
        logger.warning("python-docx not installed, cannot extract DOCX preview")
        return ""
    except Exception as e:
        logger.error(f"Error extracting DOCX preview: {e}")
        return ""
