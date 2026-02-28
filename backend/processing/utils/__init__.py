"""
Shared utilities for data processing.
"""

import sys
from pathlib import Path

# Add project root to path for backend imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .pdf_utils import extract_text_from_pdf, extract_text_from_pdfs
from .text_utils import chunk_text, clean_text
from .csv_utils import csv_to_text
from .save_utils import save_embeddings, save_graph_json

# Lazy import for get_embeddings (only imported when needed, avoids import errors in graph scripts)
def get_embeddings(*args, **kwargs):
    """Lazy import wrapper for get_embeddings."""
    from .embedding_utils import get_embeddings as _get_embeddings
    return _get_embeddings(*args, **kwargs)

__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_pdfs",
    "chunk_text",
    "clean_text",
    "get_embeddings",
    "csv_to_text",
    "save_embeddings",
    "save_graph_json",
]
