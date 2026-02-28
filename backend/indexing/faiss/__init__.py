"""
FAISS vector indexing module.
"""

from .utils import build_faiss_index, build_cosine_index, load_faiss_index

__all__ = [
    'build_faiss_index',
    'build_cosine_index',
    'load_faiss_index'
]
