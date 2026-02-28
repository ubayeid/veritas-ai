"""
Shared utilities for FAISS indexing.
"""

from .faiss_builder import (
    load_embeddings_from_json,
    build_faiss_index,
    build_cosine_index,
    save_faiss_index,
    load_faiss_index
)

__all__ = [
    "load_embeddings_from_json",
    "build_faiss_index",
    "build_cosine_index",
    "save_faiss_index",
    "load_faiss_index",
]
