"""
Retrieval module for querying vector databases and knowledge graphs.
"""

from .engines import VectorQueryEngine, GraphQueryEngine, HybridQueryEngine, load_faiss_database, get_query_embedding
from .utils import (
    get_api_client,
    get_llm_model,
    get_embedding_model,
    get_embedding_client,
    generate_local_embedding,
    is_local_embeddings_enabled,
    KnowledgeGraphQueries,
)
from .interfaces import Chatbot, app, init_query_engine, init_hybrid_engine

__all__ = [
    # Engines
    'VectorQueryEngine',
    'GraphQueryEngine',
    'HybridQueryEngine',
    'load_faiss_database',
    'get_query_embedding',
    # Utils
    'get_api_client',
    'get_llm_model',
    'get_embedding_model',
    'get_embedding_client',
    'generate_local_embedding',
    'is_local_embeddings_enabled',
    'KnowledgeGraphQueries',
    # Interfaces
    'Chatbot',
    'app',
    'init_query_engine',
    'init_hybrid_engine',
]
