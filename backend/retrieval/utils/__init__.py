"""
Utility modules for retrieval: API client, embeddings, Neo4j queries, and model configuration.
"""

from .model_config import (
    API_PROVIDER,
    get_embedding_model,
    get_llm_model,
    get_rerank_model,
    get_agent_model,
    get_model_config,
    print_model_config,
    EMBEDDING_MODE,
    EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    RERANK_TEMPERATURE,
    RERANK_MAX_TOKENS,
    AGENT_TEMPERATURE,
)
from .api_client import (
    get_api_client,
    get_embedding_client,
)
from .local_embeddings import (
    generate_local_embedding,
    is_local_embeddings_enabled,
    get_local_embedding_model,
    get_available_local_models,
)
from .neo4j_queries import KnowledgeGraphQueries
from .model_switcher import (
    switch_to_grok_llm,
    switch_to_local_embeddings,
    switch_to_openai_all,
    show_current_config,
    get_quick_switch_guide,
)

__all__ = [
    # Model Configuration
    'API_PROVIDER',
    'get_embedding_model',
    'get_llm_model',
    'get_rerank_model',
    'get_agent_model',
    'get_model_config',
    'print_model_config',
    'EMBEDDING_MODE',
    'EMBEDDING_MODEL',
    'LOCAL_EMBEDDING_MODEL',
    'LLM_MODEL',
    'LLM_TEMPERATURE',
    'LLM_MAX_TOKENS',
    'RERANK_TEMPERATURE',
    'RERANK_MAX_TOKENS',
    'AGENT_TEMPERATURE',
    # API Client
    'get_api_client',
    'get_embedding_client',
    # Local Embeddings
    'generate_local_embedding',
    'is_local_embeddings_enabled',
    'get_local_embedding_model',
    'get_available_local_models',
    # Neo4j
    'KnowledgeGraphQueries',
    # Model Switching
    'switch_to_grok_llm',
    'switch_to_local_embeddings',
    'switch_to_openai_all',
    'show_current_config',
    'get_quick_switch_guide',
]
