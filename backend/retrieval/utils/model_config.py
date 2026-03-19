"""
Centralized Model Configuration
Provides a single source of truth for all model configurations.
Makes it easy to switch between different models and providers.
"""

import os
from typing import Optional, Dict, Literal
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# API PROVIDER CONFIGURATION
# ============================================================================

API_PROVIDER = os.getenv("API_PROVIDER", "openai").lower()  # "openai" or "anthropic"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================

# Embedding Mode: "api" (OpenAI API) or "local" (sentence-transformers)
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "auto").lower()  # "auto", "api", or "local"

# API-based Embedding Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Local Embedding Models (sentence-transformers)
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Legacy: USE_LOCAL_EMBEDDINGS (for backward compatibility)
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "auto").lower()

# ============================================================================
# LLM MODEL CONFIGURATION
# ============================================================================

# OpenAI LLM Models
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "3000"))

# Anthropic LLM Models
ANTHROPIC_LLM_MODEL = os.getenv("ANTHROPIC_LLM_MODEL", "claude-3-5-sonnet-20241022")

# ============================================================================
# RERANKING CONFIGURATION
# ============================================================================

RERANK_MODEL = os.getenv("RERANK_MODEL", "")  # Empty = use LLM_MODEL
RERANK_TEMPERATURE = float(os.getenv("RERANK_TEMPERATURE", "0.1"))
RERANK_MAX_TOKENS = int(os.getenv("RERANK_MAX_TOKENS", "100"))

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
# Note: Evaluation now uses LLM_MODEL directly (judge LLM removed)

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

AGENT_LLM_MODEL = os.getenv("AGENT_LLM_MODEL", "")  # Empty = use LLM_MODEL
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.3"))

# ============================================================================
# MODEL SELECTION FUNCTIONS
# ============================================================================

def get_embedding_mode() -> Literal["api", "local"]:
    """
    Determine which embedding mode to use.
    
    Returns:
        "api" or "local"
    """
    if EMBEDDING_MODE == "local":
        return "local"
    elif EMBEDDING_MODE == "api":
        return "api"
    else:  # "auto"
        # Check legacy USE_LOCAL_EMBEDDINGS for backward compatibility (takes precedence)
        if USE_LOCAL_EMBEDDINGS == "true":
            return "local"
        elif USE_LOCAL_EMBEDDINGS == "false":
            return "api"
        else:  # "auto"
            # Use local if no OpenAI API key, else use API
            if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "":
                return "local"
            return "api"


def get_embedding_model(provider: Optional[str] = None) -> str:
    """
    Get the embedding model name based on current configuration.
    
    Args:
        provider: API provider ("openai" or "anthropic"). If None, uses API_PROVIDER.
        
    Returns:
        Model name string
    """
    mode = get_embedding_mode()
    
    if mode == "local":
        return LOCAL_EMBEDDING_MODEL
    
    # API mode
    if provider is None:
        provider = API_PROVIDER
    
    provider = provider.lower()
    
    if provider == "anthropic":
        # Anthropic doesn't have native embeddings, use OpenAI for embeddings
        use_openai_embeddings = os.getenv("USE_OPENAI_FOR_EMBEDDINGS", "true").lower() == "true"
        if use_openai_embeddings:
            return EMBEDDING_MODEL  # Use OpenAI embedding model
        return LOCAL_EMBEDDING_MODEL  # Fallback to local if OpenAI not available
    
    return EMBEDDING_MODEL


def get_llm_model(provider: Optional[str] = None) -> str:
    """
    Get the LLM model name based on current configuration.
    
    Args:
        provider: API provider ("openai" or "anthropic"). If None, uses API_PROVIDER.
        
    Returns:
        Model name string
    """
    if provider is None:
        provider = API_PROVIDER
    
    provider = provider.lower()
    
    if provider == "anthropic":
        return ANTHROPIC_LLM_MODEL
    
    return LLM_MODEL


def get_rerank_model(provider: Optional[str] = None) -> str:
    """
    Get the reranking model name.
    
    Args:
        provider: API provider. If None, uses API_PROVIDER.
        
    Returns:
        Model name string (defaults to LLM_MODEL if RERANK_MODEL not set)
    """
    if RERANK_MODEL:
        return RERANK_MODEL
    return get_llm_model(provider)




def get_agent_model(provider: Optional[str] = None) -> str:
    """
    Get the agent LLM model name.
    
    Args:
        provider: API provider. If None, uses API_PROVIDER.
        
    Returns:
        Model name string (defaults to LLM_MODEL if AGENT_LLM_MODEL not set)
    """
    if AGENT_LLM_MODEL:
        return AGENT_LLM_MODEL
    return get_llm_model(provider)


def get_model_config() -> Dict[str, any]:
    """
    Get complete model configuration as a dictionary.
    Useful for debugging and logging.
    
    Returns:
        Dictionary with all model configurations
    """
    return {
        "api_provider": API_PROVIDER,
        "embedding": {
            "mode": get_embedding_mode(),
            "model": get_embedding_model(),
            "api_model": EMBEDDING_MODEL,
            "local_model": LOCAL_EMBEDDING_MODEL,
        },
        "llm": {
            "model": get_llm_model(),
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        },
        "rerank": {
            "model": get_rerank_model(),
            "temperature": RERANK_TEMPERATURE,
            "max_tokens": RERANK_MAX_TOKENS,
        },
        "agent": {
            "model": get_agent_model(),
            "temperature": AGENT_TEMPERATURE,
        },
    }


def print_model_config():
    """Print current model configuration for debugging."""
    config = get_model_config()
    print("\n" + "="*80)
    print("MODEL CONFIGURATION")
    print("="*80)
    print(f"API Provider: {config['api_provider']}")
    print(f"\nEmbeddings:")
    print(f"  Mode: {config['embedding']['mode']}")
    print(f"  Model: {config['embedding']['model']}")
    print(f"  API Model: {config['embedding']['api_model']}")
    print(f"  Local Model: {config['embedding']['local_model']}")
    print(f"\nLLM:")
    print(f"  Model: {config['llm']['model']}")
    print(f"  Temperature: {config['llm']['temperature']}")
    print(f"  Max Tokens: {config['llm']['max_tokens']}")
    print(f"\nReranking:")
    print(f"  Model: {config['rerank']['model']}")
    print(f"  Temperature: {config['rerank']['temperature']}")
    print(f"\nAgents:")
    print(f"  Model: {config['agent']['model']}")
    print(f"  Temperature: {config['agent']['temperature']}")
    print("="*80 + "\n")
