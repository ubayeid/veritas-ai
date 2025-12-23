"""
Unified API Client Factory
Supports both OpenAI and xAI (Grok) APIs with OpenAI-compatible interface.
"""

import os
from typing import Optional
from openai import OpenAI
from openai import RateLimitError, APIError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Provider Configuration
API_PROVIDER = os.getenv("API_PROVIDER", "openai").lower()  # "openai" or "xai"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "3000"))
RERANK_TEMPERATURE = float(os.getenv("RERANK_TEMPERATURE", "0.1"))
RERANK_MAX_TOKENS = int(os.getenv("RERANK_MAX_TOKENS", "100"))

# xAI-specific configuration
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_LLM_MODEL = os.getenv("XAI_LLM_MODEL", "grok-3")  # or "grok-2", "grok-2-1212", etc. (grok-beta deprecated)
XAI_EMBEDDING_MODEL = os.getenv("XAI_EMBEDDING_MODEL", "text-embedding-3-small")  # xAI may use OpenAI embeddings

# Model mappings based on provider
MODEL_MAPPINGS = {
    "openai": {
        "llm": LLM_MODEL,
        "embedding": EMBEDDING_MODEL
    },
    "xai": {
        "llm": XAI_LLM_MODEL,
        "embedding": XAI_EMBEDDING_MODEL  # Note: xAI may not have embeddings, might need OpenAI fallback
    }
}


def get_api_client(provider: Optional[str] = None) -> OpenAI:
    """
    Get API client instance for the specified provider.
    
    Args:
        provider: API provider ("openai" or "xai"). If None, uses API_PROVIDER env var.
        
    Returns:
        OpenAI client instance configured for the specified provider
        
    Raises:
        ValueError: If API key is not found for the selected provider
    """
    if provider is None:
        provider = API_PROVIDER
    
    provider = provider.lower()
    
    if provider == "xai":
        api_key = XAI_API_KEY or OPENAI_API_KEY  # Fallback to OpenAI key if xAI key not set
        if not api_key:
            raise ValueError(
                "XAI_API_KEY not found in .env file. "
                "Please set XAI_API_KEY or set API_PROVIDER=openai to use OpenAI."
            )
        try:
            return OpenAI(
                api_key=api_key,
                base_url=XAI_BASE_URL
            )
        except TypeError:
            os.environ["XAI_API_KEY"] = api_key
            return OpenAI(base_url=XAI_BASE_URL)
    
    elif provider == "openai":
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        try:
            return OpenAI(api_key=api_key)
        except TypeError:
            os.environ["OPENAI_API_KEY"] = api_key
            return OpenAI()
    
    else:
        raise ValueError(f"Unknown API provider: {provider}. Use 'openai' or 'xai'.")


def get_llm_model(provider: Optional[str] = None) -> str:
    """
    Get the LLM model name for the specified provider.
    
    Args:
        provider: API provider ("openai" or "xai"). If None, uses API_PROVIDER env var.
        
    Returns:
        Model name string
    """
    if provider is None:
        provider = API_PROVIDER
    
    provider = provider.lower()
    return MODEL_MAPPINGS.get(provider, MODEL_MAPPINGS["openai"])["llm"]


def get_embedding_model(provider: Optional[str] = None) -> str:
    """
    Get the embedding model name for the specified provider.
    
    Note: xAI may not have native embeddings. This function may return OpenAI
    embedding model names even when using xAI provider.
    
    Args:
        provider: API provider ("openai" or "xai"). If None, uses API_PROVIDER env var.
        
    Returns:
        Model name string
    """
    if provider is None:
        provider = API_PROVIDER
    
    provider = provider.lower()
    
    # xAI doesn't have native embeddings, so we might need to use OpenAI for embeddings
    # even when using xAI for LLM
    if provider == "xai":
        # Check if we should use OpenAI for embeddings
        use_openai_embeddings = os.getenv("USE_OPENAI_FOR_EMBEDDINGS", "true").lower() == "true"
        if use_openai_embeddings:
            return MODEL_MAPPINGS["openai"]["embedding"]
    
    return MODEL_MAPPINGS.get(provider, MODEL_MAPPINGS["openai"])["embedding"]


def get_embedding_client(provider: Optional[str] = None) -> OpenAI:
    """
    Get API client for embeddings.
    
    Note: If using xAI provider but embeddings are not available from xAI,
    this will return an OpenAI client for embeddings.
    
    Args:
        provider: API provider ("openai" or "xai"). If None, uses API_PROVIDER env var.
        
    Returns:
        OpenAI client instance configured for embeddings
    """
    if provider is None:
        provider = API_PROVIDER
    
    provider = provider.lower()
    
    # xAI doesn't have native embeddings, use OpenAI for embeddings
    if provider == "xai":
        use_openai_embeddings = os.getenv("USE_OPENAI_FOR_EMBEDDINGS", "true").lower() == "true"
        if use_openai_embeddings:
            return get_api_client("openai")
    
    return get_api_client(provider)


# Backward compatibility: maintain get_openai_client() function
def get_openai_client() -> OpenAI:
    """
    Get OpenAI client instance (backward compatibility).
    Uses API_PROVIDER to determine which provider to use.
    """
    return get_api_client()

