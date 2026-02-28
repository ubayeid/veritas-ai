"""
Unified API Client Factory
Supports both OpenAI and xAI (Grok) APIs with OpenAI-compatible interface.
"""

import os
from typing import Optional
from openai import OpenAI
from openai import RateLimitError, APIError
from dotenv import load_dotenv

# Import centralized model configuration
from .model_config import (
    API_PROVIDER,
    OPENAI_API_KEY,
    XAI_API_KEY,
    XAI_BASE_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    RERANK_TEMPERATURE,
    RERANK_MAX_TOKENS,
    XAI_LLM_MODEL,
    XAI_EMBEDDING_MODEL,
    get_llm_model,
    get_embedding_model,
)

# Load environment variables
load_dotenv()


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


# get_llm_model and get_embedding_model are imported from model_config


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

