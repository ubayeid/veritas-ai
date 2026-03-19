"""
Unified API Client Factory
Supports OpenAI and Anthropic APIs with OpenAI-compatible interface.
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
    ANTHROPIC_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    RERANK_TEMPERATURE,
    RERANK_MAX_TOKENS,
    ANTHROPIC_LLM_MODEL,
    AGENT_TEMPERATURE,
    get_llm_model,
    get_embedding_model,
    get_agent_model,
)

# Load environment variables
load_dotenv()


def get_api_client(provider: Optional[str] = None) -> OpenAI:
    """
    Get API client instance for the specified provider.
    
    Note: Anthropic uses a different client library, so this function returns OpenAI client
    for embeddings. For LLM usage with Anthropic, use get_langchain_llm() instead.
    
    Args:
        provider: API provider ("openai" or "anthropic"). If None, uses API_PROVIDER env var.
        
    Returns:
        OpenAI client instance configured for the specified provider
        
    Raises:
        ValueError: If API key is not found for the selected provider
    """
    if provider is None:
        provider = API_PROVIDER
    
    provider = provider.lower()
    
    if provider == "openai":
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        try:
            return OpenAI(api_key=api_key)
        except TypeError:
            os.environ["OPENAI_API_KEY"] = api_key
            return OpenAI()
    
    elif provider == "anthropic":
        # Anthropic doesn't use OpenAI client, but we can use OpenAI for embeddings
        # For LLM, use get_langchain_llm() instead
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in .env file. "
                "Anthropic provider requires OpenAI API key for embeddings. "
                "For LLM usage, use get_langchain_llm() instead."
            )
        try:
            return OpenAI(api_key=api_key)
        except TypeError:
            os.environ["OPENAI_API_KEY"] = api_key
            return OpenAI()
    
    else:
        raise ValueError(f"Unknown API provider: {provider}. Use 'openai' or 'anthropic'.")


# get_llm_model and get_embedding_model are imported from model_config


def get_embedding_client(provider: Optional[str] = None) -> OpenAI:
    """
    Get API client for embeddings.
    
    Note: If using Anthropic provider, embeddings are not available natively,
    so this will return an OpenAI client for embeddings.
    
    Args:
        provider: API provider ("openai" or "anthropic"). If None, uses API_PROVIDER env var.
        
    Returns:
        OpenAI client instance configured for embeddings
    """
    if provider is None:
        provider = API_PROVIDER
    
    provider = provider.lower()
    
    # Anthropic doesn't have native embeddings, use OpenAI for embeddings
    if provider == "anthropic":
        use_openai_embeddings = os.getenv("USE_OPENAI_FOR_EMBEDDINGS", "true").lower() == "true"
        if use_openai_embeddings:
            return get_api_client("openai")
    
    return get_api_client(provider)


def get_langchain_llm(provider: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None, use_agent_model: bool = False):
    """
    Get LangChain LLM instance for the specified provider.
    
    Supports OpenAI (ChatOpenAI) and Anthropic (ChatAnthropic).
    
    Args:
        provider: API provider ("openai" or "anthropic"). If None, uses API_PROVIDER env var.
        model: Model name. If None, uses get_llm_model() or get_agent_model().
        temperature: Temperature setting. If None, uses LLM_TEMPERATURE or AGENT_TEMPERATURE.
        use_agent_model: If True, use agent-specific model and temperature settings.
        
    Returns:
        LangChain LLM instance (ChatOpenAI or ChatAnthropic)
        
    Raises:
        ValueError: If API key is not found for the selected provider
        ImportError: If required LangChain packages are not installed
    """
    if provider is None:
        provider = API_PROVIDER
    
    provider = provider.lower()
    
    # Get model and temperature
    if model is None:
        if use_agent_model:
            model = get_agent_model(provider)
            temp = temperature if temperature is not None else AGENT_TEMPERATURE
        else:
            model = get_llm_model(provider)
            temp = temperature if temperature is not None else LLM_TEMPERATURE
    else:
        temp = temperature if temperature is not None else (AGENT_TEMPERATURE if use_agent_model else LLM_TEMPERATURE)
    
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic package is required for Anthropic support. "
                "Install with: pip install langchain-anthropic"
            )
        
        api_key = ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env file")
        
        return ChatAnthropic(
            model=model,
            temperature=temp,
            anthropic_api_key=api_key
        )
    
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai package is required for OpenAI support. "
                "Install with: pip install langchain-openai"
            )
        
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        return ChatOpenAI(
            model=model,
            temperature=temp,
            api_key=api_key
        )
    
    else:
        raise ValueError(f"Unknown API provider: {provider}. Use 'openai' or 'anthropic'.")

