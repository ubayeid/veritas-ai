"""
Model Switching Utilities
Helper functions to easily switch between different model configurations.
"""

from typing import Dict, Optional
from .model_config import (
    print_model_config,
    get_model_config,
    API_PROVIDER,
    EMBEDDING_MODE,
    EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    LLM_MODEL,
)


def switch_to_anthropic_llm():
    """Switch LLM to Anthropic Claude while keeping embeddings."""
    print("\n" + "="*80)
    print("SWITCHING TO ANTHROPIC CLAUDE FOR LLM")
    print("="*80)
    print("\nTo use Anthropic Claude for answer generation:")
    print("1. Set API_PROVIDER=anthropic in .env")
    print("2. Set ANTHROPIC_API_KEY=your_anthropic_key in .env")
    print("3. Set ANTHROPIC_LLM_MODEL=claude-3-5-sonnet-20241022 in .env")
    print("\nEmbeddings will use OpenAI (recommended)")
    print("="*80 + "\n")


def switch_to_local_embeddings():
    """Switch embeddings to local CPU-based models."""
    print("\n" + "="*80)
    print("SWITCHING TO LOCAL EMBEDDINGS")
    print("="*80)
    print("\nTo use local embeddings:")
    print("1. Set EMBEDDING_MODE=local in .env")
    print("2. Set LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2 (or other model) in .env")
    print("\nAvailable local models:")
    print("  - all-MiniLM-L6-v2 (384 dim, fast) ⭐ Recommended")
    print("  - all-mpnet-base-v2 (768 dim, better quality)")
    print("  - all-MiniLM-L12-v2 (384 dim)")
    print("  - BAAI/bge-small-en-v1.5 (384 dim)")
    print("  - BAAI/bge-base-en-v1.5 (768 dim)")
    print("="*80 + "\n")


def switch_to_openai_all():
    """Switch everything to OpenAI."""
    print("\n" + "="*80)
    print("SWITCHING TO OPENAI FOR ALL MODELS")
    print("="*80)
    print("\nTo use OpenAI for everything:")
    print("1. Set API_PROVIDER=openai in .env")
    print("2. Set EMBEDDING_MODE=api in .env")
    print("3. Set EMBEDDING_MODEL=text-embedding-3-small in .env")
    print("4. Set LLM_MODEL=gpt-4o-mini (or gpt-4, gpt-4-turbo, gpt-4o) in .env")
    print("="*80 + "\n")


def show_current_config():
    """Show current model configuration."""
    print_model_config()


def get_quick_switch_guide() -> Dict[str, str]:
    """
    Get a quick reference guide for common model switching scenarios.
    
    Returns:
        Dictionary with scenario names and .env configurations
    """
    return {
        "anthropic_llm_openai_embeddings": """
# Anthropic Claude for LLM, OpenAI for embeddings
API_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_LLM_MODEL=claude-3-5-sonnet-20241022
EMBEDDING_MODE=api
EMBEDDING_MODEL=text-embedding-3-small
USE_OPENAI_FOR_EMBEDDINGS=true
        """,
        "local_embeddings_openai_llm": """
# Local embeddings, OpenAI LLM
EMBEDDING_MODE=local
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
API_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
        """,
        "local_embeddings_anthropic_llm": """
# Local embeddings, Anthropic Claude LLM
EMBEDDING_MODE=local
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
API_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_LLM_MODEL=claude-3-5-sonnet-20241022
        """,
        "openai_all": """
# Everything OpenAI
API_PROVIDER=openai
EMBEDDING_MODE=api
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
        """,
    }


if __name__ == "__main__":
    """CLI tool for model configuration."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "show":
            show_current_config()
        elif command == "anthropic":
            switch_to_anthropic_llm()
        elif command == "local":
            switch_to_local_embeddings()
        elif command == "openai":
            switch_to_openai_all()
        elif command == "guide":
            guides = get_quick_switch_guide()
            print("\n" + "="*80)
            print("QUICK SWITCH GUIDE")
            print("="*80)
            for name, config in guides.items():
                print(f"\n{name.upper().replace('_', ' ')}:")
                print(config)
            print("="*80 + "\n")
        else:
            print(f"Unknown command: {command}")
            print("Usage: python model_switcher.py [show|anthropic|local|openai|guide]")
    else:
        show_current_config()
