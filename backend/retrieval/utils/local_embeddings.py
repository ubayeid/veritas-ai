"""
Local CPU-based Embedding Models
Fallback when OpenAI API is unavailable (quota exceeded, no API key, etc.)
Uses sentence-transformers for CPU-based embeddings.
"""

import os
import numpy as np
from typing import Optional, List
import warnings
import sys
from io import StringIO

# Suppress HuggingFace warnings by default (set BEFORE any imports)
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', '1')

# Suppress warnings globally
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*HF Hub.*')
warnings.filterwarnings('ignore', message='.*BertModel.*')
warnings.filterwarnings('ignore', message='.*unauthenticated.*')
warnings.filterwarnings('ignore', message='.*UNEXPECTED.*')

from dotenv import load_dotenv
load_dotenv()

# Import centralized configuration
from .model_config import (
    LOCAL_EMBEDDING_MODEL,
    USE_LOCAL_EMBEDDINGS,
    get_embedding_mode,
)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")


# Cache the model to avoid reloading
_cached_model = None
_cached_model_name = None


def get_local_embedding_model(model_name: Optional[str] = None) -> Optional['SentenceTransformer']:
    """
    Get or load a local embedding model.
    
    Args:
        model_name: Model name (if None, uses LOCAL_EMBEDDING_MODEL from env)
        
    Returns:
        SentenceTransformer model instance or None if not available
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    
    if model_name is None:
        model_name = LOCAL_EMBEDDING_MODEL
    
    global _cached_model, _cached_model_name
    
    # Return cached model if it's the same
    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model
    
    try:
        verbose = os.getenv("VERBOSE", "false").lower() == "true"
        
        if not verbose:
            # Suppress stdout/stderr during model loading to hide all output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
        
        if verbose:
            print(f"Loading local embedding model: {model_name}")
        
        try:
            # Suppress transformers logging (CRITICAL level suppresses everything)
            import logging
            logging.getLogger("transformers").setLevel(logging.CRITICAL)
            logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)
            logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
            logging.getLogger("transformers.modeling_utils").setLevel(logging.CRITICAL)
            logging.getLogger("transformers.configuration_utils").setLevel(logging.CRITICAL)
            
            # Load model (all warnings/output suppressed if not verbose)
            _cached_model = SentenceTransformer(model_name)
        finally:
            if not verbose:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        _cached_model_name = model_name
        if verbose:
            print(f"Local embedding model loaded successfully")
        return _cached_model
    except Exception as e:
        verbose = os.getenv("VERBOSE", "false").lower() == "true"
        if verbose:
            print(f"Warning: Failed to load local embedding model {model_name}: {e}")
        return None


def generate_local_embedding(text: str, model_name: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Generate embedding using local CPU-based model.
    
    Args:
        text: Text to embed
        model_name: Model name (if None, uses LOCAL_EMBEDDING_MODEL from env)
        
    Returns:
        Embedding as numpy array (normalized) or None if failed
    """
    model = get_local_embedding_model(model_name)
    if model is None:
        return None
    
    try:
        # Generate embedding
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        
        # Ensure it's float32 and reshape for FAISS
        embedding = np.array(embedding, dtype='float32')
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        return embedding
    except Exception as e:
        print(f"Error generating local embedding: {e}")
        return None


def generate_local_embeddings_batch(texts: List[str], model_name: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Generate embeddings for multiple texts in batch (much faster than one-by-one).
    
    Args:
        texts: List of texts to embed
        model_name: Model name (if None, uses LOCAL_EMBEDDING_MODEL from env)
        
    Returns:
        Embeddings as numpy array (normalized, shape: [len(texts), embedding_dim]) or None if failed
    """
    if not texts:
        return None
    
    model = get_local_embedding_model(model_name)
    if model is None:
        return None
    
    try:
        # Batch encode all texts at once (much faster!)
        # Suppress progress bar unless verbose
        verbose = os.getenv("VERBOSE", "false").lower() == "true"
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=verbose)
        
        # Ensure it's float32
        embeddings = np.array(embeddings, dtype='float32')
        
        # If single text, reshape to [1, dim]
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        return embeddings
    except Exception as e:
        print(f"Error generating batch local embeddings: {e}")
        return None


def is_local_embeddings_enabled() -> bool:
    """
    Check if local embeddings should be used.
    
    Returns:
        True if local embeddings are enabled
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return False
    
    return get_embedding_mode() == "local"


def get_available_local_models() -> list:
    """
    Get list of recommended local embedding models.
    
    Returns:
        List of model names
    """
    return [
        "all-MiniLM-L6-v2",           # 384 dim, fast, good quality
        "all-mpnet-base-v2",           # 768 dim, better quality, slower
        "all-MiniLM-L12-v2",           # 384 dim, better than L6
        "BAAI/bge-small-en-v1.5",      # 384 dim, good quality
        "BAAI/bge-base-en-v1.5",      # 768 dim, high quality
    ]

