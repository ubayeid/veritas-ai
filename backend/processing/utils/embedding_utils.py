"""
Embedding generation utilities.
"""

import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Add project root to path before importing backend modules
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import unified API client and local embeddings
from backend.retrieval.utils.api_client import get_embedding_client
from backend.retrieval.utils.model_config import get_embedding_model, get_embedding_mode
from backend.retrieval.utils.local_embeddings import generate_local_embeddings_batch


def get_embeddings(
    text_chunks: List[str],
    model: str = None
) -> List[List[float]]:
    """
    Get embeddings for text chunks using API or local models based on configuration.
    
    Args:
        text_chunks: List of text chunks to embed
        model: Embedding model to use (defaults to configured model from .env)
        
    Returns:
        List of embedding vectors
        
    Raises:
        Exception: If embedding generation fails
    """
    mode = get_embedding_mode()
    
    # Use local embeddings if mode is "local"
    if mode == "local":
        if model is None:
            model = get_embedding_model()
        
        # Generate embeddings using local model
        embeddings_array = generate_local_embeddings_batch(text_chunks, model)
        if embeddings_array is None:
            raise Exception("Failed to generate local embeddings")
        
        # Convert numpy array to list of lists
        return embeddings_array.tolist()
    
    # Use API embeddings
    if model is None:
        model = get_embedding_model()
    
    client = get_embedding_client()
    
    embeddings = []
    batch_size = 100  # Process in batches
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        
        try:
            response = client.embeddings.create(
                model=model,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
        except Exception as e:
            raise Exception(f"Error generating embeddings for batch {i//batch_size + 1}: {str(e)}")
    
    return embeddings
