"""
Regenerate embeddings using local CPU-based models and rebuild FAISS databases.

This script:
1. Reads existing embedding JSON files (to extract text chunks)
2. Regenerates embeddings using local sentence-transformers model
3. Saves new embedding JSON files
4. Rebuilds FAISS databases with the new embeddings

Usage:
    python backend/processing/vector/regenerate_with_local_embeddings.py
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "retrieval"))
from backend.retrieval.utils.local_embeddings import (
    generate_local_embedding,
    LOCAL_EMBEDDING_MODEL,
    get_local_embedding_model,
)

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import save_embeddings

# Import FAISS building functions from shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "indexing" / "faiss" / "utils"))
from faiss_builder import build_faiss_index


def load_existing_embeddings(json_path: Path) -> tuple:
    """
    Load text chunks from existing embedding JSON file.
    
    Args:
        json_path: Path to existing embeddings JSON file
        
    Returns:
        Tuple of (text_chunks, metadata)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    chunks = data.get('chunks', [])
    
    # Extract text chunks
    text_chunks = [chunk.get('text', '') for chunk in chunks]
    
    return text_chunks, metadata


def regenerate_embeddings_with_local_model(
    text_chunks: List[str],
    model_name: str = None
) -> List[List[float]]:
    """
    Regenerate embeddings using local CPU-based model.
    
    Args:
        text_chunks: List of text chunks to embed
        model_name: Local model name (defaults to LOCAL_EMBEDDING_MODEL)
        
    Returns:
        List of embedding vectors
    """
    if model_name is None:
        model_name = LOCAL_EMBEDDING_MODEL
    
    print(f"Generating embeddings using local model: {model_name}")
    
    embeddings = []
    model = get_local_embedding_model(model_name)
    
    if model is None:
        raise ValueError(f"Failed to load local embedding model: {model_name}")
    
    # Process in batches for efficiency
    batch_size = 32  # Smaller batches for CPU processing
    total = len(text_chunks)
    
    for i in range(0, total, batch_size):
        batch = text_chunks[i:i + batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({i+1}-{min(i+batch_size, total)}/{total})")
        
        try:
            # Generate embeddings for batch
            batch_embeddings = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Convert to list of lists
            for emb in batch_embeddings:
                embeddings.append(emb.tolist())
                
        except Exception as e:
            print(f"  Error processing batch: {e}")
            raise
    
    print(f"Generated {len(embeddings)} embeddings")
    return embeddings


def save_new_embeddings(
    text_chunks: List[str],
    embeddings: List[List[float]],
    metadata: Dict,
    output_path: Path
):
    """
    Save regenerated embeddings to JSON file.
    
    Args:
        text_chunks: List of text chunks
        embeddings: List of embedding vectors
        metadata: Metadata dictionary
        output_path: Path to save the new embeddings file
    """
    # Update metadata with new model info
    metadata['original_model'] = metadata.get('embedding_model', 'unknown')
    metadata['embedding_model'] = LOCAL_EMBEDDING_MODEL
    metadata['embedding_dimension'] = len(embeddings[0]) if embeddings else 0
    metadata['regenerated'] = True
    
    # Use shared save utility
    save_embeddings(
        text_chunks,
        embeddings,
        metadata,
        str(output_path.parent),
        output_path.name
    )


def regenerate_database_embeddings(
    database_name: str,
    input_dir: Path,
    output_dir: Path
):
    """
    Regenerate embeddings for a specific database.
    
    Args:
        database_name: Name of database ('company', 'aiid', or 'standards')
        input_dir: Directory containing existing embedding JSON files
        output_dir: Directory to save regenerated embeddings
    """
    print(f"\n{'='*60}")
    print(f"Regenerating embeddings for: {database_name}")
    print(f"{'='*60}")
    
    # Find all embedding JSON files
    json_files = list(input_dir.glob("*_embeddings.json"))
    
    if not json_files:
        print(f"Warning: No embedding files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} embedding file(s)")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        
        # Load existing embeddings to get text chunks
        text_chunks, metadata = load_existing_embeddings(json_file)
        
        if not text_chunks:
            print(f"  Warning: No text chunks found in {json_file.name}")
            continue
        
        print(f"  Found {len(text_chunks)} text chunks")
        
        # Regenerate embeddings with local model
        try:
            embeddings = regenerate_embeddings_with_local_model(text_chunks)
        except Exception as e:
            print(f"  ERROR: Failed to generate embeddings: {e}")
            continue
        
        # Save new embeddings
        output_file = output_dir / json_file.name
        save_new_embeddings(text_chunks, embeddings, metadata, output_file)
    
    print(f"\n✓ Completed regenerating embeddings for {database_name}")


def rebuild_faiss_database(
    database_name: str,
    embeddings_dir: Path,
    faiss_output_dir: Path
):
    """
    Rebuild FAISS database from regenerated embeddings.
    
    Args:
        database_name: Name of database ('company', 'aiid', or 'standards')
        embeddings_dir: Directory containing regenerated embedding JSON files
        faiss_output_dir: Directory to save FAISS index
    """
    print(f"\n{'='*60}")
    print(f"Rebuilding FAISS database for: {database_name}")
    print(f"{'='*60}")
    
    index_name = f"{database_name}_faiss_index"
    
    try:
        # Use shared build_cosine_index function for cosine similarity
        build_cosine_index(
            str(embeddings_dir),
            str(faiss_output_dir),
            index_name
        )
        
        print(f"\n✓ FAISS database rebuilt for {database_name}")
    except Exception as e:
        print(f"ERROR: Failed to rebuild FAISS database: {e}")
        raise


def main():
    """Main function to regenerate all embeddings and rebuild FAISS databases."""
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Paths
    processed_dir = project_root / "backend" / "processed" / "vector"
    faiss_dir = project_root / "backend" / "indexing" / "faiss"
    
    # Create temporary directory for regenerated embeddings
    regenerated_dir = processed_dir / "regenerated_local"
    regenerated_dir.mkdir(parents=True, exist_ok=True)
    
    databases = {
        'company': {
            'input': processed_dir / "company",
            'output': regenerated_dir / "company",
            'faiss_output': faiss_dir / "output"
        },
        'aiid': {
            'input': processed_dir / "aiid",
            'output': regenerated_dir / "aiid",
            'faiss_output': faiss_dir / "output"
        },
        'standards': {
            'input': processed_dir / "standards",
            'output': regenerated_dir / "standards",
            'faiss_output': faiss_dir / "output"
        }
    }
    
    print("="*60)
    print("REGENERATING EMBEDDINGS WITH LOCAL MODEL")
    print("="*60)
    print(f"Local Model: {LOCAL_EMBEDDING_MODEL}")
    print(f"Output Directory: {regenerated_dir}")
    print("="*60)
    
    # Process each database
    for db_name, paths in databases.items():
        if not paths['input'].exists():
            print(f"\nSkipping {db_name}: input directory not found")
            continue
        
        try:
            # Step 1: Regenerate embeddings
            regenerate_database_embeddings(
                db_name,
                paths['input'],
                paths['output']
            )
            
            # Step 2: Rebuild FAISS database
            rebuild_faiss_database(
                db_name,
                paths['output'],
                paths['faiss_output']
            )
            
        except Exception as e:
            print(f"\nERROR processing {db_name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("REGENERATION COMPLETE")
    print("="*60)
    print(f"\nNew embeddings saved to: {regenerated_dir}")
    print(f"FAISS databases rebuilt in: {faiss_dir}")
    print("\nNext steps:")
    print("1. Update your .env to use local embeddings:")
    print("   USE_LOCAL_EMBEDDINGS=true")
    print("2. Test vector search - it should now work!")


if __name__ == "__main__":
    main()

