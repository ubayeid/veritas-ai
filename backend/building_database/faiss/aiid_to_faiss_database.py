"""
Script to build a FAISS vector database from embedding JSON files.
Combines all embeddings from a directory into a single searchable FAISS index.

Environment Variables (.env file):
    OPENAI_API_KEY: Your OpenAI API key (required for verification, not used for building)

The script reads all *_embeddings.json files from the input directory,
extracts embeddings and metadata, and creates a FAISS index.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_embeddings_from_json(json_path: Path) -> tuple:
    """
    Load embeddings and metadata from a JSON file.
    
    Args:
        json_path: Path to the embeddings JSON file
        
    Returns:
        Tuple of (embeddings array, metadata list)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata_info = data.get('metadata', {})
    chunks = data.get('chunks', [])
    
    if not chunks:
        return None, None
    
    # Extract embeddings and create metadata for each chunk
    embeddings = []
    metadata_list = []
    
    for chunk in chunks:
        embedding = chunk.get('embedding')
        if embedding:
            embeddings.append(embedding)
            # Store metadata for this chunk
            chunk_metadata = {
                'chunk_id': chunk.get('chunk_id'),
                'text': chunk.get('text'),
                'source_file': metadata_info.get('source_file'),
                'source_name': metadata_info.get('source_name'),
                'embedding_file': str(json_path.name)
            }
            metadata_list.append(chunk_metadata)
    
    if not embeddings:
        return None, None
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype='float32')
    
    return embeddings_array, metadata_list


def build_faiss_index(
    embeddings_dir: str,
    output_dir: str,
    index_name: str = "faiss_index"
):
    """
    Build a FAISS index from all embedding JSON files in a directory.
    
    Args:
        embeddings_dir: Directory containing *_embeddings.json files
        output_dir: Directory to save the FAISS index and metadata
        index_name: Base name for the output files
    """
    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.exists():
        raise ValueError(f"Embeddings directory does not exist: {embeddings_dir}")
    
    # Find all embedding JSON files
    json_files = list(embeddings_path.glob("*_embeddings.json"))
    
    if not json_files:
        raise ValueError(f"No embedding JSON files found in {embeddings_dir}")
    
    print(f"Found {len(json_files)} embedding file(s) to process")
    
    # Collect all embeddings and metadata
    all_embeddings = []
    all_metadata = []
    
    for json_file in json_files:
        print(f"Loading embeddings from {json_file.name}...")
        embeddings_array, metadata_list = load_embeddings_from_json(json_file)
        
        if embeddings_array is not None and metadata_list is not None:
            all_embeddings.append(embeddings_array)
            all_metadata.extend(metadata_list)
            print(f"  Loaded {len(metadata_list)} chunks")
        else:
            print(f"  Warning: No valid embeddings found in {json_file.name}")
    
    if not all_embeddings:
        raise ValueError("No embeddings were loaded from any files")
    
    # Combine all embeddings
    print(f"\nCombining {len(all_embeddings)} embedding arrays...")
    combined_embeddings = np.vstack(all_embeddings)
    
    dimension = combined_embeddings.shape[1]
    num_vectors = combined_embeddings.shape[0]
    
    print(f"Total vectors: {num_vectors}")
    print(f"Embedding dimension: {dimension}")
    
    # Create FAISS index
    # Using L2 (Euclidean) distance - can be changed to cosine similarity if needed
    print("\nBuilding FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to index
    index.add(combined_embeddings)
    
    print(f"Index built with {index.ntotal} vectors")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_file = output_path / f"{index_name}.index"
    faiss.write_index(index, str(index_file))
    print(f"Saved FAISS index to {index_file}")
    
    # Save metadata
    metadata_file = output_path / f"{index_name}_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(all_metadata, f)
    print(f"Saved metadata to {metadata_file}")
    
    # Save summary info as JSON
    summary = {
        "num_vectors": num_vectors,
        "dimension": dimension,
        "index_type": "IndexFlatL2",
        "distance_metric": "L2 (Euclidean)",
        "source_files": [str(f.name) for f in json_files],
        "num_source_files": len(json_files)
    }
    
    summary_file = output_path / f"{index_name}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")
    
    print(f"\n{'='*60}")
    print("FAISS database build complete!")
    print(f"{'='*60}")
    
    return index, all_metadata


def build_cosine_index(
    embeddings_dir: str,
    output_dir: str,
    index_name: str = "faiss_index_cosine"
):
    """
    Build a FAISS index optimized for cosine similarity.
    Normalizes embeddings and uses inner product for cosine similarity.
    
    Args:
        embeddings_dir: Directory containing *_embeddings.json files
        output_dir: Directory to save the FAISS index and metadata
        index_name: Base name for the output files
    """
    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.exists():
        raise ValueError(f"Embeddings directory does not exist: {embeddings_dir}")
    
    # Find all embedding JSON files
    json_files = list(embeddings_path.glob("*_embeddings.json"))
    
    if not json_files:
        raise ValueError(f"No embedding JSON files found in {embeddings_dir}")
    
    print(f"Found {len(json_files)} embedding file(s) to process")
    
    # Collect all embeddings and metadata
    all_embeddings = []
    all_metadata = []
    
    for json_file in json_files:
        print(f"Loading embeddings from {json_file.name}...")
        embeddings_array, metadata_list = load_embeddings_from_json(json_file)
        
        if embeddings_array is not None and metadata_list is not None:
            all_embeddings.append(embeddings_array)
            all_metadata.extend(metadata_list)
            print(f"  Loaded {len(metadata_list)} chunks")
        else:
            print(f"  Warning: No valid embeddings found in {json_file.name}")
    
    if not all_embeddings:
        raise ValueError("No embeddings were loaded from any files")
    
    # Combine all embeddings
    print(f"\nCombining {len(all_embeddings)} embedding arrays...")
    combined_embeddings = np.vstack(all_embeddings)
    
    dimension = combined_embeddings.shape[1]
    num_vectors = combined_embeddings.shape[0]
    
    print(f"Total vectors: {num_vectors}")
    print(f"Embedding dimension: {dimension}")
    
    # Normalize embeddings for cosine similarity
    print("\nNormalizing embeddings for cosine similarity...")
    faiss.normalize_L2(combined_embeddings)
    
    # Create FAISS index for inner product (cosine similarity on normalized vectors)
    print("Building FAISS index (cosine similarity)...")
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to index
    index.add(combined_embeddings)
    
    print(f"Index built with {index.ntotal} vectors")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_file = output_path / f"{index_name}.index"
    faiss.write_index(index, str(index_file))
    print(f"Saved FAISS index to {index_file}")
    
    # Save metadata
    metadata_file = output_path / f"{index_name}_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(all_metadata, f)
    print(f"Saved metadata to {metadata_file}")
    
    # Save summary info as JSON
    summary = {
        "num_vectors": num_vectors,
        "dimension": dimension,
        "index_type": "IndexFlatIP",
        "distance_metric": "Inner Product (Cosine Similarity)",
        "source_files": [str(f.name) for f in json_files],
        "num_source_files": len(json_files)
    }
    
    summary_file = output_path / f"{index_name}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")
    
    print(f"\n{'='*60}")
    print("FAISS database build complete!")
    print(f"{'='*60}")
    
    return index, all_metadata


if __name__ == "__main__":
    # Get project root and set paths relative to it
    # Script is in: backend/building_database/faiss/
    # So we need to go up 4 levels: faiss -> building_database -> backend -> project_root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    EMBEDDINGS_DIR = str(project_root / "backend" / "data_processing" / "processed" / "vector" / "aiid")
    OUTPUT_DIR = str(project_root / "backend" / "building_database" / "faiss" / "aiid")
    INDEX_NAME = "aiid_faiss_index"
    
    # Build FAISS index with cosine similarity (recommended for embeddings)
    print("Building FAISS index with cosine similarity...")
    build_cosine_index(EMBEDDINGS_DIR, OUTPUT_DIR, INDEX_NAME)
    
    # Uncomment below to also build L2 index
    # print("\nBuilding FAISS index with L2 distance...")
    # build_faiss_index(EMBEDDINGS_DIR, OUTPUT_DIR, f"{INDEX_NAME}_l2")

