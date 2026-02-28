"""
Shared utilities for building FAISS indexes.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss


def load_embeddings_from_json(json_path: Path) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
    """
    Load embeddings and metadata from a JSON file.
    
    Args:
        json_path: Path to the embeddings JSON file
        
    Returns:
        Tuple of (embeddings array, metadata list) or (None, None) if no valid data
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
    index_name: str = "faiss_index",
    metric: str = "L2"
) -> Tuple[faiss.Index, List[Dict]]:
    """
    Build a FAISS index from all embedding JSON files in a directory.
    
    Args:
        embeddings_dir: Directory containing *_embeddings.json files
        output_dir: Directory to save the FAISS index and metadata
        index_name: Base name for the output files
        metric: Distance metric ("L2" or "IP" for inner product)
        
    Returns:
        Tuple of (FAISS index, metadata list)
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
    print(f"\nBuilding FAISS index ({metric} distance)...")
    if metric == "L2":
        index = faiss.IndexFlatL2(dimension)
    elif metric == "IP":
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'L2' or 'IP'")
    
    # Add vectors to index
    index.add(combined_embeddings)
    
    print(f"Index built with {index.ntotal} vectors")
    
    # Save index and metadata
    save_faiss_index(index, all_metadata, output_dir, index_name, metric, json_files)
    
    return index, all_metadata


def build_cosine_index(
    embeddings_dir: str,
    output_dir: str,
    index_name: str = "faiss_index_cosine"
) -> Tuple[faiss.Index, List[Dict]]:
    """
    Build a FAISS index optimized for cosine similarity.
    Normalizes embeddings and uses inner product for cosine similarity.
    
    Args:
        embeddings_dir: Directory containing *_embeddings.json files
        output_dir: Directory to save the FAISS index and metadata
        index_name: Base name for the output files
        
    Returns:
        Tuple of (FAISS index, metadata list)
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
    
    # Save index and metadata
    save_faiss_index(index, all_metadata, output_dir, index_name, "IP", json_files, normalized=True)
    
    return index, all_metadata


def save_faiss_index(
    index: faiss.Index,
    metadata: List[Dict],
    output_dir: str,
    index_name: str,
    metric: str,
    source_files: List[Path],
    normalized: bool = False
):
    """
    Save FAISS index, metadata, and summary to disk.
    
    Args:
        index: FAISS index object
        metadata: List of metadata dictionaries
        output_dir: Directory to save files
        index_name: Base name for output files
        metric: Distance metric used
        source_files: List of source JSON files
        normalized: Whether embeddings were normalized
    """
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
        pickle.dump(metadata, f)
    print(f"Saved metadata to {metadata_file}")
    
    # Save summary info as JSON
    distance_metric = "Inner Product (Cosine Similarity)" if normalized else (
        "L2 (Euclidean)" if metric == "L2" else "Inner Product"
    )
    
    summary = {
        "num_vectors": index.ntotal,
        "dimension": index.d,
        "index_type": "IndexFlatIP" if metric == "IP" else "IndexFlatL2",
        "distance_metric": distance_metric,
        "normalized": normalized,
        "source_files": [str(f.name) for f in source_files],
        "num_source_files": len(source_files)
    }
    
    summary_file = output_path / f"{index_name}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")
    
    print(f"\n{'='*60}")
    print("FAISS database build complete!")
    print(f"{'='*60}")


def load_faiss_index(index_path: str, metadata_path: str) -> Tuple[faiss.Index, List[Dict]]:
    """
    Load FAISS index and metadata from disk.
    
    Args:
        index_path: Path to FAISS index file
        metadata_path: Path to metadata pickle file
        
    Returns:
        Tuple of (FAISS index, metadata list)
    """
    index = faiss.read_index(index_path)
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata
