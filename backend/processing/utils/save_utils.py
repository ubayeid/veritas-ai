"""
Utilities for saving processed data.
"""

import json
from pathlib import Path
from typing import List, Dict


def save_embeddings(
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: Dict,
    output_dir: str,
    filename: str = "embeddings.json"
):
    """
    Save embeddings and associated metadata to JSON file.
    
    Args:
        chunks: List of text chunks
        embeddings: List of embedding vectors
        metadata: Metadata dictionary
        output_dir: Directory to save embeddings
        filename: Output filename
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data structure
    data = {
        "metadata": metadata,
        "chunks": [
            {
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
    }
    
    # Save to JSON
    output_path = Path(output_dir) / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} embeddings to {output_path}")


def save_graph_json(
    graph_data: Dict,
    output_path: Path
):
    """
    Save graph-structured data to JSON file.
    
    Args:
        graph_data: Dictionary containing graph nodes and relationships
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved graph data to {output_path}")
