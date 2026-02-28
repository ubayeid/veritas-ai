"""
Script to convert CSV files from a directory into embeddings using OpenAI API.
Processes all CSVs in a directory, converts rows to text, chunks text, generates embeddings, and saves them.

Environment Variables (.env file):
    OPENAI_API_KEY: Your OpenAI API key (required)
    EMBEDDING_MODEL: Embedding model to use (default: "text-embedding-3-small")
                     Options: "text-embedding-3-small", "text-embedding-3-large"
    CHUNK_SIZE: Size of text chunks in characters (default: 1000)
    CHUNK_OVERLAP: Overlap between chunks in characters (default: 200)

Example .env file:
    OPENAI_API_KEY=sk-...
    EMBEDDING_MODEL=text-embedding-3-small
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
"""

import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import shared utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import csv_to_text, chunk_text, get_embeddings, save_embeddings

# Import unified API client for model config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "retrieval"))
from backend.retrieval.utils.model_config import get_embedding_model

# Configuration - can be overridden via .env file
EMBEDDING_MODEL = get_embedding_model()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def process_csv_to_embeddings(
    csv_path: str,
    output_dir: str,
    embedding_model: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None
):
    """
    Process a single CSV file and generate embeddings.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save embeddings
        embedding_model: OpenAI embedding model to use (defaults to EMBEDDING_MODEL from .env)
        chunk_size: Size of chunks in characters (defaults to CHUNK_SIZE from .env)
        chunk_overlap: Overlap between chunks in characters (defaults to CHUNK_OVERLAP from .env)
    """
    # Use provided values or fall back to environment/defaults
    embedding_model = embedding_model or EMBEDDING_MODEL
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    
    csv_name = Path(csv_path).stem
    print(f"\n{'='*60}")
    print(f"Processing CSV: {csv_path}")
    print(f"Configuration: model={embedding_model}, chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    # Convert CSV to text
    print("Converting CSV to text...")
    try:
        text = csv_to_text(csv_path)
        print(f"Converted {len(text)} characters from CSV")
    except Exception as e:
        print(f"ERROR: Failed to convert CSV {csv_path}: {str(e)}")
        return False
    
    # Chunk text
    print(f"Chunking text (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings
    print(f"Generating embeddings using {embedding_model}...")
    try:
        embeddings = get_embeddings(chunks, embedding_model)
        print(f"Generated {len(embeddings)} embeddings")
    except Exception as e:
        print(f"ERROR: Failed to generate embeddings: {str(e)}")
        return False
    
    # Prepare metadata
    metadata = {
        "source_file": str(csv_path),
        "source_name": csv_name,
        "num_chunks": len(chunks),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_dimension": len(embeddings[0]) if embeddings else 0
    }
    
    # Save embeddings
    output_filename = f"{csv_name}_embeddings.json"
    save_embeddings(chunks, embeddings, metadata, output_dir, output_filename)
    
    print(f"[OK] Successfully processed {csv_name}")
    return True


def process_directory_to_embeddings(
    input_dir: str,
    output_dir: str,
    embedding_model: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None
):
    """
    Process all CSV files in a directory and generate embeddings.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save embeddings
        embedding_model: OpenAI embedding model to use (defaults to EMBEDDING_MODEL from .env)
        chunk_size: Size of chunks in characters (defaults to CHUNK_SIZE from .env)
        chunk_overlap: Overlap between chunks in characters (defaults to CHUNK_OVERLAP from .env)
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process")
    
    # Process each CSV
    successful = 0
    failed = 0
    
    for csv_file in csv_files:
        try:
            if process_csv_to_embeddings(
                str(csv_file),
                output_dir,
                embedding_model,
                chunk_size,
                chunk_overlap
            ):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"ERROR: Unexpected error processing {csv_file.name}: {str(e)}")
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {successful} file(s)")
    print(f"Failed: {failed} file(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert AIID CSVs to embeddings")
    parser.add_argument("--input-dir", type=str, help="Input directory containing CSVs")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--embedding-model", type=str, help="Embedding model to use")
    parser.add_argument("--chunk-size", type=int, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, help="Chunk overlap in characters")
    
    args = parser.parse_args()
    
    # Get project root and set paths relative to it
    project_root = Path(__file__).parent.parent.parent.parent
    
    input_dir = args.input_dir or str(project_root / "data" / "aiid")
    output_dir = args.output_dir or str(project_root / "backend" / "processed" / "vector" / "aiid")
    
    # Process all CSVs in the directory
    process_directory_to_embeddings(
        input_dir,
        output_dir,
        args.embedding_model,
        args.chunk_size,
        args.chunk_overlap
    )

