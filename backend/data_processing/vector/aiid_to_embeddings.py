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
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import unified API client
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "searching"))
from api_client import get_embedding_client, get_embedding_model

# Initialize API client (supports both OpenAI and xAI)
def get_openai_client():
    """Get API client instance for embeddings (supports both OpenAI and xAI)."""
    return get_embedding_client()

# Configuration - can be overridden via .env file
EMBEDDING_MODEL = get_embedding_model()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def csv_to_text(csv_path: str) -> str:
    """
    Convert CSV file to text format.
    Each row is converted to a readable text format with column names.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Text representation of the CSV data
    """
    text_rows = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as file:
            # Try to detect delimiter
            sample = file.read(1024)
            file.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(file, delimiter=delimiter)
            
            # Get column names
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise ValueError(f"No columns found in CSV file: {csv_path}")
            
            # Process each row
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (row 1 is header)
                # Create a text representation of the row
                row_text_parts = []
                for key, value in row.items():
                    if value and str(value).strip():  # Only include non-empty values
                        # Clean up the value
                        clean_value = str(value).strip().replace('\n', ' ').replace('\r', ' ')
                        row_text_parts.append(f"{key}: {clean_value}")
                
                if row_text_parts:
                    row_text = f"Row {row_num}: " + " | ".join(row_text_parts)
                    text_rows.append(row_text)
    
    except Exception as e:
        raise Exception(f"Error reading CSV {csv_path}: {str(e)}")
    
    if not text_rows:
        raise ValueError(f"No data rows found in CSV file: {csv_path}")
    
    # Join all rows with newlines
    return "\n".join(text_rows)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def get_embeddings(text_chunks: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """
    Get embeddings for text chunks using OpenAI API.
    
    Args:
        text_chunks: List of text chunks to embed
        model: OpenAI embedding model to use
        
    Returns:
        List of embedding vectors
    """
    client = get_openai_client()
    embeddings = []
    
    # Process in batches to avoid rate limits
    batch_size = 100
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
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    return embeddings


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
    
    print(f"✓ Successfully processed {csv_name}")
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
    # Get project root and set paths relative to it
    project_root = Path(__file__).parent.parent.parent.parent
    INPUT_DIR = str(project_root / "data" / "aiid")
    OUTPUT_DIR = str(project_root / "backend" / "data_processing" / "processed" / "vector" / "aiid")
    
    # Process all CSVs in the directory
    process_directory_to_embeddings(INPUT_DIR, OUTPUT_DIR)

