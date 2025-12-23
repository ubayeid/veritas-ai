"""
Script to convert PDF files from a directory into embeddings using OpenAI API.
Processes all PDFs in a directory, chunks text, generates embeddings, and saves them.

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
import json
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2

# Load environment variables
load_dotenv()

# Import unified API client
import sys
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


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
    
    if not text.strip():
        raise ValueError(f"No text could be extracted from {pdf_path}")
    
    return text


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


def process_pdf_to_embeddings(
    pdf_path: str,
    output_dir: str,
    embedding_model: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None
):
    """
    Process a single PDF file and generate embeddings.
    
    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory to save embeddings
        embedding_model: OpenAI embedding model to use (defaults to EMBEDDING_MODEL from .env)
        chunk_size: Size of chunks in characters (defaults to CHUNK_SIZE from .env)
        chunk_overlap: Overlap between chunks in characters (defaults to CHUNK_OVERLAP from .env)
    """
    # Use provided values or fall back to environment/defaults
    embedding_model = embedding_model or EMBEDDING_MODEL
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    
    pdf_name = Path(pdf_path).stem
    print(f"\n{'='*60}")
    print(f"Processing PDF: {pdf_path}")
    print(f"Configuration: model={embedding_model}, chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    # Extract text from PDF
    print("Extracting text from PDF...")
    try:
        text = extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(text)} characters")
    except Exception as e:
        print(f"ERROR: Failed to extract text from {pdf_path}: {str(e)}")
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
        "source_file": str(pdf_path),
        "source_name": pdf_name,
        "num_chunks": len(chunks),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_dimension": len(embeddings[0]) if embeddings else 0
    }
    
    # Save embeddings
    output_filename = f"{pdf_name}_embeddings.json"
    save_embeddings(chunks, embeddings, metadata, output_dir, output_filename)
    
    print(f"✓ Successfully processed {pdf_name}")
    return True


def process_directory_to_embeddings(
    input_dir: str,
    output_dir: str,
    embedding_model: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None
):
    """
    Process all PDF files in a directory and generate embeddings.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save embeddings
        embedding_model: OpenAI embedding model to use (defaults to EMBEDDING_MODEL from .env)
        chunk_size: Size of chunks in characters (defaults to CHUNK_SIZE from .env)
        chunk_overlap: Overlap between chunks in characters (defaults to CHUNK_OVERLAP from .env)
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    
    # Process each PDF
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        try:
            if process_pdf_to_embeddings(
                str(pdf_file),
                output_dir,
                embedding_model,
                chunk_size,
                chunk_overlap
            ):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"ERROR: Unexpected error processing {pdf_file.name}: {str(e)}")
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
    INPUT_DIR = str(project_root / "data" / "company")
    OUTPUT_DIR = str(project_root / "backend" / "data_processing" / "processed" / "vector" / "company")
    
    # Process all PDFs in the directory
    process_directory_to_embeddings(INPUT_DIR, OUTPUT_DIR)

