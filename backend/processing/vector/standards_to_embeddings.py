"""
Script to convert PDF files into embeddings using OpenAI API.
Processes PDFs, chunks text, generates embeddings, and saves them.
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
from utils import extract_text_from_pdf, chunk_text, get_embeddings, save_embeddings

# Import unified API client for model config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "retrieval"))
from backend.retrieval.utils.model_config import get_embedding_model

# Configuration - can be overridden via .env file
EMBEDDING_MODEL = get_embedding_model()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def process_pdf_to_embeddings(
    pdf_path: str,
    output_dir: str,
    embedding_model: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None
):
    """
    Main function to process PDF and generate embeddings.
    
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
    
    print(f"[OK] Successfully processed {pdf_name}")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert standards PDF to embeddings")
    parser.add_argument("--pdf-path", type=str, help="Path to PDF file")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--embedding-model", type=str, help="Embedding model to use")
    parser.add_argument("--chunk-size", type=int, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, help="Chunk overlap in characters")
    
    args = parser.parse_args()
    
    # Get project root and set paths relative to it
    project_root = Path(__file__).parent.parent.parent.parent
    
    pdf_path = args.pdf_path or str(project_root / "data" / "standards" / "gdpr.pdf")
    output_dir = args.output_dir or str(project_root / "backend" / "processed" / "vector" / "standards")
    
    # Process PDF
    success = process_pdf_to_embeddings(
        pdf_path,
        output_dir,
        args.embedding_model,
        args.chunk_size,
        args.chunk_overlap
    )
    
    if not success:
        exit(1)
