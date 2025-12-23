"""
Script to convert PDF files into embeddings using OpenAI API.
Processes PDFs, chunks text, generates embeddings, and saves them.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2

# Load environment variables
load_dotenv()

# Import unified API client
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "searching"))
from api_client import get_embedding_client, get_embedding_model

# Initialize API client (supports both OpenAI and xAI)
client = get_embedding_client()

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
    embedding_model: str = EMBEDDING_MODEL
):
    """
    Main function to process PDF and generate embeddings.
    
    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory to save embeddings
        embedding_model: OpenAI embedding model to use
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Extract text from PDF
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters")
    
    # Chunk text
    print(f"Chunking text (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings
    print(f"Generating embeddings using {embedding_model}...")
    embeddings = get_embeddings(chunks, embedding_model)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Prepare metadata
    pdf_name = Path(pdf_path).stem
    metadata = {
        "source_file": pdf_path,
        "source_name": pdf_name,
        "num_chunks": len(chunks),
        "embedding_model": embedding_model,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_dimension": len(embeddings[0]) if embeddings else 0
    }
    
    # Save embeddings
    output_filename = f"{pdf_name}_embeddings.json"
    save_embeddings(chunks, embeddings, metadata, output_dir, output_filename)
    
    print("Processing complete!")


if __name__ == "__main__":
    # Get project root and set paths relative to it
    project_root = Path(__file__).parent.parent.parent.parent
    PDF_PATH = str(project_root / "data" / "standards" / "gdpr.pdf")
    OUTPUT_DIR = str(project_root / "backend" / "data_processing" / "processed" / "vector" / "standards")
    
    # Process PDF
    process_pdf_to_embeddings(PDF_PATH, OUTPUT_DIR)

