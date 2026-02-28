"""
PDF extraction utilities.
"""

import PyPDF2
from pathlib import Path
from typing import List


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string (cleaned)
        
    Raises:
        Exception: If PDF cannot be read
        ValueError: If no text could be extracted
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
    
    if not text.strip():
        raise ValueError(f"No text could be extracted from {pdf_path}")
    
    # Clean the extracted text
    from .text_utils import clean_text
    text = clean_text(text)
    
    return text


def extract_text_from_pdfs(pdf_paths: List[str]) -> dict:
    """
    Extract text from multiple PDF files.
    
    Args:
        pdf_paths: List of paths to PDF files
        
    Returns:
        Dictionary mapping file paths to extracted text
    """
    results = {}
    for pdf_path in pdf_paths:
        try:
            results[pdf_path] = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"Warning: Failed to extract text from {pdf_path}: {e}")
            results[pdf_path] = ""
    return results
