"""
Script to convert company documents (PDFs) into graph-structured data.
Extracts documents and clauses, then saves as JSON for Neo4j import.

This follows the same pattern as vector processing:
- Input: data/company/*.pdf
- Output: backend/data_processing/processed/graph/company_graph.json

Environment Variables (.env file):
    (None required - this is just data extraction)
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
from dotenv import load_dotenv

load_dotenv()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
    return text


def split_into_clauses(text: str, min_length: int = 50) -> List[str]:
    """
    Split text into clauses (sentences or logical units).
    
    Args:
        text: Text to split
        min_length: Minimum length for a clause
        
    Returns:
        List of clause texts
    """
    clauses = []
    
    # Split by sentences
    sentences = re.split(r'[.!?]+\s+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) >= min_length:
            clauses.append(sentence)
    
    # Also try splitting by newlines (for structured documents)
    if len(clauses) < 5:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) >= min_length and line:
                clauses.append(line)
    
    return clauses


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from clause text."""
    keywords = [
        'data', 'personal', 'information', 'privacy', 'collect', 'use', 'share',
        'consent', 'right', 'access', 'delete', 'security', 'protection', 'policy',
        'cookies', 'tracking', 'advertising', 'third party', 'user', 'account'
    ]
    
    text_lower = text.lower()
    found_keywords = [kw for kw in keywords if kw in text_lower]
    
    words = re.findall(r'\b[a-z]{4,}\b', text_lower)
    meaningful_words = [w for w in set(words) if len(w) > 4][:5]
    
    return list(set(found_keywords + meaningful_words))


def map_file_to_document_name(file_path: str) -> str:
    """
    Map file path to document name.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Document name
    """
    file_lower = file_path.lower()
    
    if 'privacy' in file_lower:
        return 'Privacy Policy'
    elif 'terms' in file_lower or 'tos' in file_lower:
        return 'Terms of Service'
    elif 'cookie' in file_lower:
        return 'Cookie Policy'
    else:
        # Use file name as document name
        return Path(file_path).stem


def get_document_url(document_name: str) -> str:
    """Get source URL for document."""
    urls = {
        'Privacy Policy': 'https://www.facebook.com/privacy/policy',
        'Terms of Service': 'https://www.facebook.com/legal/terms',
        'Cookie Policy': 'https://www.facebook.com/policies/cookies'
    }
    return urls.get(document_name, '')


def process_pdf_to_graph(pdf_path: str) -> Dict:
    """
    Process a PDF file and extract graph-structured data.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with document and clauses data
    """
    print(f"Processing: {pdf_path}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    # Split into clauses
    clauses_text = split_into_clauses(text)
    
    # Map to document name
    document_name = map_file_to_document_name(pdf_path)
    source_url = get_document_url(document_name)
    
    # Create clauses with IDs
    clauses = []
    for i, clause_text in enumerate(clauses_text):
        clause_id = f"{document_name}_clause_{i+1}"
        keywords = extract_keywords(clause_text)
        
        clauses.append({
            'id': clause_id,
            'text': clause_text,
            'document_name': document_name,
            'keywords': keywords,
            'section': ''  # Could be extracted from PDF structure
        })
    
    return {
        'document': {
            'name': document_name,
            'source_url': source_url,
            'source_file': str(pdf_path)
        },
        'clauses': clauses
    }


def save_graph_data(documents_data: List[Dict], output_path: str):
    """
    Save graph-structured data to JSON file.
    
    Args:
        documents_data: List of document dictionaries with clauses
        output_path: Path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten structure for easier Neo4j import
    all_clauses = []
    all_documents = []
    
    for doc_data in documents_data:
        all_documents.append(doc_data['document'])
        all_clauses.extend(doc_data['clauses'])
    
    graph_data = {
        'metadata': {
            'source': 'Company Documents',
            'num_documents': len(all_documents),
            'num_clauses': len(all_clauses),
            'version': '1.0'
        },
        'documents': all_documents,
        'clauses': all_clauses
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_documents)} documents and {len(all_clauses)} clauses to {output_file}")


def process_directory_to_graph(input_dir: str = None, output_dir: str = None):
    """
    Process all PDF files in a directory and save graph-structured data.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save graph data
    """
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Default paths
    if not input_dir:
        input_dir = str(project_root / "data" / "company")
    
    if not output_dir:
        output_dir = str(project_root / "backend" / "data_processing" / "processed" / "graph")
    
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
    documents_data = []
    for pdf_file in pdf_files:
        try:
            doc_data = process_pdf_to_graph(str(pdf_file))
            documents_data.append(doc_data)
        except Exception as e:
            print(f"ERROR: Failed to process {pdf_file.name}: {str(e)}")
    
    # Save graph data
    output_path = Path(output_dir) / "company_graph.json"
    print(f"\nSaving graph data to {output_path}...")
    save_graph_data(documents_data, str(output_path))
    
    print(f"\n✓ Company graph data processing complete!")
    print(f"  Processed: {len(documents_data)} documents")
    print(f"  Total clauses: {sum(len(d['clauses']) for d in documents_data)}")


if __name__ == "__main__":
    process_directory_to_graph()

