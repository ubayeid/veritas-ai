"""
Script to convert GDPR PDF into graph-structured data.
Extracts articles, sub-obligations, and topics, then saves as JSON for Neo4j import.

This follows the same pattern as vector processing:
- Input: data/standards/gdpr.pdf
- Output: backend/data_processing/processed/graph/gdpr_graph.json

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


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text."""
    gdpr_keywords = [
        'data protection', 'personal data', 'processing', 'consent', 'right to access',
        'right to erasure', 'data minimization', 'purpose limitation', 'storage limitation',
        'accuracy', 'integrity', 'confidentiality', 'accountability', 'transparency',
        'data subject', 'controller', 'processor', 'breach', 'notification', 'security',
        'privacy by design', 'privacy by default', 'impact assessment', 'supervisory authority'
    ]
    
    text_lower = text.lower()
    found_keywords = [kw for kw in gdpr_keywords if kw in text_lower]
    
    words = re.findall(r'\b[a-z]{4,}\b', text_lower)
    common_words = [w for w in set(words) if len(w) > 4][:5]
    
    return list(set(found_keywords + common_words))


def parse_gdpr_articles(text: str) -> List[Dict]:
    """
    Parse GDPR text to extract articles.
    
    Args:
        text: Extracted text from GDPR PDF
        
    Returns:
        List of article dictionaries
    """
    articles = []
    
    # Pattern to match article numbers
    article_pattern = r'(?:Article|Art\.?)\s*(\d+)[\s\n]+(.+?)(?=(?:Article|Art\.?)\s*\d+|$)'
    
    matches = re.finditer(article_pattern, text, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        article_num = match.group(1)
        article_text = match.group(2).strip()
        
        # Extract title (first sentence or line)
        lines = article_text.split('\n')
        title = lines[0].strip() if lines else f"Article {article_num}"
        
        # Extract description (first paragraph or first few sentences)
        description = article_text[:500] if len(article_text) > 500 else article_text
        
        # Extract keywords
        keywords = extract_keywords(article_text)
        
        articles.append({
            'id': f'Art{article_num}',
            'number': article_num,
            'title': title,
            'description': description,
            'keywords': keywords,
            'full_text': article_text
        })
    
    return articles


def create_sample_gdpr_data() -> List[Dict]:
    """Create sample GDPR data structure."""
    return [
        {
            'id': 'Art5',
            'title': 'Principles relating to processing of personal data',
            'description': 'Personal data shall be: (a) processed lawfully, fairly and in a transparent manner; (b) collected for specified, explicit and legitimate purposes; (c) adequate, relevant and limited to what is necessary; (d) accurate and kept up to date; (e) kept in a form which permits identification for no longer than necessary; (f) processed in a manner that ensures appropriate security.',
            'keywords': ['data minimization', 'purpose limitation', 'storage limitation', 'accuracy', 'lawfulness', 'fairness', 'transparency'],
            'sub_obligations': [
                {
                    'id': 'Art5.1.c',
                    'description': 'Personal data must be adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed (data minimisation)',
                    'keywords': ['data minimization', 'limited', 'necessary', 'relevant']
                },
                {
                    'id': 'Art5.1.f',
                    'description': 'Personal data must be processed in a manner that ensures appropriate security, including protection against unauthorised or unlawful processing and against accidental loss, destruction or damage',
                    'keywords': ['security', 'protection', 'unauthorized', 'loss', 'destruction']
                }
            ],
            'topics': ['Data Minimization', 'Purpose Limitation', 'Storage Limitation', 'Security']
        },
        {
            'id': 'Art32',
            'title': 'Security of processing',
            'description': 'Taking into account the state of the art, the costs of implementation and the nature, scope, context and purposes of processing as well as the risk of varying likelihood and severity for the rights and freedoms of natural persons, the controller and the processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk.',
            'keywords': ['security', 'technical measures', 'organizational measures', 'risk assessment', 'data protection'],
            'sub_obligations': [
                {
                    'id': 'Art32.1',
                    'description': 'Implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk',
                    'keywords': ['technical measures', 'organizational measures', 'security', 'risk']
                }
            ],
            'topics': ['Security', 'Risk Assessment', 'Technical Measures']
        },
        {
            'id': 'Art15',
            'title': 'Right of access by the data subject',
            'description': 'The data subject shall have the right to obtain from the controller confirmation as to whether or not personal data concerning him or her are being processed, and, where that is the case, access to the personal data.',
            'keywords': ['right to access', 'data subject rights', 'transparency', 'access'],
            'sub_obligations': [
                {
                    'id': 'Art15.1',
                    'description': 'Data subject has the right to obtain confirmation of processing and access to personal data',
                    'keywords': ['right to access', 'confirmation', 'data subject']
                }
            ],
            'topics': ['Data Subject Rights', 'Transparency', 'Access']
        }
    ]


def save_graph_data(articles: List[Dict], output_path: str):
    """
    Save graph-structured data to JSON file.
    
    Args:
        articles: List of article dictionaries
        output_path: Path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    graph_data = {
        'metadata': {
            'source': 'GDPR',
            'num_articles': len(articles),
            'version': '1.0'
        },
        'articles': articles
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(articles)} articles to {output_file}")


def process_gdpr_to_graph(pdf_path: str = None, output_dir: str = None):
    """
    Process GDPR PDF and save graph-structured data.
    
    Args:
        pdf_path: Path to GDPR PDF file
        output_dir: Directory to save graph data
    """
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Default paths
    if not pdf_path:
        pdf_path = str(project_root / "data" / "standards" / "gdpr.pdf")
    
    if not output_dir:
        output_dir = str(project_root / "backend" / "data_processing" / "processed" / "graph")
    
    pdf_file = Path(pdf_path)
    output_path = Path(output_dir) / "gdpr_graph.json"
    
    print(f"Processing GDPR PDF: {pdf_path}")
    
    # Try to parse from PDF
    if pdf_file.exists():
        print("Extracting text from PDF...")
        text = extract_text_from_pdf(str(pdf_file))
        print(f"Extracted {len(text)} characters")
        
        print("Parsing articles...")
        articles = parse_gdpr_articles(text)
        print(f"Found {len(articles)} articles")
    else:
        print(f"PDF not found: {pdf_path}")
        print("Using sample GDPR data...")
        articles = create_sample_gdpr_data()
    
    # Save graph data
    print(f"Saving graph data to {output_path}...")
    save_graph_data(articles, str(output_path))
    
    print("✓ GDPR graph data processing complete!")


if __name__ == "__main__":
    process_gdpr_to_graph()

