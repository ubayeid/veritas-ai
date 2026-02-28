"""
Script to convert AIID incidents CSV into graph-structured data.
Extracts incidents with risk types and system types, then saves as JSON for Neo4j import.

This follows the same pattern as vector processing:
- Input: data/aiid/incidents.csv
- Output: backend/processed/graph/aiid_graph.json

Environment Variables (.env file):
    (None required - this is just data extraction)
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# Import shared utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import save_graph_json


def determine_risk_type(description: str, title: str) -> str:
    """
    Determine risk type from incident description and title.
    
    Args:
        description: Incident description
        title: Incident title
        
    Returns:
        Risk type ('Privacy', 'Security', 'Bias', 'Safety', 'Other')
    """
    text = (description + " " + title).lower()
    
    privacy_keywords = ['privacy', 'data leak', 'data breach', 'unauthorized access', 
                       'personal information', 'data exposure', 'data misuse']
    security_keywords = ['security', 'hack', 'vulnerability', 'attack', 'breach', 
                        'unauthorized', 'malware', 'exploit']
    bias_keywords = ['bias', 'discrimination', 'unfair', 'discriminatory', 'racial', 
                    'gender', 'prejudice']
    safety_keywords = ['accident', 'crash', 'injury', 'death', 'harm', 'malfunction', 
                      'failure', 'safety']
    
    if any(kw in text for kw in privacy_keywords):
        return 'Privacy'
    elif any(kw in text for kw in security_keywords):
        return 'Security'
    elif any(kw in text for kw in bias_keywords):
        return 'Bias'
    elif any(kw in text for kw in safety_keywords):
        return 'Safety'
    else:
        return 'Other'


def determine_system_type(description: str, title: str) -> str:
    """
    Determine system type from incident description.
    
    Args:
        description: Incident description
        title: Incident title
        
    Returns:
        System type
    """
    text = (description + " " + title).lower()
    
    if 'autonomous' in text or 'self-driving' in text or 'autopilot' in text:
        return 'Autonomous Vehicle'
    elif 'facial recognition' in text or 'biometric' in text:
        return 'Biometric AI'
    elif 'chatbot' in text or 'conversational' in text:
        return 'Conversational AI'
    elif 'recommendation' in text or 'algorithm' in text:
        return 'Recommendation System'
    elif 'scheduling' in text:
        return 'Scheduling System'
    elif 'risk assessment' in text or 'risk model' in text:
        return 'Risk Assessment System'
    elif 'surgery' in text or 'medical' in text:
        return 'Medical AI'
    else:
        return 'AI System'


def read_incidents_csv(csv_path: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Read incidents from CSV file and convert to graph structure.
    
    Args:
        csv_path: Path to incidents CSV file
        limit: Optional limit on number of incidents to process
        
    Returns:
        List of incident dictionaries
    """
    incidents = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            
            incident_id = row.get('incident_id', row.get('_id', ''))
            if not incident_id:
                # Generate ID from title
                incident_id = f"AIID_{abs(hash(row.get('title', '')))}"
            
            description = row.get('description', '') or ''
            title = row.get('title', '') or ''
            
            risk_type = determine_risk_type(description, title)
            system_type = determine_system_type(description, title)
            
            incidents.append({
                'id': incident_id,
                'title': title[:500] if len(title) > 500 else title,
                'description': description[:1000] if len(description) > 1000 else description,
                'system_type': system_type,
                'risk_type': risk_type,
                'date': row.get('date', ''),
                'source': 'AIID Database'
            })
    
    return incidents


def save_graph_data(incidents: List[Dict], output_path: str):
    """
    Save graph-structured data to JSON file.
    
    Args:
        incidents: List of incident dictionaries
        output_path: Path to output JSON file
    """
    graph_data = {
        'metadata': {
            'source': 'AIID Database',
            'num_incidents': len(incidents),
            'version': '1.0'
        },
        'incidents': incidents
    }
    
    save_graph_json(graph_data, Path(output_path))
    print(f"Saved {len(incidents)} incidents")


def process_aiid_to_graph(csv_path: str = None, output_dir: str = None, limit: Optional[int] = None):
    """
    Process AIID CSV and save graph-structured data.
    
    Args:
        csv_path: Path to AIID incidents CSV file
        output_dir: Directory to save graph data
        limit: Optional limit on number of incidents to process
    """
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Default paths
    if not csv_path:
        csv_path = str(project_root / "data" / "aiid" / "incidents.csv")
    
    if not output_dir:
        output_dir = str(project_root / "backend" / "processed" / "graph")
    
    csv_file = Path(csv_path)
    output_path = Path(output_dir) / "aiid_graph.json"
    
    if not csv_file.exists():
        print(f"CSV file not found: {csv_path}")
        return
    
    print(f"Processing AIID CSV: {csv_path}")
    
    # Read and process incidents
    try:
        print("Reading incidents from CSV...")
        incidents = read_incidents_csv(str(csv_file), limit=limit)
        print(f"Processed {len(incidents)} incidents")
        
        if not incidents:
            print("Warning: No incidents found in CSV file!")
            return
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        raise
    
    # Save graph data
    try:
        print(f"Saving graph data to {output_path}...")
        save_graph_data(incidents, str(output_path))
        print("[OK] AIID graph data processing complete!")
    except Exception as e:
        print(f"ERROR: Failed to save graph data: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert AIID CSV to graph data")
    parser.add_argument("--csv-path", type=str, help="Path to AIID incidents CSV")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit number of incidents to process")
    
    args = parser.parse_args()
    
    process_aiid_to_graph(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        limit=args.limit
    )

