"""
Main script to build the complete Knowledge Graph in Neo4j.
Orchestrates the creation of GDPR structure, Facebook documents, and AIID incidents.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from neo4j_connection import Neo4jConnection
from gdpr_builder import GDPRBuilder, create_sample_gdpr_data
from facebook_documents_builder import FacebookDocumentsBuilder
from aiid_incidents_builder import AIIDIncidentsBuilder


def build_complete_graph(
    clear_existing: bool = False,
    gdpr_json_path: str = None,
    company_json_path: str = None,
    aiid_json_path: str = None,
    # Legacy parameters (for backward compatibility)
    gdpr_pdf_path: str = None,
    company_dir: str = None,
    aiid_csv_path: str = None,
    aiid_limit: int = None
):
    """
    Build the complete knowledge graph.
    
    Args:
        clear_existing: Whether to clear existing data first
        gdpr_json_path: Path to processed GDPR graph JSON (preferred)
        company_json_path: Path to processed company graph JSON (preferred)
        aiid_json_path: Path to processed AIID graph JSON (preferred)
        gdpr_pdf_path: Path to GDPR PDF (legacy, uses sample data if not provided)
        company_dir: Path to directory containing Facebook documents (legacy)
        aiid_csv_path: Path to AIID incidents CSV (legacy)
        aiid_limit: Limit on number of incidents to process (legacy)
    """
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Default paths for processed graph JSON files (preferred)
    if not gdpr_json_path:
        gdpr_json_path = str(project_root / "data" / "processed" / "graph" / "gdpr_graph.json")
    
    if not company_json_path:
        company_json_path = str(project_root / "data" / "processed" / "graph" / "company_graph.json")
    
    if not aiid_json_path:
        aiid_json_path = str(project_root / "data" / "processed" / "graph" / "aiid_graph.json")
    
    # Legacy paths (for backward compatibility)
    if not gdpr_pdf_path:
        gdpr_pdf_path = str(project_root / "data" / "standards" / "gdpr.pdf")
    
    if not company_dir:
        company_dir = str(project_root / "data" / "company")
    
    if not aiid_csv_path:
        aiid_csv_path = str(project_root / "data" / "aiid" / "incidents.csv")
    
    print("=" * 80)
    print("Building Knowledge Graph in Neo4j")
    print("=" * 80)
    
    with Neo4jConnection() as conn:
        # Verify connection
        if not conn.verify_connectivity():
            print("ERROR: Failed to connect to Neo4j.")
            print("Please ensure Neo4j is running and check your connection settings.")
            print("Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in .env file or environment.")
            return False
        
        # Clear existing data if requested
        if clear_existing:
            print("\nClearing existing data...")
            conn.clear_database()
        
        # Step 1: Build GDPR structure
        print("\n" + "=" * 80)
        print("Step 1: Creating GDPR Structure")
        print("=" * 80)
        
        gdpr_builder = GDPRBuilder(conn)
        
        # Try to load from processed graph JSON first (preferred)
        gdpr_json = Path(gdpr_json_path)
        if gdpr_json.exists():
            print(f"Loading from processed graph JSON: {gdpr_json_path}")
            gdpr_builder.build_from_json(str(gdpr_json))
        else:
            # Fallback to PDF parsing
            gdpr_pdf = Path(gdpr_pdf_path)
            if gdpr_pdf.exists():
                print(f"Building from GDPR PDF: {gdpr_pdf_path}")
                gdpr_builder.build_from_pdf(str(gdpr_pdf))
            else:
                print("GDPR JSON/PDF not found. Using sample GDPR data...")
                sample_data = create_sample_gdpr_data()
                gdpr_builder.build_from_manual_data(sample_data)
        
        # Step 2: Add Facebook documents
        print("\n" + "=" * 80)
        print("Step 2: Adding Facebook Documents")
        print("=" * 80)
        
        # Try to load from processed graph JSON first (preferred)
        company_json = Path(company_json_path)
        if company_json.exists():
            print(f"Loading from processed graph JSON: {company_json_path}")
            fb_builder = FacebookDocumentsBuilder(conn)
            fb_builder.build_from_json(str(company_json))
        else:
            # Fallback to directory processing
            company_path = Path(company_dir)
            if company_path.exists():
                fb_builder = FacebookDocumentsBuilder(conn)
                fb_builder.process_directory(str(company_path))
            else:
                print(f"Company JSON/directory not found: {company_json_path} / {company_dir}")
                print("Skipping Facebook documents...")
        
        # Step 3: Add AIID incidents
        print("\n" + "=" * 80)
        print("Step 3: Adding AIID Incidents")
        print("=" * 80)
        
        # Try to load from processed graph JSON first (preferred)
        aiid_json = Path(aiid_json_path)
        if aiid_json.exists():
            print(f"Loading from processed graph JSON: {aiid_json_path}")
            aiid_builder = AIIDIncidentsBuilder(conn)
            aiid_builder.build_from_json(str(aiid_json))
        else:
            # Fallback to CSV processing
            aiid_path = Path(aiid_csv_path)
            if aiid_path.exists():
                aiid_builder = AIIDIncidentsBuilder(conn)
                aiid_builder.process_incidents_csv(str(aiid_path), limit=aiid_limit)
            else:
                print(f"AIID JSON/CSV not found: {aiid_json_path} / {aiid_csv_path}")
                print("Skipping AIID incidents...")
        
        # Print final statistics
        print("\n" + "=" * 80)
        print("Build Complete - Database Statistics")
        print("=" * 80)
        
        stats = conn.get_stats()
        print("\nNodes:")
        for label, count in stats['nodes'].items():
            print(f"  {label}: {count}")
        
        print("\nRelationships:")
        for rel_type, count in stats['relationships'].items():
            print(f"  {rel_type}: {count}")
        
        print("\n" + "=" * 80)
        print("Knowledge Graph build completed successfully!")
        print("=" * 80)
        
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Knowledge Graph in Neo4j")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before building")
    parser.add_argument("--gdpr-json", type=str, help="Path to processed GDPR graph JSON (preferred)")
    parser.add_argument("--company-json", type=str, help="Path to processed company graph JSON (preferred)")
    parser.add_argument("--aiid-json", type=str, help="Path to processed AIID graph JSON (preferred)")
    parser.add_argument("--gdpr-pdf", type=str, help="Path to GDPR PDF file (legacy)")
    parser.add_argument("--company-dir", type=str, help="Path to company documents directory (legacy)")
    parser.add_argument("--aiid-csv", type=str, help="Path to AIID incidents CSV (legacy)")
    parser.add_argument("--aiid-limit", type=int, help="Limit number of incidents to process (legacy)")
    
    args = parser.parse_args()
    
    success = build_complete_graph(
        clear_existing=args.clear,
        gdpr_json_path=args.gdpr_json,
        company_json_path=args.company_json,
        aiid_json_path=args.aiid_json,
        gdpr_pdf_path=args.gdpr_pdf,
        company_dir=args.company_dir,
        aiid_csv_path=args.aiid_csv,
        aiid_limit=args.aiid_limit
    )
    
    sys.exit(0 if success else 1)

