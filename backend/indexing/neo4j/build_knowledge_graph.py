"""
Main script to build the complete Knowledge Graph in Neo4j.
Orchestrates the creation of GDPR structure, Facebook documents, and AIID incidents.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from backend.indexing.neo4j.utils.neo4j_connection import Neo4jConnection
from backend.indexing.neo4j.builders.gdpr_builder import GDPRBuilder
from backend.indexing.neo4j.builders.facebook_documents_builder import FacebookDocumentsBuilder
from backend.indexing.neo4j.builders.aiid_incidents_builder import AIIDIncidentsBuilder


def build_complete_graph(
    clear_existing: bool = False,
    gdpr_json_path: str = None,
    company_json_path: str = None,
    aiid_json_path: str = None
):
    """
    Build the complete knowledge graph.
    
    Args:
        clear_existing: Whether to clear existing data first
        gdpr_json_path: Path to processed GDPR graph JSON
        company_json_path: Path to processed company graph JSON
        aiid_json_path: Path to processed AIID graph JSON
    """
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Default paths for processed graph JSON files
    if not gdpr_json_path:
        gdpr_json_path = str(project_root / "backend" / "processed" / "graph" / "gdpr_graph.json")
    
    if not company_json_path:
        company_json_path = str(project_root / "backend" / "processed" / "graph" / "company_graph.json")
    
    if not aiid_json_path:
        aiid_json_path = str(project_root / "backend" / "processed" / "graph" / "aiid_graph.json")
    
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
        
        # Load from processed graph JSON
        gdpr_json = Path(gdpr_json_path)
        if gdpr_json.exists():
            print(f"Loading from processed graph JSON: {gdpr_json_path}")
            gdpr_builder.build_from_json(str(gdpr_json))
        else:
            print(f"ERROR: GDPR graph JSON not found at {gdpr_json_path}")
            print("Please run backend/processing/graph/gdpr_to_graph.py first to generate the JSON file.")
            return False
        
        # Step 2: Add Facebook documents
        print("\n" + "=" * 80)
        print("Step 2: Adding Facebook Documents")
        print("=" * 80)
        
        # Load from processed graph JSON
        company_json = Path(company_json_path)
        if company_json.exists():
            print(f"Loading from processed graph JSON: {company_json_path}")
            fb_builder = FacebookDocumentsBuilder(conn)
            fb_builder.build_from_json(str(company_json))
        else:
            print(f"WARNING: Company graph JSON not found at {company_json_path}")
            print("Skipping Facebook documents. Run backend/processing/graph/company_to_graph.py to generate the JSON file.")
        
        # Step 3: Add AIID incidents
        print("\n" + "=" * 80)
        print("Step 3: Adding AIID Incidents")
        print("=" * 80)
        
        # Load from processed graph JSON
        aiid_json = Path(aiid_json_path)
        if aiid_json.exists():
            print(f"Loading from processed graph JSON: {aiid_json_path}")
            aiid_builder = AIIDIncidentsBuilder(conn)
            aiid_builder.build_from_json(str(aiid_json))
        else:
            print(f"WARNING: AIID graph JSON not found at {aiid_json_path}")
            print("Skipping AIID incidents. Run backend/processing/graph/aiid_to_graph.py to generate the JSON file.")
        
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
    parser.add_argument("--gdpr-json", type=str, help="Path to processed GDPR graph JSON")
    parser.add_argument("--company-json", type=str, help="Path to processed company graph JSON")
    parser.add_argument("--aiid-json", type=str, help="Path to processed AIID graph JSON")
    
    args = parser.parse_args()
    
    success = build_complete_graph(
        clear_existing=args.clear,
        gdpr_json_path=args.gdpr_json,
        company_json_path=args.company_json,
        aiid_json_path=args.aiid_json
    )
    
    sys.exit(0 if success else 1)

