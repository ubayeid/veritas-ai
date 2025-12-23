"""
Test script to check hybrid search behavior.
Run this to see detailed debug output of hybrid search.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.searching.hybrid_query_engine import HybridQueryEngine

def test_hybrid_search():
    """Test hybrid search with various query types."""
    
    # Initialize engine
    base_dir = Path(__file__).parent.parent.parent
    engine = HybridQueryEngine(str(base_dir))
    
    # Test queries
    test_queries = [
        # Semantic query (should use FAISS)
        "What are the privacy policies?",
        
        # Graph query (should use Neo4j)
        "Find clauses addressing GDPR Article 5",
        
        # Hybrid query (should use both)
        "What privacy policies address data minimization requirements?",
        
        # Mismatch query (special ordering)
        "Show compliance gaps and coverage",
    ]
    
    print("\n" + "="*80)
    print("HYBRID SEARCH TESTING")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST {i}: {query}")
        print(f"{'#'*80}")
        
        try:
            result = engine.hybrid_search(
                query=query,
                top_k=10,
                use_faiss=True,
                use_graph_traversal=True
            )
            
            print(f"\n✅ Search completed successfully")
            print(f"   Total results: {result['num_results']}")
            print(f"   Sources used: {result['sources_used']}")
            print(f"   Query types: {result['query_types']}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    engine.close()
    print(f"\n{'='*80}")
    print("Testing complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_hybrid_search()

