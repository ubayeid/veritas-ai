"""
Verify data quality and completeness of the Neo4j knowledge graph.

Checks:
- Node counts by type
- Relationship counts by type
- Embedding coverage (which nodes have embeddings)
- Article-clause coverage (ADDRESSES relationships)
- Data completeness warnings
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.indexing.neo4j.utils.neo4j_connection import Neo4jConnection


def check_node_counts(conn: Neo4jConnection):
    """Check node counts by type."""
    print("\n" + "=" * 80)
    print("NODE COUNTS")
    print("=" * 80)
    
    node_types = ["Clause", "Article", "Incident", "Document", "Topic"]
    
    for node_type in node_types:
        query = f"MATCH (n:{node_type}) RETURN count(n) as count"
        result = conn.execute_query(query)
        count = result[0]['count'] if result else 0
        print(f"  {node_type}: {count}")
    
    # Total nodes
    result = conn.execute_query("MATCH (n) RETURN count(n) as count")
    total = result[0]['count'] if result else 0
    print(f"\n  Total Nodes: {total}")


def check_relationship_counts(conn: Neo4jConnection):
    """Check relationship counts by type."""
    print("\n" + "=" * 80)
    print("RELATIONSHIP COUNTS")
    print("=" * 80)
    
    rel_types = ["ADDRESSES", "VIOLATES", "COVERS", "HAS_TOPIC"]
    
    for rel_type in rel_types:
        query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
        result = conn.execute_query(query)
        count = result[0]['count'] if result else 0
        print(f"  {rel_type}: {count}")
    
    # Total relationships
    result = conn.execute_query("MATCH ()-[r]->() RETURN count(r) as count")
    total = result[0]['count'] if result else 0
    print(f"\n  Total Relationships: {total}")


def check_embedding_coverage(conn: Neo4jConnection):
    """Check which nodes have embeddings."""
    print("\n" + "=" * 80)
    print("EMBEDDING COVERAGE")
    print("=" * 80)
    
    # Check Clauses
    query = """
    MATCH (c:Clause)
    RETURN 
        count(c) as total,
        sum(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embedding
    """
    result = conn.execute_query(query)
    if result:
        total = result[0]['total']
        with_emb = result[0]['with_embedding']
        pct = (with_emb / total * 100) if total > 0 else 0
        print(f"  Clauses: {with_emb} out of {total} have embeddings ({pct:.1f}%)")
    
    # Check Articles
    query = """
    MATCH (a:Article)
    RETURN 
        count(a) as total,
        sum(CASE WHEN a.embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embedding
    """
    result = conn.execute_query(query)
    if result:
        total = result[0]['total']
        with_emb = result[0]['with_embedding']
        pct = (with_emb / total * 100) if total > 0 else 0
        print(f"  Articles: {with_emb} out of {total} have embeddings ({pct:.1f}%)")
    
    # Check Incidents
    query = """
    MATCH (i:Incident)
    RETURN 
        count(i) as total,
        sum(CASE WHEN i.embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embedding
    """
    result = conn.execute_query(query)
    if result:
        total = result[0]['total']
        with_emb = result[0]['with_embedding']
        pct = (with_emb / total * 100) if total > 0 else 0
        print(f"  Incidents: {with_emb} out of {total} have embeddings ({pct:.1f}%)")


def check_article_clause_coverage(conn: Neo4jConnection):
    """Check article-clause coverage (ADDRESSES relationships)."""
    print("\n" + "=" * 80)
    print("ARTICLE-CLAUSE COVERAGE")
    print("=" * 80)
    
    # Total articles
    query = "MATCH (a:Article) RETURN count(a) as count"
    result = conn.execute_query(query)
    total_articles = result[0]['count'] if result else 0
    
    # Articles with clauses linked
    query = """
    MATCH (a:Article)
    WHERE EXISTS { (a)<-[:ADDRESSES]-(:Clause) }
    RETURN count(a) as count
    """
    result = conn.execute_query(query)
    articles_with_clauses = result[0]['count'] if result else 0
    
    # Total ADDRESSES relationships
    query = "MATCH ()-[r:ADDRESSES]->() RETURN count(r) as count"
    result = conn.execute_query(query)
    total_addresses = result[0]['count'] if result else 0
    
    print(f"  Total Articles: {total_articles}")
    print(f"  Articles with clauses linked: {articles_with_clauses}")
    print(f"  Total ADDRESSES relationships: {total_addresses}")
    
    if total_articles > 0:
        coverage_pct = (articles_with_clauses / total_articles * 100)
        print(f"  Coverage: {coverage_pct:.1f}%")
        
        # Warning if coverage is low
        if coverage_pct < 50:
            print(f"\n  [WARNING] Low article coverage ({coverage_pct:.1f}%)")
            print("  Note: This may be expected if company policies only address")
            print("        a subset of GDPR articles. Check domain-specific coverage.")
    
    # Show articles without clauses
    query = """
    MATCH (a:Article)
    WHERE NOT EXISTS { (a)<-[:ADDRESSES]-(:Clause) }
    RETURN a.id as article_id, a.number as article_num, a.title as title
    ORDER BY a.id
    LIMIT 10
    """
    result = conn.execute_query(query)
    if result:
        print(f"\n  Articles without clauses (showing first 10):")
        for row in result:
            article_id = row.get('article_id', 'N/A')
            article_num = row.get('article_num', article_id)  # Fallback to id if number missing
            title = row.get('title', 'N/A')[:50]  # Truncate long titles
            print(f"    Article {article_num} ({article_id}): {title}")


def check_data_completeness(conn: Neo4jConnection):
    """Check for data completeness issues."""
    print("\n" + "=" * 80)
    print("DATA COMPLETENESS CHECKS")
    print("=" * 80)
    
    issues = []
    
    # Check for empty nodes
    query = """
    MATCH (c:Clause)
    WHERE c.text IS NULL OR c.text = ''
    RETURN count(c) as count
    """
    result = conn.execute_query(query)
    empty_clauses = result[0]['count'] if result else 0
    if empty_clauses > 0:
        issues.append(f"  [ISSUE] {empty_clauses} clauses have empty text")
    
    # Check for nodes without required properties
    query = """
    MATCH (a:Article)
    WHERE a.id IS NULL OR a.title IS NULL
    RETURN count(a) as count
    """
    result = conn.execute_query(query)
    articles_missing_props = result[0]['count'] if result else 0
    if articles_missing_props > 0:
        issues.append(f"  [ISSUE] {articles_missing_props} articles missing id or title")
    
    # Check for articles without number property (not critical, but good to know)
    query = """
    MATCH (a:Article)
    WHERE a.number IS NULL
    RETURN count(a) as count
    """
    result = conn.execute_query(query)
    articles_no_num = result[0]['count'] if result else 0
    if articles_no_num > 0:
        issues.append(f"  [INFO] {articles_no_num} articles missing number property (non-critical)")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  [OK] No major completeness issues detected")


def main():
    """Main verification function."""
    print("=" * 80)
    print("DATA QUALITY VERIFICATION")
    print("=" * 80)
    
    try:
        conn = Neo4jConnection()
        if not conn.verify_connectivity():
            print("\n[ERROR] Cannot connect to Neo4j database")
            print("  Make sure Neo4j is running and connection settings are correct")
            return
        
        check_node_counts(conn)
        check_relationship_counts(conn)
        check_embedding_coverage(conn)
        check_article_clause_coverage(conn)
        check_data_completeness(conn)
        
        print("\n" + "=" * 80)
        print("VERIFICATION COMPLETE")
        print("=" * 80)
        print("\nNote: Low article-clause coverage may be expected if company")
        print("      policies only address a subset of GDPR articles.")
        
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
