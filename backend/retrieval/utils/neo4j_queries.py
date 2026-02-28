"""
Neo4j Knowledge Graph Query Methods
Collection of useful queries for searching and analyzing the knowledge graph.
"""

from backend.indexing.neo4j.utils.neo4j_connection import Neo4jConnection


class KnowledgeGraphQueries:
    """Collection of useful queries for the knowledge graph."""
    
    def __init__(self, neo4j_conn: Neo4jConnection):
        """
        Initialize queries.
        
        Args:
            neo4j_conn: Neo4jConnection instance
        """
        self.conn = neo4j_conn
    
    def gdpr_coverage(self) -> list:
        """
        Find GDPR coverage - which articles are addressed by which clauses.
        
        Returns:
            List of articles with their associated clauses
        """
        query = """
        MATCH (a:Article)<-[:ADDRESSES]-(c:Clause)
        RETURN a.id as article_id, 
               a.title as article_title,
               collect(c.text) as clauses,
               count(c) as clause_count
        ORDER BY clause_count DESC
        """
        return self.conn.execute_query(query)
    
    def document_gap_analysis(self) -> list:
        """
        Find GDPR articles that are NOT addressed by any clauses.
        
        Returns:
            List of articles without coverage
        """
        query = """
        MATCH (a:Article)
        WHERE NOT (a)<-[:ADDRESSES]-(:Clause)
        RETURN a.id as article_id,
               a.title as article_title,
               a.description as description
        ORDER BY a.id
        """
        return self.conn.execute_query(query)
    
    def comprehensive_mismatch_analysis(self) -> dict:
        """
        Comprehensive analysis comparing company documents with GDPR articles.
        Shows both gaps (uncovered articles) and coverage (covered articles with clauses).
        
        Returns:
            Dictionary with gaps and coverage information
        """
        # Get gaps
        gaps_query = """
        MATCH (a:Article)
        WHERE NOT (a)<-[:ADDRESSES]-(:Clause)
        RETURN a.id as article_id,
               a.title as article_title,
               a.description as description
        ORDER BY a.id
        """
        gaps = self.conn.execute_query(gaps_query)
        
        # Get coverage with clause details
        coverage_query = """
        MATCH (a:Article)<-[:ADDRESSES]-(c:Clause)
        WITH a, collect({
            clause_id: c.id,
            clause_text: c.text,
            document_name: c.document_name
        }) as clauses
        RETURN a.id as article_id,
               a.title as article_title,
               a.description as description,
               clauses,
               size(clauses) as clause_count
        ORDER BY clause_count DESC
        """
        coverage = self.conn.execute_query(coverage_query)
        
        # Get document summary
        doc_summary_query = """
        MATCH (d:Document)-[:COVERS]->(c:Clause)
        OPTIONAL MATCH (c)-[:ADDRESSES]->(a:Article)
        RETURN d.name as document_name,
               count(DISTINCT c) as total_clauses,
               count(DISTINCT a) as articles_addressed
        ORDER BY document_name
        """
        doc_summary = self.conn.execute_query(doc_summary_query)
        
        return {
            'gaps': gaps,
            'coverage': coverage,
            'document_summary': doc_summary,
            'total_articles': len(gaps) + len(coverage),
            'covered_articles': len(coverage),
            'uncovered_articles': len(gaps),
            'coverage_percentage': round((len(coverage) / (len(gaps) + len(coverage)) * 100) if (len(gaps) + len(coverage)) > 0 else 0, 2)
        }
    
    def aiid_risk_mapping(self) -> list:
        """
        Map AIID incidents to GDPR articles they violate.
        
        Returns:
            List of incidents with their violated articles
        """
        query = """
        MATCH (i:Incident)-[:VIOLATES]->(a:Article)
        RETURN i.id as incident_id,
               i.title as incident_title,
               i.risk_type as risk_type,
               i.system_type as system_type,
               collect(a.id) as violated_articles,
               collect(a.title) as article_titles
        ORDER BY i.risk_type, i.system_type
        """
        return self.conn.execute_query(query)
    
    def find_non_compliant_clauses(self, article_id: str = None) -> list:
        """
        Find clauses that are marked as non-compliant.
        
        Args:
            article_id: Optional article ID to filter by
            
        Returns:
            List of non-compliant clauses
        """
        if article_id:
            query = """
            MATCH (c:Clause)-[:ADDRESSES]->(a:Article {id: $article_id})
            WHERE c.compliance_status = 'non-compliant'
            RETURN c.id as clause_id,
                   c.text as clause_text,
                   c.document_name as document_name,
                   a.id as article_id,
                   a.title as article_title
            """
            return self.conn.execute_query(query, {'article_id': article_id})
        else:
            query = """
            MATCH (c:Clause)
            WHERE c.compliance_status = 'non-compliant'
            RETURN c.id as clause_id,
                   c.text as clause_text,
                   c.document_name as document_name
            """
            return self.conn.execute_query(query)
    
    def find_article_by_id(self, article_id: str) -> dict:
        """
        Find a specific article by its ID.
        
        Args:
            article_id: Article ID (e.g., "Art12")
            
        Returns:
            Dictionary with article information, or None if not found
        """
        query = """
        MATCH (a:Article {id: $article_id})
        RETURN a.id as article_id,
               a.title as article_title,
               a.description as description
        LIMIT 1
        """
        result = self.conn.execute_query(query, {'article_id': article_id})
        return result[0] if result else None
    
    def find_clauses_by_multiple_articles(self, article_ids: list) -> list:
        """
        Find clauses that address ALL of the specified articles.
        
        Args:
            article_ids: List of article IDs (e.g., ["Art12", "Art13"])
            
        Returns:
            List of clauses that address all specified articles
        """
        if not article_ids:
            return []
        
        # Find clauses that address ALL articles
        # This uses a pattern where we match clauses that have ADDRESSES relationships
        # to all the specified articles
        query = """
        MATCH (c:Clause)-[:ADDRESSES]->(a:Article)
        WHERE a.id IN $article_ids
        WITH c, collect(DISTINCT a.id) as addressed_articles
        WHERE size(addressed_articles) = $article_count
        RETURN DISTINCT c.id as clause_id,
               c.text as clause_text,
               c.document_name as document_name,
               addressed_articles as article_ids
        ORDER BY c.document_name, c.id
        """
        return self.conn.execute_query(query, {
            'article_ids': article_ids,
            'article_count': len(article_ids)
        })
    
    def find_clauses_by_article(self, article_id: str) -> list:
        """
        Find all clauses addressing a specific GDPR article.
        
        Args:
            article_id: ID of the GDPR article
            
        Returns:
            List of clauses addressing the article
        """
        query = """
        MATCH (c:Clause)-[:ADDRESSES]->(a:Article {id: $article_id})
        RETURN c.id as clause_id,
               c.text as clause_text,
               c.document_name as document_name,
               a.title as article_title
        """
        return self.conn.execute_query(query, {'article_id': article_id})
    
    def find_incidents_by_article(self, article_id: str) -> list:
        """
        Find all incidents violating a specific GDPR article.
        
        Args:
            article_id: ID of the GDPR article
            
        Returns:
            List of incidents violating the article
        """
        query = """
        MATCH (i:Incident)-[:VIOLATES]->(a:Article {id: $article_id})
        RETURN i.id as incident_id,
               i.title as incident_title,
               i.description as description,
               i.risk_type as risk_type,
               i.system_type as system_type,
               a.title as article_title
        """
        return self.conn.execute_query(query, {'article_id': article_id})
    
    def document_coverage_summary(self) -> list:
        """
        Get summary of document coverage.
        
        Returns:
            Summary of clauses per document
        """
        query = """
        MATCH (d:Document)-[:COVERS]->(c:Clause)
        OPTIONAL MATCH (c)-[:ADDRESSES]->(a:Article)
        RETURN d.name as document_name,
               count(DISTINCT c) as total_clauses,
               count(DISTINCT a) as articles_addressed,
               count(DISTINCT CASE WHEN c.compliance_status = 'non-compliant' THEN c END) as non_compliant_count
        ORDER BY document_name
        """
        return self.conn.execute_query(query)
    
    def topic_analysis(self, topic_name: str = None) -> list:
        """
        Analyze topics and their related articles.
        
        Args:
            topic_name: Optional topic name to filter by
            
        Returns:
            List of topics with their articles
        """
        if topic_name:
            query = """
            MATCH (t:Topic {name: $topic_name})<-[:HAS_TOPIC]-(a:Article)
            OPTIONAL MATCH (a)<-[:ADDRESSES]-(c:Clause)
            RETURN t.name as topic_name,
                   collect(DISTINCT a.id) as article_ids,
                   collect(DISTINCT a.title) as article_titles,
                   count(DISTINCT c) as clause_count
            """
            return self.conn.execute_query(query, {'topic_name': topic_name})
        else:
            query = """
            MATCH (t:Topic)<-[:HAS_TOPIC]-(a:Article)
            OPTIONAL MATCH (a)<-[:ADDRESSES]-(c:Clause)
            RETURN t.name as topic_name,
                   count(DISTINCT a) as article_count,
                   count(DISTINCT c) as clause_count
            ORDER BY clause_count DESC
            """
            return self.conn.execute_query(query)
    
    def risk_analysis_by_article(self) -> list:
        """
        Analyze risk by GDPR article based on incidents.
        
        Returns:
            List of articles with incident counts
        """
        query = """
        MATCH (a:Article)<-[:VIOLATES]-(i:Incident)
        RETURN a.id as article_id,
               a.title as article_title,
               count(i) as incident_count,
               collect(DISTINCT i.risk_type) as risk_types,
               collect(DISTINCT i.system_type) as system_types
        ORDER BY incident_count DESC
        """
        return self.conn.execute_query(query)
    
    def compliance_status_summary(self) -> dict:
        """
        Get overall compliance status summary.
        
        Returns:
            Dictionary with compliance statistics
        """
        query = """
        MATCH (c:Clause)
        RETURN count(c) as total_clauses,
               count(CASE WHEN c.compliance_status = 'compliant' THEN 1 END) as compliant_count,
               count(CASE WHEN c.compliance_status = 'partial' THEN 1 END) as partial_count,
               count(CASE WHEN c.compliance_status = 'non-compliant' THEN 1 END) as non_compliant_count
        """
        result = self.conn.execute_query(query)
        return result[0] if result else {}


def print_query_results(results: list, title: str = "Query Results"):
    """Pretty print query results."""
    print(f"\n{'=' * 80}")
    print(title)
    print('=' * 80)
    
    if not results:
        print("No results found.")
        return
    
    for i, record in enumerate(results, 1):
        print(f"\nResult {i}:")
        for key, value in record.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value[:5]:  # Show first 5 items
                    print(f"    - {item}")
                if len(value) > 5:
                    print(f"    ... and {len(value) - 5} more")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    # Example usage
    from neo4j_connection import Neo4jConnection
    
    with Neo4jConnection() as conn:
        if not conn.verify_connectivity():
            print("Failed to connect to Neo4j.")
            exit(1)
        
        queries = KnowledgeGraphQueries(conn)
        
        # Example queries
        print("\n1. GDPR Coverage Analysis")
        results = queries.gdpr_coverage()
        print_query_results(results[:5], "GDPR Coverage (Top 5)")
        
        print("\n2. Document Gap Analysis")
        results = queries.document_gap_analysis()
        print_query_results(results, "Articles Without Coverage")
        
        print("\n3. AIID Risk Mapping")
        results = queries.aiid_risk_mapping()
        print_query_results(results[:5], "AIID Risk Mapping (Top 5)")
        
        print("\n4. Compliance Status Summary")
        summary = queries.compliance_status_summary()
        print(f"\nTotal Clauses: {summary.get('total_clauses', 0)}")
        print(f"Compliant: {summary.get('compliant_count', 0)}")
        print(f"Partial: {summary.get('partial_count', 0)}")
        print(f"Non-Compliant: {summary.get('non_compliant_count', 0)}")
        
        print("\n5. Document Coverage Summary")
        results = queries.document_coverage_summary()
        print_query_results(results, "Document Coverage")

