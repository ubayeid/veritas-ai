"""
Hybrid Query Engine: Combines Vector Search (FAISS) + Graph Traversal (Neo4j)
Simplified to use only FAISS vector search and Neo4j graph traversal.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "building_database" / "neo4j"))

from query_engine import VectorQueryEngine
from neo4j_queries import KnowledgeGraphQueries
from neo4j_connection import Neo4jConnection

load_dotenv()

# Configuration from environment variables
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
RRF_K = int(os.getenv("RRF_K", "60"))  # Reciprocal Rank Fusion constant


class HybridQueryEngine:
    """
    Hybrid Query Engine combining:
    1. Vector Search (FAISS) - Semantic similarity
    2. Graph Traversal (Neo4j) - Relationship-based queries
    
    Uses Reciprocal Rank Fusion (RRF) to merge results from different sources.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize hybrid query engine.
        
        Args:
            base_dir: Base directory of the project
        """
        self.base_dir = Path(base_dir)
        
        # Initialize vector engine (FAISS)
        self.vector_engine = VectorQueryEngine(str(base_dir))
        
        # Initialize graph engine (Neo4j)
        self.neo4j_conn = Neo4jConnection()
        if not self.neo4j_conn.verify_connectivity():
            raise ConnectionError("Failed to connect to Neo4j. Please ensure Neo4j is running.")
        
        self.graph_queries = KnowledgeGraphQueries(self.neo4j_conn)
    
    def detect_query_type(self, query: str) -> Dict[str, bool]:
        """
        Detect what type of query this is to determine search strategy.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary indicating query types
        """
        query_lower = query.lower()
        
        # Graph-specific patterns
        graph_patterns = [
            'article', 'art ', 'gdpr article', 'clause', 'incident',
            'violates', 'addresses', 'compliance', 'gap', 'coverage',
            'relationship', 'related to', 'connected', 'linked',
            'mismatch', 'compare', 'difference', 'alignment'
        ]
        
        # Relationship patterns
        relationship_patterns = [
            'find clauses for', 'clauses addressing', 'incidents violating',
            'what articles', 'which articles', 'show relationships'
        ]
        
        is_graph_query = any(pattern in query_lower for pattern in graph_patterns)
        is_relationship_query = any(pattern in query_lower for pattern in relationship_patterns)
        is_semantic_query = not is_graph_query  # Default to semantic if not graph-specific
        
        return {
            'is_graph_query': is_graph_query,
            'is_relationship_query': is_relationship_query,
            'is_semantic_query': is_semantic_query
        }
    
    def graph_traversal_search(self, query: str) -> List[Dict]:
        """
        Perform graph traversal search based on query patterns.
        
        Args:
            query: User query
            
        Returns:
            List of results from graph traversal
        """
        query_lower = query.lower()
        results = []
        
        # Extract article ID if mentioned
        import re
        article_match = re.search(r'art(?:icle)?\s*(\d+)', query_lower)
        
        if article_match:
            article_id = f"Art{article_match.group(1)}"
            
            # Find clauses addressing this article
            clauses = self.graph_queries.find_clauses_by_article(article_id)
            for clause in clauses:
                results.append({
                    'id': clause['clause_id'],
                    'text': clause['clause_text'],
                    'document_name': clause['document_name'],
                    'article_id': article_id,
                    'type': 'clause',
                    'source': 'graph_traversal',
                    'relationship': 'ADDRESSES'
                })
            
            # Find incidents violating this article
            incidents = self.graph_queries.find_incidents_by_article(article_id)
            for incident in incidents:
                results.append({
                    'id': incident['incident_id'],
                    'text': incident['description'],
                    'title': incident['incident_title'],
                    'risk_type': incident['risk_type'],
                    'article_id': article_id,
                    'type': 'incident',
                    'source': 'graph_traversal',
                    'relationship': 'VIOLATES'
                })
        
        # Compliance gap and mismatch queries
        if 'gap' in query_lower or 'missing' in query_lower or 'not covered' in query_lower or 'mismatch' in query_lower or 'compare' in query_lower or 'difference' in query_lower:
            # Use comprehensive mismatch analysis for better results
            if 'mismatch' in query_lower or 'compare' in query_lower or 'difference' in query_lower:
                mismatch_data = self.graph_queries.comprehensive_mismatch_analysis()
                
                # Add gaps (articles NOT covered by company documents)
                for gap in mismatch_data['gaps'][:15]:  # Top 15 gaps
                    results.append({
                        'id': gap['article_id'],
                        'text': gap.get('description', gap.get('article_title', '')),
                        'title': gap['article_title'],
                        'type': 'gap',
                        'source': 'graph_traversal',
                        'coverage_status': 'not_covered',
                        'analysis_type': 'mismatch'
                    })
                
                # Add coverage info (articles that ARE covered)
                for cov in mismatch_data['coverage'][:15]:  # Top 15 covered
                    clause_texts = [clause.get('clause_text', '')[:200] for clause in cov.get('clauses', [])[:2]]
                    results.append({
                        'id': cov['article_id'],
                        'text': cov.get('description', cov.get('article_title', '')),
                        'title': cov['article_title'],
                        'type': 'coverage',
                        'source': 'graph_traversal',
                        'coverage_status': 'covered',
                        'clause_count': cov.get('clause_count', 0),
                        'clause_examples': clause_texts,
                        'analysis_type': 'mismatch'
                    })
                
                # Add summary statistics
                results.append({
                    'id': 'summary',
                    'text': f"Coverage Analysis: {mismatch_data['covered_articles']} articles covered, {mismatch_data['uncovered_articles']} articles not covered ({mismatch_data['coverage_percentage']}% coverage)",
                    'title': 'Coverage Summary',
                    'type': 'summary',
                    'source': 'graph_traversal',
                    'coverage_percentage': mismatch_data['coverage_percentage'],
                    'covered_count': mismatch_data['covered_articles'],
                    'uncovered_count': mismatch_data['uncovered_articles'],
                    'analysis_type': 'mismatch'
                })
            else:
                # Simple gap analysis for "gap" or "missing" queries
                gaps = self.graph_queries.document_gap_analysis()
                for gap in gaps:
                    results.append({
                        'id': gap['article_id'],
                        'text': gap.get('description', ''),
                        'title': gap['article_title'],
                        'type': 'gap',
                        'source': 'graph_traversal',
                        'coverage_status': 'not_covered'
                    })
        
        # Risk mapping queries
        if 'risk' in query_lower or 'incident' in query_lower:
            risks = self.graph_queries.aiid_risk_mapping()
            for risk in risks[:10]:  # Limit to top 10
                results.append({
                    'id': risk['incident_id'],
                    'text': risk.get('incident_title', ''),
                    'risk_type': risk['risk_type'],
                    'violated_articles': risk['violated_articles'],
                    'type': 'risk',
                    'source': 'graph_traversal'
                })
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = None, 
                     use_faiss: bool = True, use_graph_traversal: bool = True,
                     rrf_k: int = None) -> Dict[str, Any]:
        """
        Perform hybrid search combining FAISS vector search and Neo4j graph traversal.
        Uses Reciprocal Rank Fusion (RRF) to merge results from different sources.
        
        Args:
            query: User query
            top_k: Number of results to return (defaults to DEFAULT_TOP_K from .env)
            use_faiss: Use FAISS vector search
            use_graph_traversal: Use graph traversal
            rrf_k: RRF constant (defaults to RRF_K from .env)
            
        Returns:
            Dictionary with combined results
        """
        # Use defaults from .env if not provided
        top_k = top_k if top_k is not None else DEFAULT_TOP_K
        rrf_k = rrf_k if rrf_k is not None else RRF_K
        
        # Increase top_k for mismatch queries to show both gaps and coverage
        query_lower = query.lower()
        if 'mismatch' in query_lower or 'compare' in query_lower or 'difference' in query_lower:
            top_k = max(top_k, 30)  # Ensure we show gaps + coverage + summary
        
        query_types = self.detect_query_type(query)
        
        # Separate results by source for RRF
        vector_results = []
        graph_results = []
        
        # 1. Vector Search (FAISS)
        if use_faiss and query_types['is_semantic_query']:
            try:
                faiss_results = self.vector_engine.search(
                    query=query,
                    db_names=None,  # Search all databases
                    top_k=top_k,
                    similarity_threshold=0.0
                )
                
                # Keep original similarity scores for reference, but RRF uses rank
                for result in faiss_results:
                    vector_results.append({
                        **result,
                        'source': 'faiss_vector'
                    })
            except Exception as e:
                print(f"FAISS search error: {e}")
        
        # 2. Graph Traversal
        if use_graph_traversal and query_types['is_graph_query']:
            try:
                graph_results_raw = self.graph_traversal_search(query)
                
                for result in graph_results_raw:
                    graph_results.append({
                        **result,
                        'source': 'graph_traversal'
                    })
            except Exception as e:
                print(f"Graph traversal error: {e}")
        
        # Merge results using RRF
        merged_results = self._merge_and_rank_results_rrf(
            vector_results, 
            graph_results, 
            top_k, 
            rrf_k
        )
        
        return {
            'query': query,
            'results': merged_results,
            'num_results': len(merged_results),
            'query_types': query_types,
            'sources_used': {
                'faiss': use_faiss,
                'graph_traversal': use_graph_traversal
            }
        }
    
    def _merge_and_rank_results_rrf(self, vector_results: List[Dict], 
                                     graph_results: List[Dict], 
                                     top_k: int, 
                                     k: int = RRF_K) -> List[Dict]:
        """
        Merge results from vector and graph sources using Reciprocal Rank Fusion (RRF).
        
        RRF Formula: RRF_score(d) = Σ 1 / (k + rank_i(d))
        where rank_i(d) is the rank of document d in list i (1-indexed)
        
        Args:
            vector_results: Results from vector search (already ranked by similarity)
            graph_results: Results from graph traversal
            top_k: Number of top results to return
            k: RRF constant (defaults to RRF_K from .env)
            
        Returns:
            Merged and ranked results using RRF scores
        """
        # For mismatch analysis, preserve order: summary -> coverage -> gaps
        all_results = vector_results + graph_results
        has_mismatch = any(r.get('analysis_type') == 'mismatch' for r in all_results)
        if has_mismatch:
            # Separate by type
            summary_results = [r for r in all_results if r.get('type') == 'summary']
            coverage_results = [r for r in all_results if r.get('type') == 'coverage']
            gap_results = [r for r in all_results if r.get('type') == 'gap']
            other_results = [r for r in all_results if r.get('type') not in ['summary', 'coverage', 'gap']]
            
            # Sort coverage and gaps by clause_count (for coverage) or similarity (for gaps)
            coverage_results.sort(key=lambda x: x.get('clause_count', 0), reverse=True)
            gap_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # Combine in priority order
            merged = summary_results + coverage_results + gap_results + other_results
            return merged[:top_k]
        
        # Build RRF scores
        rrf_scores = {}
        
        # Process vector results (already sorted by similarity)
        for rank, result in enumerate(vector_results, start=1):
            result_id = self._get_result_id(result)
            if result_id not in rrf_scores:
                rrf_scores[result_id] = {
                    'result': result.copy(),
                    'rrf_score': 0.0,
                    'vector_rank': None,
                    'graph_rank': None,
                    'sources': []
                }
            rrf_scores[result_id]['rrf_score'] += 1.0 / (k + rank)
            rrf_scores[result_id]['vector_rank'] = rank
            rrf_scores[result_id]['sources'].append('faiss_vector')
        
        # Process graph results
        for rank, result in enumerate(graph_results, start=1):
            result_id = self._get_result_id(result)
            if result_id not in rrf_scores:
                rrf_scores[result_id] = {
                    'result': result.copy(),
                    'rrf_score': 0.0,
                    'vector_rank': None,
                    'graph_rank': None,
                    'sources': []
                }
            rrf_scores[result_id]['rrf_score'] += 1.0 / (k + rank)
            rrf_scores[result_id]['graph_rank'] = rank
            rrf_scores[result_id]['sources'].append('graph_traversal')
        
        # Mark hybrid results (appearing in both sources)
        for result_id, data in rrf_scores.items():
            if data['vector_rank'] is not None and data['graph_rank'] is not None:
                data['result']['source'] = 'hybrid'
            elif data['vector_rank'] is not None:
                data['result']['source'] = 'faiss_vector'
            else:
                data['result']['source'] = 'graph_traversal'
            
            # Add metadata
            data['result']['rrf_score'] = data['rrf_score']
            data['result']['vector_rank'] = data['vector_rank']
            data['result']['graph_rank'] = data['graph_rank']
            data['result']['sources'] = data['sources']
        
        # Sort by RRF score (descending) and return top_k
        merged = [item['result'] for item in sorted(
            rrf_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )]
        
        return merged[:top_k]
    
    def _get_result_id(self, result: Dict) -> str:
        """
        Get a unique identifier for a result for deduplication.
        
        Args:
            result: Result dictionary
            
        Returns:
            Unique identifier string
        """
        # Try different ID fields
        if 'id' in result and result['id']:
            return str(result['id'])
        
        # Use text hash as fallback
        text = result.get('text', result.get('description', ''))
        if text:
            # Use first 100 chars as identifier
            return text[:100]
        
        # Last resort: use all available fields
        return str(result)
    
    def hybrid_query(self, query: str, top_k: int = 10,
                    rerank: bool = True, contextualize: bool = True,
                    rrf_k: int = None) -> Dict[str, Any]:
        """
        Complete hybrid query pipeline with reranking and contextualization.
        Uses Reciprocal Rank Fusion (RRF) to merge vector and graph results.
        
        Args:
            query: User query
            top_k: Number of results
            rerank: Whether to rerank results
            contextualize: Whether to generate contextualized answer
            rrf_k: RRF constant (defaults to RRF_K from .env)
            
        Returns:
            Complete query result with answer
        """
        # Perform hybrid search
        search_result = self.hybrid_search(query, top_k=top_k, rrf_k=rrf_k)
        
        results = search_result['results']
        
        # Rerank if enabled
        if rerank and results:
            results = self.vector_engine.rerank_results(query, results, top_n=min(8, len(results)))
        
        # Generate contextualized answer
        answer = None
        if contextualize and results:
            answer = self.vector_engine.contextualize_results(query, results[:8])
        
        return {
            'query': query,
            'results': results,
            'answer': answer,
            'num_results': len(results),
            'query_types': search_result['query_types'],
            'sources_used': search_result['sources_used']
        }
    
    def close(self):
        """Close Neo4j connection."""
        if self.neo4j_conn:
            self.neo4j_conn.close()


def example_hybrid_queries():
    """Example usage of hybrid query engine."""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    engine = HybridQueryEngine(str(project_root))
    
    # Example 1: Semantic query (uses vector search)
    print("=" * 80)
    print("Example 1: Semantic Query")
    print("=" * 80)
    result = engine.hybrid_query("What are the privacy policies?")
    print(f"Found {result['num_results']} results")
    print(f"Sources used: {result['sources_used']}")
    
    # Example 2: Graph query (uses graph traversal)
    print("\n" + "=" * 80)
    print("Example 2: Graph Query")
    print("=" * 80)
    result = engine.hybrid_query("Find clauses addressing GDPR Article 5")
    print(f"Found {result['num_results']} results")
    print(f"Sources used: {result['sources_used']}")
    
    # Example 3: Hybrid query (combines both)
    print("\n" + "=" * 80)
    print("Example 3: Hybrid Query")
    print("=" * 80)
    result = engine.hybrid_query("What privacy policies address data minimization requirements?")
    print(f"Found {result['num_results']} results")
    print(f"Sources used: {result['sources_used']}")
    
    engine.close()


if __name__ == "__main__":
    example_hybrid_queries()

