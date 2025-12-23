"""
Hybrid Query Engine: Combines Vector Search (FAISS) + Graph Traversal (Neo4j)
Simplified to use only FAISS vector search and Neo4j graph traversal.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
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
GRAPH_SCORE_RESULTS = os.getenv("GRAPH_SCORE_RESULTS", "true").lower() == "true"  # Score graph results by semantic similarity
GRAPH_MAX_RESULTS_FOR_RRF = int(os.getenv("GRAPH_MAX_RESULTS_FOR_RRF", "150"))  # Max graph results to send to RRF (after scoring)
GRAPH_SIMILARITY_THRESHOLD_MODE = os.getenv("GRAPH_SIMILARITY_THRESHOLD_MODE", "adaptive").lower()  # "adaptive", "fixed", "percentile"
GRAPH_SIMILARITY_THRESHOLD_FIXED = float(os.getenv("GRAPH_SIMILARITY_THRESHOLD_FIXED", "0.5"))  # Fixed threshold (if mode="fixed")
GRAPH_SIMILARITY_THRESHOLD_PERCENTILE = float(os.getenv("GRAPH_SIMILARITY_THRESHOLD_PERCENTILE", "0.3"))  # Percentile threshold (if mode="percentile", e.g., 0.3 = bottom 30% filtered)
MIN_GRAPH_RESULTS_FOR_RRF = int(os.getenv("MIN_GRAPH_RESULTS_FOR_RRF", "3"))  # Minimum graph results needed to use RRF


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
        Detect what type of query this is for informational/debugging purposes.
        Note: In hybrid search, both vector and graph search are always used if enabled,
        regardless of query type detection. This is just for informational purposes.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary indicating query types
        """
        query_lower = query.lower()
        
        # Strong graph-specific patterns (queries that clearly need graph traversal)
        strong_graph_patterns = [
            'find clauses for', 'clauses addressing', 'incidents violating',
            'show relationships', 'what articles', 'which articles',
            'compliance gap', 'coverage gap', 'not covered'
        ]
        
        # Moderate graph patterns (may benefit from graph but also semantic)
        moderate_graph_patterns = [
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
        
        # Check for strong graph patterns
        is_strong_graph_query = any(pattern in query_lower for pattern in strong_graph_patterns)
        # Check for moderate graph patterns
        has_graph_elements = any(pattern in query_lower for pattern in moderate_graph_patterns)
        is_relationship_query = any(pattern in query_lower for pattern in relationship_patterns)
        
        # In hybrid search, queries can be BOTH semantic AND graph queries
        # Most queries benefit from both methods
        is_graph_query = is_strong_graph_query or has_graph_elements
        is_semantic_query = True  # Always true - semantic search works for all queries
        
        return {
            'is_graph_query': is_graph_query,
            'is_relationship_query': is_relationship_query,
            'is_semantic_query': is_semantic_query
        }
    
    def _score_graph_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Score graph results by semantic similarity to the query.
        Adds 'similarity' score to each result for ranking.
        Uses batch embedding generation for efficiency.
        
        Args:
            query: User query
            results: List of graph results
            
        Returns:
            List of results with similarity scores added
        """
        if not results:
            return results
        
        try:
            # Import embedding functions
            from query_engine import get_query_embedding
            from local_embeddings import is_local_embeddings_enabled, generate_local_embeddings_batch
            import os
            
            # Get query embedding
            query_embedding = get_query_embedding(query)
            if query_embedding is None or query_embedding.size == 0:
                # If embedding fails, return results with default score
                for result in results:
                    result['similarity'] = 0.0
                return results
            
            # Collect all texts to embed (batch processing)
            texts_to_embed = []
            result_indices = []  # Track which result each text belongs to
            
            for idx, result in enumerate(results):
                text_to_score = result.get('text', '') or result.get('title', '') or result.get('description', '')
                if text_to_score:
                    texts_to_embed.append(text_to_score)
                    result_indices.append(idx)
                else:
                    result['similarity'] = 0.0
            
            if not texts_to_embed:
                return results
            
            # Batch embed all texts at once (much faster!)
            use_local = is_local_embeddings_enabled()
            if use_local and os.getenv("USE_LOCAL_EMBEDDINGS", "auto").lower() in ["true", "auto"]:
                # Use batch local embeddings
                result_embeddings = generate_local_embeddings_batch(texts_to_embed)
            else:
                # Fallback: try to batch via API or do one-by-one
                # For API, we'd need to batch the API calls, but for now fallback to one-by-one
                # This is still better than before because we only process texts that exist
                result_embeddings = None
                if len(texts_to_embed) > 1:
                    # Try local batch as fallback even if not primary
                    result_embeddings = generate_local_embeddings_batch(texts_to_embed)
            
            if result_embeddings is not None and result_embeddings.size > 0:
                # Calculate similarities for all results at once
                for i, result_idx in enumerate(result_indices):
                    similarity = float(np.dot(query_embedding[0], result_embeddings[i]))
                    results[result_idx]['similarity'] = similarity
            else:
                # Fallback to one-by-one if batch failed
                print(f"Warning: Batch embedding failed, falling back to one-by-one (slower)")
                for i, result_idx in enumerate(result_indices):
                    try:
                        result_embedding = get_query_embedding(texts_to_embed[i])
                        if result_embedding is not None and result_embedding.size > 0:
                            similarity = float(np.dot(query_embedding[0], result_embedding[0]))
                            results[result_idx]['similarity'] = similarity
                        else:
                            results[result_idx]['similarity'] = 0.0
                    except Exception as e:
                        print(f"Warning: Failed to score result {results[result_idx].get('id', 'unknown')}: {e}")
                        results[result_idx]['similarity'] = 0.0
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
            
        except Exception as e:
            print(f"Warning: Graph result scoring failed: {e}")
            # Return results with default scores
            for result in results:
                result['similarity'] = 0.0
        
        return results
    
    def _calculate_adaptive_similarity_threshold(
        self, 
        graph_results: List[Dict], 
        vector_results: List[Dict] = None
    ) -> float:
        """
        Calculate adaptive similarity threshold for filtering graph results.
        
        Adaptive strategies:
        1. If vector results available: Use median of vector similarities as reference
        2. If no vector results: Use percentile-based threshold (filter bottom 30%)
        3. Fallback: Use fixed threshold (0.5)
        
        Args:
            graph_results: Graph results with similarity scores
            vector_results: Optional vector results for reference
            
        Returns:
            Adaptive threshold value (0.0-1.0)
        """
        if not graph_results:
            return 0.5  # Default threshold
        
        # Get similarity scores from graph results
        graph_similarities = [r.get('similarity', 0.0) for r in graph_results if r.get('similarity') is not None]
        if not graph_similarities:
            return 0.5  # No similarities available
        
        if GRAPH_SIMILARITY_THRESHOLD_MODE == "fixed":
            return GRAPH_SIMILARITY_THRESHOLD_FIXED
        
        elif GRAPH_SIMILARITY_THRESHOLD_MODE == "percentile":
            # Filter bottom X% (e.g., bottom 30%)
            percentile_idx = int(len(graph_similarities) * GRAPH_SIMILARITY_THRESHOLD_PERCENTILE)
            sorted_sims = sorted(graph_similarities)
            if percentile_idx >= len(sorted_sims):
                return sorted_sims[0] if sorted_sims else 0.5
            return sorted_sims[percentile_idx]
        
        else:  # adaptive mode
            # Strategy 1: If vector results available, use median vector similarity as reference
            if vector_results and len(vector_results) > 0:
                vector_similarities = [r.get('similarity', 0.0) for r in vector_results if r.get('similarity') is not None]
                if vector_similarities:
                    import statistics
                    vector_median = statistics.median(vector_similarities)
                    # Use vector median as reference, but allow graph results slightly below
                    # (since graph results are scored differently)
                    threshold = max(0.3, vector_median - 0.2)  # At least 0.3, or vector_median - 0.2
                    return threshold
            
            # Strategy 2: Use percentile-based (filter bottom 30% of graph results)
            percentile_idx = int(len(graph_similarities) * 0.3)
            sorted_sims = sorted(graph_similarities)
            if percentile_idx >= len(sorted_sims):
                return sorted_sims[0] if sorted_sims else 0.5
            threshold = sorted_sims[percentile_idx]
            
            # Ensure threshold is reasonable (not too low)
            return max(0.3, threshold)
    
    def _filter_low_quality_graph_results(
        self, 
        graph_results: List[Dict], 
        vector_results: List[Dict] = None
    ) -> Tuple[List[Dict], float]:
        """
        Filter low-quality graph results based on adaptive similarity threshold.
        
        Args:
            graph_results: Graph results with similarity scores
            vector_results: Optional vector results for adaptive threshold calculation
            
        Returns:
            Tuple of (filtered_results, threshold_used)
        """
        if not graph_results:
            return [], 0.5
        
        # Calculate adaptive threshold
        threshold = self._calculate_adaptive_similarity_threshold(graph_results, vector_results)
        
        # Filter results above threshold
        filtered = [r for r in graph_results if r.get('similarity', 0.0) >= threshold]
        
        return filtered, threshold
    
    def graph_traversal_search(self, query: str, top_k: int = None, score_results: bool = None) -> List[Dict]:
        """
        Perform graph traversal search based on query patterns.
        
        Args:
            query: User query
            top_k: Optional limit on number of results (if None, returns all)
            score_results: Whether to score results by semantic similarity (default: GRAPH_SCORE_RESULTS from .env)
            
        Returns:
            List of results from graph traversal, sorted by relevance if scored
        """
        if score_results is None:
            score_results = GRAPH_SCORE_RESULTS
        
        query_lower = query.lower()
        results = []
        
        # Extract article ID if mentioned
        import re
        article_match = re.search(r'art(?:icle)?\s*(\d+)', query_lower)
        
        if article_match:
            article_id = f"Art{article_match.group(1)}"
            print(f"  → Found article ID: {article_id}")
            
            # Check if query specifically asks for clauses or incidents
            asks_for_clauses = 'clause' in query_lower or 'addressing' in query_lower
            asks_for_incidents = 'incident' in query_lower or 'violating' in query_lower or 'violates' in query_lower
            
            # If query doesn't specify, return both; otherwise prioritize what's asked
            if not asks_for_clauses and not asks_for_incidents:
                asks_for_clauses = True  # Default to clauses if not specified
                asks_for_incidents = True
            
            # Find clauses addressing this article
            if asks_for_clauses:
                print(f"  → Searching for clauses addressing {article_id}...")
                clauses = self.graph_queries.find_clauses_by_article(article_id)
                print(f"  → Found {len(clauses)} clauses")
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
            else:
                print(f"  → Skipping clauses (query asks for incidents)")
            
            # Find incidents violating this article
            if asks_for_incidents:
                print(f"  → Searching for incidents violating {article_id}...")
                incidents = self.graph_queries.find_incidents_by_article(article_id)
                print(f"  → Found {len(incidents)} incidents")
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
            else:
                print(f"  → Skipping incidents (query asks for clauses)")
        
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
        
        # Score results by semantic similarity to query
        if score_results:
            results = self._score_graph_results(query, results)
        
        # Apply top_k limit if specified
        if top_k is not None:
            results = results[:top_k]
        
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
        
        # Debug: Print query type detection
        print(f"\n{'='*80}")
        print(f"HYBRID SEARCH DEBUG")
        print(f"{'='*80}")
        print(f"Query: '{query}'")
        print(f"Top K: {top_k}")
        print(f"RRF K: {rrf_k}")
        print(f"\nQuery Type Detection:")
        print(f"  - is_semantic_query: {query_types['is_semantic_query']}")
        print(f"  - is_graph_query: {query_types['is_graph_query']}")
        print(f"  - is_relationship_query: {query_types['is_relationship_query']}")
        print(f"\nSearch Settings:")
        print(f"  - use_faiss: {use_faiss}")
        print(f"  - use_graph_traversal: {use_graph_traversal}")
        
        # Separate results by source for RRF
        vector_results = []
        graph_results = []
        
        # 1. Vector Search (FAISS)
        # Always run vector search if enabled (hybrid search should use both methods)
        if use_faiss:
            print(f"\n{'─'*80}")
            print("STEP 1: Vector Search (FAISS)")
            print(f"{'─'*80}")
            try:
                faiss_results = self.vector_engine.search(
                    query=query,
                    db_names=None,  # Search all databases
                    top_k=top_k,
                    similarity_threshold=0.0
                )
                
                print(f"  FAISS returned {len(faiss_results)} results")
                if faiss_results:
                    print(f"  Similarity range: {faiss_results[0]['similarity']:.4f} to {faiss_results[-1]['similarity']:.4f}")
                    print(f"  Top 3 results:")
                    for i, r in enumerate(faiss_results[:3], 1):
                        print(f"    {i}. [{r.get('database', 'unknown')}] similarity={r['similarity']:.4f} | {r['text'][:60]}...")
                
                # Keep original similarity scores for reference, but RRF uses rank
                for result in faiss_results:
                    vector_results.append({
                        **result,
                        'source': 'faiss_vector'
                    })
                print(f"  → Added {len(vector_results)} vector results")
            except Exception as e:
                print(f"  ❌ FAISS search error: {e}")
        else:
            print(f"\n{'─'*80}")
            print("STEP 1: Vector Search (FAISS) - SKIPPED")
            print(f"{'─'*80}")
            print(f"  Reason: use_faiss=False")
        
        # 2. Graph Traversal
        # Always run graph traversal if enabled (hybrid search should use both methods)
        if use_graph_traversal:
            print(f"\n{'─'*80}")
            print("STEP 2: Graph Traversal (Neo4j)")
            print(f"{'─'*80}")
            try:
                graph_results_raw = self.graph_traversal_search(query, top_k=None, score_results=True)
                print(f"  Graph traversal returned {len(graph_results_raw)} results")
                
                # Limit graph results before filtering for performance (results are already scored and sorted)
                if len(graph_results_raw) > GRAPH_MAX_RESULTS_FOR_RRF:
                    print(f"  → Limiting graph results from {len(graph_results_raw)} to {GRAPH_MAX_RESULTS_FOR_RRF} (top scored) before filtering")
                    graph_results_raw = graph_results_raw[:GRAPH_MAX_RESULTS_FOR_RRF]
                
                # Count by type (before filtering)
                type_counts_before = {}
                for r in graph_results_raw:
                    r_type = r.get('type', 'unknown')
                    type_counts_before[r_type] = type_counts_before.get(r_type, 0) + 1
                if type_counts_before:
                    print(f"  Results by type (before filtering): {type_counts_before}")
                
                # Apply adaptive similarity filtering (Option B)
                graph_results_filtered, threshold_used = self._filter_low_quality_graph_results(
                    graph_results_raw, 
                    vector_results if use_faiss else None
                )
                
                print(f"  → Adaptive similarity threshold: {threshold_used:.4f} (mode: {GRAPH_SIMILARITY_THRESHOLD_MODE})")
                print(f"  → Filtered from {len(graph_results_raw)} to {len(graph_results_filtered)} results")
                
                if graph_results_filtered:
                    print(f"  Top 3 filtered results:")
                    for i, r in enumerate(graph_results_filtered[:3], 1):
                        r_type = r.get('type', 'unknown')
                        similarity = r.get('similarity', 0.0)
                        r_text = r.get('text', r.get('description', ''))[:60]
                        print(f"    {i}. [{r_type}] similarity={similarity:.4f} | {r_text}...")
                
                # Only add filtered results
                for result in graph_results_filtered:
                    graph_results.append({
                        **result,
                        'source': 'graph_traversal'
                    })
                print(f"  → Added {len(graph_results)} filtered graph results to RRF")
            except Exception as e:
                print(f"  ❌ Graph traversal error: {e}")
        else:
            print(f"\n{'─'*80}")
            print("STEP 2: Graph Traversal (Neo4j) - SKIPPED")
            print(f"{'─'*80}")
            print(f"  Reason: use_graph_traversal=False")
        
        # Merge results using RRF (only if both sources have results)
        print(f"\n{'─'*80}")
        print("STEP 3: Result Merging")
        print(f"{'─'*80}")
        print(f"  Vector results: {len(vector_results)}")
        print(f"  Graph results (after filtering): {len(graph_results)}")
        print(f"  Total before merge: {len(vector_results) + len(graph_results)}")
        
        # Option B: Skip graph if filtered results are too few
        if len(graph_results) < MIN_GRAPH_RESULTS_FOR_RRF:
            if len(graph_results) > 0:
                print(f"  → Graph results ({len(graph_results)}) below minimum ({MIN_GRAPH_RESULTS_FOR_RRF}): skipping graph, using vector only")
            else:
                print(f"  → No graph results after filtering: using vector only")
            if len(vector_results) > 0:
                merged_results = vector_results[:top_k]
            else:
                merged_results = []
        # If only one source has results, skip RRF (already sorted)
        elif len(vector_results) > 0 and len(graph_results) == 0:
            print(f"  → Only vector results: skipping RRF, returning sorted vector results")
            merged_results = vector_results[:top_k]
        elif len(graph_results) > 0 and len(vector_results) == 0:
            print(f"  → Only graph results: skipping RRF, returning graph results")
            merged_results = graph_results[:top_k]
        elif len(vector_results) > 0 and len(graph_results) > 0:
            print(f"  → Both sources have results ({len(vector_results)} vector, {len(graph_results)} graph): using RRF to merge")
            merged_results = self._merge_and_rank_results_rrf(
                vector_results, 
                graph_results, 
                top_k, 
                rrf_k
            )
        else:
            print(f"  → No results from either source")
            merged_results = []
        
        print(f"  → Merged to {len(merged_results)} results")
        if merged_results:
            print(f"  Top 3 merged results:")
            for i, r in enumerate(merged_results[:3], 1):
                source = r.get('source', 'unknown')
                rrf_score = r.get('rrf_score', 0)
                v_rank = r.get('vector_rank', 'N/A')
                g_rank = r.get('graph_rank', 'N/A')
                r_text = r.get('text', r.get('description', ''))[:50]
                print(f"    {i}. [{source}] RRF={rrf_score:.6f} | V_rank={v_rank} G_rank={g_rank} | {r_text}...")
        
        print(f"\n{'='*80}\n")
        
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
            print(f"  → Using mismatch analysis ordering (summary → coverage → gaps)")
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
        
        print(f"  → Using RRF formula: score = Σ 1/(k + rank), where k={k}")
        
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
        
        # Debug: Show RRF scoring details
        if len(rrf_scores) > 0:
            print(f"  → Calculated RRF scores for {len(rrf_scores)} unique results")
            hybrid_count = sum(1 for item in rrf_scores.values() 
                             if item['vector_rank'] is not None and item['graph_rank'] is not None)
            if hybrid_count > 0:
                print(f"  → Found {hybrid_count} results appearing in both sources (hybrid)")
        
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
    
    def graph_query(self, query: str, top_k: int = None,
                   rerank: bool = True, generate_answer: bool = True) -> Dict[str, Any]:
        """
        Complete graph query pipeline with scoring, reranking, and answer generation.
        Standalone graph search method matching evaluation behavior.
        
        Args:
            query: User query
            top_k: Number of results (not applied, uses all scored results)
            rerank: Whether to rerank results (recommended: True)
            generate_answer: Whether to generate answer using LLM (recommended: True)
            
        Returns:
            Complete query result with answer
        """
        # Step 1: Graph traversal + semantic scoring
        all_graph_results = self.graph_traversal_search(query, top_k=None, score_results=True)
        # Results are already scored and sorted by similarity
        
        results = all_graph_results
        
        # Step 2: Rerank if enabled
        if rerank and results:
            results = self.vector_engine.rerank_results(query, results, top_n=min(8, len(results)))
        
        # Step 3: Generate answer if enabled
        answer = None
        if generate_answer and results:
            answer = self.vector_engine.generate_answer(query, results[:8])
        
        return {
            'query': query,
            'results': results,
            'answer': answer,
            'num_results': len(results),
            'sources_used': {'graph_traversal': True}
        }
    
    def hybrid_query(self, query: str, top_k: int = 10,
                    rerank: bool = True, generate_answer: bool = True,
                    rrf_k: int = None) -> Dict[str, Any]:
        """
        Complete hybrid query pipeline with reranking and answer generation.
        Uses Reciprocal Rank Fusion (RRF) to merge vector and graph results.
        
        Args:
            query: User query
            top_k: Number of results
            rerank: Whether to rerank results
            generate_answer: Whether to generate answer using LLM
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
        
        # Generate answer
        answer = None
        if generate_answer and results:
            answer = self.vector_engine.generate_answer(query, results[:8])
        
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

