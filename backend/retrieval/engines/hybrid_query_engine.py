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

from .query_engine import VectorQueryEngine
from .graph_query_engine import GraphQueryEngine
from ..utils.neo4j_queries import KnowledgeGraphQueries
from backend.indexing.neo4j.utils.neo4j_connection import Neo4jConnection

load_dotenv()

# Configuration from environment variables
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
    
    def _calculate_adaptive_top_k(self, query: str, vector_results: List[Dict] = None, 
                                   graph_results: List[Dict] = None) -> int:
        """
        Calculate adaptive top_k based on query characteristics and available results.
        
        Adaptive strategy:
        1. Base top_k on query type and complexity:
           - Simple semantic queries: 5-8 results
           - Graph/relationship queries: 12-15 results
           - Comparison/mismatch queries: 25-30 results
        2. Adjust based on available high-quality results:
           - If many high-quality results (>0.7 similarity), increase top_k
           - If few results, use what's available
        3. Query length/complexity:
           - Longer queries (>50 chars): more results
           - Multiple keywords: more results
        
        Args:
            query: User query string
            vector_results: Optional vector results (for quality assessment)
            graph_results: Optional graph results (for quality assessment)
            
        Returns:
            Adaptive top_k value
        """
        query_lower = query.lower()
        query_length = len(query)
        
        # Base top_k based on query type
        base_top_k = 8  # Default for simple queries
        
        # Check for comparison/mismatch queries (need more results)
        if any(keyword in query_lower for keyword in ['mismatch', 'compare', 'difference', 'gap', 'coverage']):
            base_top_k = 30  # Need to show gaps + coverage + summary
        
        # Check for graph/relationship queries (need more results)
        elif any(keyword in query_lower for keyword in ['article', 'clause', 'incident', 'relationship', 'violates', 'addresses']):
            base_top_k = 15  # Graph queries often need more context
        
        # Check for complex queries (multiple concepts)
        elif any(keyword in query_lower for keyword in ['and', 'or', 'both', 'all', 'list', 'show']):
            base_top_k = 12  # Multi-concept queries
        
        # Adjust based on query length
        if query_length > 50:
            base_top_k = int(base_top_k * 1.2)  # Longer queries may need more results
        elif query_length < 20:
            base_top_k = max(5, int(base_top_k * 0.8))  # Shorter queries need fewer
        
        # Adjust based on available high-quality results
        if vector_results is not None or graph_results is not None:
            # Count high-quality results (similarity > 0.7)
            high_quality_count = 0
            total_results = 0
            
            if vector_results:
                for r in vector_results:
                    total_results += 1
                    if r.get('similarity', 0.0) > 0.7:
                        high_quality_count += 1
            
            if graph_results:
                for r in graph_results:
                    total_results += 1
                    if r.get('similarity', 0.0) > 0.7:
                        high_quality_count += 1
            
            # If we have many high-quality results, increase top_k
            if high_quality_count > 10:
                base_top_k = max(base_top_k, min(high_quality_count + 5, 30))
            # If we have few results total, use what's available (but at least base_top_k)
            elif total_results > 0:
                base_top_k = max(base_top_k, min(total_results, 20))
        
        # Ensure reasonable bounds
        return max(5, min(base_top_k, 50))  # Between 5 and 50
    
    def _score_graph_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Delegate to graph engine's scoring method."""
        return self.graph_engine._score_graph_results(query, results)
    
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
        vector_similarities: List[float] = None
    ) -> List[Dict]:
        """Delegate to graph engine's filtering method."""
        if vector_similarities is None:
            vector_similarities = []
        return self.graph_engine._filter_low_quality_graph_results(graph_results, vector_similarities)
    
    def graph_traversal_search(self, query: str, top_k: int = None, score_results: bool = None) -> List[Dict]:
        """
        Perform graph traversal search based on query patterns.
        Delegates to GraphQueryEngine.
        
        Args:
            query: User query
            top_k: Optional limit on number of results (if None, returns all)
            score_results: Whether to score results by semantic similarity (default: GRAPH_SCORE_RESULTS from .env)
            
        Returns:
            List of results from graph traversal, sorted by relevance if scored
        """
        return self.graph_engine.search(query, top_k=top_k, score_results=score_results)
    
    def hybrid_search(self, query: str, top_k: int = None, 
                     use_faiss: bool = True, use_graph_traversal: bool = True,
                     rrf_k: int = None) -> Dict[str, Any]:
        """
        Perform hybrid search combining FAISS vector search and Neo4j graph traversal.
        Uses Reciprocal Rank Fusion (RRF) to merge results from different sources.
        
        Args:
            query: User query
            top_k: Number of results to return (if None, uses adaptive calculation)
            use_faiss: Use FAISS vector search
            use_graph_traversal: Use graph traversal
            rrf_k: RRF constant (defaults to RRF_K from .env)
            
        Returns:
            Dictionary with combined results
        """
        # Use adaptive top_k if not provided
        if top_k is None:
            # Calculate adaptive top_k (will be refined after getting initial results)
            top_k = self._calculate_adaptive_top_k(query)
        
        rrf_k = rrf_k if rrf_k is not None else RRF_K
        
        query_types = self.detect_query_type(query)
        
        # Debug output (only if VERBOSE env var is set)
        verbose = os.getenv("VERBOSE", "false").lower() == "true"
        if verbose:
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
            if verbose:
                print(f"\n{'─'*80}")
                print("STEP 1: Vector Search (FAISS)")
                print(f"{'─'*80}")
            try:
                # Use a larger initial top_k for vector search to get more candidates
                # We'll refine top_k after seeing results quality
                vector_search_k = max(top_k * 2, 20)  # Get more candidates for better adaptive selection
                faiss_results = self.vector_engine.search(
                    query=query,
                    db_names=None,  # Search all databases
                    top_k=vector_search_k,
                    similarity_threshold=0.0
                )
                
                if verbose:
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
                if verbose:
                    print(f"  -> Added {len(vector_results)} vector results")
            except Exception as e:
                if verbose:
                    print(f"  ❌ FAISS search error: {e}")
                else:
                    print(f"⚠️  Search error: {str(e)[:100]}")
        else:
            if verbose:
                print(f"\n{'─'*80}")
                print("STEP 1: Vector Search (FAISS) - SKIPPED")
                print(f"{'─'*80}")
                print(f"  Reason: use_faiss=False")
        
        # 2. Graph Traversal
        # Always run graph traversal if enabled (hybrid search should use both methods)
        if use_graph_traversal:
            if verbose:
                print(f"\n{'─'*80}")
                print("STEP 2: Graph Traversal (Neo4j)")
                print(f"{'─'*80}")
            try:
                graph_results_raw = self.graph_engine.search(query, top_k=None, score_results=True)
                if verbose:
                    print(f"  Graph traversal returned {len(graph_results_raw)} results")
                
                # Limit graph results before filtering for performance (results are already scored and sorted)
                if len(graph_results_raw) > GRAPH_MAX_RESULTS_FOR_RRF:
                    if verbose:
                        print(f"  -> Limiting graph results from {len(graph_results_raw)} to {GRAPH_MAX_RESULTS_FOR_RRF} (top scored) before filtering")
                    graph_results_raw = graph_results_raw[:GRAPH_MAX_RESULTS_FOR_RRF]
                
                # Count by type (before filtering)
                type_counts_before = {}
                for r in graph_results_raw:
                    r_type = r.get('type', 'unknown')
                    type_counts_before[r_type] = type_counts_before.get(r_type, 0) + 1
                if verbose and type_counts_before:
                    print(f"  Results by type (before filtering): {type_counts_before}")
                
                # Apply adaptive similarity filtering (Option B)
                vector_similarities = [r.get('similarity', 0.0) for r in vector_results] if use_faiss and vector_results else []
                graph_results_filtered = self._filter_low_quality_graph_results(
                    graph_results_raw, 
                    vector_similarities
                )
                
                if verbose:
                    print(f"  -> Filtered from {len(graph_results_raw)} to {len(graph_results_filtered)} results")
                    
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
                if verbose:
                    print(f"  -> Added {len(graph_results)} filtered graph results to RRF")
            except Exception as e:
                if verbose:
                    print(f"  ❌ Graph traversal error: {e}")
                # Don't print graph errors if not verbose - they're often expected
        else:
            if verbose:
                print(f"\n{'─'*80}")
                print("STEP 2: Graph Traversal (Neo4j) - SKIPPED")
                print(f"{'─'*80}")
                print(f"  Reason: use_graph_traversal=False")
        
        # Refine top_k adaptively based on actual results quality
        # Only refine if we have results to assess
        if vector_results or graph_results:
            refined_top_k = self._calculate_adaptive_top_k(query, vector_results, graph_results)
            if verbose and refined_top_k != top_k:
                print(f"  -> Adaptive top_k adjustment: {top_k} → {refined_top_k} (based on result quality)")
                top_k = refined_top_k
        
        # Merge results using RRF (only if both sources have results)
        if verbose:
            print(f"\n{'─'*80}")
            print("STEP 3: Result Merging")
            print(f"{'─'*80}")
            print(f"  Vector results: {len(vector_results)}")
            print(f"  Graph results (after filtering): {len(graph_results)}")
            print(f"  Total before merge: {len(vector_results) + len(graph_results)}")
        
        # Option B: Skip graph if filtered results are too few
        if len(graph_results) < MIN_GRAPH_RESULTS_FOR_RRF:
            if len(vector_results) > 0:
                merged_results = vector_results[:top_k]
            else:
                merged_results = []
        # If only one source has results, skip RRF (already sorted)
        elif len(vector_results) > 0 and len(graph_results) == 0:
            merged_results = vector_results[:top_k]
        elif len(graph_results) > 0 and len(vector_results) == 0:
            merged_results = graph_results[:top_k]
        elif len(vector_results) > 0 and len(graph_results) > 0:
            merged_results = self._merge_and_rank_results_rrf(
                vector_results, 
                graph_results, 
                top_k, 
                rrf_k
            )
        else:
            merged_results = []
        
        if verbose:
            print(f"  -> Merged to {len(merged_results)} results")
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
            print(f"  -> Using mismatch analysis ordering (summary → coverage → gaps)")
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
        
        print(f"  -> Using RRF formula: score = sum(1/(k + rank)), where k={k}")
        
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
            print(f"  -> Calculated RRF scores for {len(rrf_scores)} unique results")
            hybrid_count = sum(1 for item in rrf_scores.values() 
                             if item['vector_rank'] is not None and item['graph_rank'] is not None)
            if hybrid_count > 0:
                print(f"  -> Found {hybrid_count} results appearing in both sources (hybrid)")
        
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
        all_graph_results = self.graph_engine.search(query, top_k=None, score_results=True)
        # Results are already scored and sorted by similarity
        
        results = all_graph_results
        
        # Step 2: Rerank if enabled
        if rerank and results:
            try:
                results = self.vector_engine.rerank_results(query, results, top_n=min(8, len(results)))
            except Exception as e:
                error_msg = str(e)
                if '429' in error_msg or 'quota' in error_msg.lower() or 'spending limit' in error_msg.lower():
                    print(f"\n⚠️  API quota/rate limit reached. Skipping reranking. Results shown without reranking.")
                else:
                    print(f"\n⚠️  Reranking failed: {error_msg}. Results shown without reranking.")
        
        # Step 3: Generate answer if enabled
        answer = None
        if generate_answer and results:
            try:
                answer = self.vector_engine.generate_answer(query, results[:8])
            except Exception as e:
                error_msg = str(e)
                if '429' in error_msg or 'quota' in error_msg.lower() or 'spending limit' in error_msg.lower():
                    print(f"\n⚠️  API quota/rate limit reached. Cannot generate answer.")
                    print(f"   Showing {len(results)} search results instead.")
                else:
                    print(f"\n⚠️  Answer generation failed: {error_msg}")
                    print(f"   Showing {len(results)} search results instead.")
                answer = None
        
        return {
            'query': query,
            'results': results,
            'answer': answer,
            'num_results': len(results),
            'sources_used': {'graph_traversal': True}
        }
    
    def hybrid_query(self, query: str, top_k: int = None,
                    rerank: bool = True, generate_answer: bool = True,
                    rrf_k: int = None) -> Dict[str, Any]:
        """
        Complete hybrid query pipeline with reranking and answer generation.
        Uses Reciprocal Rank Fusion (RRF) to merge vector and graph results.
        
        Args:
            query: User query
            top_k: Number of results (if None, uses adaptive calculation based on query)
            rerank: Whether to rerank results
            generate_answer: Whether to generate answer using LLM
            rrf_k: RRF constant (defaults to RRF_K from .env)
            
        Returns:
            Complete query result with answer
        """
        # Perform hybrid search (top_k=None triggers adaptive calculation)
        search_result = self.hybrid_search(query, top_k=top_k, rrf_k=rrf_k)
        
        results = search_result['results']
        
        # Rerank if enabled
        if rerank and results:
            try:
                results = self.vector_engine.rerank_results(query, results, top_n=min(8, len(results)))
            except Exception as e:
                # Silently skip reranking on API errors - results are still useful
                answer = None
        
        # Generate answer
        answer = None
        if generate_answer and results:
            try:
                answer = self.vector_engine.generate_answer(query, results[:8])
            except Exception as e:
                # Silently skip answer generation on API errors - results are still useful
                answer = None
        
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

