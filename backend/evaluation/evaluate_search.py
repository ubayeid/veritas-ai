"""
Evaluation Framework for Vector vs Graph Search
Measures timing, accuracy, and other metrics for comparison.

VALIDATION METHODOLOGY:
=====================

1. LLM-Based Accuracy Evaluation (Primary Method)
   - Uses a judge LLM model (JUDGE_LLM_MODEL from .env) to evaluate results
   - For Generated Answers:
     * Evaluates: Relevance (40%), Accuracy (30%), Completeness (20%), Clarity (10%)
     * Judge receives: query + top 5 results + generated answer
     * Returns: Score 0.0-1.0 with reasoning
   
   - For Raw Search Results:
     * Evaluates: Relevance (40%), Coverage (30%), Quality (20%), Diversity (10%)
     * Judge receives: query + top 8 results with similarity scores
     * Returns: Score 0.0-1.0 with reasoning
   
   - Configuration:
     * Temperature: 0.1 (for consistency)
     * Scores clamped to [0, 1] range
     * JSON response parsed with regex fallback

2. Traditional IR Metrics (Optional, requires ground truth)
   - Precision@k: Fraction of top-k results that are relevant
   - Recall@k: Fraction of relevant items retrieved
   - MRR: Mean Reciprocal Rank (position of first relevant result)
   - NDCG@k: Normalized Discounted Cumulative Gain

3. Performance Metrics
   - Execution time (ms): Total query time
   - Answer generation time (ms): LLM generation time
   - Result count: Number of results retrieved
   - Average similarity: Mean similarity score (vector search)

4. Quality Indicators
   - Source distribution: Breakdown by database/source
   - Answer length: Characters in generated answer
   - Result diversity: How diverse are the results

The evaluation framework compares three search methods:
- Vector: Semantic similarity search using embeddings
- Graph: Knowledge graph traversal using Neo4j
- Hybrid: Combination of vector + graph with RRF fusion
"""

import time
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics
from collections import defaultdict

import sys
import os
import re

# Statistical analysis imports
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Statistical tests will be skipped. Install with: pip install scipy")
# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "searching"))
sys.path.insert(0, str(Path(__file__).parent.parent / "building_database" / "neo4j"))

from query_engine import VectorQueryEngine
from hybrid_query_engine import HybridQueryEngine
from neo4j_queries import KnowledgeGraphQueries
from neo4j_connection import Neo4jConnection
from dotenv import load_dotenv

# Import unified API client
sys.path.insert(0, str(Path(__file__).parent.parent / "searching"))
from api_client import get_api_client, get_llm_model

load_dotenv()
# Both models read from .env file - use unified client
LLM_MODEL = get_llm_model()
# Judge LLM for evaluation - reads from .env, falls back to LLM_MODEL if not set
# Set JUDGE_LLM_MODEL in .env to use a different model for evaluation
JUDGE_LLM_MODEL = os.getenv("JUDGE_LLM_MODEL", LLM_MODEL)


@dataclass
class SearchMetrics:
    """Metrics for a single search query."""
    query: str
    method: str  # 'vector', 'graph', 'hybrid'
    execution_time_ms: float
    num_results: int
    top_k: int
    # Answer generation metrics
    answer_generation_time_ms: Optional[float] = None
    generated_answer: Optional[str] = None
    answer_length: Optional[int] = None
    has_answer_generation: bool = False  # Whether answer generation was used
    # Accuracy metrics
    accuracy_score: Optional[float] = None  # LLM-based accuracy score (0-1)
    accuracy_reasoning: Optional[str] = None  # Explanation of accuracy score
    # Quality metrics (if ground truth available)
    precision_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    f1_score: Optional[float] = None  # F1 score = 2 * (precision * recall) / (precision + recall)
    mrr: Optional[float] = None  # Mean Reciprocal Rank
    map_score: Optional[float] = None  # Mean Average Precision
    ndcg_at_k: Optional[float] = None  # Normalized Discounted Cumulative Gain
    # Result quality indicators
    avg_similarity: Optional[float] = None  # For vector search
    result_diversity: Optional[float] = None  # How diverse are results
    # Source distribution
    sources: Optional[Dict[str, int]] = None
    # Coverage metrics (for compliance analysis)
    coverage_score: Optional[float] = None  # Percentage of relevant items covered


class SearchEvaluator:
    """Evaluates and compares vector vs graph search performance."""
    
    def __init__(self, base_dir: str):
        """
        Initialize evaluator.
        
        Args:
            base_dir: Base directory of the project
        """
        self.base_dir = Path(base_dir)
        
        # Initialize engines
        self.vector_engine = VectorQueryEngine(str(base_dir))
        
        # Initialize graph engine
        self.neo4j_conn = Neo4jConnection()
        if not self.neo4j_conn.verify_connectivity():
            raise ConnectionError("Failed to connect to Neo4j")
        self.graph_queries = KnowledgeGraphQueries(self.neo4j_conn)
        
        # Initialize hybrid engine
        self.hybrid_engine = HybridQueryEngine(str(base_dir))
    
    def evaluate_answer_accuracy(
        self,
        query: str,
        answer: Optional[str],
        results: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Evaluate answer accuracy using LLM-based evaluation.
        
        VALIDATION METHODOLOGY:
        Uses an LLM judge model (JUDGE_LLM_MODEL) to evaluate generated answers.
        The judge evaluates on 4 dimensions with weighted scoring:
        1. Relevance (0.0-0.4): Does the answer address the query?
        2. Accuracy (0.0-0.3): Is information factually correct based on context?
        3. Completeness (0.0-0.2): Does it cover key points from context?
        4. Clarity (0.0-0.1): Is it well-structured and understandable?
        
        The judge receives:
        - The original query
        - Top 5 search results (truncated to 500 chars each)
        - The generated answer
        
        Scoring: Returns a score 0.0-1.0 with reasoning. Uses temperature=0.1
        for consistency. Scores are clamped to [0, 1] range.
        
        Args:
            query: Original query
            answer: Generated answer (None if no answer generation)
            results: Search results used to generate answer
            
        Returns:
            Tuple of (accuracy_score, reasoning)
        """
        if answer is None or not answer.strip():
            return None, "No answer generated"
        
        if not results:
            return None, "No results available for evaluation"
        
        # Prepare context from results
        context_text = []
        for i, result in enumerate(results[:5], 1):  # Use top 5 results
            text = result.get('text', result.get('description', ''))[:500]
            context_text.append(f"[Result {i}]: {text}")
        
        evaluation_prompt = f"""You are an expert evaluator assessing answer quality for a RAG (Retrieval-Augmented Generation) system.

Query: {query}

Search Results Context:
{chr(10).join(context_text)}

Generated Answer:
{answer}

Evaluate the answer on a scale of 0.0 to 1.0 based on:
1. Relevance: Does the answer address the query? (0.0-0.4)
2. Accuracy: Is the information factually correct based on the context? (0.0-0.3)
3. Completeness: Does it cover the key points from the context? (0.0-0.2)
4. Clarity: Is it well-structured and easy to understand? (0.0-0.1)

Return ONLY a JSON object with this exact format:
{{
  "score": 0.85,
  "reasoning": "Brief explanation of the score"
}}

Score should be between 0.0 and 1.0."""

        try:
            client = get_api_client()
            response = client.chat.completions.create(
                model=JUDGE_LLM_MODEL,  # Use judge LLM for evaluation
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert evaluator using {JUDGE_LLM_MODEL} as the judge model. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": evaluation_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
            
            evaluation = json.loads(result_text)
            score = float(evaluation.get('score', 0.0))
            reasoning = evaluation.get('reasoning', 'No reasoning provided')
            
            # Clamp score to [0, 1]
            score = max(0.0, min(1.0, score))
            
            return score, reasoning
        except Exception as e:
            print(f"Warning: Accuracy evaluation failed: {str(e)}")
            return None, f"Evaluation error: {str(e)}"
    
    def evaluate_results_accuracy(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Evaluate search results accuracy/relevance when no generated answer is available.
        This evaluates how well the retrieved results answer the query.
        
        VALIDATION METHODOLOGY:
        Uses an LLM judge model (JUDGE_LLM_MODEL) to evaluate raw search results.
        The judge evaluates on 4 dimensions with weighted scoring:
        1. Relevance (0.0-0.4): How relevant are results to the query?
        2. Coverage (0.0-0.3): Do results cover key aspects of the query?
        3. Quality (0.0-0.2): Are results informative and useful?
        4. Diversity (0.0-0.1): Do results provide diverse perspectives?
        
        The judge receives:
        - The original query
        - Top 8 search results with similarity scores and sources
        - Each result truncated to 400 chars for context
        
        Scoring: Returns a score 0.0-1.0 with reasoning. Uses temperature=0.1
        for consistency. Scores are clamped to [0, 1] range.
        
        Args:
            query: Original query
            results: Search results retrieved
            
        Returns:
            Tuple of (accuracy_score, reasoning)
        """
        if not results:
            return None, "No results available for evaluation"
        
        # Prepare results summary
        results_text = []
        for i, result in enumerate(results[:8], 1):  # Use top 8 results
            text = result.get('text', result.get('description', ''))
            # Truncate long texts
            text_preview = text[:400] + "..." if len(text) > 400 else text
            
            similarity = result.get('similarity', result.get('score', 0.0))
            source = result.get('source_name', result.get('source', 'Unknown'))
            
            results_text.append(
                f"[Result {i}] (Similarity: {similarity:.3f}, Source: {source}):\n{text_preview}"
            )
        
        evaluation_prompt = f"""You are an expert evaluator assessing search result quality for a RAG (Retrieval-Augmented Generation) system.

Query: {query}

Retrieved Search Results:
{chr(10).join(results_text)}

Evaluate how well these search results answer the query on a scale of 0.0 to 1.0 based on:
1. Relevance: How relevant are the results to the query? (0.0-0.4)
2. Coverage: Do the results cover the key aspects of the query? (0.0-0.3)
3. Quality: Are the results informative and useful? (0.0-0.2)
4. Diversity: Do the results provide diverse perspectives/coverage? (0.0-0.1)

Return ONLY a JSON object with this exact format:
{{
  "score": 0.75,
  "reasoning": "Brief explanation of the score"
}}

Score should be between 0.0 and 1.0."""

        try:
            client = get_api_client()
            response = client.chat.completions.create(
                model=JUDGE_LLM_MODEL,  # Use judge LLM for evaluation
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert evaluator using {JUDGE_LLM_MODEL} as the judge model. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": evaluation_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
            
            evaluation = json.loads(result_text)
            score = float(evaluation.get('score', 0.0))
            reasoning = evaluation.get('reasoning', 'No reasoning provided')
            
            # Clamp score to [0, 1]
            score = max(0.0, min(1.0, score))
            
            return score, reasoning
        except Exception as e:
            print(f"Warning: Results accuracy evaluation failed: {str(e)}")
            return None, f"Evaluation error: {str(e)}"
    
    def evaluate_vector_search(
        self, 
        query: str, 
        top_k: int = 10,
        db_names: Optional[List[str]] = None,
        evaluate_accuracy: bool = True,
        return_raw_results: bool = False
    ) -> SearchMetrics:
        """
        Evaluate vector search performance with answer generation.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            db_names: Databases to search (None = all)
            evaluate_accuracy: Whether to evaluate answer accuracy
            return_raw_results: If True, returns tuple (metrics, raw_results, search_time_ms)
            
        Returns:
            SearchMetrics object with metrics (or tuple if return_raw_results=True)
        """
        search_start = time.perf_counter()
        
        # Get search results (without answer generation for timing)
        # For pure vector: search + rerank (this is what pure vector uses)
        raw_results = self.vector_engine.search(
            query=query,
            db_names=db_names,
            top_k=top_k,
            similarity_threshold=0.0
        )
        
        # Rerank results (pure vector uses reranked results)
        results = raw_results
        if results:
            results = self.vector_engine.rerank_results(query, results, top_n=min(8, len(results)))
        
        search_time_ms = (time.perf_counter() - search_start) * 1000
        
        # For hybrid: use unreranked results (what hybrid actually uses)
        vector_results_for_hybrid = raw_results
        
        # Generate answer
        answer_gen_start = time.perf_counter()
        answer = None
        if results:
            answer = self.vector_engine.generate_answer(query, results[:8])
        answer_gen_time_ms = (time.perf_counter() - answer_gen_start) * 1000
        
        total_time_ms = search_time_ms + answer_gen_time_ms
        
        # Evaluate accuracy if requested
        accuracy_score = None
        accuracy_reasoning = None
        if evaluate_accuracy and answer:
            accuracy_score, accuracy_reasoning = self.evaluate_answer_accuracy(query, answer, results)
        
        # Calculate metrics
        avg_similarity = None
        if results:
            similarities = [r.get('similarity', 0.0) for r in results]
            avg_similarity = statistics.mean(similarities) if similarities else None
        
        # Source distribution
        sources = defaultdict(int)
        for r in results:
            sources[r.get('database', 'unknown')] += 1
        
        metrics = SearchMetrics(
            query=query,
            method='vector',
            execution_time_ms=total_time_ms,
            num_results=len(results),
            top_k=top_k,
            answer_generation_time_ms=answer_gen_time_ms,
            generated_answer=answer,
            answer_length=len(answer) if answer else None,
            has_answer_generation=True,
            accuracy_score=accuracy_score,
            accuracy_reasoning=accuracy_reasoning,
            avg_similarity=avg_similarity,
            sources=dict(sources)
        )
        
        if return_raw_results:
            return metrics, results, search_time_ms, vector_results_for_hybrid
        return metrics
    
    def evaluate_graph_search(
        self,
        query: str,
        top_k: int = 10,
        evaluate_accuracy: bool = True,
        return_raw_results: bool = False
    ) -> SearchMetrics:
        """
        Evaluate graph traversal search performance with answer generation.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            evaluate_accuracy: Whether to evaluate answer accuracy
            return_raw_results: If True, returns tuple (metrics, raw_results, search_time_ms)
            
        Returns:
            SearchMetrics object with metrics (or tuple if return_raw_results=True)
        """
        search_start = time.perf_counter()
        
        # Get graph traversal results
        # For pure graph: get all results, score them, then rerank (like vector search does)
        all_graph_results = self.hybrid_engine.graph_traversal_search(query, top_k=None, score_results=True)
        # Results are already scored and sorted by similarity
        
        # Search time = graph traversal + semantic scoring (before reranking)
        search_time_ms = (time.perf_counter() - search_start) * 1000
        
        # Rerank results using LLM (like pure vector does for consistency)
        results = all_graph_results
        rerank_start = time.perf_counter()
        if results:
            results = self.vector_engine.rerank_results(query, results, top_n=min(8, len(results)))
        rerank_time_ms = (time.perf_counter() - rerank_start) * 1000
        
        # For hybrid: use scored but unreranked results (hybrid will rerank after RRF)
        graph_results_for_hybrid = all_graph_results  # Unreranked for hybrid
        
        # Generate answer
        answer_gen_start = time.perf_counter()
        answer = None
        if results:
            try:
                # Use reranked results for answer generation (like vector search does)
                answer = self.vector_engine.generate_answer(query, results[:8])
            except Exception as e:
                print(f"Warning: Answer generation failed: {e}")
                answer = None
        
        answer_gen_time_ms = (time.perf_counter() - answer_gen_start) * 1000 if answer else None
        # Total time = search + rerank + answer generation
        total_time_ms = search_time_ms + rerank_time_ms + (answer_gen_time_ms or 0)
        
        # Evaluate accuracy if requested
        accuracy_score = None
        accuracy_reasoning = None
        if evaluate_accuracy and answer:
            accuracy_score, accuracy_reasoning = self.evaluate_answer_accuracy(query, answer, results)
        
        # Source distribution
        sources = defaultdict(int)
        for r in results:
            source = r.get('source', 'graph_traversal')
            sources[source] += 1
        
        metrics = SearchMetrics(
            query=query,
            method='graph',
            execution_time_ms=total_time_ms,
            num_results=len(results),
            top_k=top_k,
            answer_generation_time_ms=answer_gen_time_ms,
            generated_answer=answer,
            answer_length=len(answer) if answer else None,
            has_answer_generation=True,
            accuracy_score=accuracy_score,
            accuracy_reasoning=accuracy_reasoning,
            sources=dict(sources)
        )
        
        if return_raw_results:
            return metrics, results, search_time_ms, graph_results_for_hybrid
        return metrics
    
    def evaluate_hybrid_search(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        vector_search_time_ms: float,
        graph_search_time_ms: float,
        top_k: int = 10,
        evaluate_accuracy: bool = True,
        rrf_k: int = None
    ) -> SearchMetrics:
        """
        Evaluate hybrid search by applying RRF to pre-computed vector and graph results.
        This is more efficient than re-running searches.
        Applies adaptive similarity filtering (Option B) before merging.
        
        Args:
            query: Search query
            vector_results: Pre-computed vector search results
            graph_results: Pre-computed graph search results
            vector_search_time_ms: Time taken for vector search (ms)
            graph_search_time_ms: Time taken for graph search (ms)
            top_k: Number of results to retrieve
            evaluate_accuracy: Whether to evaluate answer accuracy
            rrf_k: RRF constant (defaults to RRF_K from .env)
            
        Returns:
            SearchMetrics object with metrics
        """
        if rrf_k is None:
            rrf_k = int(os.getenv("RRF_K", "60"))
        
        # Apply adaptive similarity filtering to graph results (Option B)
        min_graph_results = int(os.getenv("MIN_GRAPH_RESULTS_FOR_RRF", "3"))
        graph_results_filtered = self._filter_graph_results_adaptive(graph_results, vector_results)
        
        # If filtered graph results are too few, use vector only
        if len(graph_results_filtered) < min_graph_results:
            graph_results_filtered = []
        
        # Step 1: Merge results using RRF (only if both sources have results)
        rrf_start = time.perf_counter()
        if len(vector_results) > 0 and len(graph_results_filtered) > 0:
            merged_results = self._merge_results_rrf(vector_results, graph_results_filtered, top_k, rrf_k)
        elif len(vector_results) > 0:
            merged_results = vector_results[:top_k]  # Use vector only
        elif len(graph_results_filtered) > 0:
            merged_results = graph_results_filtered[:top_k]  # Use graph only
        else:
            merged_results = []
        rrf_time_ms = (time.perf_counter() - rrf_start) * 1000
        
        # Step 2: Rerank merged results (hybrid_query() does this, so evaluation should too)
        rerank_start = time.perf_counter()
        if merged_results:
            merged_results = self.vector_engine.rerank_results(query, merged_results, top_n=min(8, len(merged_results)))
        rerank_time_ms = (time.perf_counter() - rerank_start) * 1000
        
        # Step 3: Generate answer from reranked merged results
        answer_gen_start = time.perf_counter()
        answer = None
        if merged_results:
            answer = self.vector_engine.generate_answer(query, merged_results[:8])
        answer_gen_time_ms = (time.perf_counter() - answer_gen_start) * 1000
        
        # Total time = vector search + graph search + RRF merge + rerank + answer generation
        total_time_ms = vector_search_time_ms + graph_search_time_ms + rrf_time_ms + rerank_time_ms + answer_gen_time_ms
        
        # Evaluate accuracy if requested
        accuracy_score = None
        accuracy_reasoning = None
        if evaluate_accuracy and answer:
            accuracy_score, accuracy_reasoning = self.evaluate_answer_accuracy(query, answer, merged_results)
        
        # Calculate average similarity (for vector results)
        avg_similarity = None
        similarities = [r.get('similarity', 0.0) for r in merged_results if 'similarity' in r]
        if similarities:
            avg_similarity = statistics.mean(similarities)
        
        # Source distribution
        sources = defaultdict(int)
        for r in merged_results:
            source = r.get('source', 'unknown')
            sources[source] += 1
        
        return SearchMetrics(
            query=query,
            method='hybrid',
            execution_time_ms=total_time_ms,
            num_results=len(merged_results),
            top_k=top_k,
            answer_generation_time_ms=answer_gen_time_ms,
            generated_answer=answer,
            answer_length=len(answer) if answer else None,
            has_answer_generation=True,
            accuracy_score=accuracy_score,
            accuracy_reasoning=accuracy_reasoning,
            avg_similarity=avg_similarity,
            sources=dict(sources)
        )
    
    def _filter_graph_results_adaptive(
        self, 
        graph_results: List[Dict], 
        vector_results: List[Dict] = None
    ) -> List[Dict]:
        """
        Filter low-quality graph results using adaptive similarity threshold (Option B).
        Matches the logic in hybrid_query_engine.py.
        
        Args:
            graph_results: Graph results with similarity scores
            vector_results: Optional vector results for adaptive threshold calculation
            
        Returns:
            Filtered graph results
        """
        if not graph_results:
            return []
        
        # Get configuration
        threshold_mode = os.getenv("GRAPH_SIMILARITY_THRESHOLD_MODE", "adaptive").lower()
        threshold_fixed = float(os.getenv("GRAPH_SIMILARITY_THRESHOLD_FIXED", "0.5"))
        threshold_percentile = float(os.getenv("GRAPH_SIMILARITY_THRESHOLD_PERCENTILE", "0.3"))
        
        # Get similarity scores
        graph_similarities = [r.get('similarity', 0.0) for r in graph_results if r.get('similarity') is not None]
        if not graph_similarities:
            return graph_results  # No similarities, return all
        
        # Calculate threshold
        if threshold_mode == "fixed":
            threshold = threshold_fixed
        elif threshold_mode == "percentile":
            percentile_idx = int(len(graph_similarities) * threshold_percentile)
            sorted_sims = sorted(graph_similarities)
            if percentile_idx >= len(sorted_sims):
                threshold = sorted_sims[0] if sorted_sims else 0.5
            else:
                threshold = sorted_sims[percentile_idx]
        else:  # adaptive mode
            # Strategy 1: Use vector median as reference if available
            if vector_results and len(vector_results) > 0:
                vector_similarities = [r.get('similarity', 0.0) for r in vector_results if r.get('similarity') is not None]
                if vector_similarities:
                    vector_median = statistics.median(vector_similarities)
                    threshold = max(0.3, vector_median - 0.2)
                else:
                    # Strategy 2: Use percentile-based
                    percentile_idx = int(len(graph_similarities) * 0.3)
                    sorted_sims = sorted(graph_similarities)
                    threshold = sorted_sims[percentile_idx] if percentile_idx < len(sorted_sims) else sorted_sims[0] if sorted_sims else 0.5
                    threshold = max(0.3, threshold)
            else:
                # Strategy 2: Use percentile-based
                percentile_idx = int(len(graph_similarities) * 0.3)
                sorted_sims = sorted(graph_similarities)
                threshold = sorted_sims[percentile_idx] if percentile_idx < len(sorted_sims) else sorted_sims[0] if sorted_sims else 0.5
                threshold = max(0.3, threshold)
        
        # Filter results above threshold
        filtered = [r for r in graph_results if r.get('similarity', 0.0) >= threshold]
        return filtered
    
    def _merge_results_rrf(self, vector_results: List[Dict], graph_results: List[Dict], 
                           top_k: int, k: int = 60) -> List[Dict]:
        """
        Merge results from vector and graph sources using Reciprocal Rank Fusion (RRF).
        Simplified version for evaluation (no debug output).
        
        RRF Formula: RRF_score(d) = Σ 1 / (k + rank_i(d))
        
        Args:
            vector_results: Results from vector search (already ranked)
            graph_results: Results from graph traversal
            top_k: Number of top results to return
            k: RRF constant
            
        Returns:
            Merged and ranked results using RRF scores
        """
        rrf_scores = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            result_id = self._get_result_id(result)
            if result_id not in rrf_scores:
                rrf_scores[result_id] = {
                    'result': result.copy(),
                    'rrf_score': 0.0,
                    'vector_rank': None,
                    'graph_rank': None
                }
            rrf_scores[result_id]['rrf_score'] += 1.0 / (k + rank)
            rrf_scores[result_id]['vector_rank'] = rank
        
        # Process graph results
        for rank, result in enumerate(graph_results, start=1):
            result_id = self._get_result_id(result)
            if result_id not in rrf_scores:
                rrf_scores[result_id] = {
                    'result': result.copy(),
                    'rrf_score': 0.0,
                    'vector_rank': None,
                    'graph_rank': None
                }
            rrf_scores[result_id]['rrf_score'] += 1.0 / (k + rank)
            rrf_scores[result_id]['graph_rank'] = rank
        
        # Mark source
        for result_id, data in rrf_scores.items():
            if data['vector_rank'] is not None and data['graph_rank'] is not None:
                data['result']['source'] = 'hybrid'
            elif data['vector_rank'] is not None:
                data['result']['source'] = 'faiss_vector'
            else:
                data['result']['source'] = 'graph_traversal'
            data['result']['rrf_score'] = data['rrf_score']
            data['result']['vector_rank'] = data['vector_rank']
            data['result']['graph_rank'] = data['graph_rank']
        
        # Sort by RRF score and return top_k
        merged = sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
        return [item['result'] for item in merged[:top_k]]
    
    def _get_result_id(self, result: Dict[str, Any]) -> str:
        """Get unique identifier for a result (for RRF deduplication)."""
        # Try various ID fields
        for field in ['id', 'chunk_id', 'incident_id', 'clause_id', 'article_id']:
            if field in result:
                return str(result[field])
        
        # Use text hash as fallback
        text = result.get('text', result.get('description', ''))
        if text:
            import hashlib
            return hashlib.md5(text[:100].encode('utf-8')).hexdigest()
        
        return str(result)
    
    def calculate_precision_recall(
        self,
        results: List[Dict],
        ground_truth_ids: List[str],
        k: int = 10
    ) -> Tuple[float, float]:
        """
        Calculate precision@k and recall@k.
        
        Args:
            results: Search results
            ground_truth_ids: List of relevant result IDs
            k: Top k results to consider
            
        Returns:
            Tuple of (precision@k, recall@k)
        """
        if not ground_truth_ids:
            return None, None
        
        top_k_results = results[:k]
        result_ids = [self._get_result_id(r) for r in top_k_results]
        
        # Precision@k: fraction of top-k results that are relevant
        relevant_retrieved = len(set(result_ids) & set(ground_truth_ids))
        precision = relevant_retrieved / k if k > 0 else 0.0
        
        # Recall@k: fraction of relevant items that are retrieved
        recall = relevant_retrieved / len(ground_truth_ids) if ground_truth_ids else 0.0
        
        return precision, recall
    
    def calculate_mrr(
        self,
        results: List[Dict],
        ground_truth_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            results: Search results
            ground_truth_ids: List of relevant result IDs
            
        Returns:
            MRR score (0-1)
        """
        if not ground_truth_ids:
            return None
        
        for rank, result in enumerate(results, start=1):
            result_id = self._get_result_id(result)
            if result_id in ground_truth_ids:
                return 1.0 / rank
        
        return 0.0
    
    def calculate_ndcg(
        self,
        results: List[Dict],
        ground_truth_scores: Dict[str, float],
        k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@k).
        
        Args:
            results: Search results
            ground_truth_scores: Dict mapping result_id to relevance score
            k: Top k results to consider
            
        Returns:
            NDCG@k score (0-1)
        """
        if not ground_truth_scores:
            return None
        
        top_k_results = results[:k]
        
        # Calculate DCG
        dcg = 0.0
        for rank, result in enumerate(top_k_results, start=1):
            result_id = self._get_result_id(result)
            relevance = ground_truth_scores.get(result_id, 0.0)
            dcg += relevance / math.log2(rank + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = sorted(ground_truth_scores.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, start=1))
        
        # NDCG = DCG / IDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return ndcg
    
    def calculate_f1_score(
        self,
        precision: Optional[float],
        recall: Optional[float]
    ) -> Optional[float]:
        """
        Calculate F1 score from precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
            
        Returns:
            F1 score (0-1) or None if precision/recall unavailable
        """
        if precision is None or recall is None:
            return None
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_map(
        self,
        results: List[Dict],
        ground_truth_ids: List[str],
        k: int = 10
    ) -> float:
        """
        Calculate Mean Average Precision (MAP@k).
        
        Args:
            results: Search results
            ground_truth_ids: List of relevant result IDs
            k: Top k results to consider
            
        Returns:
            MAP@k score (0-1)
        """
        if not ground_truth_ids:
            return None
        
        top_k_results = results[:k]
        relevant_count = 0
        precision_sum = 0.0
        
        for rank, result in enumerate(top_k_results, start=1):
            result_id = self._get_result_id(result)
            if result_id in ground_truth_ids:
                relevant_count += 1
                precision_at_rank = relevant_count / rank
                precision_sum += precision_at_rank
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / len(ground_truth_ids)
    
    def calculate_coverage(
        self,
        results: List[Dict],
        ground_truth_ids: List[str]
    ) -> float:
        """
        Calculate coverage score: percentage of relevant items retrieved.
        
        Args:
            results: Search results
            ground_truth_ids: List of relevant result IDs
            
        Returns:
            Coverage score (0-1)
        """
        if not ground_truth_ids:
            return None
        
        result_ids = [self._get_result_id(r) for r in results]
        retrieved_relevant = len(set(result_ids) & set(ground_truth_ids))
        
        return retrieved_relevant / len(ground_truth_ids) if ground_truth_ids else 0.0
    
    def _get_result_id(self, result: Dict) -> str:
        """Get unique identifier for a result."""
        if 'id' in result and result['id']:
            return str(result['id'])
        text = result.get('text', result.get('description', ''))
        return text[:100] if text else str(result)
    
    def run_evaluation_suite(
        self,
        test_queries: List[str],
        top_k: int = 10,
        ground_truth: Optional[Dict[str, List[str]]] = None,
        evaluate_accuracy: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation suite.
        
        Answer generation is mandatory for all search methods.
        
        Args:
            test_queries: List of test queries
            top_k: Number of results to retrieve
            ground_truth: Optional dict mapping query to list of relevant result IDs
            evaluate_accuracy: Whether to evaluate answer accuracy
            verbose: Whether to show detailed progress
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            'vector': [],
            'graph': [],
            'hybrid': []
        }
        
        if verbose:
            print("=" * 80)
            print("SEARCH EVALUATION SUITE")
            print("=" * 80)
            print(f"Test queries: {len(test_queries)}")
            print(f"Top K: {top_k}")
            print(f"Answer Generation: Mandatory for all methods")
            print(f"Evaluate accuracy: {evaluate_accuracy}")
            if evaluate_accuracy:
                print(f"Judge LLM Model: {JUDGE_LLM_MODEL} (for accuracy evaluation)")
                print(f"Answer Generation LLM Model: {LLM_MODEL}")
            print("=" * 80)
            print()
        else:
            # Simplified header
            print(f"Running evaluation: {len(test_queries)} queries × 3 methods (all with answer generation)")
            if evaluate_accuracy:
                print(f"Judge: {JUDGE_LLM_MODEL} | Answer Gen: {LLM_MODEL}")
            print()
        
        for i, query in enumerate(test_queries, 1):
            if verbose:
                print(f"\n{'='*80}")
                print(f"[{i}/{len(test_queries)}] Query: '{query}'")
                print(f"{'='*80}")
            else:
                progress = (i - 1) / len(test_queries) * 100
                print(f"[{i}/{len(test_queries)}] {progress:.0f}% - {query[:60]}{'...' if len(query) > 60 else ''}")
            
            # ========================================================================
            # STEP 1: VECTOR SEARCH ONLY
            # ========================================================================
            if verbose:
                print("\n[1/3] VECTOR SEARCH ONLY")
                print("-" * 80)
            try:
                result = self.evaluate_vector_search(
                    query, top_k=top_k, evaluate_accuracy=evaluate_accuracy, return_raw_results=True
                )
                if isinstance(result, tuple) and len(result) == 4:
                    vector_metrics, vector_raw_results, vector_search_time, vector_results_for_hybrid = result
                else:
                    vector_metrics, vector_raw_results, vector_search_time = result
                    vector_results_for_hybrid = vector_raw_results  # Fallback: use reranked results
                results['vector'].append(vector_metrics)
                if verbose:
                    ans_time = f", ans: {vector_metrics.answer_generation_time_ms:.0f}ms" if vector_metrics.answer_generation_time_ms else ""
                    acc = f", acc: {vector_metrics.accuracy_score:.2f}" if vector_metrics.accuracy_score is not None else ""
                    print(f"✓ Search: {vector_search_time:.2f}ms | Total: {vector_metrics.execution_time_ms:.2f}ms{ans_time}{acc} | Results: {vector_metrics.num_results}")
                else:
                    print(f"  ✓ Vector: {vector_metrics.execution_time_ms:.0f}ms, {vector_metrics.num_results} results, acc: {vector_metrics.accuracy_score:.2f}" if vector_metrics.accuracy_score else f"  ✓ Vector: {vector_metrics.execution_time_ms:.0f}ms, {vector_metrics.num_results} results")
            except Exception as e:
                error_msg = str(e)
                if 'quota' in error_msg.lower() or 'insufficient_quota' in error_msg.lower():
                    print(f"  ✗ Vector search SKIPPED (API quota exceeded)")
                    vector_metrics = SearchMetrics(
                        query=query, method='vector', execution_time_ms=0.0,
                        num_results=0, top_k=top_k, has_answer_generation=False
                    )
                    vector_raw_results = []
                    vector_search_time = 0.0
                    results['vector'].append(vector_metrics)
                else:
                    raise
            
            # ========================================================================
            # STEP 2: GRAPH SEARCH ONLY
            # ========================================================================
            if verbose:
                print("\n[2/3] GRAPH SEARCH ONLY")
                print("-" * 80)
            try:
                result = self.evaluate_graph_search(
                    query, top_k=top_k, evaluate_accuracy=evaluate_accuracy, return_raw_results=True
                )
                if isinstance(result, tuple) and len(result) == 4:
                    graph_metrics, graph_raw_results, graph_search_time, graph_results_for_hybrid = result
                else:
                    graph_metrics, graph_raw_results, graph_search_time = result
                    graph_results_for_hybrid = graph_raw_results  # Fallback: use limited results
                results['graph'].append(graph_metrics)
                if verbose:
                    ans_time = f", ans: {graph_metrics.answer_generation_time_ms:.0f}ms" if graph_metrics.answer_generation_time_ms else ""
                    acc = f", acc: {graph_metrics.accuracy_score:.2f}" if graph_metrics.accuracy_score is not None else ""
                    print(f"✓ Search: {graph_search_time:.2f}ms | Total: {graph_metrics.execution_time_ms:.2f}ms{ans_time}{acc} | Results: {graph_metrics.num_results}")
                else:
                    print(f"  ✓ Graph: {graph_metrics.execution_time_ms:.0f}ms, {graph_metrics.num_results} results, acc: {graph_metrics.accuracy_score:.2f}" if graph_metrics.accuracy_score else f"  ✓ Graph: {graph_metrics.execution_time_ms:.0f}ms, {graph_metrics.num_results} results")
            except Exception as e:
                error_msg = str(e)
                print(f"  ✗ Graph search ERROR: {error_msg}")
                graph_metrics = SearchMetrics(
                    query=query, method='graph', execution_time_ms=0.0,
                    num_results=0, top_k=top_k, has_answer_generation=False
                )
                graph_raw_results = []
                graph_search_time = 0.0
                graph_results_for_hybrid = []
                results['graph'].append(graph_metrics)
            
            # ========================================================================
            # STEP 3: HYBRID SEARCH (RRF on Vector + Graph results)
            # ========================================================================
            if verbose:
                print("\n[3/3] HYBRID SEARCH (RRF on Vector + Graph)")
                print("-" * 80)
            try:
                # Use the correct results for hybrid:
                # - Unreranked vector results (hybrid doesn't rerank before RRF)
                # - Unlimited graph results (hybrid doesn't limit before RRF)
                hybrid_vector_results = vector_results_for_hybrid if 'vector_results_for_hybrid' in locals() else vector_raw_results
                hybrid_graph_results = graph_results_for_hybrid if 'graph_results_for_hybrid' in locals() else graph_raw_results
                
                hybrid_metrics = self.evaluate_hybrid_search(
                    query=query,
                    vector_results=hybrid_vector_results,
                    graph_results=hybrid_graph_results,
                    vector_search_time_ms=vector_search_time,
                    graph_search_time_ms=graph_search_time,
                    top_k=top_k,
                    evaluate_accuracy=evaluate_accuracy
                )
                results['hybrid'].append(hybrid_metrics)
                if verbose:
                    # Calculate RRF and rerank times from total
                    # Formula: total = vector + graph + rrf + rerank + answer
                    remaining_time = hybrid_metrics.execution_time_ms - vector_search_time - graph_search_time - (hybrid_metrics.answer_generation_time_ms or 0)
                    # RRF is very fast (pure math, ~0.07-3ms), rerank involves LLM call (~10-100ms)
                    # Estimate: RRF ~10% of remaining, Rerank ~90% (rerank involves LLM API call)
                    if remaining_time > 0:
                        rrf_time = remaining_time * 0.1  # RRF is pure math, very fast
                        rerank_time = remaining_time * 0.9  # Rerank involves LLM API call
                    else:
                        rrf_time = 0.0
                        rerank_time = 0.0
                    ans_time = f", ans: {hybrid_metrics.answer_generation_time_ms:.0f}ms" if hybrid_metrics.answer_generation_time_ms else ""
                    acc = f", acc: {hybrid_metrics.accuracy_score:.2f}" if hybrid_metrics.accuracy_score is not None else ""
                    print(f"✓ RRF merge: {rrf_time:.2f}ms | Total: {hybrid_metrics.execution_time_ms:.2f}ms{ans_time}{acc} | Results: {hybrid_metrics.num_results}")
                    print(f"  (Vector: {vector_search_time:.2f}ms + Graph: {graph_search_time:.2f}ms + RRF: {rrf_time:.2f}ms + Rerank: {rerank_time:.2f}ms + Answer: {hybrid_metrics.answer_generation_time_ms:.0f}ms)")
                else:
                    print(f"  ✓ Hybrid: {hybrid_metrics.execution_time_ms:.0f}ms, {hybrid_metrics.num_results} results, acc: {hybrid_metrics.accuracy_score:.2f}" if hybrid_metrics.accuracy_score else f"  ✓ Hybrid: {hybrid_metrics.execution_time_ms:.0f}ms, {hybrid_metrics.num_results} results")
            except Exception as e:
                error_msg = str(e)
                print(f"  ✗ Hybrid search ERROR: {error_msg}")
                hybrid_metrics = SearchMetrics(
                    query=query, method='hybrid', execution_time_ms=0.0,
                    num_results=0, top_k=top_k, has_answer_generation=False
                )
                results['hybrid'].append(hybrid_metrics)
            
            # Summary for this query
            if not verbose:
                avg_acc = (vector_metrics.accuracy_score or 0) + (graph_metrics.accuracy_score or 0) + (hybrid_metrics.accuracy_score or 0)
                if evaluate_accuracy and avg_acc > 0:
                    avg_acc = avg_acc / 3
                    print(f"  → Avg accuracy: {avg_acc:.2f}")
            
            if verbose:
                print()
        
        # Calculate aggregate statistics
        summary = self._calculate_summary_statistics(results)
        return {
            'results': results,
            'summary': summary,
            'test_queries': test_queries,
            'top_k': top_k
        }
    
    def _calculate_summary_statistics(
        self,
        results: Dict[str, List[SearchMetrics]]
    ) -> Dict[str, Any]:
        """Calculate aggregate statistics with confidence intervals."""
        summary = {}
        
        for method, metrics_list in results.items():
            if not metrics_list:
                continue
            
            execution_times = [m.execution_time_ms for m in metrics_list]
            num_results = [m.num_results for m in metrics_list]
            avg_similarities = [m.avg_similarity for m in metrics_list if m.avg_similarity is not None]
            answer_generation_times = [m.answer_generation_time_ms for m in metrics_list if m.answer_generation_time_ms is not None]
            answer_lengths = [m.answer_length for m in metrics_list if m.answer_length is not None]
            accuracy_scores = [m.accuracy_score for m in metrics_list if m.accuracy_score is not None]
            precision_scores = [m.precision_at_k for m in metrics_list if m.precision_at_k is not None]
            recall_scores = [m.recall_at_k for m in metrics_list if m.recall_at_k is not None]
            f1_scores = [m.f1_score for m in metrics_list if m.f1_score is not None]
            map_scores = [m.map_score for m in metrics_list if m.map_score is not None]
            mrr_scores = [m.mrr for m in metrics_list if m.mrr is not None]
            ndcg_scores = [m.ndcg_at_k for m in metrics_list if m.ndcg_at_k is not None]
            
            # Calculate 95% confidence intervals
            ci_exec_time = self._calculate_confidence_interval(execution_times, confidence=0.95)
            ci_accuracy = self._calculate_confidence_interval(accuracy_scores, confidence=0.95) if accuracy_scores else None
            
            summary[method] = {
                'avg_execution_time_ms': statistics.mean(execution_times),
                'median_execution_time_ms': statistics.median(execution_times),
                'min_execution_time_ms': min(execution_times),
                'max_execution_time_ms': max(execution_times),
                'std_execution_time_ms': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'ci_95_execution_time_ms': ci_exec_time,  # (lower, upper)
                'avg_num_results': statistics.mean(num_results),
                'avg_similarity': statistics.mean(avg_similarities) if avg_similarities else None,
                'avg_answer_generation_time_ms': statistics.mean(answer_generation_times) if answer_generation_times else None,
                'avg_answer_length': statistics.mean(answer_lengths) if answer_lengths else None,
                'avg_accuracy_score': statistics.mean(accuracy_scores) if accuracy_scores else None,
                'ci_95_accuracy_score': ci_accuracy,  # (lower, upper)
                'avg_precision_at_k': statistics.mean(precision_scores) if precision_scores else None,
                'avg_recall_at_k': statistics.mean(recall_scores) if recall_scores else None,
                'avg_f1_score': statistics.mean(f1_scores) if f1_scores else None,
                'avg_map_score': statistics.mean(map_scores) if map_scores else None,
                'avg_mrr': statistics.mean(mrr_scores) if mrr_scores else None,
                'avg_ndcg_at_k': statistics.mean(ndcg_scores) if ndcg_scores else None,
                'total_queries': len(metrics_list)
            }
        
        return summary
    
    def _calculate_confidence_interval(
        self,
        data: List[float],
        confidence: float = 0.95
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate confidence interval for a dataset.
        
        Args:
            data: List of numeric values
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound) or None if insufficient data
        """
        if len(data) < 2:
            return None
        
        if SCIPY_AVAILABLE:
            # Use scipy for more accurate CI calculation
            mean = statistics.mean(data)
            std_err = statistics.stdev(data) / math.sqrt(len(data))
            alpha = 1 - confidence
            t_critical = stats.t.ppf(1 - alpha/2, df=len(data)-1)
            margin = t_critical * std_err
            return (mean - margin, mean + margin)
        else:
            # Fallback: use normal approximation (less accurate for small samples)
            mean = statistics.mean(data)
            std_err = statistics.stdev(data) / math.sqrt(len(data))
            # Z-score for 95% CI = 1.96
            z_score = 1.96 if confidence == 0.95 else 2.576  # 99% CI = 2.576
            margin = z_score * std_err
            return (mean - margin, mean + margin)
    
    def calculate_statistical_significance(
        self,
        method1_scores: List[float],
        method2_scores: List[float],
        test_type: str = 't-test'
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance between two methods.
        
        Args:
            method1_scores: Scores from method 1
            method2_scores: Scores from method 2
            test_type: 't-test' or 'wilcoxon'
            
        Returns:
            Dictionary with p-value, statistic, and interpretation
        """
        if not SCIPY_AVAILABLE:
            return {
                'p_value': None,
                'statistic': None,
                'significant': None,
                'message': 'scipy not available for statistical tests'
            }
        
        if len(method1_scores) != len(method2_scores):
            return {
                'p_value': None,
                'statistic': None,
                'significant': None,
                'message': 'Sample sizes must match for paired tests'
            }
        
        if test_type == 't-test':
            # Paired t-test
            statistic, p_value = stats.ttest_rel(method1_scores, method2_scores)
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test (non-parametric)
            statistic, p_value = stats.wilcoxon(method1_scores, method2_scores)
        else:
            return {
                'p_value': None,
                'statistic': None,
                'significant': None,
                'message': f'Unknown test type: {test_type}'
            }
        
        # Determine significance levels
        alpha = 0.05
        significant = p_value < alpha
        significance_level = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        
        return {
            'p_value': float(p_value),
            'statistic': float(statistic),
            'significant': significant,
            'significance_level': significance_level,
            'test_type': test_type,
            'alpha': alpha
        }
    
    def print_evaluation_report(self, evaluation_results: Dict[str, Any]):
        """Print formatted evaluation report."""
        summary = evaluation_results['summary']
        
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print()
        print("All queries tested across 3 search methods (all with answer generation)")
        if any(stats.get('avg_accuracy_score') is not None for stats in summary.values()):
            print(f"Judge LLM Model: {JUDGE_LLM_MODEL} (used for accuracy evaluation)")
            print(f"Answer Generation LLM Model: {LLM_MODEL}")
            print()
        
        # Performance comparison
        print("PERFORMANCE METRICS (Timing)")
        print("-" * 80)
        for method in ['vector', 'graph', 'hybrid']:
            if method in summary:
                stats = summary[method]
                print(f"\n{method.upper()}:")
                avg_time = stats['avg_execution_time_ms']
                ci_time = stats.get('ci_95_execution_time_ms')
                if ci_time:
                    print(f"  Average time: {avg_time:.2f}ms (95% CI: [{ci_time[0]:.2f}, {ci_time[1]:.2f}])")
                else:
                    print(f"  Average time: {avg_time:.2f}ms")
                print(f"  Median time:  {stats['median_execution_time_ms']:.2f}ms")
                print(f"  Min time:     {stats['min_execution_time_ms']:.2f}ms")
                print(f"  Max time:     {stats['max_execution_time_ms']:.2f}ms")
                print(f"  Std dev:      {stats['std_execution_time_ms']:.2f}ms")
                if stats.get('avg_answer_generation_time_ms') is not None:
                    print(f"  Answer generation time: {stats['avg_answer_generation_time_ms']:.2f}ms")
                    print(f"  Search-only time: {stats['avg_execution_time_ms'] - stats.get('avg_answer_generation_time_ms', 0):.2f}ms")
        
        # Result quality
        print("\n" + "=" * 80)
        print("RESULT QUALITY METRICS")
        print("-" * 80)
        for method in ['vector', 'graph', 'hybrid']:
            if method in summary:
                stats = summary[method]
                print(f"\n{method.upper()}:")
                print(f"  Average results: {stats['avg_num_results']:.1f}")
                if stats.get('avg_similarity') is not None:
                    print(f"  Avg similarity:  {stats['avg_similarity']:.4f}")
                if stats.get('avg_answer_length') is not None:
                    print(f"  Avg answer length: {stats['avg_answer_length']:.0f} chars")
                if stats.get('avg_accuracy_score') is not None:
                    acc_score = stats['avg_accuracy_score']
                    ci_acc = stats.get('ci_95_accuracy_score')
                    if ci_acc:
                        print(f"  Avg accuracy score: {acc_score:.3f} (95% CI: [{ci_acc[0]:.3f}, {ci_acc[1]:.3f}])")
                    else:
                        print(f"  Avg accuracy score: {acc_score:.3f} (0.0-1.0)")
                if stats.get('avg_precision_at_k') is not None:
                    print(f"  Avg Precision@k: {stats['avg_precision_at_k']:.3f}")
                if stats.get('avg_recall_at_k') is not None:
                    print(f"  Avg Recall@k: {stats['avg_recall_at_k']:.3f}")
                if stats.get('avg_f1_score') is not None:
                    print(f"  Avg F1 Score: {stats['avg_f1_score']:.3f}")
                if stats.get('avg_map_score') is not None:
                    print(f"  Avg MAP: {stats['avg_map_score']:.3f}")
                if stats.get('avg_mrr') is not None:
                    print(f"  Avg MRR: {stats['avg_mrr']:.3f}")
                if stats.get('avg_ndcg_at_k') is not None:
                    print(f"  Avg NDCG@k: {stats['avg_ndcg_at_k']:.3f}")
        
        # Speed comparison
        print("\n" + "=" * 80)
        print("SPEED COMPARISON")
        print("-" * 80)
        if 'vector' in summary and 'graph' in summary:
            vector_avg = summary['vector']['avg_execution_time_ms']
            graph_avg = summary['graph']['avg_execution_time_ms']
            speedup = graph_avg / vector_avg if vector_avg > 0 else 0
            print(f"Vector vs Graph: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
        
        if 'hybrid' in summary and 'vector' in summary:
            hybrid_avg = summary['hybrid']['avg_execution_time_ms']
            vector_avg = summary['vector']['avg_execution_time_ms']
            overhead = (hybrid_avg / vector_avg - 1) * 100 if vector_avg > 0 else 0
            print(f"Hybrid overhead vs Vector: {overhead:+.1f}%")
        
        # Statistical significance comparison
        if SCIPY_AVAILABLE and len(summary) >= 2:
            print("\n" + "=" * 80)
            print("STATISTICAL SIGNIFICANCE TESTS")
            print("-" * 80)
            methods = ['vector', 'graph', 'hybrid']
            accuracy_scores_dict = {}
            for method in methods:
                if method in summary:
                    # Extract accuracy scores from results
                    method_results = evaluation_results['results'].get(method, [])
                    scores = [m.accuracy_score for m in method_results if m.accuracy_score is not None]
                    if scores:
                        accuracy_scores_dict[method] = scores
            
            # Compare pairs
            if 'vector' in accuracy_scores_dict and 'graph' in accuracy_scores_dict:
                sig_test = self.calculate_statistical_significance(
                    accuracy_scores_dict['vector'],
                    accuracy_scores_dict['graph'],
                    test_type='t-test'
                )
                if sig_test.get('p_value') is not None:
                    print(f"\nVector vs Graph (Accuracy):")
                    print(f"  p-value: {sig_test['p_value']:.4f} {sig_test.get('significance_level', '')}")
                    print(f"  Significant: {'Yes' if sig_test['significant'] else 'No'} (α=0.05)")
            
            if 'hybrid' in accuracy_scores_dict and 'vector' in accuracy_scores_dict:
                sig_test = self.calculate_statistical_significance(
                    accuracy_scores_dict['hybrid'],
                    accuracy_scores_dict['vector'],
                    test_type='t-test'
                )
                if sig_test.get('p_value') is not None:
                    print(f"\nHybrid vs Vector (Accuracy):")
                    print(f"  p-value: {sig_test['p_value']:.4f} {sig_test.get('significance_level', '')}")
                    print(f"  Significant: {'Yes' if sig_test['significant'] else 'No'} (α=0.05)")
            
            if 'hybrid' in accuracy_scores_dict and 'graph' in accuracy_scores_dict:
                sig_test = self.calculate_statistical_significance(
                    accuracy_scores_dict['hybrid'],
                    accuracy_scores_dict['graph'],
                    test_type='t-test'
                )
                if sig_test.get('p_value') is not None:
                    print(f"\nHybrid vs Graph (Accuracy):")
                    print(f"  p-value: {sig_test['p_value']:.4f} {sig_test.get('significance_level', '')}")
                    print(f"  Significant: {'Yes' if sig_test['significant'] else 'No'} (α=0.05)")
        
        print("\n" + "=" * 80)
    
    def save_evaluation_results(
        self,
        evaluation_results: Dict[str, Any],
        output_file: str
    ):
        """Save evaluation results to JSON file."""
        # Convert SearchMetrics objects to dicts
        serializable_results = {}
        for method, metrics_list in evaluation_results['results'].items():
            serializable_results[method] = [asdict(m) for m in metrics_list]
        
        output = {
            'results': serializable_results,
            'summary': evaluation_results['summary'],
            'test_queries': evaluation_results['test_queries'],
            'top_k': evaluation_results['top_k']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nEvaluation results saved to: {output_file}")


def main():
    """Example usage of the evaluation framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate vector vs graph search")
    parser.add_argument("--queries-file", type=str, help="JSON file with test queries")
    parser.add_argument("--output", type=str, default="evaluation_results.json", 
                       help="Output file for results")
    parser.add_argument("--top-k", type=int, default=10, help="Top K results")
    parser.add_argument("--no-accuracy", action="store_true",
                       help="Skip accuracy evaluation (faster)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Show detailed progress (default: True)")
    parser.add_argument("--quiet", action="store_true",
                       help="Show simplified progress view (overrides --verbose)")
    
    args = parser.parse_args()
    
    # Handle quiet flag
    verbose = args.verbose and not args.quiet
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Initialize evaluator
    evaluator = SearchEvaluator(str(project_root))
    
    # Test queries (default set)
    test_queries = [
        "What are the privacy policies?",
        "Find GDPR articles about data minimization",
        "What clauses address GDPR Article 5?",
        "Find incidents related to data breaches",
        "What are the compliance gaps?",
        "Show me information about user consent",
        "What are the data processing requirements?",
        "Find clauses about data retention"
    ]
    
    # Load queries from file if provided
    if args.queries_file:
        queries_path = Path(args.queries_file)
        if not queries_path.is_absolute():
            # Try relative to evaluation directory first
            eval_dir_path = Path(__file__).parent / queries_path
            # Try relative to project root
            root_path = project_root / queries_path
            if eval_dir_path.exists():
                queries_path = eval_dir_path
            elif root_path.exists():
                queries_path = root_path
            else:
                # Default to evaluation directory
                queries_path = eval_dir_path
        with open(queries_path, 'r') as f:
            data = json.load(f)
            test_queries = data.get('queries', test_queries)
    
    # Run evaluation (answer generation is mandatory)
    results = evaluator.run_evaluation_suite(
        test_queries=test_queries,
        top_k=args.top_k,
        evaluate_accuracy=not args.no_accuracy,
        verbose=verbose
    )
    
    # Print report
    evaluator.print_evaluation_report(results)
    
    # Save results
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    evaluator.save_evaluation_results(results, str(output_path))
    
    # Cleanup
    evaluator.neo4j_conn.close()


if __name__ == "__main__":
    main()

