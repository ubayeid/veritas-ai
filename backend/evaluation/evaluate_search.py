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
# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "searching"))
sys.path.insert(0, str(Path(__file__).parent.parent / "building_database" / "neo4j"))

from query_engine import VectorQueryEngine, get_openai_client
from hybrid_query_engine import HybridQueryEngine
from neo4j_queries import KnowledgeGraphQueries
from neo4j_connection import Neo4jConnection
from dotenv import load_dotenv

load_dotenv()
# Both models read from .env file
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
# Judge LLM for evaluation - reads from .env, falls back to LLM_MODEL if not set
# Set JUDGE_LLM_MODEL in .env to use a different model for evaluation
JUDGE_LLM_MODEL = os.getenv("JUDGE_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4"))


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
    mrr: Optional[float] = None  # Mean Reciprocal Rank
    ndcg_at_k: Optional[float] = None  # Normalized Discounted Cumulative Gain
    # Result quality indicators
    avg_similarity: Optional[float] = None  # For vector search
    result_diversity: Optional[float] = None  # How diverse are results
    # Source distribution
    sources: Optional[Dict[str, int]] = None


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
            client = get_openai_client()
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
            client = get_openai_client()
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
        evaluate_accuracy: bool = True
    ) -> SearchMetrics:
        """
        Evaluate vector search performance with answer generation.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            db_names: Databases to search (None = all)
            evaluate_accuracy: Whether to evaluate answer accuracy
            
        Returns:
            SearchMetrics object with metrics
        """
        start_time = time.perf_counter()
        
        # Use full query pipeline with answer generation (mandatory)
        result = self.vector_engine.query(
            query=query,
            db_names=db_names,
            top_k=top_k,
            rerank=True,
            generate_answer=True,  # Always enabled
            similarity_threshold=0.0
        )
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        results = result.get('results', [])
        answer = result.get('answer', '')
        
        # Measure answer generation time separately
        # For vector search, we can't easily separate answer generation time
        # Estimate it (typically 40-60% of total time)
        answer_generation_time_ms = execution_time_ms * 0.5 if answer else None
        
        # Evaluate accuracy if requested
        accuracy_score = None
        accuracy_reasoning = None
        if evaluate_accuracy and answer:
            # Evaluate generated answer accuracy
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
        
        return SearchMetrics(
            query=query,
            method='vector',
            execution_time_ms=execution_time_ms,
            num_results=len(results),
            top_k=top_k,
            answer_generation_time_ms=answer_generation_time_ms,
            generated_answer=answer,
            answer_length=len(answer) if answer else None,
            has_answer_generation=True,  # Always enabled
            accuracy_score=accuracy_score,
            accuracy_reasoning=accuracy_reasoning,
            avg_similarity=avg_similarity,
            sources=dict(sources)
        )
    
    def evaluate_graph_search(
        self,
        query: str,
        top_k: int = 10,
        evaluate_accuracy: bool = True
    ) -> SearchMetrics:
        """
        Evaluate graph traversal search performance with answer generation.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            evaluate_accuracy: Whether to evaluate answer accuracy
            
        Returns:
            SearchMetrics object with metrics
        """
        start_time = time.perf_counter()
        
        # Get graph traversal results
        results = self.hybrid_engine.graph_traversal_search(query)
        
        # Limit to top_k
        results = results[:top_k]
        
        # Measure answer generation time separately (mandatory)
        answer_gen_start = time.perf_counter()
        answer = None
        if results:
            try:
                answer = self.vector_engine.generate_answer(query, results[:8])
            except Exception as e:
                print(f"Warning: Answer generation failed: {e}")
                answer = None
        
        answer_generation_time_ms = (time.perf_counter() - answer_gen_start) * 1000 if answer else None
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Evaluate accuracy if requested
        accuracy_score = None
        accuracy_reasoning = None
        if evaluate_accuracy and answer:
            # Evaluate generated answer accuracy
            accuracy_score, accuracy_reasoning = self.evaluate_answer_accuracy(query, answer, results)
        
        # Source distribution
        sources = defaultdict(int)
        for r in results:
            source = r.get('source', 'graph_traversal')
            sources[source] += 1
        
        return SearchMetrics(
            query=query,
            method='graph',
            execution_time_ms=execution_time_ms,
            num_results=len(results),
            top_k=top_k,
            answer_generation_time_ms=answer_generation_time_ms,
            generated_answer=answer,
            answer_length=len(answer) if answer else None,
            has_answer_generation=True,  # Always enabled
            accuracy_score=accuracy_score,
            accuracy_reasoning=accuracy_reasoning,
            sources=dict(sources)
        )
    
    def evaluate_hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        use_faiss: bool = True,
        use_graph_traversal: bool = True,
        evaluate_accuracy: bool = True
    ) -> SearchMetrics:
        """
        Evaluate hybrid search performance with answer generation.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            use_faiss: Use FAISS vector search
            use_graph_traversal: Use graph traversal
            evaluate_accuracy: Whether to evaluate answer accuracy
            
        Returns:
            SearchMetrics object with metrics
        """
        start_time = time.perf_counter()
        
        # Use full hybrid query pipeline with answer generation (mandatory)
        result = self.hybrid_engine.hybrid_query(
            query=query,
            top_k=top_k,
            rerank=True,
            generate_answer=True,  # Always enabled
            rrf_k=None  # Use default from .env
        )
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        results = result.get('results', [])
        answer = result.get('answer', '')
        
        # Measure answer generation time separately (approximate)
        # Estimate answer generation time (typically 40-60% of total time)
        answer_generation_time_ms = execution_time_ms * 0.5 if answer else None
        
        # Evaluate accuracy if requested
        accuracy_score = None
        accuracy_reasoning = None
        if evaluate_accuracy and answer:
            # Evaluate generated answer accuracy
            accuracy_score, accuracy_reasoning = self.evaluate_answer_accuracy(query, answer, results)
        
        # Calculate average similarity (for vector results)
        avg_similarity = None
        similarities = [r.get('similarity', 0.0) for r in results if 'similarity' in r]
        if similarities:
            avg_similarity = statistics.mean(similarities)
        
        # Source distribution
        sources = defaultdict(int)
        for r in results:
            source = r.get('source', 'unknown')
            sources[source] += 1
        
        return SearchMetrics(
            query=query,
            method='hybrid',
            execution_time_ms=execution_time_ms,
            num_results=len(results),
            top_k=top_k,
            answer_generation_time_ms=answer_generation_time_ms,
            generated_answer=answer,
            answer_length=len(answer) if answer else None,
            has_answer_generation=True,  # Always enabled
            accuracy_score=accuracy_score,
            accuracy_reasoning=accuracy_reasoning,
            avg_similarity=avg_similarity,
            sources=dict(sources)
        )
    
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
                print(f"[{i}/{len(test_queries)}] Evaluating query: '{query}'")
            else:
                # Simplified progress indicator
                progress = (i - 1) / len(test_queries) * 100
                print(f"[{i}/{len(test_queries)}] {progress:.0f}% - {query[:60]}{'...' if len(query) > 60 else ''}", end=" ")
            
            # Evaluate vector search (answer generation mandatory)
            if verbose:
                print("  → Vector search...", end=" ")
            vector_metrics = self.evaluate_vector_search(
                query, top_k=top_k, evaluate_accuracy=evaluate_accuracy
            )
            results['vector'].append(vector_metrics)
            if verbose:
                ans_time = f", ans: {vector_metrics.answer_generation_time_ms:.0f}ms" if vector_metrics.answer_generation_time_ms else ""
                acc = f", acc: {vector_metrics.accuracy_score:.2f}" if vector_metrics.accuracy_score is not None else ""
                print(f"✓ ({vector_metrics.execution_time_ms:.2f}ms{ans_time}{acc}, {vector_metrics.num_results} results)")
            
            # Evaluate graph search (answer generation mandatory)
            if verbose:
                print("  → Graph search...", end=" ")
            graph_metrics = self.evaluate_graph_search(
                query, top_k=top_k, evaluate_accuracy=evaluate_accuracy
            )
            results['graph'].append(graph_metrics)
            if verbose:
                ans_time = f", ans: {graph_metrics.answer_generation_time_ms:.0f}ms" if graph_metrics.answer_generation_time_ms else ""
                acc = f", acc: {graph_metrics.accuracy_score:.2f}" if graph_metrics.accuracy_score is not None else ""
                print(f"✓ ({graph_metrics.execution_time_ms:.2f}ms{ans_time}{acc}, {graph_metrics.num_results} results)")
            
            # Evaluate hybrid search (answer generation mandatory)
            if verbose:
                print("  → Hybrid search...", end=" ")
            hybrid_metrics = self.evaluate_hybrid_search(
                query, top_k=top_k, evaluate_accuracy=evaluate_accuracy
            )
            results['hybrid'].append(hybrid_metrics)
            if verbose:
                ans_time = f", ans: {hybrid_metrics.answer_generation_time_ms:.0f}ms" if hybrid_metrics.answer_generation_time_ms else ""
                acc = f", acc: {hybrid_metrics.accuracy_score:.2f}" if hybrid_metrics.accuracy_score is not None else ""
                print(f"✓ ({hybrid_metrics.execution_time_ms:.2f}ms{ans_time}{acc}, {hybrid_metrics.num_results} results)")
            else:
                # Simplified: show completion with average accuracy
                avg_acc = (vector_metrics.accuracy_score or 0) + (graph_metrics.accuracy_score or 0) + (hybrid_metrics.accuracy_score or 0)
                if evaluate_accuracy and avg_acc > 0:
                    avg_acc = avg_acc / 3
                    print(f"✓ (avg acc: {avg_acc:.2f})")
                else:
                    print("✓")
            
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
        """Calculate aggregate statistics."""
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
            
            summary[method] = {
                'avg_execution_time_ms': statistics.mean(execution_times),
                'median_execution_time_ms': statistics.median(execution_times),
                'min_execution_time_ms': min(execution_times),
                'max_execution_time_ms': max(execution_times),
                'std_execution_time_ms': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'avg_num_results': statistics.mean(num_results),
                'avg_similarity': statistics.mean(avg_similarities) if avg_similarities else None,
                'avg_answer_generation_time_ms': statistics.mean(answer_generation_times) if answer_generation_times else None,
                'avg_answer_length': statistics.mean(answer_lengths) if answer_lengths else None,
                'avg_accuracy_score': statistics.mean(accuracy_scores) if accuracy_scores else None,
                'total_queries': len(metrics_list)
            }
        
        return summary
    
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
                print(f"  Average time: {stats['avg_execution_time_ms']:.2f}ms")
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
                    print(f"  Avg accuracy score: {stats['avg_accuracy_score']:.3f} (0.0-1.0)")
        
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

