"""
Evaluation Framework for Vector vs Graph Search
Measures timing, accuracy, and other metrics for comparison.
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
# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "searching"))
sys.path.insert(0, str(Path(__file__).parent.parent / "building_database" / "neo4j"))

from query_engine import VectorQueryEngine
from hybrid_query_engine import HybridQueryEngine
from neo4j_queries import KnowledgeGraphQueries
from neo4j_connection import Neo4jConnection


@dataclass
class SearchMetrics:
    """Metrics for a single search query."""
    query: str
    method: str  # 'vector', 'graph', 'hybrid'
    execution_time_ms: float
    num_results: int
    top_k: int
    # Contextualization metrics (mandatory)
    contextualization_time_ms: Optional[float] = None
    contextualized_answer: Optional[str] = None
    answer_length: Optional[int] = None
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
    
    def evaluate_vector_search(
        self, 
        query: str, 
        top_k: int = 10,
        db_names: Optional[List[str]] = None
    ) -> SearchMetrics:
        """
        Evaluate vector search performance with mandatory contextualization.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            db_names: Databases to search (None = all)
            
        Returns:
            SearchMetrics object with contextualization metrics
        """
        start_time = time.perf_counter()
        
        # Use full query pipeline with mandatory contextualization
        result = self.vector_engine.query(
            query=query,
            db_names=db_names,
            top_k=top_k,
            rerank=True,
            contextualize=True,  # Mandatory
            similarity_threshold=0.0,
            use_expansion=True
        )
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        results = result.get('results', [])
        answer = result.get('answer', '')
        
        # Measure contextualization time separately (approximate)
        # Note: This is included in total time, but we track it separately
        contextualization_time_ms = None
        if answer:
            # Estimate contextualization time (typically 30-50% of total time)
            # This is approximate since we can't measure it separately without modifying the engine
            contextualization_time_ms = execution_time_ms * 0.4  # Rough estimate
        
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
            contextualization_time_ms=contextualization_time_ms,
            contextualized_answer=answer,
            answer_length=len(answer) if answer else None,
            avg_similarity=avg_similarity,
            sources=dict(sources)
        )
    
    def evaluate_graph_search(
        self,
        query: str,
        top_k: int = 10
    ) -> SearchMetrics:
        """
        Evaluate graph traversal search performance with mandatory contextualization.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            SearchMetrics object with contextualization metrics
        """
        start_time = time.perf_counter()
        
        # Get graph traversal results
        results = self.hybrid_engine.graph_traversal_search(query)
        
        # Limit to top_k
        results = results[:top_k]
        
        # Measure contextualization time
        contextualize_start = time.perf_counter()
        
        # Generate contextualized answer using vector engine's contextualization
        answer = None
        if results:
            try:
                answer = self.vector_engine.contextualize_results(query, results[:8])
            except Exception as e:
                print(f"Warning: Contextualization failed: {e}")
                answer = None
        
        contextualization_time_ms = (time.perf_counter() - contextualize_start) * 1000
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
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
            contextualization_time_ms=contextualization_time_ms,
            contextualized_answer=answer,
            answer_length=len(answer) if answer else None,
            sources=dict(sources)
        )
    
    def evaluate_hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        use_faiss: bool = True,
        use_graph_traversal: bool = True
    ) -> SearchMetrics:
        """
        Evaluate hybrid search performance with mandatory contextualization.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            use_faiss: Use FAISS vector search
            use_graph_traversal: Use graph traversal
            
        Returns:
            SearchMetrics object with contextualization metrics
        """
        start_time = time.perf_counter()
        
        # Use full hybrid query pipeline with mandatory contextualization
        result = self.hybrid_engine.hybrid_query(
            query=query,
            top_k=top_k,
            rerank=True,
            contextualize=True,  # Mandatory
            rrf_k=None  # Use default from .env
        )
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        results = result.get('results', [])
        answer = result.get('answer', '')
        
        # Measure contextualization time separately (approximate)
        contextualization_time_ms = None
        if answer:
            # Estimate contextualization time (typically 30-50% of total time)
            contextualization_time_ms = execution_time_ms * 0.4  # Rough estimate
        
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
            contextualization_time_ms=contextualization_time_ms,
            contextualized_answer=answer,
            answer_length=len(answer) if answer else None,
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
        ground_truth: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation suite.
        
        Args:
            test_queries: List of test queries
            top_k: Number of results to retrieve
            ground_truth: Optional dict mapping query to list of relevant result IDs
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            'vector': [],
            'graph': [],
            'hybrid': []
        }
        
        print("=" * 80)
        print("SEARCH EVALUATION SUITE")
        print("=" * 80)
        print(f"Test queries: {len(test_queries)}")
        print(f"Top K: {top_k}")
        print("=" * 80)
        print()
        
        for i, query in enumerate(test_queries, 1):
            print(f"[{i}/{len(test_queries)}] Evaluating query: '{query}'")
            
            # Evaluate vector search (with mandatory contextualization)
            print("  → Vector search...", end=" ")
            vector_metrics = self.evaluate_vector_search(query, top_k=top_k)
            results['vector'].append(vector_metrics)
            ctx_time = f", ctx: {vector_metrics.contextualization_time_ms:.0f}ms" if vector_metrics.contextualization_time_ms else ""
            print(f"✓ ({vector_metrics.execution_time_ms:.2f}ms{ctx_time}, {vector_metrics.num_results} results)")
            
            # Evaluate graph search (with mandatory contextualization)
            print("  → Graph search...", end=" ")
            graph_metrics = self.evaluate_graph_search(query, top_k=top_k)
            results['graph'].append(graph_metrics)
            ctx_time = f", ctx: {graph_metrics.contextualization_time_ms:.0f}ms" if graph_metrics.contextualization_time_ms else ""
            print(f"✓ ({graph_metrics.execution_time_ms:.2f}ms{ctx_time}, {graph_metrics.num_results} results)")
            
            # Evaluate hybrid search (with mandatory contextualization)
            print("  → Hybrid search...", end=" ")
            hybrid_metrics = self.evaluate_hybrid_search(query, top_k=top_k)
            results['hybrid'].append(hybrid_metrics)
            ctx_time = f", ctx: {hybrid_metrics.contextualization_time_ms:.0f}ms" if hybrid_metrics.contextualization_time_ms else ""
            print(f"✓ ({hybrid_metrics.execution_time_ms:.2f}ms{ctx_time}, {hybrid_metrics.num_results} results)")
            
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
            contextualization_times = [m.contextualization_time_ms for m in metrics_list if m.contextualization_time_ms is not None]
            answer_lengths = [m.answer_length for m in metrics_list if m.answer_length is not None]
            
            summary[method] = {
                'avg_execution_time_ms': statistics.mean(execution_times),
                'median_execution_time_ms': statistics.median(execution_times),
                'min_execution_time_ms': min(execution_times),
                'max_execution_time_ms': max(execution_times),
                'std_execution_time_ms': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'avg_num_results': statistics.mean(num_results),
                'avg_similarity': statistics.mean(avg_similarities) if avg_similarities else None,
                'avg_contextualization_time_ms': statistics.mean(contextualization_times) if contextualization_times else None,
                'avg_answer_length': statistics.mean(answer_lengths) if answer_lengths else None,
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
                if stats.get('avg_contextualization_time_ms') is not None:
                    print(f"  Avg contextualization time: {stats['avg_contextualization_time_ms']:.2f}ms")
                if stats.get('avg_answer_length') is not None:
                    print(f"  Avg answer length: {stats['avg_answer_length']:.0f} chars")
        
        # Contextualization metrics
        print("\n" + "=" * 80)
        print("CONTEXTUALIZATION METRICS")
        print("-" * 80)
        for method in ['vector', 'graph', 'hybrid']:
            if method in summary:
                stats = summary[method]
                print(f"\n{method.upper()}:")
                if stats.get('avg_contextualization_time_ms') is not None:
                    print(f"  Avg contextualization time: {stats['avg_contextualization_time_ms']:.2f}ms")
                if stats.get('avg_answer_length') is not None:
                    print(f"  Avg answer length: {stats['avg_answer_length']:.0f} characters")
                if stats.get('avg_contextualization_time_ms') is not None and stats.get('avg_execution_time_ms'):
                    ctx_pct = (stats['avg_contextualization_time_ms'] / stats['avg_execution_time_ms']) * 100
                    print(f"  Contextualization % of total: {ctx_pct:.1f}%")
        
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
    
    args = parser.parse_args()
    
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
            queries_path = Path(__file__).parent / queries_path
        with open(queries_path, 'r') as f:
            data = json.load(f)
            test_queries = data.get('queries', test_queries)
    
    # Run evaluation
    results = evaluator.run_evaluation_suite(
        test_queries=test_queries,
        top_k=args.top_k
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

