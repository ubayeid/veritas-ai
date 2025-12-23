"""
Ablation Study Framework for Compliance RAG System

This module provides functionality to run systematic ablation studies
to understand the contribution of different components.

Ablation Studies:
1. RRF vs Simple Concatenation
2. With/Without LLM Reranking
3. Different RRF_k values
4. Different top_k values
5. Different similarity thresholds
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import sys

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "searching"))
from evaluate_search import SearchEvaluator


@dataclass
class AblationConfig:
    """Configuration for an ablation study."""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AblationStudyRunner:
    """Runs systematic ablation studies."""
    
    def __init__(self, base_dir: str):
        """
        Initialize ablation study runner.
        
        Args:
            base_dir: Base directory of the project
        """
        self.base_dir = Path(base_dir)
        self.evaluator = SearchEvaluator(str(base_dir))
    
    def run_rrf_ablation(
        self,
        test_queries: List[str],
        rrf_k_values: List[int] = [30, 60, 90, 120],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Ablation study: Test different RRF_k values.
        
        Args:
            test_queries: List of test queries
            rrf_k_values: List of RRF_k values to test
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with results for each RRF_k value
        """
        results = {}
        
        print("=" * 80)
        print("ABLATION STUDY: RRF_k Values")
        print("=" * 80)
        print(f"Testing RRF_k values: {rrf_k_values}")
        print(f"Queries: {len(test_queries)}")
        print()
        
        for rrf_k in rrf_k_values:
            print(f"Testing RRF_k = {rrf_k}...")
            # Note: This requires modifying hybrid_query_engine to accept rrf_k parameter
            # For now, this is a template
            config = AblationConfig(
                name=f"RRF_k_{rrf_k}",
                description=f"Hybrid search with RRF_k={rrf_k}",
                parameters={'rrf_k': rrf_k, 'top_k': top_k}
            )
            results[f"rrf_k_{rrf_k}"] = {
                'config': config.to_dict(),
                'note': 'Implementation requires modifying hybrid_query_engine to accept rrf_k parameter'
            }
        
        return results
    
    def run_reranking_ablation(
        self,
        test_queries: List[str],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Ablation study: With/Without LLM reranking.
        
        Args:
            test_queries: List of test queries
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with results for with/without reranking
        """
        results = {}
        
        print("=" * 80)
        print("ABLATION STUDY: LLM Reranking")
        print("=" * 80)
        print("Testing: With reranking vs Without reranking")
        print(f"Queries: {len(test_queries)}")
        print()
        
        # With reranking
        print("Testing WITH reranking...")
        eval_results_with = self.evaluator.run_evaluation_suite(
            test_queries=test_queries,
            top_k=top_k,
            evaluate_accuracy=True,
            verbose=False
        )
        results['with_reranking'] = eval_results_with['summary']
        
        # Without reranking (requires modifying query_engine)
        print("Testing WITHOUT reranking...")
        results['without_reranking'] = {
            'note': 'Implementation requires modifying query_engine to disable reranking'
        }
        
        return results
    
    def run_topk_ablation(
        self,
        test_queries: List[str],
        top_k_values: List[int] = [5, 10, 15, 20]
    ) -> Dict[str, Any]:
        """
        Ablation study: Test different top_k values.
        
        Args:
            test_queries: List of test queries
            top_k_values: List of top_k values to test
            
        Returns:
            Dictionary with results for each top_k value
        """
        results = {}
        
        print("=" * 80)
        print("ABLATION STUDY: Top-K Values")
        print("=" * 80)
        print(f"Testing top_k values: {top_k_values}")
        print(f"Queries: {len(test_queries)}")
        print()
        
        for top_k in top_k_values:
            print(f"Testing top_k = {top_k}...")
            eval_results = self.evaluator.run_evaluation_suite(
                test_queries=test_queries,
                top_k=top_k,
                evaluate_accuracy=True,
                verbose=False
            )
            results[f"top_k_{top_k}"] = eval_results['summary']
        
        return results
    
    def run_component_ablation(
        self,
        test_queries: List[str],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Ablation study: Test individual components.
        
        Tests:
        - Vector only
        - Graph only
        - Hybrid (both)
        
        Args:
            test_queries: List of test queries
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with results for each component
        """
        results = {}
        
        print("=" * 80)
        print("ABLATION STUDY: Component Analysis")
        print("=" * 80)
        print("Testing: Vector only, Graph only, Hybrid")
        print(f"Queries: {len(test_queries)}")
        print()
        
        # Run full evaluation (includes all components)
        eval_results = self.evaluator.run_evaluation_suite(
            test_queries=test_queries,
            top_k=top_k,
            evaluate_accuracy=True,
            verbose=False
        )
        
        results['vector_only'] = eval_results['summary'].get('vector', {})
        results['graph_only'] = eval_results['summary'].get('graph', {})
        results['hybrid'] = eval_results['summary'].get('hybrid', {})
        
        return results
    
    def save_ablation_results(
        self,
        results: Dict[str, Any],
        output_file: str
    ):
        """Save ablation study results to JSON file."""
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = Path(__file__).parent / output_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAblation results saved to: {output_path}")


def main():
    """Example usage of ablation study framework."""
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--study", type=str, choices=['rrf', 'reranking', 'topk', 'components', 'all'],
                       default='components', help="Which ablation study to run")
    parser.add_argument("--queries-file", type=str, help="JSON file with test queries")
    parser.add_argument("--output", type=str, default="ablation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Initialize runner
    runner = AblationStudyRunner(str(project_root))
    
    # Load test queries
    test_queries = [
        "What are the privacy policies?",
        "Find GDPR articles about data minimization",
        "What clauses address GDPR Article 5?",
        "Find incidents related to data breaches",
        "What are the compliance gaps?"
    ]
    
    if args.queries_file:
        queries_path = Path(args.queries_file)
        if not queries_path.is_absolute():
            queries_path = Path(__file__).parent / queries_path
        with open(queries_path, 'r') as f:
            data = json.load(f)
            test_queries = data.get('queries', test_queries)
    
    # Run selected study
    results = {}
    
    if args.study == 'rrf' or args.study == 'all':
        results['rrf_ablation'] = runner.run_rrf_ablation(test_queries)
    
    if args.study == 'reranking' or args.study == 'all':
        results['reranking_ablation'] = runner.run_reranking_ablation(test_queries)
    
    if args.study == 'topk' or args.study == 'all':
        results['topk_ablation'] = runner.run_topk_ablation(test_queries)
    
    if args.study == 'components' or args.study == 'all':
        results['component_ablation'] = runner.run_component_ablation(test_queries)
    
    # Save results
    runner.save_ablation_results(results, args.output)
    
    # Cleanup
    runner.evaluator.neo4j_conn.close()


if __name__ == "__main__":
    main()

