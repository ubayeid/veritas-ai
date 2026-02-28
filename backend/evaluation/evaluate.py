"""
Unified Evaluation Script
Combines retrieval evaluation and metrics calculation.
"""

import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.evaluation.ir_evaluation import IREvaluator


def safe_print(*args, **kwargs):
    """Print with encoding error handling for Windows."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback: encode problematic characters
        try:
            encoded_args = [str(arg).encode('ascii', 'replace').decode('ascii') if isinstance(arg, str) else arg for arg in args]
            print(*encoded_args, **kwargs)
        except:
            pass  # Skip printing if still fails


def export_chunks_to_csv(all_results: List[Dict], output_csv: str):
    """Export retrieval chunks to CSV for manual relevance labeling.
    
    Format: One row per chunk with columns:
    - Query ID, Query Text, Query Type
    - Method (vector/graph/hybrid)
    - Rank (1-based)
    - Chunk ID, Chunk Text, Similarity, Source
    - Relevance (empty column for manual labeling: 0=not relevant, 1=partially relevant, 2=highly relevant)
    """
    rows = []
    
    for result in all_results:
        query_id = result['query_id']
        query_text = result['query']
        query_type = result['query_type']
        chunks_by_method = result.get('chunks_by_method', {})
        
        # Create one row per chunk (long format for easier labeling)
        for method in ['vector', 'graph', 'hybrid']:
            chunks = chunks_by_method.get(method, [])
            
            for rank, chunk in enumerate(chunks, start=1):
                row = {
                    'Query ID': query_id,
                    'Query Text': query_text,
                    'Query Type': query_type,
                    'Method': method.upper(),
                    'Rank': rank,
                    'Chunk ID': chunk.get('chunk_id', ''),
                    'Chunk Text': chunk.get('text', ''),
                    'Similarity': chunk.get('similarity', ''),
                    'Source': chunk.get('source_name', ''),
                    'Relevance': ''  # Empty column for manual labeling (0, 1, or 2)
                }
                rows.append(row)
    
    # Fieldnames
    fieldnames = [
        'Query ID', 'Query Text', 'Query Type', 'Method', 'Rank',
        'Chunk ID', 'Chunk Text', 'Similarity', 'Source', 'Relevance'
    ]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    total_queries = len(set(r['Query ID'] for r in rows))
    total_chunks = len(rows)
    print(f"\nExported {total_queries} queries with {total_chunks} chunks to: {output_csv}")
    print(f"  Format: One row per chunk (long format) for manual relevance labeling")
    print(f"  Relevance labels: 0=not relevant, 1=partially relevant, 2=highly relevant")
    print(f"  Methods: Vector, Graph, Hybrid (top-k chunks per method per query)")


def run_retrieval_evaluation(
    queries_file: str,
    output_csv: str,
    top_k: int = 8,
    generate_answer: bool = True,
    export_chunks: bool = False
):
    """Run evaluation and export to CSV with answers only (no retrieval chunks) or chunks for review."""
    # Load queries
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    
    queries = queries_data.get('queries', [])
    
    print("=" * 80)
    if export_chunks:
        print("EVALUATION: EXPORT RETRIEVAL CHUNKS FOR MANUAL REVIEW")
    else:
        print("EVALUATION: ANSWERS ONLY")
    print("=" * 80)
    print(f"Total queries: {len(queries)}")
    print(f"Top-K retrievals per method: {top_k}")
    print(f"Generate answers: {generate_answer}")
    if export_chunks:
        print(f"Output: Retrieval chunks (top-{top_k} per method) for manual review")
    else:
        print(f"Output: Answers only (retrieval chunks not included in CSV)")
    print()
    
    # Initialize evaluator
    evaluator = IREvaluator(str(project_root))
    
    all_results = []
    for i, q_data in enumerate(queries, 1):
        query_id = q_data['id']
        query = q_data['query']
        query_type = q_data.get('type', 'unknown')
        
        safe_print(f"[{i}/{len(queries)}] {query_id} ({query_type}): {query[:60]}...")
        
        try:
            result = evaluator.evaluate_query(
                query_id=query_id,
                query=query,
                top_k=top_k,
                generate_answer=generate_answer
            )
            
            result_dict = {
                'query_id': query_id,
                'query': query,
                'query_type': query_type,
                'answer': result.answer if result.answer else '',  # Backward compatibility
                'answers_by_method': getattr(result, 'answers_by_method', {}),
                'chunks_by_method': {}
            }
            
            for method in ['vector', 'graph', 'hybrid']:
                chunks = result.chunks_by_method.get(method, [])
                result_dict['chunks_by_method'][method] = [
                    {
                        'chunk_id': c.chunk_id,
                        'text': c.text,
                        'similarity': c.similarity,
                        'source_name': c.source_name,
                        'database': c.database,
                        'source': c.source
                    }
                    for c in chunks
                ]
            
            all_results.append(result_dict)
            
            # Display progress
            vector_count = len(result.chunks_by_method.get('vector', []))
            graph_count = len(result.chunks_by_method.get('graph', []))
            hybrid_count = len(result.chunks_by_method.get('hybrid', []))
            
            if export_chunks:
                # Simple progress output when exporting chunks
                safe_print(f"  Retrievals - Vector: {vector_count} | Graph: {graph_count} | Hybrid: {hybrid_count}")
            else:
                # Full output with answers
                answers_by_method = getattr(result, 'answers_by_method', {}) or {}
                print(f"\n{'='*80}")
                print(f"QUERY: {query}")
                print(f"{'='*80}")
                print(f"Retrievals - Vector: {vector_count} | Graph: {graph_count} | Hybrid: {hybrid_count}")
                print(f"\n{'-'*80}")
                
                # Display Vector Answer
                vector_ans = answers_by_method.get('vector', '')
                if vector_ans:
                    print(f"\n[VECTOR ANSWER]")
                    print(f"{vector_ans}")
                else:
                    print(f"\n[VECTOR ANSWER] - No answer generated")
                
                # Display Graph Answer
                graph_ans = answers_by_method.get('graph', '')
                if graph_ans:
                    print(f"\n[GRAPH ANSWER]")
                    print(f"{graph_ans}")
                else:
                    print(f"\n[GRAPH ANSWER] - No answer generated")
                
                # Display Hybrid Answer
                hybrid_ans = answers_by_method.get('hybrid', '')
                if hybrid_ans:
                    print(f"\n[HYBRID ANSWER]")
                    print(f"{hybrid_ans}")
                else:
                    print(f"\n[HYBRID ANSWER] - No answer generated")
                
                print(f"\n{'-'*80}\n")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Export to CSV
    if export_chunks:
        # Export chunks for manual review
        export_chunks_to_csv(all_results, output_csv)
    else:
        # Export answers only (original format)
        rows = []
        for result in all_results:
            query_id = result['query_id']
            query_text = result['query']
            query_type = result['query_type']
            answers_by_method = result.get('answers_by_method', {})
            
            # Create row with only query info and answers
            row = {
                'Query ID': query_id,
                'Query Text': query_text,
                'Query Type': query_type,
                'vector ans': answers_by_method.get('vector', '') or '',
                'Graph ans': answers_by_method.get('graph', '') or '',
                'hybrid ans': answers_by_method.get('hybrid', '') or ''
            }
            
            rows.append(row)
        
        # Fieldnames: Query info + three answer columns
        fieldnames = ['Query ID', 'Query Text', 'Query Type', 'vector ans', 'Graph ans', 'hybrid ans']
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\nExported {len(rows)} rows to: {output_csv}")
        print(f"  Format: Answers only (no retrieval chunks)")
        print(f"  Columns: Query ID, Query Text, Query Type, vector ans, Graph ans, hybrid ans")
    
    # Close graph engine connection
    try:
        evaluator.graph_engine.close()
    except:
        pass  # Connection may already be closed or not exist


def calculate_metrics(
    labels_file: str,
    output_file: Optional[str] = None,
    k: int = 10,
    per_method: bool = True
):
    """Calculate IR metrics from labeled CSV file."""
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required for metrics calculation. Install with: pip install pandas")
        return
    
    # Load labeled CSV
    df = pd.read_csv(labels_file)
    
    if 'Relevance' not in df.columns:
        print("Error: CSV file must have 'Relevance' column with values 0, 1, or 2")
        return
    
    evaluator = IREvaluator(str(project_root))
    
    if per_method:
        # Calculate per-method metrics
        methods = ['VECTOR', 'GRAPH', 'HYBRID']
        all_method_metrics = {}
        
        for method in methods:
            method_df = df[df['Method'] == method].copy()
            if method_df.empty:
                continue
            
            # Group by query
            per_query_metrics = []
            for query_id in method_df['Query ID'].unique():
                query_df = method_df[method_df['Query ID'] == query_id]
                query_df = query_df.sort_values('Rank').head(k)
                
                # Calculate metrics
                relevant_count = (query_df['Relevance'] >= 1).sum()
                total_relevant = (df[df['Query ID'] == query_id]['Relevance'] >= 1).sum()
                
                precision = relevant_count / k if k > 0 else 0.0
                recall = relevant_count / total_relevant if total_relevant > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # MRR
                mrr = 0.0
                for rank, rel in enumerate(query_df['Relevance'], start=1):
                    if rel >= 1:
                        mrr = 1.0 / rank
                        break
                
                # MAP
                map_score = 0.0
                relevant_so_far = 0
                precision_sum = 0.0
                for rank, rel in enumerate(query_df['Relevance'], start=1):
                    if rel >= 1:
                        relevant_so_far += 1
                        precision_sum += relevant_so_far / rank
                if total_relevant > 0:
                    map_score = precision_sum / total_relevant
                
                # NDCG
                import math
                dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(query_df['Relevance'], start=1))
                ideal_relevances = sorted(df[df['Query ID'] == query_id]['Relevance'], reverse=True)[:k]
                idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, start=1))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                
                per_query_metrics.append({
                    'query_id': query_id,
                    'precision_at_k': precision,
                    'recall_at_k': recall,
                    'f1_at_k': f1,
                    'mrr': mrr,
                    'map': map_score,
                    'ndcg_at_k': ndcg
                })
            
            # Aggregate
            if per_query_metrics:
                aggregate = {
                    'avg_precision_at_k': sum(m['precision_at_k'] for m in per_query_metrics) / len(per_query_metrics),
                    'avg_recall_at_k': sum(m['recall_at_k'] for m in per_query_metrics) / len(per_query_metrics),
                    'avg_f1_at_k': sum(m['f1_at_k'] for m in per_query_metrics) / len(per_query_metrics),
                    'avg_mrr': sum(m['mrr'] for m in per_query_metrics) / len(per_query_metrics),
                    'avg_map': sum(m['map'] for m in per_query_metrics) / len(per_query_metrics),
                    'avg_ndcg_at_k': sum(m['ndcg_at_k'] for m in per_query_metrics) / len(per_query_metrics),
                    'total_queries': len(per_query_metrics)
                }
                
                all_method_metrics[method.lower()] = {
                    'per_query': per_query_metrics,
                    'aggregate': aggregate
                }
        
        # Print comparison table
        print("\n" + "=" * 80)
        print("PER-METHOD METRICS COMPARISON")
        print("=" * 80)
        print(f"\n{'Method':<20} {'P@{k}':<10} {'R@{k}':<10} {'F1@{k}':<10} {'MRR':<10} {'MAP':<10} {'NDCG@{k}':<10}")
        print("-" * 80)
        
        for method in ['vector', 'graph', 'hybrid']:
            if method in all_method_metrics:
                agg = all_method_metrics[method]['aggregate']
                print(f"{method.capitalize():<20} "
                      f"{agg['avg_precision_at_k']:<10.3f} "
                      f"{agg['avg_recall_at_k']:<10.3f} "
                      f"{agg['avg_f1_at_k']:<10.3f} "
                      f"{agg['avg_mrr']:<10.3f} "
                      f"{agg['avg_map']:<10.3f} "
                      f"{agg['avg_ndcg_at_k']:<10.3f}")
        
        if output_file:
            output_data = {
                "k": k,
                "methods": all_method_metrics,
                "comparison_table": {
                    method: all_method_metrics[method]['aggregate']
                    for method in all_method_metrics.keys()
                }
            }
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nMetrics saved to: {output_file}")
    
    # Close graph engine connection
    try:
        evaluator.graph_engine.close()
    except:
        pass  # Connection may already be closed or not exist


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified evaluation script")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run evaluation
    eval_parser = subparsers.add_parser('run', help='Run retrieval evaluation')
    eval_parser.add_argument('--queries', type=str,
                            default='backend/evaluation/evaluation_queries_50.json',
                            help='Queries JSON file')
    eval_parser.add_argument('--output', type=str,
                            default='backend/evaluation/retrieval_results.csv',
                            help='Output CSV file')
    eval_parser.add_argument('--top-k', type=int, default=8,
                            help='Top-K retrievals per method (default: 8)')
    eval_parser.add_argument('--no-answer', action='store_true',
                            help='Skip answer generation (faster, no answers in CSV)')
    eval_parser.add_argument('--export-chunks', action='store_true',
                            help='Export retrieval chunks for manual review (instead of answers)')
    
    # Calculate metrics
    metrics_parser = subparsers.add_parser('metrics', help='Calculate IR metrics')
    metrics_parser.add_argument('--labels', type=str, required=True,
                               help='Labeled CSV file')
    metrics_parser.add_argument('--output', type=str,
                               help='Output JSON file for metrics')
    metrics_parser.add_argument('--k', type=int, default=10,
                               help='Top K for metrics')
    metrics_parser.add_argument('--per-method', action='store_true', default=True,
                               help='Calculate per-method metrics')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        run_retrieval_evaluation(
            queries_file=args.queries,
            output_csv=args.output,
            top_k=args.top_k,
            generate_answer=not args.no_answer,
            export_chunks=args.export_chunks
        )
    elif args.command == 'metrics':
        calculate_metrics(
            labels_file=args.labels,
            output_file=args.output,
            k=args.k,
            per_method=args.per_method
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
