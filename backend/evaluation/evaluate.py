"""
Unified Evaluation Script
Combines retrieval evaluation and metrics calculation.
"""

import json
import csv
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.evaluation.ir_evaluation import IREvaluator, ChunkResult
from backend.evaluation.get_metrics import calculate_accuracy_metrics, display_metrics


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
    - Vector Answer, Graph Answer, Hybrid Answer (generated answers for the query; written once per query)
    - Answers Row (TRUE on the single row that contains the answers for that query)
    - Relevance (empty column for manual labeling: 0=not relevant, 1=partially relevant, 2=highly relevant)
    """
    rows = []
    
    for result in all_results:
        query_id = result['query_id']
        query_text = result['query']
        query_type = result['query_type']
        chunks_by_method = result.get('chunks_by_method', {})
        answers_by_method = result.get('answers_by_method', {})
        
        # Get answers for this query (same for all chunks of this query)
        vector_answer = answers_by_method.get('vector', '') or ''
        graph_answer = answers_by_method.get('graph', '') or ''
        hybrid_answer = answers_by_method.get('hybrid', '') or ''
        wrote_answers_for_query = False
        
        # Create one row per chunk (long format for easier labeling)
        for method in ['vector', 'graph', 'hybrid']:
            chunks = chunks_by_method.get(method, [])
            
            for rank, chunk in enumerate(chunks, start=1):
                # Write answers once per query to avoid repeating large text in every row.
                include_answers = (not wrote_answers_for_query)
                if include_answers:
                    wrote_answers_for_query = True

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
                    'Vector Answer': vector_answer if include_answers else '',
                    'Graph Answer': graph_answer if include_answers else '',
                    'Hybrid Answer': hybrid_answer if include_answers else '',
                    'Answers Row': 'TRUE' if include_answers else 'FALSE',
                    'Relevance': ''  # Empty column for manual labeling (0, 1, or 2)
                }
                rows.append(row)
    
    # Fieldnames
    fieldnames = [
        'Query ID', 'Query Text', 'Query Type', 'Method', 'Rank',
        'Chunk ID', 'Chunk Text', 'Similarity', 'Source',
        'Vector Answer', 'Graph Answer', 'Hybrid Answer', 'Answers Row', 'Relevance'
    ]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    total_queries = len(set(r['Query ID'] for r in rows))
    total_chunks = len(rows)
    print(f"\nExported {total_queries} queries with {total_chunks} chunks to: {output_csv}")
    print(f"  Format: One row per chunk (long format) for manual relevance labeling")
    print(f"  Includes generated answers for each query (Vector, Graph, Hybrid)")
    print(f"  Relevance labels: 0=not relevant, 1=partially relevant, 2=highly relevant")
    print(f"  Methods: Vector, Graph, Hybrid (top-k chunks per method per query)")


def _graph_result_allowed(db_name: str, chunk: Dict[str, Any]) -> bool:
    """Heuristic DB filter for graph results."""
    ctype = (chunk.get("type") or "").lower()

    if db_name == "company":
        return ctype in {"clause", "coverage"}
    if db_name == "aiid":
        return ctype in {"incident", "risk"}
    if db_name == "standards":
        return ctype in {"article"}

    return True


def _filter_graph_chunks_for_db(db_name: str, graph_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in graph_chunks if _graph_result_allowed(db_name, c)]


def _chunk_to_row(chunk: Any) -> Dict[str, Any]:
    """Normalize ChunkResult/dict chunk to serializable dict."""
    if isinstance(chunk, ChunkResult):
        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "similarity": chunk.similarity,
            "source_name": chunk.source_name,
            "database": chunk.database,
            "source": chunk.source,
        }
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "text": chunk.get("text", ""),
        "similarity": chunk.get("similarity", ""),
        "source_name": chunk.get("source_name", ""),
        "database": chunk.get("database", ""),
        "source": chunk.get("source", ""),
    }


def export_chunks_to_csv_with_timing(
    all_results: List[Dict[str, Any]],
    output_csv: str,
    include_methods: Optional[List[str]] = None,
) -> None:
    """Chunk export including timing columns (used for --db mode)."""
    methods = include_methods or ["vector", "graph", "hybrid"]
    rows: List[Dict[str, Any]] = []

    for result in all_results:
        query_id = result["query_id"]
        query_text = result["query"]
        query_type = result["query_type"]
        chunks_by_method = result.get("chunks_by_method", {}) or {}
        answers_by_method = result.get("answers_by_method", {}) or {}
        timing_breakdown_by_method = result.get("timing_breakdown_by_method", {}) or {}

        wrote_answers_for_query = False
        for method in methods:
            chunks = chunks_by_method.get(method, []) or []
            for rank, chunk in enumerate(chunks, start=1):
                include_answers = not wrote_answers_for_query
                if include_answers:
                    wrote_answers_for_query = True
                timing = timing_breakdown_by_method.get(method, {}) or {}

                rows.append(
                    {
                        "Query ID": query_id,
                        "Query Text": query_text,
                        "Query Type": query_type,
                        "Method": method.upper(),
                        "Rank": rank,
                        "Chunk ID": chunk.get("chunk_id", ""),
                        "Chunk Text": chunk.get("text", ""),
                        "Similarity": chunk.get("similarity", ""),
                        "Source": chunk.get("source_name", ""),
                        "Vector Answer": (answers_by_method.get("vector") or "") if include_answers else "",
                        "Graph Answer": (answers_by_method.get("graph") or "") if include_answers else "",
                        "Hybrid Answer": (answers_by_method.get("hybrid") or "") if include_answers else "",
                        "Answers Row": "TRUE" if include_answers else "FALSE",
                        "total_s": timing.get("total_s", ""),
                        "retrieval_s": timing.get("retrieval_s", ""),
                        "answer_s": timing.get("answer_s", ""),
                        "non_answer_s": timing.get("non_answer_s", ""),
                        "Relevance": "",
                    }
                )

    fieldnames = [
        "Query ID",
        "Query Text",
        "Query Type",
        "Method",
        "Rank",
        "Chunk ID",
        "Chunk Text",
        "Similarity",
        "Source",
        "Vector Answer",
        "Graph Answer",
        "Hybrid Answer",
        "Answers Row",
        "total_s",
        "retrieval_s",
        "answer_s",
        "non_answer_s",
        "Relevance",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_queries = len(set(r["Query ID"] for r in rows))
    print(f"\nExported {total_queries} queries with {len(rows)} timed chunk rows to: {output_csv}")


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
        print("EVALUATION: EXPORT RETRIEVAL CHUNKS WITH ANSWERS FOR MANUAL REVIEW")
    else:
        print("EVALUATION: ANSWERS ONLY")
    print("=" * 80)
    print(f"Total queries: {len(queries)}")
    print(f"Top-K retrievals per method: {top_k}")
    print(f"Generate answers: {generate_answer}")
    if export_chunks:
        print(f"Output: Retrieval chunks (top-{top_k} per method) + generated answers for manual review")
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
                'timing_by_method': getattr(result, 'timing_by_method', {}),
                'timing_breakdown_by_method': getattr(result, 'timing_breakdown_by_method', {}),
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
    
    # Save results to JSON (includes timing)
    results_json = output_csv.replace('.csv', '_results.json')
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults with timing saved to: {results_json}")
    
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


def run_retrieval_evaluation_by_db(
    db_name: str,
    queries_file: str,
    output_csv: str,
    top_k: int = 8,
    generate_answer: bool = True,
    export_chunks: bool = True,
):
    """
    Run retrieval evaluation restricted to one DB domain.

    db_name: company | aiid | standards
    """
    with open(queries_file, "r", encoding="utf-8") as f:
        queries_data = json.load(f)
    queries = queries_data.get("queries", [])

    print("=" * 80)
    print(f"EVALUATION BY DB: {db_name.upper()}")
    print("=" * 80)
    print(f"Total queries: {len(queries)}")
    print(f"Top-K retrievals per method: {top_k}")
    print(f"Generate answers: {generate_answer}")
    print()

    evaluator = IREvaluator(str(project_root))
    all_results: List[Dict[str, Any]] = []

    for i, q_data in enumerate(queries, 1):
        query_id = q_data["id"]
        query = q_data["query"]
        query_type = q_data.get("type", "unknown")
        safe_print(f"[{i}/{len(queries)}] {query_id} ({query_type}): {query[:60]}...")

        # Vector restricted to one FAISS db
        vec_total_start = time.perf_counter()
        vec_ret_start = time.perf_counter()
        vector_chunks = evaluator.vector_engine.search(
            query=query, db_names=[db_name], top_k=top_k, similarity_threshold=0.0
        )
        vec_ret_s = time.perf_counter() - vec_ret_start
        vector_chunk_objs = [evaluator._result_to_chunk(r, "vector") for r in vector_chunks]
        vec_ans_s = 0.0
        vec_ans = None
        if generate_answer and vector_chunk_objs:
            ans_start = time.perf_counter()
            vec_ans, _ = evaluator.generate_answer_with_attribution(
                query, vector_chunk_objs, top_n=min(8, len(vector_chunk_objs))
            )
            vec_ans_s = time.perf_counter() - ans_start
        vec_total_s = time.perf_counter() - vec_total_start

        # Graph restricted by entity types for selected db
        graph_total_start = time.perf_counter()
        graph_ret_start = time.perf_counter()
        graph_raw = evaluator.graph_engine.search(query=query, top_k=None, score_results=True)
        graph_ret_s = time.perf_counter() - graph_ret_start
        graph_filtered = _filter_graph_chunks_for_db(db_name, graph_raw)[:top_k]
        graph_chunk_objs = [evaluator._result_to_chunk(r, "graph") for r in graph_filtered]
        graph_ans_s = 0.0
        graph_ans = None
        if generate_answer and graph_chunk_objs:
            ans_start = time.perf_counter()
            graph_ans, _ = evaluator.generate_answer_with_attribution(
                query, graph_chunk_objs, top_n=min(8, len(graph_chunk_objs))
            )
            graph_ans_s = time.perf_counter() - ans_start
        graph_total_s = time.perf_counter() - graph_total_start

        # DB-restricted hybrid approximation (interleave restricted vector + graph)
        hybrid_total_start = time.perf_counter()
        hybrid_ret_start = time.perf_counter()
        merged: List[Dict[str, Any]] = []
        for a, b in zip(vector_chunks, graph_filtered):
            merged.append({**a, "source": "faiss_vector"})
            merged.append({**b, "source": "graph_traversal"})
        if len(vector_chunks) > len(graph_filtered):
            merged.extend([{**r, "source": "faiss_vector"} for r in vector_chunks[len(graph_filtered):]])
        elif len(graph_filtered) > len(vector_chunks):
            merged.extend([{**r, "source": "graph_traversal"} for r in graph_filtered[len(vector_chunks):]])
        merged = merged[:top_k]
        hybrid_ret_s = time.perf_counter() - hybrid_ret_start
        hybrid_chunk_objs = [evaluator._result_to_chunk(r, "hybrid") for r in merged]
        hybrid_ans_s = 0.0
        hybrid_ans = None
        if generate_answer and hybrid_chunk_objs:
            ans_start = time.perf_counter()
            hybrid_ans, _ = evaluator.generate_answer_with_attribution(
                query, hybrid_chunk_objs, top_n=min(8, len(hybrid_chunk_objs))
            )
            hybrid_ans_s = time.perf_counter() - ans_start
        hybrid_total_s = time.perf_counter() - hybrid_total_start

        result_dict = {
            "query_id": query_id,
            "query": query,
            "query_type": query_type,
            "answers_by_method": {"vector": vec_ans, "graph": graph_ans, "hybrid": hybrid_ans},
            "timing_breakdown_by_method": {
                "vector": {
                    "total_s": float(vec_total_s),
                    "retrieval_s": float(vec_ret_s),
                    "answer_s": float(vec_ans_s),
                    "non_answer_s": float(vec_total_s - vec_ans_s),
                },
                "graph": {
                    "total_s": float(graph_total_s),
                    "retrieval_s": float(graph_ret_s),
                    "answer_s": float(graph_ans_s),
                    "non_answer_s": float(graph_total_s - graph_ans_s),
                },
                "hybrid": {
                    "total_s": float(hybrid_total_s),
                    "retrieval_s": float(hybrid_ret_s),
                    "answer_s": float(hybrid_ans_s),
                    "non_answer_s": float(hybrid_total_s - hybrid_ans_s),
                },
            },
            "chunks_by_method": {
                "vector": [_chunk_to_row(c) for c in vector_chunk_objs],
                "graph": [_chunk_to_row(c) for c in graph_chunk_objs],
                "hybrid": [_chunk_to_row(c) for c in hybrid_chunk_objs],
            },
        }
        all_results.append(result_dict)

        safe_print(
            f"  Retrievals - Vector: {len(vector_chunk_objs)} | "
            f"Graph: {len(graph_chunk_objs)} | Hybrid: {len(hybrid_chunk_objs)}"
        )

    # Save JSON
    results_json = output_csv.replace(".csv", "_results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults with timing saved to: {results_json}")

    if export_chunks:
        export_chunks_to_csv_with_timing(all_results, output_csv)
    else:
        export_chunks_to_csv(all_results, output_csv)

    try:
        evaluator.graph_engine.close()
    except Exception:
        pass


def calculate_metrics(
    labels_file: str,
    output_file: Optional[str] = None,
    k: int = 10,
    per_method: bool = True
):
    """Calculate IR metrics from labeled CSV file using get_metrics."""
    all_method_metrics = calculate_accuracy_metrics(labels_file, k=k)
    if not all_method_metrics:
        return  # get_metrics already printed error/warning
    display_metrics(all_method_metrics, {}, k=k)
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


def add_answers_to_existing_csv(
    input_csv: str,
    output_csv: str,
    queries_file: str,
    top_k: int = 8
):
    """Add generated answers to an existing CSV file without losing relevance labels.
    
    This function reads an existing CSV with chunks and relevance labels,
    generates answers for queries that don't have them, and saves an updated CSV.
    """
    from collections import defaultdict
    
    # Load existing CSV
    rows = []
    queries_seen = set()
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        
        for row in reader:
            rows.append(row)
            query_id = row.get('Query ID', '')
            if query_id:
                queries_seen.add(query_id)
    
    print(f"Loaded {len(rows)} rows for {len(queries_seen)} queries")
    
    # Load query definitions
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    
    queries_dict = {}
    for q_data in queries_data.get('queries', []):
        queries_dict[q_data['id']] = {
            'query': q_data['query'],
            'type': q_data.get('type', 'unknown')
        }
    
    # Check which queries already have answers
    queries_with_answers = set()
    for row in rows:
        query_id = row.get('Query ID', '')
        vector_ans = row.get('Vector Answer', '').strip()
        graph_ans = row.get('Graph Answer', '').strip()
        hybrid_ans = row.get('Hybrid Answer', '').strip()
        
        if vector_ans or graph_ans or hybrid_ans:
            queries_with_answers.add(query_id)
    
    # Find queries that need answers
    queries_to_process = {}
    for query_id, query_info in queries_dict.items():
        if query_id not in queries_with_answers:
            queries_to_process[query_id] = query_info
    
    if not queries_to_process:
        print("\nAll queries already have answers. No action needed.")
        # Still copy to output file
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV copied to: {output_csv}")
        return
    
    print(f"\nFound {len(queries_to_process)} queries without answers")
    
    # Generate answers
    evaluator = IREvaluator(str(project_root))
    answers_by_query = {}
    
    print(f"Generating answers for {len(queries_to_process)} queries...")
    
    for i, (query_id, query_info) in enumerate(queries_to_process.items(), 1):
        query = query_info['query']
        safe_print(f"  [{i}/{len(queries_to_process)}] {query_id}: {query[:60]}...", end=' ', flush=True)
        
        try:
            result = evaluator.evaluate_query(
                query_id=query_id,
                query=query,
                top_k=top_k,
                generate_answer=True
            )
            
            answers_by_method = getattr(result, 'answers_by_method', {}) or {}
            answers_by_query[query_id] = {
                'vector': answers_by_method.get('vector', '') or '',
                'graph': answers_by_method.get('graph', '') or '',
                'hybrid': answers_by_method.get('hybrid', '') or ''
            }
            safe_print("[OK]")
        except Exception as e:
            safe_print(f"[ERROR: {e}]")
            answers_by_query[query_id] = {
                'vector': '',
                'graph': '',
                'hybrid': ''
            }
    
    # Close connection
    try:
        evaluator.graph_engine.close()
    except:
        pass
    
    # Update CSV with answers
    # Ensure answer columns exist in fieldnames
    answer_columns = ['Vector Answer', 'Graph Answer', 'Hybrid Answer']
    for col in answer_columns:
        if col not in fieldnames:
            # Insert before 'Relevance' column if it exists, otherwise append
            if 'Relevance' in fieldnames:
                idx = fieldnames.index('Relevance')
                fieldnames.insert(idx, col)
            else:
                fieldnames.append(col)
    
    # Update rows with answers
    for row in rows:
        query_id = row.get('Query ID', '')
        if query_id in answers_by_query:
            answers = answers_by_query[query_id]
            row['Vector Answer'] = answers['vector']
            row['Graph Answer'] = answers['graph']
            row['Hybrid Answer'] = answers['hybrid']
    
    # Write updated CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nUpdated CSV saved to: {output_csv}")


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
                            default='',
                            help='Output CSV file (default depends on --db)')
    eval_parser.add_argument('--top-k', type=int, default=8,
                            help='Top-K retrievals per method (default: 8)')
    eval_parser.add_argument('--db', type=str, default='all',
                            choices=['all', 'company', 'aiid', 'standards'],
                            help='Evaluation scope: all methods (default) or DB-restricted mode')
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
    
    # Add answers to existing CSV
    add_answers_parser = subparsers.add_parser('add-answers', help='Add answers to existing CSV file')
    add_answers_parser.add_argument('--input', type=str, required=True,
                                   help='Input CSV file (with chunks, may have relevance labels)')
    add_answers_parser.add_argument('--output', type=str, required=True,
                                   help='Output CSV file (with answers added)')
    add_answers_parser.add_argument('--queries', type=str,
                                   default='backend/evaluation/evaluation_queries_50.json',
                                   help='Queries JSON file')
    add_answers_parser.add_argument('--top-k', type=int, default=8,
                                   help='Top-K for retrieval (default: 8)')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        output_csv = args.output.strip() if args.output else ''
        if not output_csv:
            if args.db == 'all':
                output_csv = 'backend/evaluation/data/labeling_csv/chunks_for_labeling_combined.csv'
            else:
                output_csv = f'backend/evaluation/data/labeling_csv/chunks_for_labeling_{args.db}.csv'

        if args.db == 'all':
            run_retrieval_evaluation(
                queries_file=args.queries,
                output_csv=output_csv,
                top_k=args.top_k,
                generate_answer=not args.no_answer,
                export_chunks=args.export_chunks
            )
        else:
            run_retrieval_evaluation_by_db(
                db_name=args.db,
                queries_file=args.queries,
                output_csv=output_csv,
                top_k=args.top_k,
                generate_answer=not args.no_answer,
                export_chunks=True,
            )
    elif args.command == 'metrics':
        calculate_metrics(
            labels_file=args.labels,
            output_file=args.output,
            k=args.k,
            per_method=args.per_method
        )
    elif args.command == 'add-answers':
        print("=" * 80)
        print("ADD ANSWERS TO EXISTING CSV")
        print("=" * 80)
        print(f"Input CSV: {args.input}")
        print(f"Output CSV: {args.output}")
        print()
        
        input_file = Path(args.input)
        queries_file = project_root / args.queries
        
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}")
            sys.exit(1)
        
        if not queries_file.exists():
            print(f"Error: Queries file not found: {queries_file}")
            sys.exit(1)
        
        add_answers_to_existing_csv(
            input_csv=str(input_file),
            output_csv=args.output,
            queries_file=str(queries_file),
            top_k=args.top_k
        )
        
        print("\n" + "=" * 80)
        print("COMPLETE")
        print("=" * 80)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
