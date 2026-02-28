"""
IR Evaluation System with Chunk Pooling and Relevance Labeling
- Runs queries through all 3 search methods
- Collects and pools chunks from all methods
- Generates answers with chunk attribution
- Outputs format for manual relevance labeling
- Calculates IR metrics (Precision@K, Recall@K, NDCG, MAP, MRR)
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import math

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.retrieval.engines.query_engine import VectorQueryEngine
from backend.retrieval.engines.graph_query_engine import GraphQueryEngine
from backend.retrieval.engines.hybrid_query_engine import HybridQueryEngine


@dataclass
class ChunkResult:
    """Represents a chunk result from search."""
    chunk_id: str
    text: str
    source: str  # 'vector', 'graph', 'hybrid'
    method: str  # Which method retrieved it
    similarity: Optional[float] = None
    database: Optional[str] = None
    source_name: Optional[str] = None
    source_file: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class QueryResult:
    """Results for a single query."""
    query_id: str
    query: str
    answer: Optional[str] = None  # Deprecated: kept for backward compatibility
    answers_by_method: Dict[str, Optional[str]] = field(default_factory=dict)  # method -> answer
    chunks_by_method: Dict[str, List[ChunkResult]] = field(default_factory=dict)  # method -> chunks
    pooled_chunks: List[ChunkResult] = field(default_factory=list)  # Deduplicated chunks
    relevance_labels: Dict[str, int] = field(default_factory=dict)  # chunk_id -> relevance (0, 1, 2)


class IREvaluator:
    """IR Evaluation with chunk pooling and relevance labeling."""
    
    def __init__(self, base_dir: str):
        """Initialize evaluator."""
        self.base_dir = Path(base_dir)
        self.vector_engine = VectorQueryEngine(str(base_dir))
        self.graph_engine = GraphQueryEngine(str(base_dir))
        self.hybrid_engine = HybridQueryEngine(str(base_dir))
    
    def _get_chunk_id(self, result: Dict[str, Any]) -> str:
        """Generate unique chunk ID from result."""
        # Try explicit IDs first
        for field in ['id', 'chunk_id', 'incident_id', 'clause_id', 'article_id']:
            if field in result and result[field]:
                return str(result[field])
        
        # Construct ID from source and text
        source_name = result.get('source_name', result.get('document_name', 'unknown'))
        chunk_id = result.get('chunk_id', '')
        text = result.get('text', result.get('description', ''))[:200]
        
        if source_name and chunk_id:
            return f"{source_name}_{chunk_id}"
        
        # Fallback: hash of text
        if text:
            return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
        
        return str(hash(str(result)))
    
    def _result_to_chunk(self, result: Dict[str, Any], method: str) -> ChunkResult:
        """Convert search result to ChunkResult."""
        chunk_id = self._get_chunk_id(result)
        text = result.get('text', result.get('description', ''))
        
        return ChunkResult(
            chunk_id=chunk_id,
            text=text,
            source=result.get('source', method),
            method=method,
            similarity=result.get('similarity'),
            database=result.get('database'),
            source_name=result.get('source_name', result.get('document_name')),
            source_file=result.get('source_file'),
            metadata={k: v for k, v in result.items() if k not in ['text', 'description', 'similarity', 'database', 'source_name', 'source_file', 'source']}
        )
    
    def run_vector_search(self, query: str, top_k: int = 20) -> List[ChunkResult]:
        """Run vector search and return chunks."""
        results = self.vector_engine.search(
            query=query,
            db_names=None,  # Search all databases
            top_k=top_k,
            similarity_threshold=0.0
        )
        
        # Skip reranking to avoid API calls during evaluation
        # Results from FAISS are already sorted by similarity score
        # Reranking would require additional API calls that may hit rate limits
        
        return [self._result_to_chunk(r, 'vector') for r in results]
    
    def run_graph_search(self, query: str, top_k: int = 20) -> List[ChunkResult]:
        """Run graph search and return chunks."""
        results = self.graph_engine.search(
            query=query,
            top_k=None,  # Get all results
            score_results=True
        )
        
        # Limit to top_k
        results = results[:top_k]
        
        return [self._result_to_chunk(r, 'graph') for r in results]
    
    def run_hybrid_search(self, query: str, top_k: int = 20) -> List[ChunkResult]:
        """Run hybrid search and return chunks."""
        search_result = self.hybrid_engine.hybrid_search(
            query=query,
            top_k=top_k,
            use_faiss=True,
            use_graph_traversal=True
        )
        
        results = search_result.get('results', [])
        
        return [self._result_to_chunk(r, 'hybrid') for r in results]
    
    def pool_chunks(self, chunks_by_method: Dict[str, List[ChunkResult]]) -> List[ChunkResult]:
        """
        Pool and deduplicate chunks from all methods.
        
        Args:
            chunks_by_method: Dict mapping method name to list of chunks
            
        Returns:
            Deduplicated list of chunks with source tracking
        """
        chunk_map = {}  # chunk_id -> ChunkResult
        
        # Process chunks from each method
        for method, chunks in chunks_by_method.items():
            for chunk in chunks:
                if chunk.chunk_id not in chunk_map:
                    # New chunk
                    chunk_map[chunk.chunk_id] = chunk
                else:
                    # Existing chunk - update source to include this method
                    existing = chunk_map[chunk.chunk_id]
                    # Track which methods found this chunk
                    if existing.source != 'hybrid':
                        if existing.method != method:
                            existing.source = 'hybrid'  # Found by multiple methods
                    # Keep highest similarity score
                    if chunk.similarity is not None:
                        if existing.similarity is None or chunk.similarity > existing.similarity:
                            existing.similarity = chunk.similarity
        
        # Sort by similarity (descending)
        pooled = sorted(chunk_map.values(), key=lambda c: c.similarity or 0.0, reverse=True)
        
        return pooled
    
    def generate_answer_with_attribution(
        self,
        query: str,
        chunks: List[ChunkResult],
        top_n: int = 8
    ) -> Tuple[Optional[str], List[ChunkResult]]:
        """
        Generate answer and return chunks used for attribution.
        
        Returns:
            Tuple of (answer, chunks_used)
        """
        if not chunks:
            return None, []
        
        # Convert ChunkResult back to dict format for answer generation
        results_for_llm = []
        for chunk in chunks[:top_n]:
            results_for_llm.append({
                'text': chunk.text,
                'similarity': chunk.similarity,
                'source_name': chunk.source_name,
                'database': chunk.database,
                'chunk_id': chunk.chunk_id
            })
        
        try:
            answer = self.vector_engine.generate_answer(query, results_for_llm)
            if not answer or answer.strip() == "":
                print(f"Warning: Answer generation returned empty string")
                return None, []
            chunks_used = chunks[:top_n]
            return answer, chunks_used
        except Exception as e:
            import traceback
            print(f"Warning: Answer generation failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None, []
    
    def evaluate_query(
        self,
        query_id: str,
        query: str,
        top_k: int = 20,
        generate_answer: bool = True
    ) -> QueryResult:
        """
        Evaluate a single query through all methods.
        
        Returns:
            QueryResult with chunks from all methods and pooled chunks
        """
        # Run all three methods
        vector_chunks = self.run_vector_search(query, top_k=top_k)
        graph_chunks = self.run_graph_search(query, top_k=top_k)
        hybrid_chunks = self.run_hybrid_search(query, top_k=top_k)
        
        chunks_by_method = {
            'vector': vector_chunks,
            'graph': graph_chunks,
            'hybrid': hybrid_chunks
        }
        
        # Pool chunks (deduplicate)
        pooled_chunks = self.pool_chunks(chunks_by_method)
        
        # Generate answers for each method
        answers_by_method = {}
        answer = None  # Keep for backward compatibility (use first answer)
        if generate_answer:
            for method in ['vector', 'graph', 'hybrid']:
                method_chunks = chunks_by_method.get(method, [])
                if not method_chunks:
                    print(f"  Warning: No chunks for {method} method, skipping answer generation")
                    answers_by_method[method] = None
                    continue
                try:
                    method_answer, _ = self.generate_answer_with_attribution(query, method_chunks)
                    if method_answer:
                        answers_by_method[method] = method_answer
                        # Set first answer as default for backward compatibility
                        if answer is None:
                            answer = method_answer
                    else:
                        print(f"  Warning: Answer generation returned None for {method}")
                        answers_by_method[method] = None
                except Exception as e:
                    import traceback
                    print(f"  Warning: Answer generation failed for {query_id} ({method}): {e}")
                    print(f"  Traceback: {traceback.format_exc()}")
                    answers_by_method[method] = None
        
        return QueryResult(
            query_id=query_id,
            query=query,
            answer=answer,  # Backward compatibility
            answers_by_method=answers_by_method,
            chunks_by_method=chunks_by_method,
            pooled_chunks=pooled_chunks,
            relevance_labels=None  # Will be filled after manual labeling
        )
    
    def calculate_ir_metrics(
        self,
        query_result: QueryResult,
        k: int = 10,
        method: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate IR metrics based on relevance labels.
        
        Relevance scale: 2 (highly relevant), 1 (partially relevant), 0 (not relevant)
        
        Args:
            query_result: QueryResult with chunks and relevance labels
            k: Top-K cutoff for metrics
            method: If specified, calculate metrics for this method's chunks only.
                    Options: 'vector', 'graph', 'hybrid'. If None, uses pooled chunks.
        
        Returns:
            Dictionary with metrics
        """
        if not query_result.relevance_labels:
            return {}
        
        # Get chunks to evaluate
        if method and method in query_result.chunks_by_method:
            # Use specific method's chunks
            top_k_chunks = query_result.chunks_by_method[method][:k]
        else:
            # Use pooled chunks (default)
            top_k_chunks = query_result.pooled_chunks[:k]
        
        # Calculate metrics
        metrics = {}
        
        # Precision@K: fraction of top-k that are relevant (score >= 1)
        relevant_count = sum(1 for c in top_k_chunks if query_result.relevance_labels.get(c.chunk_id, 0) >= 1)
        metrics['precision_at_k'] = relevant_count / k if k > 0 else 0.0
        
        # Recall@K: fraction of all relevant chunks retrieved
        # Total relevant is from the pooled ground truth labels
        total_relevant = sum(1 for score in query_result.relevance_labels.values() if score >= 1)
        metrics['recall_at_k'] = relevant_count / total_relevant if total_relevant > 0 else 0.0
        
        # F1@K
        if metrics['precision_at_k'] + metrics['recall_at_k'] > 0:
            metrics['f1_at_k'] = 2 * (metrics['precision_at_k'] * metrics['recall_at_k']) / (metrics['precision_at_k'] + metrics['recall_at_k'])
        else:
            metrics['f1_at_k'] = 0.0
        
        # MRR: Mean Reciprocal Rank (position of first relevant result)
        for rank, chunk in enumerate(top_k_chunks, start=1):
            if query_result.relevance_labels.get(chunk.chunk_id, 0) >= 1:
                metrics['mrr'] = 1.0 / rank
                break
        else:
            metrics['mrr'] = 0.0
        
        # MAP: Mean Average Precision
        relevant_count_so_far = 0
        precision_sum = 0.0
        for rank, chunk in enumerate(top_k_chunks, start=1):
            relevance = query_result.relevance_labels.get(chunk.chunk_id, 0)
            if relevance >= 1:
                relevant_count_so_far += 1
                precision_at_rank = relevant_count_so_far / rank
                precision_sum += precision_at_rank
        
        if total_relevant > 0:
            metrics['map'] = precision_sum / total_relevant
        else:
            metrics['map'] = 0.0
        
        # NDCG@K: Normalized Discounted Cumulative Gain
        dcg = 0.0
        for rank, chunk in enumerate(top_k_chunks, start=1):
            relevance_score = query_result.relevance_labels.get(chunk.chunk_id, 0)
            dcg += relevance_score / math.log2(rank + 1)
        
        # Ideal DCG: sort all relevance scores descending
        ideal_relevances = sorted(query_result.relevance_labels.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, start=1))
        
        metrics['ndcg_at_k'] = dcg / idcg if idcg > 0 else 0.0
        
        return metrics
    
    def save_for_labeling(
        self,
        query_results: List[QueryResult],
        output_file: str
    ):
        """
        Save query results in format suitable for manual relevance labeling.
        
        Format:
        {
            "queries": [
                {
                    "query_id": "Q001",
                    "query": "...",
                    "answer": "...",
                    "chunks": [
                        {
                            "chunk_id": "...",
                            "text": "...",
                            "source": "vector|graph|hybrid",
                            "method": "vector|graph|hybrid",
                            "similarity": 0.85,
                            "relevance": null  // To be filled: 0, 1, or 2
                        }
                    ],
                    "methods": {
                        "vector": [chunk_ids in order],
                        "graph": [chunk_ids in order],
                        "hybrid": [chunk_ids in order]
                    }
                }
            ]
        }
        """
        output = {
            "metadata": {
                "total_queries": len(query_results),
                "relevance_scale": {
                    "2": "highly relevant",
                    "1": "partially relevant",
                    "0": "not relevant"
                }
            },
            "queries": []
        }
        
        for qr in query_results:
            query_data = {
                "query_id": qr.query_id,
                "query": qr.query,
                "answer": qr.answer,
                "chunks": [],
                "methods": {}
            }
            
            # Track chunk data by ID
            chunk_map = {}
            
            # Add pooled chunks (deduplicated) for labeling
            for chunk in qr.pooled_chunks:
                chunk_data = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text[:500],  # Truncate for labeling
                    "source": chunk.source,
                    "method": chunk.method,
                    "similarity": chunk.similarity,
                    "source_name": chunk.source_name,
                    "relevance": None  # To be filled manually
                }
                query_data["chunks"].append(chunk_data)
                chunk_map[chunk.chunk_id] = chunk_data
            
            # Save method-specific chunk order (for per-method evaluation)
            for method in ['vector', 'graph', 'hybrid']:
                if method in qr.chunks_by_method:
                    method_chunk_ids = [c.chunk_id for c in qr.chunks_by_method[method]]
                    query_data["methods"][method] = method_chunk_ids
            
            output["queries"].append(query_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(query_results)} queries for manual labeling: {output_file}")
        print(f"  Total chunks to label: {sum(len(qr.pooled_chunks) for qr in query_results)}")
    
    def load_labels(self, labeled_file: str) -> Dict[str, Dict[str, int]]:
        """
        Load relevance labels from labeled file.
        
        Returns:
            Dict mapping query_id -> chunk_id -> relevance_score
        """
        with open(labeled_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        labels = {}
        for query_data in data.get('queries', []):
            query_id = query_data['query_id']
            labels[query_id] = {}
            
            for chunk in query_data.get('chunks', []):
                chunk_id = chunk['chunk_id']
                relevance = chunk.get('relevance')
                if relevance is not None:
                    labels[query_id][chunk_id] = int(relevance)
        
        return labels
    
    def apply_labels(
        self,
        query_results: List[QueryResult],
        labels: Dict[str, Dict[str, int]]
    ):
        """Apply relevance labels to query results."""
        for qr in query_results:
            if qr.query_id in labels:
                qr.relevance_labels = labels[qr.query_id]
    
    def calculate_aggregate_metrics(
        self,
        query_results: List[QueryResult],
        k: int = 10,
        method: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate aggregate IR metrics across all queries.
        
        Args:
            query_results: List of QueryResult objects
            k: Top-K cutoff for metrics
            method: If specified, calculate metrics for this method only.
                    Options: 'vector', 'graph', 'hybrid'. If None, uses pooled chunks.
        
        Returns:
            Dictionary with aggregate metrics
        """
        all_metrics = []
        
        for qr in query_results:
            if qr.relevance_labels:
                metrics = self.calculate_ir_metrics(qr, k=k, method=method)
                metrics['query_id'] = qr.query_id
                all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        # Aggregate statistics
        aggregate = {}
        for metric_name in ['precision_at_k', 'recall_at_k', 'f1_at_k', 'mrr', 'map', 'ndcg_at_k']:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                aggregate[f'avg_{metric_name}'] = sum(values) / len(values)
                aggregate[f'min_{metric_name}'] = min(values)
                aggregate[f'max_{metric_name}'] = max(values)
        
        aggregate['total_queries'] = len(all_metrics)
        
        return aggregate


def main():
    """Demo: Run evaluation on a small subset of queries."""
    import sys
    from pathlib import Path
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Load queries
    queries_file = Path(__file__).parent / "evaluation_queries.json"
    if not queries_file.exists():
        print(f"Error: {queries_file} not found. Run generate_50_queries.py first.")
        sys.exit(1)
    
    with open(queries_file, 'r') as f:
        queries_data = json.load(f)
    
    # Demo: Use first 5 queries
    demo_queries = queries_data['queries'][:5]
    
    print("=" * 80)
    print("IR EVALUATION DEMO")
    print("=" * 80)
    print(f"Running evaluation on {len(demo_queries)} queries (demo)")
    print()
    
    evaluator = IREvaluator(str(project_root))
    
    query_results = []
    for i, q_data in enumerate(demo_queries, 1):
        query_id = q_data['id']
        query = q_data['query']
        
        print(f"[{i}/{len(demo_queries)}] {query_id}: {query[:60]}...")
        
        try:
            result = evaluator.evaluate_query(
                query_id=query_id,
                query=query,
                top_k=20,
                generate_answer=False  # Skip answer generation for demo (API quota)
            )
            query_results.append(result)
            
            print(f"  [OK] Vector: {len(result.chunks_by_method['vector'])} chunks")
            print(f"  [OK] Graph: {len(result.chunks_by_method['graph'])} chunks")
            print(f"  [OK] Hybrid: {len(result.chunks_by_method['hybrid'])} chunks")
            print(f"  [OK] Pooled: {len(result.pooled_chunks)} unique chunks")
        except Exception as e:
            print(f"  [ERROR] Error: {e}")
    
    # Save for labeling
    output_file = Path(__file__).parent / "demo_labeling.json"
    evaluator.save_for_labeling(query_results, str(output_file))
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Open {output_file}")
    print(f"2. Label each chunk with relevance: 2 (highly), 1 (partial), 0 (not relevant)")
    print(f"3. Save the labeled file")
    print(f"4. Run: python backend/evaluation/calculate_metrics.py --labels demo_labeling.json")
    
    # Cleanup
    evaluator.neo4j_conn.close()


if __name__ == "__main__":
    main()
