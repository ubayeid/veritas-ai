# Search Evaluation Framework

Comprehensive evaluation framework for comparing Vector Search vs Graph Search vs Hybrid Search.

## Metrics Evaluated

### 1. **Performance Metrics (Timing)**
- **Average Execution Time**: Mean time per query (includes contextualization)
- **Median Execution Time**: Median time per query
- **Min/Max Execution Time**: Range of execution times
- **Standard Deviation**: Variability in execution times
- **Contextualization Time**: Time spent generating LLM answers
- **Speed Comparison**: Relative performance between methods

### 2. **Quality Metrics**
- **Number of Results**: How many results each method returns
- **Average Similarity**: Mean similarity score (for vector search)
- **Contextualized Answer**: LLM-generated answer (mandatory for all methods)
- **Answer Length**: Length of generated contextualized answer
- **Precision@K**: Fraction of top-K results that are relevant (requires ground truth)
- **Recall@K**: Fraction of relevant items retrieved (requires ground truth)
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant result
- **NDCG@K**: Normalized Discounted Cumulative Gain (requires relevance scores)

### 3. **Result Characteristics**
- **Source Distribution**: Which databases/sources contribute results
- **Result Diversity**: How diverse are the results
- **Coverage**: How well each method covers different query types

### 4. **Contextualization (Mandatory)**
- All search methods (vector, graph, hybrid) now include mandatory contextualization
- Uses LLM to generate comprehensive answers from search results
- Contextualization time is tracked separately

## Usage

### Basic Evaluation

```bash
# Run evaluation with default test queries
python backend/evaluation/evaluate_search.py

# Specify custom queries file
python backend/evaluation/evaluate_search.py --queries-file backend/evaluation/test_queries.json

# Set top-K value
python backend/evaluation/evaluate_search.py --top-k 20

# Save results to custom file
python backend/evaluation/evaluate_search.py --output my_results.json
``

### Performance Comparison

**Speed:**
- Vector search: Typically fastest (FAISS is optimized)
- Graph search: Depends on query complexity and graph size
- Hybrid: Sum of both + RRF overhead

**Expected Results (with contextualization):**
- Vector: 2000-5000ms (includes LLM contextualization)
- Graph: 2000-5500ms (includes LLM contextualization)
- Hybrid: 2000-6000ms (vector + graph + RRF + LLM contextualization)

**Note:** Contextualization adds significant time (typically 1500-4000ms) but provides comprehensive answers.

### Quality Comparison

**Result Count:**
- Vector: Returns top-K by similarity
- Graph: Returns all matching relationships
- Hybrid: Combines both (may have duplicates)

**Relevance:**
- Vector: Good for semantic similarity
- Graph: Good for explicit relationships
- Hybrid: Best overall coverage

## Advanced: Ground Truth Evaluation

To evaluate accuracy metrics (precision, recall, MRR, NDCG), you need ground truth data:

```python
# Ground truth: mapping query to relevant result IDs
ground_truth = {
    "What are the privacy policies?": [
        "clause_123",
        "clause_456",
        "clause_789"
    ],
    "Find GDPR articles about data minimization": [
        "Art5",
        "Art25"
    ]
}

# Run evaluation with ground truth
results = evaluator.run_evaluation_suite(
    test_queries=list(ground_truth.keys()),
    top_k=10,
    ground_truth=ground_truth
)
```

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "results": {
    "vector": [
      {
        "query": "...",
        "method": "vector",
        "execution_time_ms": 45.23,
        "num_results": 10,
        "top_k": 10,
        "avg_similarity": 0.85,
        "sources": {"company": 5, "standards": 5}
      }
    ],
    "graph": [...],
    "hybrid": [...]
  },
  "summary": {
    "vector": {
      "avg_execution_time_ms": 45.23,
      "median_execution_time_ms": 42.10,
      "avg_num_results": 9.5,
      "avg_similarity": 0.85
    },
    "graph": {...},
    "hybrid": {...}
  },
  "test_queries": [...],
  "top_k": 10
}
```

## Tips for Evaluation

1. **Warm-up Runs**: Run a few queries first to warm up caches
2. **Multiple Runs**: Run each query multiple times and average for stable metrics
3. **Query Diversity**: Include different query types (semantic, graph, hybrid)
4. **Top-K Variation**: Test with different top-K values (5, 10, 20)
5. **Database Selection**: Test with specific databases vs all databases

## Troubleshooting

**Neo4j Connection Error:**
- Ensure Neo4j is running
- Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env

**No Results:**
- Check if FAISS indexes are built
- Verify graph data is loaded in Neo4j
- Check query type detection logic

**Slow Performance:**
- Check network latency to Neo4j
- Verify FAISS indexes are loaded
- Consider reducing top_k for testing

