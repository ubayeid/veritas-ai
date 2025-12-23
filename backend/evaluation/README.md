# Search Evaluation Framework

Compare Vector, Graph, and Hybrid search methods with timing and accuracy metrics.

## Quick Start

### Run Complete Evaluation

```bash
python backend/evaluation/evaluate_search.py
```

This tests all 3 methods (Vector, Graph, Hybrid) with answer generation (mandatory), showing execution time and accuracy for each.

## What It Does

For each query, it runs **3 search methods** (all with answer generation):
1. Vector - WITH answer generation
2. Graph - WITH answer generation
3. Hybrid - WITH answer generation

**Note:** Answer generation is mandatory for all search methods.

## Prerequisites

Before running:
- ✅ Neo4j is running
- ✅ `.env` file has `OPENAI_API_KEY`
- ✅ FAISS indexes are built
- ✅ Neo4j connection details in `.env`

## Command Options

```bash
# Run evaluation (answer generation is mandatory)
python backend/evaluation/evaluate_search.py [--queries-file FILE] [--top-k K] [--output FILE] [--no-accuracy] [--quiet]
```

**Example:**
```bash
# Full evaluation with all queries
python backend/evaluation/evaluate_search.py --queries-file backend/evaluation/test_queries.json --top-k 15 --output my_results.json

# Simplified output (quiet mode)
python backend/evaluation/evaluate_search.py --queries-file backend/evaluation/test_queries.json --quiet
```

**Options:**
- `--queries-file FILE` - Use custom queries file (default: uses built-in queries)
- `--top-k K` - Number of results to retrieve (default: 10)
- `--output FILE` - Custom output filename (default: evaluation_results.json)

## Output

The report shows:

### Execution Time & Accuracy Comparison
```
Method       Execution Time (ms)        Answer Gen Time    Accuracy Score
----------------------------------------------------------------------------------------------------
VECTOR       2500.23ms                  1200ms             0.850
GRAPH        2800.45ms                  1500ms             0.820
HYBRID       3200.67ms                  1800ms             0.880
```

Plus detailed breakdowns, summary tables, and performance comparisons.

## Metrics Explained

### Execution Time
- **Total Time**: Search + LLM answer generation (answer generation is mandatory)
- **Answer Generation Time**: Time spent generating the answer
- **Search-Only Time**: Time for retrieval only (calculated as total - answer generation)

### Accuracy Score (0.0-1.0)
- Evaluates generated answer quality (relevance, accuracy, completeness, clarity)
- Uses **Judge LLM** model (configurable via `JUDGE_LLM_MODEL` in `.env`)

## Configuration

### Environment Variables (.env)

```bash
# Required
OPENAI_API_KEY=sk-your-key-here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Optional (with defaults)
LLM_MODEL=gpt-4                          # For answer generation (and evaluation if JUDGE_LLM_MODEL not set)
JUDGE_LLM_MODEL=gpt-4o                   # For accuracy evaluation (optional - defaults to LLM_MODEL if not set)
```

## Time Estimates

- **With accuracy**: ~2-5 min per query
- **Without accuracy**: ~10-30 sec per query
- **Full comparison (10 queries)**: ~20-50 min with accuracy, ~2-5 min without

## Troubleshooting

**Neo4j Connection Error**
- Make sure Neo4j is running
- Check `.env` has correct connection details

**FAISS Index Not Found**
- Build indexes first (see `backend/building_database/faiss/`)

**OpenAI API Error**
- Add `OPENAI_API_KEY` to `.env`

**Slow Performance**
- Use `--no-accuracy` flag
- Reduce `--top-k` value
- Use fewer test queries

## Output Files

Results are saved to `evaluation_results.json` (or custom filename) with:
- All query results
- Summary statistics
- Timing and accuracy metrics
- Comparison data

## Advanced Usage

For more details on metrics, ground truth evaluation, and advanced features, see the code documentation in `evaluate_search.py`.
