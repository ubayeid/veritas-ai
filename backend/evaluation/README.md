# Search Evaluation Framework

Compare Vector, Graph, and Hybrid search methods with timing and accuracy metrics.

## Quick Start

### Run Complete Evaluation

```bash
python backend/evaluation/evaluate_search.py --compare-contextualization
```

This tests all 3 methods (Vector, Graph, Hybrid) with and without contextualization, showing execution time and accuracy for each.

## What It Does

For each query, it runs **6 combinations**:
1. Vector - WITH contextualization
2. Vector - WITHOUT contextualization  
3. Graph - WITH contextualization
4. Graph - WITHOUT contextualization
5. Hybrid - WITH contextualization
6. Hybrid - WITHOUT contextualization

## Prerequisites

Before running:
- ✅ Neo4j is running
- ✅ `.env` file has `OPENAI_API_KEY`
- ✅ FAISS indexes are built
- ✅ Neo4j connection details in `.env`

## Command Options

```bash
python backend/evaluation/evaluate_search.py --compare-contextualization [--queries-file FILE] [--top-k K] [--output FILE] [--no-accuracy]
```

**Example:**
```bash
python backend/evaluation/evaluate_search.py --compare-contextualization --queries-file backend/evaluation/test_queries.json --top-k 10 --output my_results.json
```

**Options:**
- `--queries-file FILE` - Use custom queries file (default: uses built-in queries)
- `--top-k K` - Number of results to retrieve (default: 10)
- `--output FILE` - Custom output filename (default: evaluation_results.json)
- `--no-accuracy` - Skip accuracy evaluation for faster runs

## Output

The report shows:

### Execution Time & Accuracy Comparison
```
Method       Mode                        Execution Time (ms)        Accuracy Score
----------------------------------------------------------------------------------------------------
VECTOR       WITH Contextualization     2500.23ms (ctx: 1200ms)   0.850
             WITHOUT Contextualization   800.15ms                  0.720
             Overhead                    1700.08ms (+212.5%)       

GRAPH        WITH Contextualization     2800.45ms (ctx: 1500ms)   0.820
             WITHOUT Contextualization   900.30ms                  0.750
             Overhead                    1900.15ms (+211.1%)       

HYBRID       WITH Contextualization     3200.67ms (ctx: 1800ms)   0.880
             WITHOUT Contextualization  1200.50ms                  0.780
             Overhead                    2000.17ms (+166.7%)       
```

Plus detailed breakdowns, summary tables, and performance comparisons.

## Metrics Explained

### Execution Time
- **WITH Contextualization**: Total time (search + LLM answer generation)
- **WITHOUT Contextualization**: Search-only time
- **Overhead**: Additional time cost of contextualization

### Accuracy Score (0.0-1.0)
- **WITH Contextualization**: Evaluates generated answer quality (relevance, accuracy, completeness, clarity)
- **WITHOUT Contextualization**: Evaluates search results quality/relevance
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
LLM_MODEL=gpt-4                          # For contextualization (and evaluation if JUDGE_LLM_MODEL not set)
JUDGE_LLM_MODEL=gpt-4-turbo              # For accuracy evaluation (optional - defaults to LLM_MODEL if not set)
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
