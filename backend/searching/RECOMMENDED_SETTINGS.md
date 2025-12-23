# Recommended Settings for Vector Search

## Optimal Configuration

These are the **recommended defaults** for best balance of speed, quality, and cost:

```python
{
    'top_k': 10,                    # ✅ Optimal balance
    'similarity_threshold': 0.0,    # ✅ Include all, rank by similarity
    'rerank': True,                 # ✅ Improves relevance
    'generate_answer': True,        # ✅ Provides comprehensive answers
    'db_names': None                # ✅ Search all databases
}
```

## Parameter Details

### `top_k = 10` ✅ Recommended
- **Why**: Good balance between coverage and speed
- **Too low (5)**: May miss relevant results
- **Too high (50+)**: Slower, more noise, diminishing returns
- **For answer generation**: 8-10 is optimal (LLM context window)

### `similarity_threshold = 0.0` ✅ Recommended
- **Why**: Include all results, let ranking handle relevance
- **Range**: -1.0 to 1.0 (cosine similarity)
- **Higher (0.7+)**: High precision, low recall
- **Lower (-0.5)**: More inclusive
- **Default (0.0)**: Works best for most queries

### `rerank = True` ✅ Recommended
- **Why**: Improves relevance beyond similarity scores
- **Cost**: ~$0.01 per query, +2-3 seconds
- **Benefit**: Better final order for answer generation
- **When to disable**: Speed > quality, budget constraints

### `generate_answer = True` ✅ Recommended
- **Why**: Provides comprehensive, synthesized answers
- **Cost**: ~$0.03 per query, +3-5 seconds
- **Benefit**: Much better user experience
- **When to disable**: Raw results only needed

### `db_names = None` ✅ Recommended
- **Why**: Most comprehensive results
- **Alternative**: Specify databases if you know what you need
- **Example**: `['company', 'standards']` to skip AIID

## Usage Examples

### Basic (Recommended Settings)
```python
result = vector_engine.query(
    query="What are the privacy policies?",
    # All defaults are optimal ✅
)
```

### Explicit Recommended Settings
```python
result = vector_engine.query(
    query="What are the privacy policies?",
    db_names=None,              # ✅ Search all
    top_k=10,                   # ✅ Optimal
    rerank=True,                # ✅ Improve quality
    generate_answer=True,       # ✅ Generate answers
    similarity_threshold=0.0    # ✅ Include all
)
```

### Fast Mode (Speed Priority)
```python
result = vector_engine.query(
    query="What are the privacy policies?",
    top_k=5,                    # Fewer results
    rerank=False,               # Skip reranking
    generate_answer=True,       # Still generate answer
    similarity_threshold=0.0
)
```

### High Quality Mode (Quality Priority)
```python
result = vector_engine.query(
    query="What are the privacy policies?",
    top_k=15,                   # More results
    rerank=True,                # Enable reranking
    generate_answer=True,       # Generate answer
    similarity_threshold=0.0
)
```

## When to Adjust

### Increase `top_k` (15-20) when:
- Complex queries needing more context
- Answer generation needs more sources
- Want broader coverage

### Decrease `top_k` (5-8) when:
- Simple, focused queries
- Speed is critical
- Want only most relevant

### Adjust `similarity_threshold` when:
- Too many irrelevant results → increase to 0.5-0.7
- Too few results → decrease to -0.5 or -1.0
- Default (0.0) works for most cases ✅

### Disable `rerank` when:
- Speed > quality
- Similarity scores already good
- Budget constraints

## Performance Estimates

| Configuration | Speed | Quality | Cost/Query |
|--------------|-------|---------|------------|
| **Recommended** ✅ | Medium | High | ~$0.04 |
| Fast Mode | Fast | Medium | ~$0.03 |
| High Quality | Slow | Highest | ~$0.05 |

## Summary

**Use the recommended defaults** - they provide the best balance:
- ✅ `top_k=10`
- ✅ `similarity_threshold=0.0`
- ✅ `rerank=True`
- ✅ `generate_answer=True`
- ✅ `db_names=None`

These are already set as defaults throughout the codebase.

