# Search Architecture Documentation

## Overview

This document describes the complete search architecture, including vector search, graph search, and hybrid search methods.

---

## Search Methods

### 1. Pure Vector Search

**Method:** `VectorQueryEngine.query()`

**Flow:**
```
Query → Vector Search (FAISS, top_k=30) → LLM Reranking → Answer Generation
```

**Characteristics:**
- Searches across all FAISS databases (company, aiid, standards)
- Returns top_k results sorted by similarity
- Reranks using LLM to top 8
- Generates answer from reranked results

**API Endpoint:** `POST /api/query`

---

### 2. Pure Graph Search

**Method:** `HybridQueryEngine.graph_query()`

**Flow:**
```
Query → Graph Traversal (Neo4j) → Semantic Scoring → LLM Reranking → Answer Generation
```

**Characteristics:**
- Queries Neo4j knowledge graph
- Scores all results by semantic similarity
- Uses all scored results (no limit)
- Reranks using LLM to top 8
- Generates answer from reranked results

**API Endpoint:** `POST /api/graph_query`

---

### 3. Hybrid Search

**Method:** `HybridQueryEngine.hybrid_query()`

**Flow:**
```
Query
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Vector Part                                                 │
│ • Search FAISS (top_k=30)                                   │
│ • NO reranking (passes unreranked to RRF)                  │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Graph Part                                                   │
│ • Traverse Neo4j (all results)                              │
│ • Score all by semantic similarity                           │
│ • Limit to top 150 (GRAPH_MAX_RESULTS_FOR_RRF)              │
│ • NO reranking (passes scored but unreranked to RRF)       │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ RRF Merge                                                    │
│ • Combine vector + graph results                            │
│ • Calculate RRF scores: RRF_score = Σ 1/(k + rank_i)       │
│ • Sort by RRF score                                         │
│ • Apply top_k limit                                         │
└─────────────────────────────────────────────────────────────┘
  ↓
LLM Reranking → Answer Generation
```

**Characteristics:**
- Combines vector and graph results using RRF
- Single reranking after RRF merge (avoids double reranking)
- Optimized for fusion quality

**API Endpoint:** `POST /api/hybrid_query`

---

## Key Differences: Pure vs Hybrid Components

### Vector Component

| Aspect | Pure Vector | Hybrid's Vector |
|--------|------------|----------------|
| **Search** | top_k=30 | top_k=30 (same) |
| **Reranking** | ✅ Yes (before answer) | ❌ No (before RRF) |
| **Results Used** | Reranked (8) | Unreranked (30) |

**Why:** Pure vector reranks for standalone quality. Hybrid's vector doesn't rerank before RRF to avoid redundancy (reranks once after RRF).

### Graph Component

| Aspect | Pure Graph | Hybrid's Graph |
|--------|-----------|----------------|
| **Scoring** | ✅ All results | ✅ All results (same) |
| **Limit** | ❌ No limit | ✅ Limit to 150 |
| **Reranking** | ✅ Yes (before answer) | ❌ No (before RRF) |
| **Results Used** | All scored + reranked | Top 150 scored (unreranked) |

**Why:** 
- Pure graph uses all results for standalone quality
- Hybrid's graph limits to 150 for RRF performance optimization
- Hybrid's graph doesn't rerank before RRF (reranks once after RRF)

---

## Why Similarity Scoring is Essential for RRF

RRF (Reciprocal Rank Fusion) uses the **rank/position** of results:

```
RRF_score = Σ 1/(k + rank_i)
```

Where `rank_i` = position in the list (1st, 2nd, 3rd...)

**Without similarity scoring (unsorted):**
- Rank is arbitrary → RRF fusion is meaningless

**With similarity scoring (sorted):**
- Rank reflects relevance → RRF fusion works correctly

**Therefore:** Graph results MUST be scored and sorted before RRF merge.

---

## Configuration

### Environment Variables (.env)

```env
# Vector Search
DEFAULT_TOP_K=10
EMBEDDING_MODEL=text-embedding-3-small

# Graph Search
GRAPH_SCORE_RESULTS=true              # Enable semantic scoring
GRAPH_MAX_RESULTS_FOR_RRF=150        # Max graph results before RRF

# RRF
RRF_K=60                              # RRF constant

# LLM
LLM_MODEL=grok-3                      # LLM model
RERANK_TEMPERATURE=0.1
RERANK_MAX_TOKENS=100

# API Provider
API_PROVIDER=xai                      # or "openai"
XAI_API_KEY=sk-...
OPENAI_API_KEY=sk-...

# Local Embeddings
USE_LOCAL_EMBEDDINGS=auto             # "auto", "true", or "false"
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## Performance Considerations

### Graph Results Limit

- **Why limit to 150?** RRF with 200+ results is slower
- **Quality maintained:** Top 150 scored results are sufficient
- **Configurable:** Set `GRAPH_MAX_RESULTS_FOR_RRF` in `.env`

### Reranking Strategy

- **Pure methods:** Rerank before answer (standalone quality)
- **Hybrid:** Single reranking after RRF (avoids redundancy)

---

## API Endpoints

### Vector Search
```
POST /api/query
{
    "query": "user query",
    "top_k": 10,
    "rerank": true,
    "generate_answer": true
}
```

### Graph Search
```
POST /api/graph_query
{
    "query": "user query",
    "rerank": true,
    "generate_answer": true
}
```

### Hybrid Search
```
POST /api/hybrid_query
{
    "query": "user query",
    "top_k": 10,
    "rerank": true,
    "generate_answer": true
}
```

---

## Evaluation

The evaluation framework (`backend/evaluation/evaluate_search.py`) tests all three methods:

- **Pure Vector:** Tests `query()` method
- **Pure Graph:** Tests `graph_query()` method
- **Hybrid:** Tests `hybrid_query()` method

**Evaluation matches actual program:** All methods have corresponding implementations.

