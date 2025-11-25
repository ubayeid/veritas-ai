# Technical Documentation

Complete technical documentation for the Compliance RAG system.

## Table of Contents
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Graph Structure](#graph-structure)
- [Search Pipeline](#search-pipeline)
- [Components](#components)

---

## Architecture

### System Overview

```
┌─────────────────┐
│  Company Docs   │  ┌──────────────┐
│  (PDFs)         │  │  GDPR PDF    │
└────────┬────────┘  └──────┬───────┘
         │                   │
         ▼                   ▼
    ┌─────────────────────────────┐
    │   Data Processing Pipeline  │
    │  - Extract text             │
    │  - Create embeddings        │
    │  - Build graph structure    │
    └───────────┬─────────────────┘
                │
         ┌──────┴──────┐
         ▼             ▼
    ┌─────────┐   ┌─────────┐
    │  FAISS  │   │  Neo4j  │
    │  Vector │   │  Graph  │
    │  DB     │   │  DB     │
    └────┬────┘   └────┬────┘
         │              │
         └──────┬───────┘
                ▼
         ┌─────────────┐
         │  Hybrid     │
         │  Query      │
         │  Engine     │
         └──────┬──────┘
                ▼
         ┌─────────────┐
         │  Chatbot    │
         │  Interface  │
         └─────────────┘
```

### Main Entry Points

1. **API Server** (`api_server.py`) - Flask REST API
   - `/api/query` - Vector search
   - `/api/hybrid_query` - Hybrid search
   - `/api/chat` - Chatbot interface

2. **Interactive Chatbot** (`run_chatbot.py`) - CLI interface

3. **Query Engine** (`query_engine.py`) - Core search engine
   - Vector search with FAISS
   - Query expansion
   - Result reranking
   - Answer contextualization

4. **Hybrid Query Engine** (`hybrid_query_engine.py`) - Advanced search
   - Combines FAISS + Neo4j
   - Graph traversal
   - Relationship queries

---

## Data Flow

### Complete Pipeline

```
┌───────────────────────────────────────────────────────────────┐
│                        RAW DATA                               │
│  data/company/*.pdf (3 PDFs)                                  │
│  data/standards/gdpr.pdf                                      │
│  data/aiid/incidents.csv (graph)                              │
│  data/aiid/classifications_*.csv (embeddings only)            │
└────────────────────┬──────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│  VECTOR PATH     │    │   GRAPH PATH     │
│                  │    │                  │
│  Step 1:         │    │  Step 1:         │
│  Generate        │    │  Generate        │
│  Embeddings      │    │  Graph JSON      │
│                  │    │                  │
│  Output:         │    │  Output:         │
│  *_embeddings.json│   │  *_graph.json    │
└────────┬─────────┘    └────────┬─────────┘
         │                      │
         ▼                      ▼
┌──────────────────┐    ┌──────────────────┐
│  Step 2:         │    │  Step 2:         │
│  Build FAISS     │    │  Build Neo4j     │
│                  │    │  Graph           │
│  Output:         │    │                  │
│  *.index         │    │  Output:         │
│  *_metadata.pkl  │    │  Neo4j Database  │
└────────┬─────────┘    └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Step 3:         │
         │  Connect         │
         │  Embeddings      │
         │  to Neo4j        │
         │                  │
         │  Adds embeddings │
         │  to nodes        │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  HYBRID SEARCH   │
         │                  │
         │  ✅ FAISS        │
         │  ✅ Graph        │
         │     Traversal    │
         └──────────────────┘
```

### Processing Steps

**Phase 1: Data Processing**
- Extract text from PDFs/CSVs
- Generate embeddings (OpenAI API)
- Create graph structure (JSON)

**Phase 2: Database Building**
- Build FAISS indexes from embeddings
- Build Neo4j graph from JSON
- Add embeddings to Neo4j nodes

**Phase 3: Integration**
- Link clauses to articles (similarity matching)
- Create relationships in Neo4j
- Enable hybrid search

---

## Graph Structure

### How the Graph is Created

**Step 1: Process Raw Data to JSON**

- **Company Documents**: 3 PDFs → 3 Document nodes, ~292 Clause nodes
- **GDPR**: 1 PDF → 102 Article nodes + Topics + SubObligations
- **AIID**: incidents.csv → Incident nodes (classification CSVs → embeddings only)

**Step 2: Build Neo4j Graph**

**Nodes:**
- `Article` - GDPR articles (id, title, description)
- `Clause` - Company document clauses (id, text, document_name)
- `Document` - Company documents (name, source_url)
- `Incident` - AIID incidents (id, title, risk_type)
- `Topic` - GDPR topics
- `SubObligation` - Article sub-obligations

**Relationships:**
- `COVERS` - Document → Clause
- `ADDRESSES` - Clause → Article (via similarity matching)
- `VIOLATES` - Incident → Article
- `HAS_TOPIC` - Article → Topic
- `HAS_SUB_OBLIGATION` - Article → SubObligation

**Step 3: Add Embeddings**
- Generate embeddings for clauses and articles
- Store as node properties (not used for Neo4j vector search)
- Used for creating ADDRESSES relationships

**Step 4: Link Clauses to Articles**
- Compare clause embeddings with article embeddings
- Create ADDRESSES relationships where similarity > threshold (0.45)
- Store similarity score on relationship

### Final Graph Contains

- **102 Article nodes** (GDPR articles)
- **~292 Clause nodes** (from 3 company documents)
- **3 Document nodes** (Privacy Policy, Terms of Service, Cookie Policy)
- **Multiple Incident nodes** (from incidents.csv)
- **ADDRESSES relationships** (clauses → articles)
- **VIOLATES relationships** (incidents → articles)

---

## Search Pipeline

### Complete Query Flow

```
1. User Query
   ↓
2. Query Expansion (optional)
   ├─ Uses: query_expansion_prompt.txt
   └─ Output: Refined query + alternatives
   ↓
3. Vector Search (FAISS)
   ├─ Searches across databases
   └─ Returns: Top K results with similarity scores
   ↓
4. Reranking (optional)
   ├─ Uses: rerank_prompt.txt
   ├─ Model: GPT-4
   └─ Output: Reordered results by relevance
   ↓
5. Contextualization (optional)
   ├─ Uses: contextualize_prompt.txt
   ├─ Model: GPT-4
   └─ Output: Comprehensive answer
   ↓
6. Final Response
   └─ Returns: Results + Answer
```

### Hybrid Search Flow

```
User Query
    ↓
Detect Query Type
    ├─ Graph query? → Neo4j traversal
    └─ Semantic query? → FAISS search
    ↓
Execute Both (if hybrid)
    ├─ FAISS: Vector similarity search
    └─ Neo4j: Graph traversal
    ↓
Merge Results
    ├─ Combine scores
    └─ Deduplicate
    ↓
Rerank & Contextualize
    └─ Return final results
```

---

## Components

### Data Processing

**Location**: `backend/data_processing/`

- **`graph/`** - Convert PDFs/CSVs to graph JSON
  - `gdpr_to_graph.py` - Process GDPR PDF
  - `company_to_graph.py` - Process company PDFs
  - `aiid_to_graph.py` - Process AIID CSV

- **`vector/`** - Generate embeddings
  - `standards_to_embeddings.py` - GDPR embeddings
  - `company_to_embeddings.py` - Company document embeddings
  - `aiid_to_embeddings.py` - AIID embeddings

### Database Building

**Location**: `backend/building_database/`

- **`faiss/`** - Build FAISS indexes
  - `standards_to_faiss_database.py`
  - `company_to_faiss_database.py`
  - `aiid_to_faiss_database.py`

- **`neo4j/`** - Build Neo4j graph
  - `build_knowledge_graph.py` - Create nodes and relationships
  - `add_embeddings.py` - Add embeddings to nodes
  - `link_clauses_to_articles.py` - Create ADDRESSES relationships

### Search & Query

**Location**: `backend/searching/`

- **`query_engine.py`** - Vector search engine
  - `search()` - Basic vector search
  - `search_with_expansion()` - With query expansion
  - `rerank_results()` - LLM-based reranking
  - `contextualize_results()` - Generate answers
  - `query()` - Complete pipeline

- **`hybrid_query_engine.py`** - Hybrid search
  - `hybrid_search()` - Combine vector + graph
  - `hybrid_query()` - Complete hybrid pipeline

- **`chatbot.py`** - Interactive chatbot interface
- **`api_server.py`** - Flask REST API

### Prompts

**Location**: `backend/searching/prompts/`

- **`query_expansion_prompt.txt`** - Expand/refine queries
- **`rerank_prompt.txt`** - Rerank results by relevance
- **`contextualize_prompt.txt`** - Generate comprehensive answers

---

## Key Files

### Processing Scripts
- `backend/data_processing/graph/*_to_graph.py` - Graph processing
- `backend/data_processing/vector/*_to_embeddings.py` - Embedding generation

### Database Builders
- `backend/building_database/faiss/*_to_faiss_database.py` - FAISS indexes
- `backend/building_database/neo4j/build_knowledge_graph.py` - Neo4j graph
- `backend/building_database/neo4j/add_embeddings.py` - Add embeddings
- `backend/building_database/neo4j/link_clauses_to_articles.py` - Link relationships

### Search Engines
- `backend/searching/query_engine.py` - Vector search
- `backend/searching/hybrid_query_engine.py` - Hybrid search
- `backend/searching/chatbot.py` - Chatbot interface
- `backend/searching/api_server.py` - REST API

---

## Configuration

All parameters are configurable via `.env` file:

```bash
# Required
OPENAI_API_KEY=your_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Optional
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
ADDRESSES_SIMILARITY_THRESHOLD=0.45
DEFAULT_TOP_K=10
RRF_K=60
```

---

For usage examples, see [USAGE.md](USAGE.md).

