# Compliance RAG System - Web Interface

Modern web-based interface for querying the Compliance RAG System with support for Vector, Graph, and Hybrid search modes.

## Features

- 🎨 **Modern UI**: Clean, responsive design
- 🔍 **Three Search Modes**: Vector, Graph, and Hybrid
- 💬 **Real-time Chat**: Interactive Q&A interface
- ⚙️ **Configurable Settings**: Top-K, Reranking, Answer Generation
- 📊 **Mode Indicators**: Visual badges showing which method was used

## Quick Start

### 1. Start API Server

```bash
# Terminal 1: Start the API server
python backend/retrieval/interfaces/api_server.py
```

The API will run on `http://localhost:5000`

### 2. Start Frontend Server

```bash
# Terminal 2: Start the frontend server
python frontend/scripts/start_server.py
```

The frontend will run on `http://localhost:8000`

### 3. Open Browser

Navigate to: **http://localhost:8000**

## Usage

1. **Select Search Mode**: Choose from dropdown:
   - **Vector**: FAISS semantic similarity search
   - **Graph**: Neo4j knowledge graph traversal
   - **Hybrid**: Combined vector + graph with RRF

2. **Configure Settings** (optional):
   - **Top K**: Number of results to retrieve (default: 8)
   - **Rerank**: Use LLM to rerank results
   - **Generate Answer**: Generate answer using LLM

3. **Ask Questions**: Type your question and click Send

## API Endpoints

The frontend uses the unified `/api/query` endpoint:

```javascript
POST /api/query
{
  "query": "your question",
  "mode": "vector|graph|hybrid",
  "top_k": 8,
  "rerank": true,
  "generate_answer": true
}
```

## Architecture

```
Frontend (localhost:8000)
    ↓ HTTP POST
API Server (localhost:5000)
    ↓ Mode Routing
Query Engines:
    - VectorQueryEngine (FAISS)
    - GraphQueryEngine (Neo4j)
    - HybridQueryEngine (RRF)
```

## Files

- `static/index.html` - Main HTML structure
- `static/styles.css` - Styling
- `static/app.js` - JavaScript client logic
- `scripts/start_server.py` - Simple HTTP server

## Troubleshooting

**API Connection Error:**
- Ensure API server is running on port 5000
- Check CORS settings in `api_server.py`
- Verify API_BASE_URL in `app.js`

**No Results:**
- Check Neo4j is running (for graph/hybrid modes)
- Verify FAISS indexes are built (for vector/hybrid modes)
- Check API server logs for errors
