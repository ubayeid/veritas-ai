# Compliance RAG Chatbot

Interactive chatbot interface for querying processed vector data from company policies, AIID incidents, and regulatory standards.

## Features

- **FAISS Vector Search**: Fast semantic search across all databases (company, AIID, standards)
- **Hybrid Search**: Combines FAISS vector search + Neo4j graph traversal (use `--hybrid` flag)
- **LLM Reranking**: Use configurable LLM model to rerank search results by relevance
- **Contextualized Answers**: Generate comprehensive answers based on search results
- **Interactive Interface**: Command-line chatbot with helpful commands
- **Conversation History**: Track queries and answers during session
- **Fully Configurable**: All parameters customizable via `.env` file (see `CONFIGURATION.md`)

## Requirements

- Python 3.8+
- OpenAI API key (set in `.env` file)
- FAISS indexes built (see `backend/indexing/faiss/`)

## Installation

1. Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Set up your `.env` file (copy from `.env.example`):
```
OPENAI_API_KEY=your_api_key_here
EMBEDDING_MODEL=text-embedding-3-small  # Optional, defaults to this
LLM_MODEL=gpt-4  # Optional, defaults to gpt-4
CHUNK_SIZE=1000  # Optional, defaults to 1000
CHUNK_OVERLAP=200  # Optional, defaults to 200
# See CONFIGURATION.md for all options
```

3. Ensure FAISS databases are built:
   - `backend/indexing/faiss/output/company_faiss_index.*`
   - `backend/indexing/faiss/output/aiid_faiss_index.*`
   - `backend/indexing/faiss/output/standards_faiss_index.*`

## Usage

### Running the Chatbot

```bash
# Vector-only mode (default)
python backend/retrieval/scripts/run_chatbot.py

# Hybrid mode (FAISS + Neo4j graph traversal) - Recommended
python backend/retrieval/scripts/run_chatbot.py --hybrid
```

### Interactive Commands

- Type your question to search the database
- `!help` - Show help message
- `!databases` - List available databases
- `!history` - Show conversation history
- `!clear` - Clear conversation history
- `!settings` - Change search settings
- `!mode` - Switch between vector/hybrid search modes
- `!quit` or `!exit` - Exit chatbot

### Example Queries

```
What are the privacy policies?
Find incidents related to data breaches
What GDPR requirements apply?
Show me company policies about data handling
```

### Programmatic Usage

```python
from backend.retrieval.engines.query_engine import VectorQueryEngine

# Initialize engine
engine = VectorQueryEngine(base_dir="/path/to/project")

# Simple search
results = engine.search(
    query="What are the privacy policies?",
    db_names=['company', 'standards'],
    top_k=10
)

# Full query with reranking and contextualization
result = engine.query(
    query="What GDPR requirements apply?",
    db_names=['standards'],
    top_k=10,
    rerank=True,
    contextualize=True
)

print(result['answer'])
```

## Architecture

### Directory Structure

```
backend/retrieval/
├── engines/                 # Query engines
│   ├── __init__.py
│   ├── query_engine.py      # Vector query engine
│   └── hybrid_query_engine.py  # Hybrid query engine
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── api_client.py        # API client for LLMs
│   ├── local_embeddings.py  # Local embedding utilities
│   └── neo4j_queries.py     # Neo4j query utilities
├── interfaces/              # User interfaces
│   ├── __init__.py
│   ├── api_server.py        # Flask API server
│   └── chatbot.py           # Interactive chatbot
├── scripts/                 # CLI scripts
│   ├── run_chatbot.py       # Run chatbot script
│   └── start_server.py      # Start API server script
├── __init__.py
├── README.md
└── RECOMMENDED_SETTINGS.md
```

### Components

1. **engines/query_engine.py**: Core query engine with vector search, reranking, and contextualization
2. **engines/hybrid_query_engine.py**: Hybrid query engine combining vector and graph search
3. **interfaces/chatbot.py**: Interactive chatbot interface
4. **interfaces/api_server.py**: Flask API server for programmatic access
5. **utils/**: Utility modules for API clients, embeddings, and Neo4j queries
6. **prompts/**: Prompt templates (in `backend/generation/prompts/`)
   - `rerank_prompt.txt`: Prompt for LLM-based result reranking
   - `answer_generation_prompt.txt`: Prompt for generating contextualized answers

### Query Pipeline

1. **Search**: 
   - FAISS: Convert query to embedding and search FAISS indexes
   - Hybrid: Also traverse Neo4j graph relationships
2. **Rerank** (optional): Use configurable LLM model to rerank results by relevance
3. **Contextualize** (optional): Generate comprehensive answer from top results using LLM

### Database Structure

The query engine supports three databases:
- **company**: Company policies and documents
- **aiid**: AI Incident Database entries
- **standards**: Regulatory standards (e.g., GDPR)

## Configuration

All parameters are customizable via `.env` file. See `CONFIGURATION.md` for complete list.

### Recommended Settings (Best Balance)

**Optimal defaults for pure vector search:**
- `top_k`: 10 ✅ (good balance of speed and quality)
- `similarity_threshold`: 0.0 ✅ (include all, rank by similarity)
- `rerank`: True ✅ (improves relevance)
- `generate_answer`: True ✅ (provides comprehensive answers)
- `db_names`: None ✅ (search all databases)

See `RECOMMENDED_SETTINGS.md` for detailed guidance.

### Key Search Settings (via .env)

- **Top K**: Adaptive based on query type and result quality (no fixed default)
- **LLM_MODEL**: Model for reranking/answers (default: gpt-4)
- **LLM_TEMPERATURE**: Creativity for answers (default: 0.3)
- **LLM_MAX_TOKENS**: Max tokens in answers (default: 3000)
- **RRF_K**: Reciprocal Rank Fusion constant (default: 60)

### Custom Prompts

You can customize prompts by editing:
- `backend/prompts/rerank.txt` (consolidated prompts directory)
- `backend/prompts/answer_generation.txt` (consolidated prompts directory)

## Hybrid Search Mode

When using `--hybrid` flag, the system combines:
- **FAISS Vector Search**: Semantic similarity across all databases
- **Neo4j Graph Traversal**: Relationship-based queries (clauses → articles, incidents → articles)

This provides both semantic understanding and structural relationships for comprehensive answers.

## Troubleshooting

### Database Not Found

Ensure FAISS indexes are built:
```bash
python backend/indexing/faiss/build_faiss_index.py --source company
python backend/indexing/faiss/build_faiss_index.py --source aiid
python backend/indexing/faiss/build_faiss_index.py --source standards
```

### OpenAI API Errors

- Check your API key in `.env`
- Ensure you have sufficient API credits
- Check rate limits if making many queries

### Import Errors

Make sure you're running from the project root or have the correct Python path set.

