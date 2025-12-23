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
- FAISS indexes built (see `backend/building_database/faiss/`)

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
   - `backend/building_database/faiss/company/company_faiss_index.*`
   - `backend/building_database/faiss/aiid/aiid_faiss_index.*`
   - `backend/building_database/faiss/standards/standards_faiss_index.*`

## Usage

### Running the Chatbot

```bash
# Vector-only mode (default)
python backend/searching/run_chatbot.py

# Hybrid mode (FAISS + Neo4j graph traversal) - Recommended
python backend/searching/run_chatbot.py --hybrid
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
from backend.searching.query_engine import VectorQueryEngine

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

### Components

1. **query_engine.py**: Core query engine with vector search, reranking, and contextualization
2. **chatbot.py**: Interactive chatbot interface
3. **prompts/**: Prompt templates for reranking and contextualization
   - `rerank_prompt.txt`: Prompt for LLM-based result reranking
   - `contextualize_prompt.txt`: Prompt for generating contextualized answers

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

- **DEFAULT_TOP_K**: Number of results to retrieve (default: 10, recommended: 10)
- **LLM_MODEL**: Model for reranking/answers (default: gpt-4)
- **LLM_TEMPERATURE**: Creativity for answers (default: 0.3)
- **LLM_MAX_TOKENS**: Max tokens in answers (default: 3000)
- **RRF_K**: Reciprocal Rank Fusion constant (default: 60)

### Custom Prompts

You can customize prompts by editing:
- `backend/searching/prompts/rerank_prompt.txt`
- `backend/searching/prompts/answer_generation_prompt.txt`

## Hybrid Search Mode

When using `--hybrid` flag, the system combines:
- **FAISS Vector Search**: Semantic similarity across all databases
- **Neo4j Graph Traversal**: Relationship-based queries (clauses → articles, incidents → articles)

This provides both semantic understanding and structural relationships for comprehensive answers.

## Troubleshooting

### Database Not Found

Ensure FAISS indexes are built:
```bash
python backend/building_database/faiss/company_to_faiss_database.py
python backend/building_database/faiss/aiid_to_faiss_database.py
python backend/building_database/faiss/standards_to_faiss_database.py
```

### OpenAI API Errors

- Check your API key in `.env`
- Ensure you have sufficient API credits
- Check rate limits if making many queries

### Import Errors

Make sure you're running from the project root or have the correct Python path set.

