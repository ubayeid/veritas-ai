# Setup and Configuration Guide

Complete guide for installing, configuring, and running the Compliance RAG System.

## Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Building the System](#building-the-system)
6. [Running the Application](#running-the-application)
7. [Troubleshooting](#troubleshooting)

---

## System Overview

The Compliance RAG System is a hybrid search and analysis platform that:

- **Analyzes Compliance**: Compares company documents against GDPR requirements
- **Identifies Gaps**: Finds missing coverage in company policies
- **Provides Insights**: Generates actionable recommendations for compliance improvements
- **Searches Intelligently**: Combines semantic vector search with knowledge graph relationships

### Key Components

- **Web Interface**: Modern browser-based chat interface
- **Command-Line Chatbot**: Interactive terminal interface
- **Agentic System**: Autonomous compliance analysis agent
- **REST API**: Programmatic access to search capabilities

---

## Prerequisites

Before installing the Compliance RAG System, ensure you have:

1. **Python 3.8 or higher**
   - Check version: `python --version` or `python3 --version`
   - Download from: https://www.python.org/downloads/

2. **Neo4j Database**
   - **Option A**: Neo4j Desktop (Recommended for Windows/Mac)
     - Download from: https://neo4j.com/download/
     - Install and create a new database
   - **Option B**: Neo4j Docker Container (Recommended for Linux/WSL)
     ```bash
     docker run -d --name neo4j \
       -p 7474:7474 -p 7687:7687 \
       -e NEO4J_AUTH=neo4j/password \
       neo4j:latest
     ```

3. **OpenAI API Key**
   - Sign up at: https://platform.openai.com/
   - Generate an API key from your account dashboard
   - Keep this key secure - you'll need it for configuration

4. **Git** (Optional, for cloning the repository)
   - Download from: https://git-scm.com/downloads

---

## Installation

### Step 1: Download/Clone the Project

**Option A: Clone from Git Repository**
```bash
git clone <repository-url>
cd comp_rag
```

**Option B: Download and Extract**
- Download the project ZIP file
- Extract to your desired location
- Navigate to the project directory

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac/WSL:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:
- `openai` - For embeddings and LLM queries
- `neo4j` - For graph database connectivity
- `faiss-cpu` - For vector similarity search
- `flask` - For web API server
- `PyPDF2` - For PDF processing
- `python-dotenv` - For environment configuration

### Step 4: Verify Installation

Check that all packages installed correctly:
```bash
python -c "import openai, neo4j, faiss, flask; print('All packages installed successfully!')"
```

---

## Configuration

### Environment Variables Setup

Create a `.env` file in the project root:

```env
# API Provider (openai or xai)
API_PROVIDER=xai

# API Keys
XAI_API_KEY=sk-...
OPENAI_API_KEY=sk-...  # Required for embeddings even when using xAI

# LLM Configuration
LLM_MODEL=grok-3  # or gpt-4
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=3000

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-small
USE_LOCAL_EMBEDDINGS=auto  # "auto", "true", or "false"
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Search Configuration
DEFAULT_TOP_K=10
GRAPH_SCORE_RESULTS=true
GRAPH_MAX_RESULTS_FOR_RRF=150
RRF_K=60

# Reranking Configuration
RERANK_TEMPERATURE=0.1
RERANK_MAX_TOKENS=100
```

---

## xAI (Grok) Setup

### Configuration

1. Get your xAI API key from [x.ai](https://x.ai)
2. Set in `.env`:
   ```env
   API_PROVIDER=xai
   XAI_API_KEY=sk-...
   XAI_LLM_MODEL=grok-3
   ```

3. **Important:** You still need `OPENAI_API_KEY` for embeddings (xAI doesn't provide embeddings yet)

### Model Notes

- `grok-beta` is deprecated, use `grok-3`
- xAI uses OpenAI-compatible API, so code works seamlessly

---

## Local Embeddings Setup

### When to Use Local Embeddings

- **`USE_LOCAL_EMBEDDINGS=true`**: Always use local (no API calls)
- **`USE_LOCAL_EMBEDDINGS=auto`**: Use local if API unavailable (default)
- **`USE_LOCAL_EMBEDDINGS=false`**: Always use API

### Installation

```bash
pip install sentence-transformers
```

### Model Selection

Default: `all-MiniLM-L6-v2` (384 dimensions)
- Fast and efficient
- Good quality for most use cases
- Lower memory usage

Other options:
- `all-mpnet-base-v2` (768 dimensions) - Better quality, slower
- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support

### Regenerating Embeddings

If you switch to local embeddings, you need to regenerate FAISS databases:

```bash
python backend/data_processing/vector/regenerate_with_local_embeddings.py
```

This will:
1. Regenerate all embeddings using local model
2. Rebuild FAISS databases with correct dimensions

**Note:** Local embeddings (384-dim) are incompatible with OpenAI embeddings (1536-dim). You must rebuild FAISS databases when switching.

---

## Neo4j Setup

### Installation

1. **Desktop:** Download from [neo4j.com](https://neo4j.com/download/)
2. **Docker:**
   ```bash
   docker run -d --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password \
     neo4j:latest
   ```

### Configuration

Set in `.env`:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### Verify Connection

```python
from backend.building_database.neo4j.neo4j_connection import Neo4jConnection

conn = Neo4jConnection()
if conn.verify_connectivity():
    print("✅ Neo4j connected")
else:
    print("❌ Neo4j connection failed")
```

---

## Database Building

**⚠️ Important**: This step processes raw data and builds databases. It only needs to be run once (or when you add new data).

### Option 1: Automated Build (Recommended)

**Windows PowerShell:**
```powershell
# Complete pipeline (all steps)
.\build.ps1 -Complete

# Or step by step:
.\build.ps1 -ProcessGraph
.\build.ps1 -ProcessVector
.\build.ps1 -BuildFaiss
.\build.ps1 -BuildNeo4j
.\build.ps1 -AddEmbeddings
.\build.ps1 -LinkRelationships
```

**Linux/Mac/WSL (using Makefile):**
```bash
# Complete pipeline (all steps)
make complete

# Or step by step:
make process-graph      # Process raw data to graph JSON
make process-vector     # Generate embeddings
make build-faiss         # Build FAISS indexes
make build-neo4j         # Build Neo4j graph (requires Neo4j running)
make add-embeddings      # Add embeddings to Neo4j nodes
make link-relationships  # Link clauses to articles
```

### Option 2: Manual Build

If automated scripts are not available, run these commands in order:

```bash
# 1. Process graph data
python backend/data_processing/graph/gdpr_to_graph.py
python backend/data_processing/graph/company_to_graph.py
python backend/data_processing/graph/aiid_to_graph.py

# 2. Generate embeddings
python backend/data_processing/vector/standards_to_embeddings.py
python backend/data_processing/vector/company_to_embeddings.py
python backend/data_processing/vector/aiid_to_embeddings.py

# 3. Build FAISS indexes
python backend/building_database/faiss/company_to_faiss_database.py
python backend/building_database/faiss/standards_to_faiss_database.py
python backend/building_database/faiss/aiid_to_faiss_database.py

# 4. Build Neo4j graph (requires Neo4j running)
python backend/building_database/neo4j/build_knowledge_graph.py

# 5. Add embeddings to Neo4j
python backend/building_database/neo4j/add_embeddings.py \
    --json-dir backend/data_processing/processed/vector/company \
    --node-type Clause
python backend/building_database/neo4j/add_embeddings.py \
    --json-dir backend/data_processing/processed/vector/standards \
    --node-type Article

# 6. Link clauses to articles
python backend/building_database/neo4j/link_clauses_to_articles.py
```

### Build Verification

After building, verify the databases were created:

**Check FAISS Indexes:**
```bash
# Should exist:
ls backend/building_database/faiss/company/company_faiss_index.index
ls backend/building_database/faiss/standards/standards_faiss_index.index
ls backend/building_database/faiss/aiid/aiid_faiss_index.index
```

**Check Neo4j:**
- Open Neo4j Browser: http://localhost:7474
- Run query: `MATCH (n) RETURN count(n) as node_count`
- Should show nodes created

---

## Running the Application

### Option 1: Web Interface (Recommended)

**Windows PowerShell:**
```powershell
.\start_web_app.ps1
```

**Linux/Mac/WSL:**
```bash
make start-web-app
```

**Manual Start (Two Terminals):**

**Terminal 1 - Backend API Server:**
```bash
python backend/searching/api_server.py
```

**Terminal 2 - Frontend Server:**
```bash
python frontend/start_server.py
```

Then open **http://localhost:8000** in your web browser.

### Option 2: Command-Line Chatbot

**Vector-only mode:**
```bash
python backend/searching/run_chatbot.py
```

**Hybrid mode (recommended):**
```bash
python backend/searching/run_chatbot.py --hybrid
```

### Option 3: Agentic System

```bash
# Using Makefile
make run-agent

# Or directly
python backend/agentic/run_agent.py
```

---

## Troubleshooting

### Installation Issues

**Problem: `pip` command not found**
- **Solution**: Install Python with pip included, or use `python -m pip` instead

**Problem: Virtual environment activation fails**
- **Windows**: Use `.\venv\Scripts\Activate.ps1` (PowerShell) or `venv\Scripts\activate.bat` (CMD)
- **Linux/Mac**: Ensure you're using `source venv/bin/activate` (not `./venv/bin/activate`)

**Problem: Package installation fails**
- **Solution**: Upgrade pip: `python -m pip install --upgrade pip`
- Try installing packages individually to identify the problematic package

### Configuration Issues

**Problem: OpenAI API key not working**
- **Solution**: 
  - Verify key is correct in `.env` file
  - Check for extra spaces or quotes
  - Ensure you have API credits in your OpenAI account
  - Test key: `python -c "import openai; import os; from dotenv import load_dotenv; load_dotenv(); client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY')); print('API key valid!')"`

**Problem: Neo4j connection fails**
- **Solution**:
  - Verify Neo4j is running: Check Neo4j Desktop or Docker container
  - Test connection: `python -c "from neo4j import GraphDatabase; import os; from dotenv import load_dotenv; load_dotenv(); driver = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))); driver.verify_connectivity(); print('Connected!')"`
  - Check URI format: Should be `bolt://localhost:7687` (not `http://`)
  - Verify username/password match your Neo4j instance

**Problem: API key not found**
- **Solution:** Check `.env` file exists and has correct keys

**Problem: Dimension mismatch in FAISS**
- **Solution:** Rebuild FAISS databases with matching embedding model

**Problem: Local embeddings not working**
- **Error:** "sentence-transformers not available"
- **Solution:** `pip install sentence-transformers`

### Build Issues

**Problem: FAISS index build fails**
- **Solution**:
  - Ensure embeddings were generated first
  - Check that processed vector JSON files exist in `backend/data_processing/processed/vector/`
  - Verify sufficient disk space

**Problem: Neo4j build fails**
- **Solution**:
  - Ensure Neo4j is running before building
  - Check Neo4j connection settings in `.env`
  - Verify processed graph JSON files exist in `backend/data_processing/processed/graph/`
  - Check Neo4j logs for specific errors

**Problem: Embedding generation fails**
- **Solution**:
  - Verify OpenAI API key is set correctly
  - Check API rate limits and credits
  - Ensure source PDFs exist in `data/` directory
  - Check internet connection (API calls required)

### Runtime Issues

**Problem: Web interface shows "Cannot connect to API"**
- **Solution**:
  - Ensure backend API server is running (`python backend/searching/api_server.py`)
  - Check backend server logs for errors
  - Verify API URL in `frontend/app.js` matches backend server (default: `http://localhost:5000/api`)
  - Check firewall settings

**Problem: No search results found**
- **Solution**:
  - Verify databases were built successfully
  - Check that FAISS indexes exist
  - Try lowering similarity threshold in settings
  - Use `!databases` command to verify available databases
  - Try different query phrasing

**Problem: CORS errors in browser**
- **Solution**:
  - Ensure you're accessing via web server (not file://)
  - Use `python frontend/start_server.py` to serve frontend
  - Check that Flask-CORS is installed: `pip install flask-cors`

**Problem: Import errors**
- **Solution**:
  - Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `.\venv\Scripts\Activate.ps1` (Windows)
  - Ensure you're running from project root directory
  - Reinstall dependencies: `pip install -r requirements.txt`

**Problem: Out of memory errors**
- **Solution**:
  - Reduce `CHUNK_SIZE` in `.env` file
  - Process fewer documents at once
  - Close other applications to free memory
  - Consider using a machine with more RAM

### Performance Issues

**Problem: Slow search results**
- **Solution**:
  - Reduce `DEFAULT_TOP_K` in settings
  - Disable reranking if not needed
  - Use vector-only mode instead of hybrid (faster)
  - Check Neo4j query performance

**Problem: High API costs**
- **Solution**:
  - Use smaller embedding model: `EMBEDDING_MODEL=text-embedding-3-small`
  - Use cheaper LLM model: `LLM_MODEL=gpt-3.5-turbo`
  - Reduce `LLM_MAX_TOKENS` in `.env`
  - Disable contextualization for faster, cheaper queries

### Getting Help

If you encounter issues not covered here:

1. **Check Logs**: Review error messages in terminal/console
2. **Verify Setup**: Ensure all prerequisites are installed correctly
3. **Test Components**: Test individual components (Neo4j, OpenAI API) separately
4. **Review Documentation**: Check [USAGE.md](USAGE.md) and [TECHNICAL.md](TECHNICAL.md)
5. **Check Issues**: Review project issue tracker (if available)

---

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.10+ |
| RAM | 4 GB | 8 GB+ |
| Disk Space | 2 GB | 5 GB+ |
| Internet | Required (for API calls) | Broadband |
| Browser | Chrome/Firefox/Safari/Edge | Latest version |

---

## Configuration Reference

See [SEARCH_ARCHITECTURE.md](SEARCH_ARCHITECTURE.md) for detailed configuration options.

