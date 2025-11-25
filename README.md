# Compliance RAG System

A hybrid search system that analyzes company documents against GDPR requirements using vector search and knowledge graphs.

---

## 🎯 What It Does

This system helps companies:
- **Find compliance gaps** between their policies and GDPR requirements
- **Identify coverage** - which GDPR articles are addressed in company documents
- **Get actionable insights** - specific recommendations for compliance improvements
- **Search intelligently** - combines semantic search with relationship analysis

---

## ✨ Key Features

### 1. **Hybrid Search**
- **Vector Search (FAISS)** - Fast semantic similarity matching across all data
- **Graph Traversal (Neo4j)** - Relationship-based queries (clauses → articles, incidents → articles)
- **Combined Results** - Merges semantic search with structural relationships

### 2. **Compliance Analysis**
- **Gap Detection** - Find GDPR articles not covered by company documents
- **Coverage Analysis** - See which articles ARE addressed
- **Clause Mapping** - Link company clauses to GDPR articles

### 3. **Intelligent Answers**
- **Contextualized Responses** - LLM-generated answers from search results
- **Source Citations** - Know where information comes from
- **Actionable Recommendations** - What to fix and how

---

## 🏗️ Architecture

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

---

## 🚀 Quick Start

### 1. **Setup**

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (.env)
# Copy .env.example to .env and edit with your values
cp .env.example .env
# Edit .env with your API keys and preferences

# Required:
OPENAI_API_KEY=your_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Optional (see CONFIGURATION.md for all options):
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### 2. **Build the System** (One-Time Setup)

**⚠️ Important**: This step processes raw data and builds databases. It only needs to be run once (or when you add new data).

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
make process-vector      # Generate embeddings
make build-faiss         # Build FAISS indexes
make build-neo4j         # Build Neo4j graph (requires Neo4j running)
make add-embeddings      # Add embeddings to Neo4j nodes
make link-relationships  # Link clauses to articles
```

**Option C: Manual (Individual Scripts)**
```bash
# Process graph data
python backend/data_processing/graph/gdpr_to_graph.py
python backend/data_processing/graph/company_to_graph.py
python backend/data_processing/graph/aiid_to_graph.py

# Generate embeddings
python backend/data_processing/vector/standards_to_embeddings.py
python backend/data_processing/vector/company_to_embeddings.py
python backend/data_processing/vector/aiid_to_embeddings.py

# Build FAISS indexes
python backend/building_database/faiss/company_to_faiss_database.py
python backend/building_database/faiss/standards_to_faiss_database.py
python backend/building_database/faiss/aiid_to_faiss_database.py

# Build Neo4j graph (requires Neo4j running)
python backend/building_database/neo4j/build_knowledge_graph.py

# Add embeddings to Neo4j
python backend/building_database/neo4j/add_embeddings.py \
    --json-dir backend/data_processing/processed/vector/company \
    --node-type Clause
python backend/building_database/neo4j/add_embeddings.py \
    --json-dir backend/data_processing/processed/vector/standards \
    --node-type Article

# Link clauses to articles
python backend/building_database/neo4j/link_clauses_to_articles.py
```

### 3. **Start the Web Application** (Runtime)

**⚠️ Important**: This starts the servers to use the system. Run this every time you want to query the chatbot.

**Windows PowerShell:**
```powershell
.\start_web_app.ps1
# Or using build script:
.\build.ps1 -StartWebApp
```

**Linux/Mac/WSL:**
```bash
# Option 1: Using Makefile
make start-web-app

# Option 2: Manual (two terminals)
# Terminal 1 - Backend API Server
python backend/searching/api_server.py

# Terminal 2 - Frontend Server
python frontend/start_server.py
```

Then open **http://localhost:8000** in your browser.

### 4. **Run Chatbot** (Alternative: Command Line)

```bash
# Vector-only mode
python backend/searching/run_chatbot.py

# Hybrid mode (recommended)
python backend/searching/run_chatbot.py --hybrid
```

### 5. **Run Agentic System** (New!)

```bash
# Using Makefile (recommended)
make run-agent

# Or directly
source venv/bin/activate
python backend/agentic/run_agent.py
```

See [docs/AGENTIC_SYSTEM.md](docs/AGENTIC_SYSTEM.md) for details.

---

## 💬 Example Queries

### Compliance Analysis
```
"What are the mismatches between company data and GDPR data?"
"Find GDPR articles not covered by company documents"
"Which clauses address GDPR Article 5?"
```

### Gap Detection
```
"What are the compliance gaps?"
"Find missing GDPR coverage"
```

### Incident Analysis
```
"Find incidents related to data breaches"
"What AIID incidents violate GDPR Article 5?"
```

---

## 📊 Current Results

### Database Statistics
- **Total Articles:** 102 GDPR articles
- **Coverage:** 8 articles covered (7.84%)
- **Gaps:** 94 articles not covered
- **Relationships:** 21 clause-to-article links
- **Clauses:** 89 company document clauses

### Example Coverage
- **Article 12:** 10 clauses (Transparent information)
- **Article 62:** 4 clauses (Joint operations)
- **Article 45:** 2 clauses (Data transfers)

---

## 🔧 System Components

### Data Processing
- **`backend/data_processing/`** - Extract and process PDFs/CSVs
- **`backend/data_processing/graph/`** - Convert to graph structure
- **`backend/data_processing/vector/`** - Generate embeddings

### Database Building
- **`backend/building_database/neo4j/`** - Build knowledge graph
- **`backend/building_database/faiss/`** - Build vector database
- **`backend/building_database/neo4j/link_clauses_to_articles.py`** - Create relationships

### Search & Query
- **`backend/searching/`** - Search engines and chatbot
- **`backend/searching/hybrid_query_engine.py`** - Hybrid search logic
- **`backend/searching/chatbot.py`** - Interactive interface

---

## 📁 Project Structure

```
comp_rag/
├── data/
│   ├── company/              # Raw company PDFs (3 documents: Privacy Policy, Terms of Service, Cookie Policy)
│   ├── standards/            # GDPR PDF
│   ├── aiid/                 # AIID data (incidents.csv → graph, classification CSVs → embeddings only)
│   └── processed/
│       ├── graph/            # Processed graph JSONs
│       └── vector/           # Embeddings JSONs
│
├── backend/
│   ├── data_processing/      # Data extraction & processing
│   ├── building_database/    # Build Neo4j & FAISS
│   └── searching/            # Search engines & chatbot
│
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

---

## 🎓 How It Works

### 1. **Data Ingestion**
- Company documents (3 PDFs: Privacy Policy, Terms of Service, Cookie Policy)
- GDPR regulation (PDF)
- AIID incident database (incidents.csv → graph nodes, classification CSVs → embeddings only)

### 2. **Processing**
- Extract text and structure
- Generate embeddings (OpenAI)
- Create graph nodes and relationships

### 3. **Storage**
- **FAISS** - Fast vector similarity search
- **Neo4j** - Graph database for relationships

### 4. **Querying**
- **FAISS Vector Search** - Find semantically similar content across all databases
- **Graph Traversal** - Follow relationships (clauses → articles, incidents → articles)
- **Hybrid** - Combines FAISS vector search + Neo4j graph traversal for comprehensive results

### 5. **Answer Generation**
- Rerank results by relevance
- Generate contextualized answer using LLM
- Cite sources and provide recommendations

---

## 🔍 Search Modes

### Vector-Only Mode
- Fast semantic search
- Good for general queries
- Uses FAISS database

### Hybrid Mode (Recommended)
- Combines FAISS vector search + Neo4j graph traversal
- Best for compliance analysis
- Shows relationships and coverage
- No redundant Neo4j vector search (simplified architecture)

**Switch modes:** Type `!mode` in chatbot

---

## 📈 Key Achievements

✅ **Simplified Hybrid Search** - FAISS Vector Search + Neo4j Graph Traversal (removed redundant Neo4j vector search)  
✅ **Compliance Gap Detection** - Identifies 94 gaps  
✅ **Coverage Analysis** - Shows 8 articles with coverage  
✅ **Clause Mapping** - Links company clauses to GDPR articles  
✅ **Intelligent Answers** - LLM-generated contextual responses  
✅ **Company-Specific Insights** - Focused on actual company documents  

---

## 🛠️ Requirements

- **Python 3.8+**
- **Neo4j** (Desktop or Docker)
- **OpenAI API Key** (for embeddings and LLM)
- **Dependencies:** See `requirements.txt`

## ⚙️ Configuration

All parameters are customizable via `.env` file. See `CONFIGURATION.md` for details.

**Quick customization:**
- `CHUNK_SIZE` - Size of text chunks (default: 1000)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 200)
- `LLM_MODEL` - Model for answers (default: gpt-4)
- `EMBEDDING_MODEL` - Embedding model (default: text-embedding-3-small)
- `ADDRESSES_SIMILARITY_THRESHOLD` - For linking clauses to articles (default: 0.45)

Copy `.env.example` to `.env` and customize as needed!

---

## 📚 Documentation

- **[docs/USAGE.md](docs/USAGE.md)** - Usage guide with sample questions and Cypher queries
- **[docs/TECHNICAL.md](docs/TECHNICAL.md)** - Technical architecture and data flow
- **[docs/AGENTIC_SYSTEM.md](docs/AGENTIC_SYSTEM.md)** - Agentic system guide

---

## 🎯 Use Cases

1. **Compliance Audits** - Find gaps in GDPR coverage
2. **Policy Review** - Compare company policies with regulations
3. **Risk Assessment** - Link incidents to violated articles
4. **Documentation** - Map clauses to requirements

---

## 💡 Tips

- **Use Hybrid Mode** for compliance queries
- **Lower similarity threshold** (0.4-0.45) for more relationships
- **Check Neo4j Browser** to visualize the graph
- **Use `!help`** in chatbot for commands

---

## 📞 Support

For issues or questions:
- See [docs/USAGE.md](docs/USAGE.md) for usage examples and troubleshooting
- See [docs/TECHNICAL.md](docs/TECHNICAL.md) for technical details
- See [docs/AGENTIC_SYSTEM.md](docs/AGENTIC_SYSTEM.md) for agentic system guide

---

**Built for:** Compliance analysis and GDPR gap detection  
**Technology:** Python, Neo4j, FAISS, OpenAI, LLMs  
**Status:** ✅ Working - Coverage analysis functional

