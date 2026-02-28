# Veritas AI - Agentic Compliance Monitoring System

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**An intelligent hybrid RAG system with multi-agent architecture for automated compliance monitoring and risk assessment**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 🎯 Overview

**Veritas AI** is a comprehensive compliance monitoring system that combines **hybrid retrieval-augmented generation (RAG)** with a **multi-agent architecture** to automatically analyze company policies against regulatory standards (e.g., GDPR, AI Act) and identify compliance gaps.

### Problem Statement

Organizations struggle to maintain compliance with evolving regulations like GDPR and the EU AI Act. Manual compliance audits are:
- **Time-consuming** - Require expert review of hundreds of documents
- **Error-prone** - Easy to miss gaps or violations
- **Expensive** - Require specialized legal/compliance teams
- **Reactive** - Only catch issues after they occur

**Veritas AI solves this** by automating compliance monitoring using AI agents that continuously analyze policies, detect gaps, and assess risks.

### Key Innovation

Unlike traditional compliance tools, Veritas AI uses:
- **Hybrid Search**: Combines FAISS vector similarity with Neo4j knowledge graph traversal
- **Multi-Agent System**: Four specialized agents working together for comprehensive analysis
- **Automated Gap Detection**: Identifies missing coverage and policy violations automatically
- **Incident-Based Risk Assessment**: Learns from historical AI incidents to predict risks

---

## ✨ Features

### 🔍 **Three Search Modes**
- **Vector Search**: Fast semantic similarity using FAISS
- **Graph Traversal**: Relationship-based queries using Neo4j knowledge graphs
- **Hybrid Search**: Combined approach with Reciprocal Rank Fusion (RRF) for best results

### 🤖 **Multi-Agent Architecture**
1. **Monitoring Agent** - Observes AI application behavior in real-time
2. **Decision Making Agent** - Evaluates compliance risk levels
3. **Compliance Verification Agent** - Identifies specific policy violations
4. **Orchestration Agent** - Coordinates agents and makes final decisions

### 💡 **Core Capabilities**
- ✅ **Compliance Gap Detection** - Find missing regulatory coverage
- ✅ **Policy Mapping** - Link company clauses to regulatory articles
- ✅ **Risk Assessment** - Evaluate compliance risks with scoring
- ✅ **Incident Analysis** - Learn from historical AI incidents (AIID database)
- ✅ **Interactive CLI** - Query system via command line
- ✅ **Web Interface** - Modern browser-based UI
- ✅ **REST API** - Full API for integration
- ✅ **Evaluation Framework** - Built-in IR metrics for research

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Neo4j** (Desktop or Docker) - [Download](https://neo4j.com/download/)
- **OpenAI API Key** - [Get one here](https://platform.openai.com/api-keys)

### Installation

```bash
# Clone the repository
git clone https://github.com/ubayeid/veritas-ai.git
cd veritas-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and Neo4j credentials
```

### Configuration

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Optional (with defaults)
API_PROVIDER=openai
LLM_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### One-Time Database Setup

**⚠️ Important**: Build the databases before first use (one-time setup):

```bash
# Build FAISS vector indexes
python backend/indexing/faiss/build_faiss_index.py --source company
python backend/indexing/faiss/build_faiss_index.py --source aiid
python backend/indexing/faiss/build_faiss_index.py --source standards

# Build Neo4j knowledge graph (requires Neo4j running)
python backend/indexing/neo4j/build_knowledge_graph.py

# Add embeddings to Neo4j nodes
python backend/indexing/neo4j/scripts/add_embeddings.py \
    --json-dir backend/processing/vector/company \
    --node-type Clause

python backend/indexing/neo4j/scripts/add_embeddings.py \
    --json-dir backend/processing/vector/standards \
    --node-type Article

# Link clauses to articles
python backend/indexing/neo4j/scripts/link_clauses_to_articles.py
```

### Usage

#### 1. Interactive CLI (Query/Answer)

```bash
# Vector mode (FAISS semantic search)
python query.py interactive --mode vector

# Graph mode (Neo4j graph traversal)
python query.py interactive --mode graph

# Hybrid mode (recommended - combines both)
python query.py interactive --mode hybrid
```

**Interactive Commands:**
- Type your question to search
- `!help` - Show help
- `!mode` - Switch between vector/graph/hybrid modes
- `!settings` - Adjust search parameters
- `!quit` - Exit

#### 2. Multi-Agent Orchestration

```bash
# Run 4-agent compliance monitoring system
python query.py agent
```

**Agent Commands:**
- Type events/queries to evaluate compliance
- `!audit` - Show audit log
- `!quit` - Exit

#### 3. Web Interface

```bash
# Terminal 1: Start API server
python backend/retrieval/interfaces/api_server.py

# Terminal 2: Start frontend server
python frontend/scripts/start_server.py

# Open browser: http://localhost:8000
```

#### 4. Evaluation (Research)

```bash
# Run evaluation on 50 queries
python query.py evaluate

# Custom queries and output
python query.py evaluate \
    --queries my_queries.json \
    --output results.csv \
    --top-k 8
```

---

## 📊 Example Queries

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

### Risk Assessment
```
"Does this AI system comply with AI Act?"
"Evaluate compliance risk for facial recognition system"
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  (CLI / Web UI / REST API)                              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Multi-Agent Orchestration                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │ Monitoring   │ │ Decision    │ │ Compliance   │     │
│  │ Agent        │ │ Making      │ │ Verification │     │
│  └──────────────┘ └──────────────┘ └──────────────┘     │
│                    ┌──────────────┐                      │
│                    │ Orchestration│                      │
│                    │ Agent        │                      │
│                    └──────────────┘                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Hybrid Query Engine                         │
│  ┌──────────────┐         ┌──────────────┐             │
│  │ Vector       │         │ Graph        │             │
│  │ (FAISS)      │ + RRF + │ (Neo4j)      │             │
│  └──────────────┘         └──────────────┘             │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────┐         ┌──────────────┐
│ FAISS        │         │ Neo4j        │
│ Indexes     │         │ Knowledge    │
│ (Vectors)   │         │ Graph        │
└──────────────┘         └──────────────┘
```

### Data Flow

1. **Data Ingestion**: Company documents (PDFs), regulatory standards (GDPR), AIID incidents (CSV)
2. **Processing**: Extract text → Generate embeddings → Build graph structure
3. **Storage**: FAISS (vectors) + Neo4j (graph relationships)
4. **Querying**: Hybrid search combines semantic similarity + graph traversal
5. **Analysis**: Multi-agent system evaluates compliance and generates insights

### Project Structure

```
veritas-ai/
├── query.py                    # Main CLI entry point
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
│
├── backend/
│   ├── agents/                # Multi-agent architecture
│   │   ├── monitoring/        # Monitoring Agent
│   │   ├── decision_making/   # Decision Making Agent
│   │   ├── compliance/        # Compliance Verification Agent
│   │   ├── orchestration/     # Orchestration Agent
│   │   ├── core/              # Base classes
│   │   └── utils/             # Utilities
│   │
│   ├── retrieval/             # Core query engines
│   │   ├── engines/           # Vector, Graph, Hybrid engines
│   │   ├── interfaces/        # API server, chatbot
│   │   └── utils/             # Helper functions
│   │
│   ├── indexing/              # Database building (one-time setup)
│   │   ├── faiss/            # FAISS index builder
│   │   └── neo4j/            # Neo4j graph builder
│   │
│   ├── processing/            # Data processing pipeline
│   │   ├── graph/            # Graph data processing
│   │   ├── vector/           # Embedding generation
│   │   └── utils/            # Processing utilities
│   │
│   ├── evaluation/            # Evaluation framework
│   │   └── ir_evaluation.py  # IR metrics calculation
│   │
│   └── prompts/               # LLM prompt templates
│
├── frontend/                   # Web interface
│   ├── static/                # HTML, CSS, JavaScript
│   └── scripts/                # Server scripts
│
└── data/                       # Data directory (not in repo)
    ├── company/               # Company documents (PDFs)
    ├── standards/             # Regulatory standards (PDFs)
    └── aiid/                  # AIID incident database (CSV)
```

---

## 📚 Documentation

The project structure is self-documenting with clear module organization:
- `backend/agents/` - Multi-agent architecture implementation
- `backend/retrieval/` - Query engines (Vector, Graph, Hybrid)
- `backend/indexing/` - Database building scripts
- `backend/processing/` - Data processing pipeline
- `backend/evaluation/` - Evaluation framework
- `frontend/` - Web interface

---

## 🧪 Evaluation

The system includes a built-in evaluation framework for research:

```bash
# Run evaluation on 50 queries
python query.py evaluate

# Export results for analysis
python query.py evaluate --export-chunks --output results.csv
```

**Metrics Supported:**
- Precision@K, Recall@K, F1@K
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG@K)

---

## 🔧 Configuration

All parameters are customizable via `.env` file:

```env
# Search Settings
TOP_K=8                    # Number of results to retrieve
SIMILARITY_THRESHOLD=0.0   # Minimum similarity score
RERANK=true                # Use LLM reranking
GENERATE_ANSWER=true       # Generate contextualized answers

# Model Settings
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=3000
EMBEDDING_MODEL=text-embedding-3-small

# Hybrid Search
RRF_K=60                   # Reciprocal Rank Fusion constant
```

---

## 🔧 Troubleshooting

### Common Issues

**Neo4j Connection Errors:**
```bash
# Check if Neo4j is running
# Windows: Check Neo4j Desktop or service status
# Linux/Mac: sudo systemctl status neo4j

# Verify connection settings in .env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

**FAISS Index Not Found:**
```bash
# Build the indexes first
python backend/indexing/faiss/build_faiss_index.py --source company
python backend/indexing/faiss/build_faiss_index.py --source aiid
python backend/indexing/faiss/build_faiss_index.py --source standards
```

**OpenAI API Errors:**
- Verify your API key in `.env`
- Check API balance at https://platform.openai.com/usage
- System will fallback to local embeddings if API unavailable

**Import Errors:**
```bash
# Ensure you're in the project root directory
# Install all dependencies
pip install -r requirements.txt
```

**Port Already in Use:**
```bash
# Change port in .env or kill existing process
# Windows: netstat -ano | findstr :5000
# Linux/Mac: lsof -i :5000
```

## 🤝 Contributing

We welcome contributions! 

### Development Setup

```bash
# Clone and setup
git clone https://github.com/ubayeid/veritas-ai.git
cd veritas-ai
pip install -r requirements.txt

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python query.py interactive --mode hybrid

# Commit and push
git commit -m "Add your feature"
git push origin feature/your-feature-name
```

### Code Style

- Follow PEP 8 for Python code
- Add docstrings to all functions and classes
- Write tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 💡 Lessons Learned

Combining hybrid RAG (vector + graph) with multi-agent orchestration enables automated compliance monitoring that scales beyond manual audits while maintaining interpretability through structured knowledge graphs.

## 🙏 Acknowledgments

- **AIID Database** - AI Incident Database for incident data
- **LangChain & LangGraph** - Agent framework
- **LlamaIndex** - RAG abstractions
- **Neo4j** - Graph database
- **FAISS** - Vector similarity search

---

## 📞 Contact & Support

- **Author**: Ubayeid U.
- **Repository**: [https://github.com/ubayeid/veritas-ai](https://github.com/ubayeid/veritas-ai)
- **Issues**: [GitHub Issues](https://github.com/ubayeid/veritas-ai/issues)

---

## 🎯 Use Cases

1. **Compliance Audits** - Automated GDPR/AI Act compliance checking
2. **Policy Review** - Compare company policies against regulations
3. **Risk Assessment** - Identify potential compliance risks
4. **Gap Analysis** - Find missing regulatory coverage
5. **Incident Learning** - Learn from historical AI incidents

---

## ⭐ Features in Detail

### Hybrid Search
- **Vector Search**: Semantic similarity using FAISS for fast retrieval
- **Graph Traversal**: Relationship-based queries using Neo4j (clauses → articles, incidents → articles)
- **Reciprocal Rank Fusion**: Combines both methods for optimal results

### Multi-Agent System
- **Modular Design**: Each agent has a specific role
- **LangGraph Integration**: State machine-based agent orchestration
- **Audit Logging**: Complete audit trail of all decisions
- **Extensible**: Easy to add new agents

### Evaluation Framework
- **IR Metrics**: Standard information retrieval metrics
- **Answer Generation**: LLM-generated answers for evaluation
- **Export Support**: CSV export for analysis

---

<div align="center">

**Built with ❤️ for automated compliance monitoring**

[⭐ Star this repo](https://github.com/ubayeid/veritas-ai) if you find it useful!

</div>
