# Usage Guide

Complete guide to using the Compliance RAG system.

## Table of Contents
- [Quick Start](#quick-start)
- [Web Interface Usage](#web-interface-usage)
- [Command-Line Chatbot Usage](#command-line-chatbot-usage)
- [Agentic System Usage](#agentic-system-usage)
- [Sample Questions](#sample-questions)
- [Cypher Queries](#cypher-queries)
- [API Usage](#api-usage)
- [Tips for Best Results](#tips-for-best-results)

---

## Quick Start

### 1. Setup

```bash
# Install dependencies
make install

# Build the system (one-time)
make complete
```

### 2. Run Chatbot

```bash
# Interactive chatbot
make start-web-app
# Then open http://localhost:8000

# Or command line
python backend/searching/run_chatbot.py --hybrid
```

### 3. Run Agent

```bash
# Agentic system
make run-agent
```

---

## Web Interface Usage

1. **Start the Application** (see [SETUP.md](SETUP.md#running-the-application))

2. **Open Browser**
   - Navigate to: http://localhost:8000
   - You should see the chat interface

3. **Configure Settings** (Optional)
   - Click the settings icon
   - Select databases to search (Company, AIID, Standards)
   - Adjust Top K results (1-50)
   - Enable/disable reranking and answer generation
   - Set similarity threshold (0.0-1.0)

4. **Ask Questions**
   - Type your question in the input box
   - Click "Send" or press Enter
   - View the AI-generated answer and search results

5. **Review Results**
   - Read the contextualized answer at the top
   - Scroll down to see individual search results
   - Each result shows similarity score and source

---

## Command-Line Chatbot Usage

**Starting the Chatbot:**
```bash
python backend/searching/run_chatbot.py --hybrid
```

**Available Commands:**
- Type your question to search
- `!help` - Show help message
- `!databases` - List available databases
- `!history` - Show conversation history
- `!clear` - Clear conversation history
- `!settings` - Change search settings
- `!mode` - Switch between vector/hybrid search modes
- `!quit` or `!exit` - Exit chatbot

**Example Session:**
```
> What are the privacy policies?
[Searching...]
[Answer generated]
> !databases
Available databases: company, aiid, standards
> !quit
```

---

## Agentic System Usage

**Starting the Agent:**
```bash
python backend/agentic/run_agent.py
```

**Example Goals:**
```
Goal: Find compliance gaps
[AGENT] Planning for goal: Find compliance gaps
[AGENT] Created plan with 2 steps:
  Step 1: Find GDPR articles not covered by company documents
  Step 2: Generate comprehensive compliance report

[AGENT] Executing plan...
[AGENT] Step 1/2: Find GDPR articles not covered...
✓ Step 1: Found 94 compliance gaps
[AGENT] Step 2/2: Generate comprehensive compliance report...
✓ Step 2: Report generated
```

---

## Sample Questions

### Vector Search Questions (Semantic Similarity)

**Privacy & Data Protection:**
- "What are the privacy policies?"
- "How does the company handle personal data?"
- "What information is collected about users?"
- "Explain data retention policies"

**GDPR Compliance:**
- "What are GDPR requirements for data processing?"
- "What rights do data subjects have?"
- "What are the legal bases for processing personal data?"

### Graph Traversal Questions (Hybrid Mode)

**Article & Clause Relationships:**
- "Find clauses addressing GDPR Article 5"
- "Which clauses cover GDPR Article 6?"
- "Show clauses that address GDPR Article 13"

**Incident Analysis:**
- "Find incidents related to data breaches"
- "What AIID incidents violate GDPR Article 5?"
- "Show incidents that relate to privacy violations"

**Compliance Gaps:**
- "What GDPR articles are not covered by company documents?"
- "Find gaps in compliance coverage"
- "Which articles have no corresponding clauses?"

### Hybrid Questions (Best Results)

**Compliance Analysis:**
- "What are the mismatches between company data and GDPR data?"
- "Find GDPR articles not covered by company documents"
- "Which clauses address GDPR Article 5?"
- "What compliance gaps exist?"
- "Compare company privacy policy with GDPR requirements"
- "How does the company's data handling align with GDPR?"
- "What compliance gaps exist in the company documents?"

**Incident & Risk Analysis:**
- "Find incidents related to data breaches"
- "What AIID incidents violate GDPR Article 5?"
- "Show incidents that relate to privacy violations"
- "Find relevant AIID incidents related to GDPR violations"
- "What incidents show patterns of non-compliance?"
- "Show incidents that relate to the company's privacy policy"

---

## Cypher Queries

### Basic Queries

**Find All Articles:**
```cypher
MATCH (a:Article)
RETURN a.id, a.title
LIMIT 10
```

**Find Clauses Addressing an Article:**
```cypher
MATCH (c:Clause)-[:ADDRESSES]->(a:Article {id: 'Article 5'})
RETURN c.text, a.title
```

**Find Compliance Gaps:**
```cypher
MATCH (a:Article)
WHERE NOT EXISTS {
  (c:Clause)-[:ADDRESSES]->(a)
}
RETURN a.id, a.title
```

**Find Incidents Violating Articles:**
```cypher
MATCH (i:Incident)-[:VIOLATES]->(a:Article)
RETURN i.title, a.id, a.title
LIMIT 10
```

### Advanced Queries

**Coverage Analysis:**
```cypher
MATCH (a:Article)
OPTIONAL MATCH (c:Clause)-[:ADDRESSES]->(a)
RETURN a.id, a.title, COUNT(c) as clause_count
ORDER BY clause_count ASC
```

**Document Coverage:**
```cypher
MATCH (d:Document)-[:COVERS]->(c:Clause)-[:ADDRESSES]->(a:Article)
RETURN d.name, COUNT(DISTINCT a) as articles_covered
```

**Related Incidents:**
```cypher
MATCH (i:Incident)-[:VIOLATES]->(a:Article)<-[:ADDRESSES]-(c:Clause)
RETURN i.title, a.id, c.text
LIMIT 20
```

---

## API Usage

### REST API Endpoints

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

**Vector Search:**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the privacy policies?",
    "db_names": ["company", "standards"],
    "top_k": 10
  }'
```

**Hybrid Search:**
```bash
curl -X POST http://localhost:5000/api/hybrid_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find compliance gaps",
    "top_k": 10
  }'
```

### Python API

```python
from backend.searching.query_engine import VectorQueryEngine

engine = VectorQueryEngine("/path/to/project")

# Simple search
results = engine.search(
    query="What are the privacy policies?",
    db_names=['company', 'standards'],
    top_k=10
)

# Full query with reranking
result = engine.query(
    query="Find compliance gaps",
    db_names=['standards'],
    top_k=10,
    rerank=True,
    contextualize=True
)
```

---

## Agentic Mode

### Using the Agent

```python
from backend.agentic import ComplianceAgent

agent = ComplianceAgent("/path/to/project")

# Execute a goal
result = agent.execute_goal("Find compliance gaps and generate report")

# Check results
print(f"Success: {result['success']}")
print(f"Steps: {result['steps_executed']}")
```

### Example Goals

- "Find compliance gaps"
- "Compare privacy policy with GDPR"
- "Generate full compliance report"
- "Find incidents related to data breaches"
- "Map all clauses to GDPR articles"

See [AGENTIC_SYSTEM.md](AGENTIC_SYSTEM.md) for complete agent documentation.

---

## Tips for Best Results

1. **Use Hybrid Mode** for:
   - Relationship queries (clauses → articles)
   - Compliance gap analysis
   - Incident analysis
   - Complex queries requiring both semantic and structural understanding

2. **Use Vector Mode** for:
   - General semantic searches
   - Content understanding
   - Concept exploration
   - Faster queries when relationships aren't needed

3. **Query Expansion**: Enabled by default - improves search quality by expanding queries with related terms

4. **Reranking**: Enabled by default - improves result relevance using LLM-based ranking

5. **Contextualization**: Enabled by default - generates comprehensive answers from search results

6. **Database Selection**: Choose specific databases (Company, AIID, Standards) to focus your search

7. **Similarity Threshold**: Lower threshold (0.0-0.3) for broader results, higher (0.5-0.8) for more precise matches

---

## Troubleshooting

### No Results Found
- Try lowering similarity threshold
- Check if databases are loaded (`!databases` in chatbot)
- Try different query phrasing
- Verify databases were built successfully (see [SETUP.md](SETUP.md#build-verification))

### Neo4j Connection Issues
- Check Neo4j is running: `make check-neo4j` or verify in Neo4j Desktop
- Verify connection settings in `.env`
- Try `make setup-neo4j-docker` for WSL users

### Import Errors
- Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `.\venv\Scripts\Activate.ps1` (Windows)
- Or use Makefile commands: `make run-agent`
- Ensure you're running from project root directory

For more detailed troubleshooting, see [SETUP.md](SETUP.md#troubleshooting).

---

For more information, see:
- [Setup Guide](SETUP.md) - Installation and configuration
- [Technical Documentation](TECHNICAL.md) - System architecture
- [Search Architecture](SEARCH_ARCHITECTURE.md) - Detailed search documentation
- [Agentic System Guide](AGENTIC_SYSTEM.md) - Agent documentation
- Main [README.md](../README.md) - Project overview

