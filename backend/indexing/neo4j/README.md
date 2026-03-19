# Neo4j Knowledge Graph Builder

This module builds an enhanced Knowledge Graph in Neo4j that combines:
1. **GDPR structure** (102 articles, sub-obligations, topics)
2. **Company documents** (3 PDFs: Privacy Policy, Terms of Service, Cookie Policy) → ~292 clauses
3. **AIID incidents** (from incidents.csv only; classification CSVs used for embeddings, not graph)

## Architecture

```
┌─────────────────┐
│   GDPR Articles │
│  SubObligations │
│     Topics      │
└────────┬────────┘
         │ HAS_TOPIC
         │ HAS_SUB_OBLIGATION
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│    Documents     │─────▶│     Clauses      │
│  (Facebook)      │COVERS│  (Split text)    │
└─────────────────┘      └────────┬─────────┘
                                  │ ADDRESSES
                                  ▼
                           ┌─────────────────┐
                           │  GDPR Articles   │
                           └────────┬────────┘
                                    ▲
                                    │ VIOLATES
                           ┌────────┴─────────┐
                           │    Incidents    │
                           │   (AIID DB)     │
                           └─────────────────┘
```

## Quick Start

### 1. Prerequisites

- **Neo4j installed and running**
  - Download from [neo4j.com](https://neo4j.com/download/)
  - Or use Docker: `docker run -p 7474:7474 -p 7687:7687 neo4j:latest`
  - Default credentials: `neo4j` / `password` (change on first login)

- **Python dependencies**
  ```bash
  pip install neo4j>=5.0.0
  ```

- **Environment variables** (create `.env` file in project root)
  ```env
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=your_password
  OPENAI_API_KEY=your_key  # Optional, for embeddings
  ```

### 2. Verify Connection

```bash
python -c "from backend.indexing.neo4j import Neo4jConnection; Neo4jConnection().verify_connectivity()"
```

### 3. Build the Knowledge Graph

```bash
# Build everything (uses default paths)
python backend/indexing/neo4j/build_knowledge_graph.py

# Clear existing data and rebuild
python backend/indexing/neo4j/build_knowledge_graph.py --clear
```

## Directory Structure

Your project follows a clean separation of concerns:

```
comp_rag/
├── backend/
│   ├── processing/
│   │   ├── vector/                    # Step 1: Convert to embeddings
│   │   │   ├── company_to_embeddings.py
│   │   │   ├── standards_to_embeddings.py
│   │   │   └── aiid_to_embeddings.py
│   │   └── graph/                     # Step 1: Convert to graph JSON
│   │       ├── company_to_graph.py
│   │       ├── gdpr_to_graph.py
│   │       └── aiid_to_graph.py
│   │
│   └── indexing/
│       ├── faiss/                     # Step 2: Build FAISS indexes
│       │   ├── build_faiss_index.py
│       │   └── output/
│       └── neo4j/                     # Step 2: Build Neo4j graph
│           ├── utils/
│           │   └── neo4j_connection.py
│           ├── builders/
│           │   ├── gdpr_builder.py
│           │   ├── facebook_documents_builder.py
│           │   └── aiid_incidents_builder.py
│           ├── scripts/
│           │   ├── add_embeddings.py
│           │   └── link_clauses_to_articles.py
│           └── build_knowledge_graph.py
│
└── data/
    ├── company/                       # Raw: Company PDFs
    ├── standards/                     # Raw: GDPR PDF
    └── aiid/                          # Raw: AIID CSV
```

## Complete Workflow

### Step 1: Process Data to Graph JSON

```bash
# Process company documents → graph JSON
python backend/processing/graph/company_to_graph.py

# Process GDPR → graph JSON
python backend/processing/graph/gdpr_to_graph.py

# Process AIID → graph JSON
python backend/processing/graph/aiid_to_graph.py
```

### Step 2: Build Neo4j Graph

```bash
# Build complete graph from JSON files
python backend/indexing/neo4j/build_knowledge_graph.py
```

### Step 3: Connect FAISS Embeddings (Optional)

```bash
# Connect company document embeddings → Clauses
python backend/indexing/neo4j/scripts/add_embeddings.py \
    --json-dir backend/processed/vector/company \
    --node-type Clause \
    --similarity-threshold 0.3

# Connect standards (GDPR) embeddings → Articles
python backend/indexing/neo4j/scripts/add_embeddings.py \
    --json-dir backend/processed/vector/standards \
    --node-type Article \
    --similarity-threshold 0.2
```

## Usage

### Build Individual Components

#### GDPR Structure

```python
from neo4j_connection import Neo4jConnection
from gdpr_builder import GDPRBuilder

with Neo4jConnection() as conn:
    builder = GDPRBuilder(conn)
    
    # Option 1: Load from JSON (recommended)
    builder.build_from_json("backend/processed/graph/gdpr_graph.json")
    
    # Option 2: Parse from PDF
    builder.build_from_pdf("data/standards/gdpr.pdf")
```

#### Facebook Documents

```python
from facebook_documents_builder import FacebookDocumentsBuilder

with Neo4jConnection() as conn:
    builder = FacebookDocumentsBuilder(conn)
    
    # Option 1: Load from JSON (recommended)
    builder.build_from_json("backend/processed/graph/company_graph.json")
    
    # Option 2: Process PDFs directly
    builder.process_directory("data/company")
```

#### AIID Incidents

```python
from aiid_incidents_builder import AIIDIncidentsBuilder

with Neo4jConnection() as conn:
    builder = AIIDIncidentsBuilder(conn)
    
    # Option 1: Load from JSON (recommended)
    builder.build_from_json("backend/processed/graph/aiid_graph.json")
    
    # Option 2: Process CSV directly
    builder.process_incidents_csv("data/aiid/incidents.csv", limit=100)
```

### Query the Graph

```python
from backend.indexing.neo4j import Neo4jConnection
from backend.retrieval.neo4j_queries import KnowledgeGraphQueries

with Neo4jConnection() as conn:
    queries = KnowledgeGraphQueries(conn)
    
    # GDPR coverage analysis
    results = queries.gdpr_coverage()
    
    # Document gap analysis
    gaps = queries.document_gap_analysis()
    
    # AIID risk mapping
    risks = queries.aiid_risk_mapping()
    
    # Find non-compliant clauses
    non_compliant = queries.find_non_compliant_clauses(article_id="Art5")
```

### Example Queries

See `backend/retrieval/neo4j_queries.py` for a full collection of query methods.

## Node Types

### Article
- **Label**: `Article`
- **Properties**: `id`, `title`, `description`, `keywords`, `number`, `embedding` (optional)
- **Relationships**: 
  - `HAS_SUB_OBLIGATION` → SubObligation
  - `HAS_TOPIC` → Topic

### SubObligation
- **Label**: `SubObligation`
- **Properties**: `id`, `description`, `keywords`
- **Relationships**: 
  - `HAS_SUB_OBLIGATION` ← Article

### Topic
- **Label**: `Topic`
- **Properties**: `name`
- **Relationships**: 
  - `HAS_TOPIC` ← Article

### Document
- **Label**: `Document`
- **Properties**: `name`, `source_url`
- **Relationships**: 
  - `COVERS` → Clause

### Clause
- **Label**: `Clause`
- **Properties**: `id`, `text`, `document_name`, `section`, `keywords`, `compliance_status`, `embedding` (optional)
- **Relationships**: 
  - `COVERS` ← Document
  - `ADDRESSES` → Article

### Incident
- **Label**: `Incident`
- **Properties**: `id`, `title`, `description`, `system_type`, `risk_type`, `source`, `date`
- **Relationships**: 
  - `VIOLATES` → Article

## Embeddings Connection

### How It Works

The `scripts/add_embeddings.py` script connects FAISS embeddings to Neo4j nodes:

```
┌─────────────────────┐
│  FAISS JSON Files   │
│  (Embeddings)       │
│                     │
│  chunks: [         │
│    {text, embedding}│
│  ]                  │
└──────────┬──────────┘
           │
           │ Load & Match
           ▼
┌─────────────────────┐
│  Neo4j Nodes        │
│                     │
│  Clause {           │
│    text: "...",     │
│    embedding: [...] │  ← Stored here
│  }                  │
└─────────────────────┘
```

### Matching Process

1. **Load JSON File**: Reads embeddings from `backend/processed/vector/`
2. **Detect Node Type**: Auto-detects based on source file path
3. **Note**: Embeddings are stored on nodes but NOT used for vector search in Neo4j
4. **Purpose**: Embeddings are used for creating ADDRESSES relationships via similarity matching
   - `standards/` or `gdpr` → Article nodes
   - `company/` → Clause nodes
3. **Match Chunk to Node**: Uses word overlap similarity
4. **Store Embedding**: Sets `embedding` property on matched node

### Usage

```bash
# Load from directory (auto-detects node type)
python backend/indexing/neo4j/scripts/add_embeddings.py \
    --json-dir backend/processed/vector/company

# Specify node type explicitly
python backend/indexing/neo4j/scripts/add_embeddings.py \
    --json-dir backend/processed/vector/standards \
    --node-type Article

# Adjust similarity threshold (default: 0.3)
python backend/indexing/neo4j/scripts/add_embeddings.py \
    --json-dir backend/processed/vector/company \
    --similarity-threshold 0.2
```

### Parameters

- `--json-file PATH`: Load from single JSON file
- `--json-dir PATH`: Load from directory of JSON files
- `--node-type TYPE`: `Clause`, `Article`, or `auto` (default: `auto`)
- `--similarity-threshold FLOAT`: Minimum similarity (0-1, default: 0.3)

### Verification

```cypher
// Count nodes with embeddings
MATCH (n)
WHERE n.embedding IS NOT NULL
RETURN labels(n)[0] as label, count(n) as count

// View sample clause with embedding
MATCH (c:Clause)
WHERE c.embedding IS NOT NULL
RETURN c.id, c.text, size(c.embedding) as embedding_size
LIMIT 5
```

## Common Cypher Queries

### Find all clauses addressing Art5

```cypher
MATCH (c:Clause)-[:ADDRESSES]->(a:Article {id:'Art5'})
WHERE c.compliance_status='non-compliant'
RETURN c.text, c.document_name
```

### GDPR Coverage

```cypher
MATCH (a:Article)<-[:ADDRESSES]-(c:Clause)
RETURN a.id, a.title, collect(c.text)
```

### Document Gap Analysis

```cypher
MATCH (a:Article)
WHERE NOT (a)<-[:ADDRESSES]-(:Clause)
RETURN a.id, a.title
```

### AIID Risk Mapping

```cypher
MATCH (i:Incident)-[:VIOLATES]->(a:Article)
RETURN i.id, i.description, a.id, a.title
```

### Compliance Status by Document

```cypher
MATCH (d:Document)-[:COVERS]->(c:Clause)
RETURN d.name,
       count(c) as total_clauses,
       count(CASE WHEN c.compliance_status='compliant' THEN 1 END) as compliant,
       count(CASE WHEN c.compliance_status='non-compliant' THEN 1 END) as non_compliant
```

## Compliance Status

Clauses can have a `compliance_status` property with values:
- `compliant`: Clause fully addresses the GDPR requirement
- `partial`: Clause partially addresses the requirement
- `non-compliant`: Clause does not meet the requirement

Set compliance status:

```python
builder = FacebookDocumentsBuilder(conn)
builder.set_clause_compliance_status("clause_id", "non-compliant")
```

## Troubleshooting

### Connection Issues

1. Verify Neo4j is running: `neo4j status`
2. Check connection settings in `.env`
3. Test connection: `python -c "from backend.indexing.neo4j import Neo4jConnection; Neo4jConnection().verify_connectivity()"`

### Import Errors

Make sure you're running scripts from the project root or adjust import paths:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
```

### Memory Issues

For large datasets:
- Process in batches (use `limit` parameter)
- Clear database before rebuilding (`--clear` flag)
- Increase Neo4j heap memory in `neo4j.conf`

### Embeddings Not Matching

1. **Check text overlap**: Compare chunk text vs node text
2. **Lower threshold**: Try `--similarity-threshold 0.2`
3. **Verify node creation**: Ensure `build_knowledge_graph.py` ran successfully
4. **Check source mapping**: Verify document names match

## Directory Structure

```
backend/indexing/neo4j/
├── utils/
│   ├── __init__.py
│   └── neo4j_connection.py      # Connection handler for Neo4j
├── builders/
│   ├── __init__.py
│   ├── gdpr_builder.py          # Creates GDPR Article/SubObligation/Topic nodes
│   ├── facebook_documents_builder.py  # Creates Document/Clause nodes
│   └── aiid_incidents_builder.py # Creates Incident nodes
├── scripts/
│   ├── add_embeddings.py        # Connects FAISS embeddings to Neo4j nodes
│   └── link_clauses_to_articles.py  # Links clauses to articles via similarity
├── build_knowledge_graph.py     # Main orchestrator script
├── __init__.py
└── README.md
```

## Files Overview

- **`utils/neo4j_connection.py`**: Connection handler for Neo4j
- **`build_knowledge_graph.py`**: Main orchestrator script
- **`builders/gdpr_builder.py`**: Creates GDPR Article/SubObligation/Topic nodes
- **`builders/facebook_documents_builder.py`**: Creates Document/Clause nodes (processes 3 PDFs → 3 Document nodes)
- **`builders/aiid_incidents_builder.py`**: Creates Incident nodes
- **`scripts/add_embeddings.py`**: Connects FAISS embeddings to Neo4j nodes
- **`scripts/link_clauses_to_articles.py`**: Automatically links clauses to articles using semantic similarity

## References

- [Neo4j Python Driver Documentation](https://neo4j.com/docs/python-manual/current/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [GDPR Official Text](https://gdpr-info.eu/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
