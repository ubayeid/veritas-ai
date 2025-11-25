# Graph Data Processing

This directory contains scripts to convert raw data into graph-structured JSON files for Neo4j import.

## Directory Structure

Following the same pattern as vector processing:

```
backend/data_processing/
├── vector/          # Converts data → embeddings → backend/data_processing/processed/vector/
└── graph/           # Converts data → graph JSON → backend/data_processing/processed/graph/
```

## Workflow

1. **Data Processing** (`backend/data_processing/graph/`)
   - Extract structured data from PDFs/CSVs
   - Save as JSON files in `backend/data_processing/processed/graph/`

2. **Graph Building** (`backend/building_database/neo4j/`)
   - Load JSON files from `backend/data_processing/processed/graph/`
   - Build Neo4j knowledge graph

## Scripts

### `gdpr_to_graph.py`
Converts GDPR PDF into graph-structured JSON.

**Input**: `data/standards/gdpr.pdf`  
**Output**: `backend/data_processing/processed/graph/gdpr_graph.json`

**Usage**:
```bash
python backend/data_processing/graph/gdpr_to_graph.py
```

### `company_to_graph.py`
Converts company PDFs into graph-structured JSON.

**Input**: `data/company/*.pdf` (3 PDFs: Privacy Policy, Terms of Service, Cookie Policy)  
**Output**: `backend/data_processing/processed/graph/company_graph.json`  
**Result**: 3 Document nodes, ~292 Clause nodes

**Usage**:
```bash
python backend/data_processing/graph/company_to_graph.py
```

### `aiid_to_graph.py`
Converts AIID CSV into graph-structured JSON.

**Input**: `data/aiid/incidents.csv` (only this CSV goes to graph)  
**Output**: `backend/data_processing/processed/graph/aiid_graph.json`  
**Note**: Classification CSVs (CSETv0, CSETv1, GMF, MIT) are processed for embeddings only, not graph nodes

**Usage**:
```bash
python backend/data_processing/graph/aiid_to_graph.py
```

## Complete Workflow

```bash
# Step 1: Process data into graph JSON files
python backend/data_processing/graph/gdpr_to_graph.py
python backend/data_processing/graph/company_to_graph.py
python backend/data_processing/graph/aiid_to_graph.py

# Step 2: Build Neo4j graph from JSON files
python backend/building_database/neo4j/build_knowledge_graph.py
```

## Output Format

### GDPR Graph JSON
```json
{
  "metadata": {
    "source": "GDPR",
    "num_articles": 3,
    "version": "1.0"
  },
  "articles": [
    {
      "id": "Art5",
      "title": "...",
      "description": "...",
      "keywords": [...],
      "sub_obligations": [...],
      "topics": [...]
    }
  ]
}
```

### Company Graph JSON
```json
{
  "metadata": {
    "source": "Company Documents",
    "num_documents": 3,
    "num_clauses": 292,
    "version": "1.0"
  },
  "documents": [
    {
      "name": "Privacy Policy",
      "source_url": "...",
      "source_file": "..."
    }
  ],
  "clauses": [
    {
      "id": "Privacy Policy_clause_1",
      "text": "...",
      "document_name": "Privacy Policy",
      "keywords": [...]
    }
  ]
}
```

### AIID Graph JSON
```json
{
  "metadata": {
    "source": "AIID Database",
    "num_incidents": 100,
    "version": "1.0"
  },
  "incidents": [
    {
      "id": "AIID_123",
      "title": "...",
      "description": "...",
      "system_type": "...",
      "risk_type": "...",
      "date": "..."
    }
  ]
}
```

## Parallel Structure with Vector Processing

| Step | Vector Processing | Graph Processing |
|------|------------------|-----------------|
| **Input** | PDFs/CSVs | PDFs/CSVs |
| **Processing** | `backend/data_processing/vector/` | `backend/data_processing/graph/` |
| **Output** | `backend/data_processing/processed/vector/*.json` | `backend/data_processing/processed/graph/*.json` |
| **Building** | `backend/building_database/faiss/` | `backend/building_database/neo4j/` |
| **Final DB** | FAISS indexes | Neo4j graph |

This ensures consistency across your data processing pipeline!

