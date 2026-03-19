# Data Processing Module

This module handles data extraction, processing, and conversion into formats suitable for indexing.

**Note**: Processed outputs are saved to `backend/processed/` (not inside this directory).

## Structure

```
backend/processing/
├── utils/              # Shared utilities
│   ├── pdf_utils.py    # PDF extraction
│   ├── text_utils.py   # Text chunking
│   ├── embedding_utils.py  # Embedding generation
│   ├── csv_utils.py    # CSV processing
│   └── save_utils.py   # Saving utilities
│
├── graph/              # Graph processing
│   ├── gdpr_to_graph.py
│   ├── company_to_graph.py
│   └── aiid_to_graph.py
│
└── vector/             # Vector processing
    ├── company_to_embeddings.py
    ├── standards_to_embeddings.py
    ├── aiid_to_embeddings.py
    └── regenerate_with_local_embeddings.py
```

## Output Location

All processed data is saved to `backend/processed/`:
- `backend/processed/vector/` - Processed embeddings (JSON files)
- `backend/processed/graph/` - Processed graph data (JSON files)

## Usage

### Graph Processing
```bash
python backend/processing/graph/gdpr_to_graph.py
python backend/processing/graph/company_to_graph.py
python backend/processing/graph/aiid_to_graph.py
```

### Vector Processing
```bash
python backend/processing/vector/company_to_embeddings.py
python backend/processing/vector/standards_to_embeddings.py
python backend/processing/vector/aiid_to_embeddings.py
```

### Regenerate with Local Embeddings
```bash
python backend/processing/vector/regenerate_with_local_embeddings.py
```

## Shared Utilities

All processing scripts use shared utilities from `utils/`:
- `extract_text_from_pdf()` - PDF text extraction
- `chunk_text()` - Text chunking
- `get_embeddings()` - Embedding generation
- `csv_to_text()` - CSV to text conversion
- `save_embeddings()` - Save embeddings to JSON
- `save_graph_json()` - Save graph data to JSON
