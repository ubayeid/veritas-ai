# FAISS Indexing

This directory contains scripts and utilities for building FAISS vector indexes from processed embeddings.

## Structure

```
backend/indexing/faiss/
├── utils/                    # Shared utilities
│   ├── __init__.py
│   └── faiss_builder.py     # Core FAISS building functions
│
├── build_faiss_index.py     # Unified build script
│
└── output/                  # Output directory (all indexes stored here)
```

## Usage

### Recommended: Unified Build Script

```bash
# Build cosine similarity index (recommended for embeddings)
python backend/indexing/faiss/build_index.py --source company
python backend/indexing/faiss/build_index.py --source aiid
python backend/indexing/faiss/build_index.py --source standards

# Build with different metrics
python backend/indexing/faiss/build_index.py --source company --metric L2
python backend/indexing/faiss/build_index.py --source company --metric IP

# Custom paths
python backend/indexing/faiss/build_index.py \
    --source company \
    --embeddings-dir custom/path/to/embeddings \
    --output-dir custom/path/to/output \
    --index-name custom_index_name
```

### Programmatic Usage

```python
from backend.indexing.faiss.utils import build_cosine_index, build_faiss_index

# Build cosine similarity index (recommended)
index, metadata = build_cosine_index(
    embeddings_dir="backend/processed/vector/company",
    output_dir="backend/indexing/faiss/output",
    index_name="company_faiss_index"
)

# Build L2 index
index, metadata = build_faiss_index(
    embeddings_dir="backend/processed/vector/company",
    output_dir="backend/indexing/faiss/output",
    index_name="company_faiss_index",
    metric="L2"
)
```

## Shared Utilities

The `utils/faiss_builder.py` module provides:

- `load_embeddings_from_json()` - Load embeddings from JSON files
- `build_faiss_index()` - Build FAISS index with specified metric (L2 or IP)
- `build_cosine_index()` - Build cosine similarity index (normalized IP)
- `save_faiss_index()` - Save index, metadata, and summary
- `load_faiss_index()` - Load saved index and metadata

## Output Files

Each index directory contains:
- `{index_name}.index` - FAISS index file
- `{index_name}_metadata.pkl` - Pickled metadata list
- `{index_name}_summary.json` - JSON summary with statistics

## Input Data

Reads from `backend/processed/vector/{source}/` containing `*_embeddings.json` files.
