"""
Unified script to build FAISS indexes for any data source.
Replaces the individual *_to_faiss_database.py scripts.
"""

import argparse
from pathlib import Path
from typing import Optional

from utils import build_faiss_index, build_cosine_index


def main():
    """Main function to build FAISS indexes."""
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index from processed embeddings"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["company", "aiid", "standards"],
        help="Data source: company, aiid, or standards"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        help="Directory containing embedding JSON files (defaults to backend/processed/vector/{source})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for FAISS index (defaults to backend/indexing/faiss/{source})"
    )
    parser.add_argument(
        "--index-name",
        type=str,
        help="Name for the index (defaults to {source}_faiss_index)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["L2", "IP", "cosine"],
        default="cosine",
        help="Distance metric: L2, IP, or cosine (default: cosine)"
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Set default paths
    embeddings_dir = args.embeddings_dir or str(project_root / "backend" / "processed" / "vector" / args.source)
    output_dir = args.output_dir or str(project_root / "backend" / "indexing" / "faiss" / "output")
    index_name = args.index_name or f"{args.source}_faiss_index"
    
    # Build index based on metric
    if args.metric == "cosine":
        print(f"Building cosine similarity index for {args.source}...")
        build_cosine_index(embeddings_dir, output_dir, index_name)
    else:
        print(f"Building {args.metric} index for {args.source}...")
        build_faiss_index(embeddings_dir, output_dir, index_name, args.metric)


if __name__ == "__main__":
    main()
