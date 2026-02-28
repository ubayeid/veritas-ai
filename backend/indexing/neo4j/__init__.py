"""
Neo4j Knowledge Graph Builder
Builds enhanced knowledge graph combining GDPR, Facebook documents, and AIID incidents.
"""

from .utils.neo4j_connection import Neo4jConnection
from .builders.gdpr_builder import GDPRBuilder, create_sample_gdpr_data
from .builders.facebook_documents_builder import FacebookDocumentsBuilder
from .builders.aiid_incidents_builder import AIIDIncidentsBuilder

# Import queries from searching module (moved for better organization)
import sys
import importlib.util
from pathlib import Path

# Get the path to the retrieval module
_searching_path = Path(__file__).parent.parent.parent / "retrieval" / "utils" / "neo4j_queries.py"

# Use importlib to load the module (works for both runtime and IDE)
_spec = importlib.util.spec_from_file_location("neo4j_queries", _searching_path)
_neo4j_queries_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_neo4j_queries_module)
KnowledgeGraphQueries = _neo4j_queries_module.KnowledgeGraphQueries

__all__ = [
    'Neo4jConnection',
    'GDPRBuilder',
    'create_sample_gdpr_data',
    'FacebookDocumentsBuilder',
    'AIIDIncidentsBuilder',
    'KnowledgeGraphQueries'
]

