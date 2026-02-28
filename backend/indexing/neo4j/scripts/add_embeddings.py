"""
Add vector embeddings to Neo4j nodes for semantic search.
Uses existing embeddings from the vector database or generates new ones.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import utilities
from backend.indexing.neo4j.utils.neo4j_connection import Neo4jConnection
from backend.retrieval.utils.api_client import get_embedding_client
from backend.retrieval.utils.model_config import get_embedding_model
from dotenv import load_dotenv

load_dotenv()


class EmbeddingAdder:
    """Adds vector embeddings to Neo4j nodes."""
    
    def __init__(self, neo4j_conn: Neo4jConnection):
        """
        Initialize embedding adder.
        
        Args:
            neo4j_conn: Neo4jConnection instance
        """
        self.conn = neo4j_conn
        
        # Initialize API client (supports both OpenAI and xAI)
        self.client = get_embedding_client()
        self.embedding_model = get_embedding_model()
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def add_embedding_to_clause(self, clause_id: str, text: str) -> bool:
        """
        Add embedding to a Clause node.
        
        Args:
            clause_id: ID of the clause
            text: Text of the clause
        """
        embedding = self.get_embedding(text)
        if not embedding:
            return False
        
        query = """
        MATCH (c:Clause {id: $clause_id})
        SET c.embedding = $embedding
        RETURN c
        """
        
        try:
            self.conn.execute_write(query, {
                'clause_id': clause_id,
                'embedding': embedding
            })
            return True
        except Exception as e:
            print(f"Error adding embedding to clause {clause_id}: {e}")
            return False
    
    def add_embedding_to_article(self, article_id: str, text: str) -> bool:
        """
        Add embedding to an Article node.
        
        Args:
            article_id: ID of the article
            text: Text of the article (description)
        """
        embedding = self.get_embedding(text)
        if not embedding:
            return False
        
        query = """
        MATCH (a:Article {id: $article_id})
        SET a.embedding = $embedding
        RETURN a
        """
        
        try:
            self.conn.execute_write(query, {
                'article_id': article_id,
                'embedding': embedding
            })
            return True
        except Exception as e:
            print(f"Error adding embedding to article {article_id}: {e}")
            return False
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity based on word overlap.
        Returns a score between 0 and 1.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def match_chunk_to_clause(self, chunk_text: str, document_name: str = None, 
                              similarity_threshold: float = 0.3) -> Optional[str]:
        """
        Find the best matching clause for a chunk based on text similarity.
        
        Args:
            chunk_text: Text from the embedding chunk
            document_name: Optional document name to filter by
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            Clause ID if match found, None otherwise
        """
        # Get all clauses (optionally filtered by document)
        if document_name:
            query = """
            MATCH (c:Clause)
            WHERE c.document_name = $document_name AND c.embedding IS NULL
            RETURN c.id as id, c.text as text
            """
            clauses = self.conn.execute_query(query, {'document_name': document_name})
        else:
            query = """
            MATCH (c:Clause)
            WHERE c.embedding IS NULL
            RETURN c.id as id, c.text as text
            """
            clauses = self.conn.execute_query(query)
        
        if not clauses:
            return None
        
        # Find best match
        best_match = None
        best_score = 0.0
        
        for clause in clauses:
            similarity = self.calculate_text_similarity(chunk_text, clause['text'])
            if similarity > best_score and similarity >= similarity_threshold:
                best_score = similarity
                best_match = clause['id']
        
        return best_match
    
    def match_chunk_to_article(self, chunk_text: str, 
                               similarity_threshold: float = 0.2) -> Optional[str]:
        """
        Find the best matching article for a chunk based on text similarity.
        
        Args:
            chunk_text: Text from the embedding chunk
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            Article ID if match found, None otherwise
        """
        query = """
        MATCH (a:Article)
        WHERE a.embedding IS NULL
        RETURN a.id as id, a.title as title, a.description as description
        """
        articles = self.conn.execute_query(query)
        
        if not articles:
            return None
        
        # Find best match
        best_match = None
        best_score = 0.0
        
        for article in articles:
            # Combine title and description for matching
            article_text = f"{article.get('title', '')} {article.get('description', '')}"
            similarity = self.calculate_text_similarity(chunk_text, article_text)
            if similarity > best_score and similarity >= similarity_threshold:
                best_score = similarity
                best_match = article['id']
        
        return best_match
    
    def map_source_file_to_document_name(self, source_file: str) -> Optional[str]:
        """
        Map source file path to document name.
        
        Args:
            source_file: Path to source file
            
        Returns:
            Document name or None
        """
        source_lower = source_file.lower()
        
        # Map common file patterns to document names
        if 'privacy' in source_lower:
            return 'Privacy Policy'
        elif 'terms' in source_lower or 'tos' in source_lower:
            return 'Terms of Service'
        elif 'cookie' in source_lower:
            return 'Cookie Policy'
        elif 'gdpr' in source_lower:
            return None  # GDPR is articles, not documents
        else:
            return None
    
    def load_embeddings_from_json(self, json_path: str, node_type: str = "auto", 
                                  similarity_threshold: float = 0.3):
        """
        Load embeddings from existing JSON files and add to Neo4j nodes.
        Intelligently matches chunks to nodes based on text similarity.
        
        Args:
            json_path: Path to embeddings JSON file
            node_type: Type of node ('Clause', 'Article', 'auto' for auto-detect)
            similarity_threshold: Minimum similarity for matching (0-1)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get('chunks', [])
        metadata = data.get('metadata', {})
        source_file = metadata.get('source_file', '')
        source_name = metadata.get('source_name', '')
        
        print(f"Loading {len(chunks)} embeddings from {json_path}...")
        print(f"Source: {source_file or source_name}")
        
        # Auto-detect node type based on source file
        if node_type == "auto":
            if 'gdpr' in source_file.lower() or 'standards' in source_file.lower():
                node_type = "Article"
            else:
                node_type = "Clause"
        
        document_name = self.map_source_file_to_document_name(source_file or source_name)
        
        matched_count = 0
        skipped_count = 0
        
        for i, chunk_data in enumerate(chunks):
            chunk_text = chunk_data.get('text', '')
            embedding = chunk_data.get('embedding', [])
            
            if not embedding or not chunk_text:
                skipped_count += 1
                continue
            
            matched = False
            
            if node_type == "Clause":
                clause_id = self.match_chunk_to_clause(
                    chunk_text, 
                    document_name=document_name,
                    similarity_threshold=similarity_threshold
                )
                
                if clause_id:
                    query = """
                    MATCH (c:Clause {id: $clause_id})
                    SET c.embedding = $embedding
                    RETURN c.id as id
                    """
                    try:
                        self.conn.execute_write(query, {
                            'clause_id': clause_id,
                            'embedding': embedding
                        })
                        matched_count += 1
                        matched = True
                    except Exception as e:
                        print(f"Error adding embedding to clause {clause_id}: {e}")
            
            elif node_type == "Article":
                article_id = self.match_chunk_to_article(
                    chunk_text,
                    similarity_threshold=similarity_threshold
                )
                
                if article_id:
                    query = """
                    MATCH (a:Article {id: $article_id})
                    SET a.embedding = $embedding
                    RETURN a.id as id
                    """
                    try:
                        self.conn.execute_write(query, {
                            'article_id': article_id,
                            'embedding': embedding
                        })
                        matched_count += 1
                        matched = True
                    except Exception as e:
                        print(f"Error adding embedding to article {article_id}: {e}")
            
            if not matched:
                skipped_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks, matched: {matched_count}, skipped: {skipped_count}")
        
        print(f"\nCompleted: Matched {matched_count} embeddings, Skipped {skipped_count} chunks")
    
    def load_embeddings_from_directory(self, embeddings_dir: str, node_type: str = "auto",
                                       similarity_threshold: float = 0.3):
        """
        Load embeddings from all JSON files in a directory.
        
        Args:
            embeddings_dir: Directory containing *_embeddings.json files
            node_type: Type of node ('Clause', 'Article', 'auto')
            similarity_threshold: Minimum similarity for matching
        """
        embeddings_path = Path(embeddings_dir)
        json_files = list(embeddings_path.glob("*_embeddings.json"))
        
        if not json_files:
            print(f"No embedding JSON files found in {embeddings_dir}")
            return
        
        print(f"Found {len(json_files)} embedding files")
        
        for json_file in json_files:
            print(f"\nProcessing {json_file.name}...")
            self.load_embeddings_from_json(
                str(json_file),
                node_type=node_type,
                similarity_threshold=similarity_threshold
            )
    
    def add_embeddings_to_all_clauses(self, limit: Optional[int] = None):
        """
        Add embeddings to all Clause nodes that don't have them.
        
        Args:
            limit: Optional limit on number of clauses to process
        """
        query = """
        MATCH (c:Clause)
        WHERE c.embedding IS NULL
        RETURN c.id as id, c.text as text
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        clauses = self.conn.execute_query(query)
        
        print(f"Adding embeddings to {len(clauses)} clauses...")
        
        for i, clause in enumerate(clauses):
            clause_id = clause['id']
            clause_text = clause['text']
            
            self.add_embedding_to_clause(clause_id, clause_text)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} clauses...")
    
    def add_embeddings_to_all_articles(self):
        """Add embeddings to all Article nodes that don't have them."""
        query = """
        MATCH (a:Article)
        WHERE a.embedding IS NULL
        RETURN a.id as id, a.description as description
        """
        
        articles = self.conn.execute_query(query)
        
        print(f"Adding embeddings to {len(articles)} articles...")
        
        for article in articles:
            article_id = article['id']
            description = article['description']
            
            # Combine title and description for embedding
            text = f"{article.get('title', '')} {description}"
            self.add_embedding_to_article(article_id, text)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add embeddings to Neo4j nodes")
    parser.add_argument("--clauses", action="store_true", help="Generate embeddings for clauses (requires OpenAI API)")
    parser.add_argument("--articles", action="store_true", help="Generate embeddings for articles (requires OpenAI API)")
    parser.add_argument("--all", action="store_true", help="Generate embeddings for all nodes (requires OpenAI API)")
    parser.add_argument("--limit", type=int, help="Limit number of nodes to process")
    
    # Options for loading from existing FAISS embeddings
    parser.add_argument("--json-file", type=str, help="Load embeddings from a single JSON file")
    parser.add_argument("--json-dir", type=str, help="Load embeddings from all JSON files in directory")
    parser.add_argument("--node-type", type=str, choices=['Clause', 'Article', 'auto'], 
                       default='auto', help="Node type for JSON loading (default: auto-detect)")
    parser.add_argument("--similarity-threshold", type=float, default=0.3,
                       help="Minimum similarity threshold for matching (0-1, default: 0.3)")
    
    args = parser.parse_args()
    
    with Neo4jConnection() as conn:
        if not conn.verify_connectivity():
            print("Failed to connect to Neo4j.")
            exit(1)
        
        adder = EmbeddingAdder(conn)
        
        # Load from existing FAISS JSON files
        if args.json_file:
            print(f"Loading embeddings from JSON file: {args.json_file}")
            adder.load_embeddings_from_json(
                args.json_file,
                node_type=args.node_type,
                similarity_threshold=args.similarity_threshold
            )
        elif args.json_dir:
            print(f"Loading embeddings from directory: {args.json_dir}")
            adder.load_embeddings_from_directory(
                args.json_dir,
                node_type=args.node_type,
                similarity_threshold=args.similarity_threshold
            )
        # Generate new embeddings (requires OpenAI API)
        elif args.all or args.clauses:
            print("Generating new embeddings for clauses (requires OpenAI API)...")
            adder.add_embeddings_to_all_clauses(limit=args.limit)
        elif args.articles:
            print("Generating new embeddings for articles (requires OpenAI API)...")
            adder.add_embeddings_to_all_articles()
        else:
            print("Please specify one of:")
            print("  --json-file PATH      : Load from single JSON file")
            print("  --json-dir PATH       : Load from directory of JSON files")
            print("  --clauses             : Generate embeddings for clauses")
            print("  --articles            : Generate embeddings for articles")
            print("\nExample:")
            print("  python add_embeddings.py --json-dir backend/processed/vector/company")
            print("  python add_embeddings.py --json-file backend/processed/vector/standards/gdpr_embeddings.json")

