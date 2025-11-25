"""
Automatically link Company Clauses to GDPR Articles using semantic similarity.
Creates ADDRESSES relationships between clauses and articles based on embedding similarity.
"""

import sys
import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "searching"))
from neo4j_connection import Neo4jConnection
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables
DEFAULT_ADDRESSES_THRESHOLD = float(os.getenv("ADDRESSES_SIMILARITY_THRESHOLD", "0.45"))
DEFAULT_MAX_LINKS_PER_CLAUSE = int(os.getenv("MAX_LINKS_PER_CLAUSE", "3"))
DEFAULT_MAX_LINKS_PER_ARTICLE = int(os.getenv("MAX_LINKS_PER_ARTICLE", "10"))

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def link_clauses_to_articles(
    similarity_threshold: float = None,
    max_links_per_clause: int = None,
    max_links_per_article: int = None
):
    """
    Link clauses to articles based on embedding similarity.
    
    Args:
        similarity_threshold: Minimum cosine similarity to create relationship (0-1)
                              Defaults to ADDRESSES_SIMILARITY_THRESHOLD from .env
        max_links_per_clause: Maximum number of articles to link per clause
                              Defaults to MAX_LINKS_PER_CLAUSE from .env
        max_links_per_article: Maximum number of clauses to link per article
                              Defaults to MAX_LINKS_PER_ARTICLE from .env
    """
    # Use defaults from .env if not provided
    similarity_threshold = similarity_threshold if similarity_threshold is not None else DEFAULT_ADDRESSES_THRESHOLD
    max_links_per_clause = max_links_per_clause if max_links_per_clause is not None else DEFAULT_MAX_LINKS_PER_CLAUSE
    max_links_per_article = max_links_per_article if max_links_per_article is not None else DEFAULT_MAX_LINKS_PER_ARTICLE
    conn = Neo4jConnection()
    
    if not conn.verify_connectivity():
        print("❌ Failed to connect to Neo4j")
        return
    
    print("="*80)
    print("LINKING CLAUSES TO ARTICLES")
    print("="*80)
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Max links per clause: {max_links_per_clause}")
    print(f"Max links per article: {max_links_per_article}")
    print("="*80)
    print()
    
    # Get all clauses with embeddings
    print("Loading clauses with embeddings...")
    clause_query = """
    MATCH (c:Clause)
    WHERE c.embedding IS NOT NULL
    RETURN c.id as clause_id, 
           c.text as clause_text,
           c.document_name as document_name,
           c.embedding as embedding
    """
    clauses = conn.execute_query(clause_query)
    print(f"Found {len(clauses)} clauses with embeddings")
    
    if not clauses:
        print("⚠ No clauses with embeddings found!")
        print("  Run add_embeddings.py first to add embeddings to clauses.")
        return
    
    # Get all articles with embeddings
    print("Loading articles with embeddings...")
    article_query = """
    MATCH (a:Article)
    WHERE a.embedding IS NOT NULL
    RETURN a.id as article_id,
           a.title as article_title,
           a.description as description,
           a.embedding as embedding
    """
    articles = conn.execute_query(article_query)
    print(f"Found {len(articles)} articles with embeddings")
    
    if not articles:
        print("⚠ No articles with embeddings found!")
        print("  Run add_embeddings.py first to add embeddings to articles.")
        return
    
    print()
    print("Calculating similarities and creating relationships...")
    print("-"*80)
    
    # Track statistics
    total_links_created = 0
    total_links_skipped = 0
    clause_links = {}  # Track links per clause
    article_links = {}  # Track links per article
    
    # Convert embeddings to numpy arrays
    clause_embeddings = {}
    for clause in clauses:
        if clause.get('embedding'):
            clause_embeddings[clause['clause_id']] = np.array(clause['embedding'])
    
    article_embeddings = {}
    for article in articles:
        if article.get('embedding'):
            article_embeddings[article['article_id']] = np.array(article['embedding'])
    
    # Calculate similarities and create relationships
    all_similarities = []  # For debugging
    
    for clause in clauses:
        clause_id = clause['clause_id']
        clause_emb = clause_embeddings.get(clause_id)
        
        if clause_emb is None:
            continue
        
        # Calculate similarity with all articles
        similarities = []
        for article in articles:
            article_id = article['article_id']
            article_emb = article_embeddings.get(article_id)
            
            if article_emb is None:
                continue
            
            similarity = cosine_similarity(clause_emb, article_emb)
            all_similarities.append(similarity)  # Track for debugging
            
            if similarity >= similarity_threshold:
                similarities.append({
                    'article_id': article_id,
                    'article_title': article['article_title'],
                    'similarity': similarity
                })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Limit links per clause
        similarities = similarities[:max_links_per_clause]
        
        # Create relationships
        for sim_data in similarities:
            article_id = sim_data['article_id']
            similarity = sim_data['similarity']
            
            # Check limits
            if clause_id not in clause_links:
                clause_links[clause_id] = 0
            if article_id not in article_links:
                article_links[article_id] = 0
            
            if clause_links[clause_id] >= max_links_per_clause:
                total_links_skipped += 1
                continue
            
            if article_links[article_id] >= max_links_per_article:
                total_links_skipped += 1
                continue
            
            # Create relationship
            link_query = """
            MATCH (c:Clause {id: $clause_id})
            MATCH (a:Article {id: $article_id})
            MERGE (c)-[r:ADDRESSES]->(a)
            SET r.similarity_score = $similarity
            RETURN c.id as clause_id, a.id as article_id, r.similarity_score as similarity
            """
            
            try:
                result = conn.execute_write(link_query, {
                    'clause_id': clause_id,
                    'article_id': article_id,
                    'similarity': float(similarity)
                })
                
                if result:
                    total_links_created += 1
                    clause_links[clause_id] = clause_links.get(clause_id, 0) + 1
                    article_links[article_id] = article_links.get(article_id, 0) + 1
                    
                    if total_links_created % 10 == 0:
                        print(f"  Created {total_links_created} relationships...")
            except Exception as e:
                print(f"  ⚠ Error linking {clause_id} to {article_id}: {e}")
                total_links_skipped += 1
    
    print("-"*80)
    print()
    
    # Debug: Show similarity statistics
    if all_similarities:
        print("="*80)
        print("SIMILARITY STATISTICS")
        print("="*80)
        print(f"Total similarity calculations: {len(all_similarities)}")
        print(f"Max similarity: {max(all_similarities):.4f}")
        print(f"Min similarity: {min(all_similarities):.4f}")
        print(f"Mean similarity: {np.mean(all_similarities):.4f}")
        print(f"Median similarity: {np.median(all_similarities):.4f}")
        print(f"Similarities >= {similarity_threshold}: {sum(1 for s in all_similarities if s >= similarity_threshold)}")
        print(f"Similarities >= 0.6: {sum(1 for s in all_similarities if s >= 0.6)}")
        print(f"Similarities >= 0.5: {sum(1 for s in all_similarities if s >= 0.5)}")
        print("="*80)
        print()
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"[OK] Relationships created: {total_links_created}")
    print(f"[WARNING] Relationships skipped: {total_links_skipped}")
    print(f"[INFO] Clauses with links: {len([c for c in clause_links.values() if c > 0])}")
    print(f"[INFO] Articles with links: {len([a for a in article_links.values() if a > 0])}")
    print("="*80)
    
    if total_links_created == 0 and all_similarities:
        print()
        print("[WARNING] No relationships created!")
        max_sim = max(all_similarities) if all_similarities else 0
        print(f"   Max similarity found: {max_sim:.4f}")
        print(f"   Current threshold: {similarity_threshold}")
        if max_sim < similarity_threshold:
            print(f"   Suggested: --similarity-threshold {max_sim:.2f}")
        print()
    
    # Verify results
    print()
    print("Verifying relationships...")
    verify_query = """
    MATCH (c:Clause)-[r:ADDRESSES]->(a:Article)
    RETURN count(r) as total_relationships,
           count(DISTINCT c) as clauses_with_links,
           count(DISTINCT a) as articles_with_links
    """
    verify_result = conn.execute_query(verify_query)
    if verify_result:
        stats = verify_result[0]
        print(f"[OK] Total ADDRESSES relationships: {stats['total_relationships']}")
        print(f"[OK] Clauses with relationships: {stats['clauses_with_links']}")
        print(f"[OK] Articles with relationships: {stats['articles_with_links']}")
    
    conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Link clauses to articles using embeddings")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help=f"Minimum similarity to create relationship (0-1, default: {DEFAULT_ADDRESSES_THRESHOLD} from env or 0.45)"
    )
    parser.add_argument(
        "--max-links-per-clause",
        type=int,
        default=None,
        help=f"Maximum articles to link per clause (default: {DEFAULT_MAX_LINKS_PER_CLAUSE} from env or 3)"
    )
    parser.add_argument(
        "--max-links-per-article",
        type=int,
        default=None,
        help=f"Maximum clauses to link per article (default: {DEFAULT_MAX_LINKS_PER_ARTICLE} from env or 10)"
    )
    
    args = parser.parse_args()
    
    link_clauses_to_articles(
        similarity_threshold=args.similarity_threshold,
        max_links_per_clause=args.max_links_per_clause,
        max_links_per_article=args.max_links_per_article
    )

