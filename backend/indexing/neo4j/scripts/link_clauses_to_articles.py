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
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from backend.indexing.neo4j.utils.neo4j_connection import Neo4jConnection
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables
DEFAULT_ADDRESSES_THRESHOLD = float(os.getenv("ADDRESSES_SIMILARITY_THRESHOLD", "0.45"))
# Note: MAX_LINKS_PER_CLAUSE and MAX_LINKS_PER_ARTICLE are now adaptive based on similarity quality

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def _calculate_adaptive_max_links(
    similarities: List[float],
    similarity_threshold: float,
    base_max_links: int = None,
    link_type: str = "clause"
) -> int:
    """
    Calculate adaptive maximum links based on similarity quality distribution.
    
    Adaptive strategy:
    1. Base on similarity distribution:
       - Many high-quality matches (>0.7): allow more links
       - Few high-quality matches: limit links
    2. Quality tiers:
       - Excellent (>0.75): up to 2x base
       - Good (>0.65): up to 1.5x base
       - Fair (>threshold): base amount
    3. Distribution-based:
       - If many matches above threshold, allow more links
       - If few matches, be conservative
    
    Args:
        similarities: List of similarity scores for potential links
        similarity_threshold: Minimum threshold for linking
        base_max_links: Base maximum links (if None, calculates from data)
        link_type: "clause" or "article" (for different base values)
        
    Returns:
        Adaptive maximum links value
    """
    if not similarities:
        return 3 if link_type == "clause" else 10
    
    # Filter similarities above threshold
    valid_similarities = [s for s in similarities if s >= similarity_threshold]
    
    if not valid_similarities:
        return 1  # At least allow 1 if above threshold
    
    # Calculate base max links if not provided
    if base_max_links is None:
        base_max_links = 3 if link_type == "clause" else 10
    
    # Count matches in quality tiers
    excellent_count = sum(1 for s in valid_similarities if s > 0.75)
    good_count = sum(1 for s in valid_similarities if s > 0.65)
    fair_count = len(valid_similarities)
    
    # Calculate adaptive max based on quality distribution
    if excellent_count >= 3:
        # Many excellent matches: allow more links
        adaptive_max = min(base_max_links * 2, fair_count)
    elif good_count >= 5:
        # Many good matches: allow moderate increase
        adaptive_max = min(int(base_max_links * 1.5), fair_count)
    elif fair_count >= 3:
        # Several fair matches: use base
        adaptive_max = base_max_links
    else:
        # Few matches: be conservative but allow what's available
        adaptive_max = min(base_max_links, fair_count)
    
    # Ensure reasonable bounds
    if link_type == "clause":
        return max(1, min(adaptive_max, 10))  # Clauses: 1-10 links
    else:
        return max(3, min(adaptive_max, 25))  # Articles: 3-25 links

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
                              If None, uses adaptive calculation based on similarity quality
        max_links_per_article: Maximum number of clauses to link per article
                              If None, uses adaptive calculation based on similarity quality
    """
    # Use defaults from .env if not provided
    similarity_threshold = similarity_threshold if similarity_threshold is not None else DEFAULT_ADDRESSES_THRESHOLD
    
    # Track if adaptive mode is enabled
    use_adaptive_clause = max_links_per_clause is None
    use_adaptive_article = max_links_per_article is None
    
    # Set initial values (will be refined adaptively)
    if max_links_per_clause is None:
        max_links_per_clause = 3  # Will be refined per clause
    if max_links_per_article is None:
        max_links_per_article = 10  # Will be refined per article
    conn = Neo4jConnection()
    
    if not conn.verify_connectivity():
        print("❌ Failed to connect to Neo4j")
        return
    
    print("="*80)
    print("LINKING CLAUSES TO ARTICLES")
    print("="*80)
    print(f"Similarity threshold: {similarity_threshold}")
    if use_adaptive_clause:
        print(f"Max links per clause: Adaptive (based on similarity quality)")
    else:
        print(f"Max links per clause: {max_links_per_clause} (fixed)")
    if use_adaptive_article:
        print(f"Max links per article: Adaptive (based on similarity quality)")
    else:
        print(f"Max links per article: {max_links_per_article} (fixed)")
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
    article_similarities_map = {}  # Track similarities per article for adaptive calculation
    
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
            
            # Track similarities per article for adaptive calculation
            if article_id not in article_similarities_map:
                article_similarities_map[article_id] = []
            article_similarities_map[article_id].append(similarity)
            
            if similarity >= similarity_threshold:
                similarities.append({
                    'article_id': article_id,
                    'article_title': article['article_title'],
                    'similarity': similarity
                })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Calculate adaptive max links per clause if enabled
        if use_adaptive_clause:
            clause_similarities = [s['similarity'] for s in similarities]
            adaptive_max_clause = _calculate_adaptive_max_links(
                clause_similarities,
                similarity_threshold,
                base_max_links=3,
                link_type="clause"
            )
            # Limit links per clause using adaptive value
            similarities = similarities[:adaptive_max_clause]
        else:
            # Use fixed limit
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
            
            # Calculate adaptive limits if enabled
            if use_adaptive_clause:
                # Get all similarities for this clause to calculate adaptive limit
                clause_all_sims = [s['similarity'] for s in similarities]
                adaptive_max_clause = _calculate_adaptive_max_links(
                    clause_all_sims,
                    similarity_threshold,
                    base_max_links=3,
                    link_type="clause"
                )
                if clause_links[clause_id] >= adaptive_max_clause:
                    total_links_skipped += 1
                    continue
            else:
                if clause_links[clause_id] >= max_links_per_clause:
                    total_links_skipped += 1
                    continue
            
            if use_adaptive_article:
                # Get all similarities for this article from the map
                article_similarities = article_similarities_map.get(article_id, [similarity])
                # Filter to only those above threshold
                article_valid_sims = [s for s in article_similarities if s >= similarity_threshold]
                adaptive_max_article = _calculate_adaptive_max_links(
                    article_valid_sims,
                    similarity_threshold,
                    base_max_links=10,
                    link_type="article"
                )
                if article_links[article_id] >= adaptive_max_article:
                    total_links_skipped += 1
                    continue
            else:
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
        help="Maximum articles to link per clause (default: None = adaptive based on similarity quality)"
    )
    parser.add_argument(
        "--max-links-per-article",
        type=int,
        default=None,
        help="Maximum clauses to link per article (default: None = adaptive based on similarity quality)"
    )
    
    args = parser.parse_args()
    
    link_clauses_to_articles(
        similarity_threshold=args.similarity_threshold,
        max_links_per_clause=args.max_links_per_clause,
        max_links_per_article=args.max_links_per_article
    )

