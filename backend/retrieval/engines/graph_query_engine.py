"""
Graph Query Engine: Standalone Neo4j graph traversal search
Provides relationship-based queries for compliance and regulatory data.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from difflib import SequenceMatcher
import numpy as np

# Add paths for imports
from ..utils.neo4j_queries import KnowledgeGraphQueries
from backend.indexing.neo4j.utils.neo4j_connection import Neo4jConnection
from .query_engine import get_query_embedding
from ..utils.api_client import get_api_client
from ..utils.model_config import get_llm_model, LLM_TEMPERATURE, LLM_MAX_TOKENS

load_dotenv()

# Configuration from environment variables
GRAPH_SCORE_RESULTS = os.getenv("GRAPH_SCORE_RESULTS", "true").lower() == "true"
GRAPH_SIMILARITY_THRESHOLD_MODE = os.getenv("GRAPH_SIMILARITY_THRESHOLD_MODE", "adaptive").lower()
GRAPH_SIMILARITY_THRESHOLD_FIXED = float(os.getenv("GRAPH_SIMILARITY_THRESHOLD_FIXED", "0.5"))
GRAPH_SIMILARITY_THRESHOLD_PERCENTILE = float(os.getenv("GRAPH_SIMILARITY_THRESHOLD_PERCENTILE", "0.3"))
# Enable/disable LLM-based intent classification (can be disabled for faster queries)
GRAPH_USE_INTENT_CLASSIFICATION = os.getenv("GRAPH_USE_INTENT_CLASSIFICATION", "true").lower() == "true"


class GraphQueryEngine:
    """
    Graph Query Engine for Neo4j knowledge graph traversal.
    
    Provides standalone graph search functionality for:
    - Article-to-clause relationships
    - Incident-to-article violations
    - Compliance gap analysis
    - Risk mapping
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize graph query engine.
        
        Args:
            base_dir: Base directory of the project
        """
        self.base_dir = Path(base_dir)
        
        # Initialize Neo4j connection
        self.neo4j_conn = Neo4jConnection()
        if not self.neo4j_conn.verify_connectivity():
            raise ConnectionError("Failed to connect to Neo4j. Please ensure Neo4j is running.")
        
        self.graph_queries = KnowledgeGraphQueries(self.neo4j_conn)
        
        # Initialize API client for LLM-based intent classification
        self.api_client = None
        if GRAPH_USE_INTENT_CLASSIFICATION:
            try:
                self.api_client = get_api_client()
            except Exception as e:
                print(f"Warning: Could not initialize API client for intent classification: {e}")
                print("  Intent classification will be disabled. Set GRAPH_USE_INTENT_CLASSIFICATION=false to suppress this warning.")
        
        # Cache for relationship validation
        self._relationship_cache = {
            'clauses_by_article': {},  # article_id -> bool (has clauses)
            'incidents_by_article': {},  # article_id -> bool (has incidents)
            'topics': None  # List of all topics
        }
    
    def search(self, query: str, top_k: int = None, score_results: bool = None) -> List[Dict[str, Any]]:
        """
        Perform graph traversal search based on query patterns.
        
        Args:
            query: User query string
            top_k: Optional limit on number of results (if None, returns all)
            score_results: Whether to score results by semantic similarity (default: GRAPH_SCORE_RESULTS from .env)
            
        Returns:
            List of results from graph traversal, sorted by relevance if scored
        """
        if score_results is None:
            score_results = GRAPH_SCORE_RESULTS
        
        # Recommendation 3: Query Intent Classification using LLM
        query_intent = self._classify_query_intent(query) if GRAPH_USE_INTENT_CLASSIFICATION else None
        
        query_lower = query.lower()
        results = []
        
        # Use intent classification if available, otherwise fall back to pattern matching
        if query_intent:
            intent_type = query_intent.get('intent_type', 'general')
            extracted = query_intent.get('extracted_entities', {})
            
            # Extract article IDs from intent classification
            article_ids_from_intent = [f"Art{aid}" for aid in extracted.get('article_ids', [])]
            
            # Use intent classification to guide query processing
            is_gap_query = intent_type == 'gap_analysis'
            asks_for_clauses = 'clauses' in extracted.get('query_asks_for', []) or intent_type == 'clause_query'
            asks_for_incidents = 'incidents' in extracted.get('query_asks_for', []) or intent_type == 'incident_query'
            detected_topics = extracted.get('topics', [])
            
            # Extract article IDs (prefer intent classification, fall back to regex)
            import re
            article_matches = re.findall(r'art(?:icle)?\s*(\d+)', query_lower)
            article_ids = article_ids_from_intent if article_ids_from_intent else [f"Art{match}" for match in article_matches]
            
            if query_intent.get('confidence', 0.0) > 0.7:
                print(f"  -> Intent classified as: {intent_type} (confidence: {query_intent.get('confidence', 0.0):.2f})")
        else:
            # Fallback to pattern matching if intent classification unavailable
            # Check for gap/mismatch queries FIRST (before any other processing)
            is_gap_query = ('gap' in query_lower or 'missing' in query_lower or 
                           'not covered' in query_lower or 'not covered by' in query_lower or
                           'mismatch' in query_lower or 'compare' in query_lower or 
                           'difference' in query_lower or 'uncovered' in query_lower)
            
            # Extract ALL article IDs if mentioned (support multiple articles)
            import re
            article_matches = re.findall(r'art(?:icle)?\s*(\d+)', query_lower)
            article_ids = [f"Art{match}" for match in article_matches]
            
            # Initialize these variables for pattern matching fallback
            asks_for_clauses = None  # Will be determined later
            asks_for_incidents = None
            detected_topics = []
            
            # Set asks_for_incidents from pattern matching if not set by intent
            if 'incident' in query_lower or 'violating' in query_lower or 'violates' in query_lower:
                asks_for_incidents = True
        
        # Check for topic-based queries (e.g., "data subject rights", "privacy", "consent")
        # Only if NOT a gap query (gap queries take priority)
        detected_topic = None
        if not is_gap_query:
            # Use topics from intent classification if available
            if detected_topics:
                detected_topic = detected_topics[0]  # Use first detected topic
            else:
                # Fallback to keyword-based detection
                topic_keywords = {
                    'data subject rights': ['data subject', 'rights', 'access', 'rectification', 'erasure', 'portability'],
                    'privacy': ['privacy', 'data protection', 'personal data'],
                    'consent': ['consent', 'lawful basis'],
                    'security': ['security', 'data breach', 'integrity'],
                    'transparency': ['transparency', 'information', 'notice'],
                    'accountability': ['accountability', 'responsibility', 'compliance']
                }
                
                for topic, keywords in topic_keywords.items():
                    if any(kw in query_lower for kw in keywords):
                        detected_topic = topic
                        break
        
        # Handle incident queries without article IDs FIRST (before topic-based article search)
        # If query asks for incidents but no specific article, search incidents directly
        # Check asks_for_incidents explicitly (could be True, False, or None)
        if (asks_for_incidents is True or 
            (asks_for_incidents is None and ('incident' in query_lower or 'violating' in query_lower))) and not article_ids:
            # Query asks for incidents but no specific article - search all incidents
            print(f"  -> Searching for incidents...")
            risks = self.graph_queries.aiid_risk_mapping()
            
            # Filter by topic if detected
            if detected_topic:
                print(f"  -> Filtering incidents by topic: {detected_topic}")
                topic_keywords = detected_topic.split()
                filtered_risks = []
                for risk in risks:
                    # Check if topic keywords appear in incident title, risk_type, or violated articles
                    incident_text = (risk.get('incident_title', '') + ' ' + 
                                    risk.get('risk_type', '') + ' ' + 
                                    ' '.join(risk.get('violated_articles', []))).lower()
                    if any(kw.lower() in incident_text for kw in topic_keywords):
                        filtered_risks.append(risk)
                risks = filtered_risks[:20]  # Limit to top 20
                print(f"  -> Found {len(risks)} incidents matching topic")
            else:
                risks = risks[:20]  # Limit to top 20
            
            for risk in risks:
                results.append({
                    'id': risk['incident_id'],
                    'text': risk.get('incident_title', ''),
                    'title': risk.get('incident_title', ''),
                    'risk_type': risk['risk_type'],
                    'violated_articles': risk['violated_articles'],
                    'type': 'incident',
                    'source': 'graph_traversal',
                    'relationship': 'VIOLATES'
                })
        
        # Handle topic-based queries (but skip if query asks for incidents - incidents already handled above)
        elif detected_topic and not article_ids:
            print(f"  -> Detected topic: {detected_topic}")
            # Recommendation 2: Use fuzzy and semantic matching for topics
            try:
                # First, get all topics to find matching ones
                all_topics = self.graph_queries.topic_analysis()
                matching_topic_names = []
                topic_scores = []  # Store similarity scores
                
                # Search for topics using fuzzy and semantic matching
                for topic_data in all_topics:
                    topic_name = topic_data.get('topic_name', '')
                    if not topic_name:
                        continue
                    
                    # Fuzzy matching
                    fuzzy_score = self._fuzzy_match_topic(detected_topic, topic_name)
                    
                    # Semantic matching (if fuzzy score is decent)
                    semantic_score = 0.0
                    if fuzzy_score > 0.3:  # Only do semantic matching if fuzzy match is reasonable
                        semantic_score = self._semantic_match_topic(detected_topic, topic_name)
                    
                    # Combined score (weighted: 40% fuzzy, 60% semantic)
                    combined_score = (fuzzy_score * 0.4) + (semantic_score * 0.6)
                    
                    # Threshold: accept if combined score > 0.5 or fuzzy > 0.7
                    if combined_score > 0.5 or fuzzy_score > 0.7:
                        matching_topic_names.append(topic_name)
                        topic_scores.append((topic_name, combined_score))
                
                # Sort by score (highest first)
                topic_scores.sort(key=lambda x: x[1], reverse=True)
                matching_topic_names = [name for name, _ in topic_scores]
                
                if matching_topic_names:
                    # Get articles for each matching topic
                    all_article_ids = []
                    for topic_name in matching_topic_names:
                        # Query articles for this specific topic
                        topic_results = self.graph_queries.topic_analysis(topic_name=topic_name)
                        if topic_results:
                            article_ids_list = topic_results[0].get('article_ids', [])
                            if isinstance(article_ids_list, list):
                                all_article_ids.extend(article_ids_list)
                    
                    # Get unique article IDs
                    article_ids = list(set(all_article_ids))
                    print(f"  -> Found {len(article_ids)} articles related to '{detected_topic}'")
                    
                    # Return articles with their information
                    for article_id in article_ids[:20]:  # Limit to top 20
                        # Get article details
                        article_info = self.graph_queries.find_article_by_id(article_id)
                        if article_info:
                            results.append({
                                'id': article_id,
                                'text': article_info.get('description', article_info.get('title', '')),
                                'title': article_info.get('title', article_id),
                                'type': 'article',
                                'source': 'graph_traversal',
                                'topic': detected_topic,
                                'relationship': 'HAS_TOPIC'
                            })
                else:
                    # Fallback: search for articles containing topic keywords in title/description
                    print(f"  -> No exact topic match found, searching articles by keywords...")
                    # Try to find articles by searching their titles/descriptions for topic keywords
                    try:
                        # Get all articles and filter by keywords
                        # Use a direct Cypher query through the connection
                        all_articles_query = """
                        MATCH (a:Article)
                        RETURN a.id as article_id,
                               a.title as article_title,
                               a.description as description
                        ORDER BY a.id
                        """
                        # Access the connection through graph_queries
                        all_articles = self.graph_queries.conn.execute_query(all_articles_query)
                        
                        # Filter articles that contain topic keywords
                        topic_keywords_list = detected_topic.split()
                        matching_articles = []
                        for article in all_articles:
                            title = (article.get('article_title', '') or '').lower()
                            desc = (article.get('description', '') or '').lower()
                            text = f"{title} {desc}"
                            
                            # Check if any keyword matches
                            if any(kw in text for kw in topic_keywords_list):
                                matching_articles.append(article)
                        
                        if matching_articles:
                            print(f"  -> Found {len(matching_articles)} articles matching topic keywords")
                            for article in matching_articles[:20]:  # Limit to top 20
                                results.append({
                                    'id': article.get('article_id'),
                                    'text': article.get('description', article.get('article_title', '')),
                                    'title': article.get('article_title', article.get('article_id')),
                                    'type': 'article',
                                    'source': 'graph_traversal',
                                    'topic': detected_topic,
                                    'relationship': 'KEYWORD_MATCH'
                                })
                        else:
                            print(f"  -> No articles found matching topic keywords")
                    except Exception as e:
                        import traceback
                        verbose = os.getenv("VERBOSE", "false").lower() == "true"
                        if verbose:
                            print(f"  -> Error in keyword search: {e}")
                            traceback.print_exc()
                    # This will also be handled by the fallback semantic search below
            except Exception as e:
                import traceback
                verbose = os.getenv("VERBOSE", "false").lower() == "true"
                if verbose:
                    print(f"  -> Error in topic search: {e}")
                    traceback.print_exc()
                # Continue to fallback
        
        # Handle article-based queries (single or multiple articles)
        if article_ids:
            print(f"  -> Found article ID(s): {', '.join(article_ids)}")
            
            # Check if query specifically asks for clauses or incidents
            # Use intent classification if available, otherwise use pattern matching
            if asks_for_clauses is None:
                asks_for_clauses = 'clause' in query_lower or 'addressing' in query_lower or 'address' in query_lower
            if asks_for_incidents is None:
                asks_for_incidents = 'incident' in query_lower or 'violating' in query_lower or 'violates' in query_lower
            
            # If query doesn't specify, return both; otherwise prioritize what's asked
            if not asks_for_clauses and not asks_for_incidents:
                asks_for_clauses = True  # Default to clauses if not specified
                asks_for_incidents = True
            
            # Handle multiple articles: find clauses that address ALL mentioned articles
            if len(article_ids) > 1 and asks_for_clauses:
                print(f"  -> Searching for clauses addressing ALL articles: {', '.join(article_ids)}...")
                # Find clauses that address all mentioned articles
                clauses = self.graph_queries.find_clauses_by_multiple_articles(article_ids)
                print(f"  -> Found {len(clauses)} clauses addressing all articles")
                
                if clauses:
                    for clause in clauses:
                        results.append({
                            'id': clause['clause_id'],
                            'text': clause['clause_text'],
                            'document_name': clause['document_name'],
                            'article_ids': article_ids,
                            'type': 'clause',
                            'source': 'graph_traversal',
                            'relationship': 'ADDRESSES'
                        })
                else:
                    # Fallback: find clauses addressing ANY of the articles
                    print(f"  -> No clauses address all articles. Searching for clauses addressing ANY article...")
                    all_clauses = []
                    for article_id in article_ids:
                        article_clauses = self.graph_queries.find_clauses_by_article(article_id)
                        for clause in article_clauses:
                            # Avoid duplicates
                            clause_id = clause.get('clause_id')
                            if not any(r.get('id') == clause_id for r in all_clauses):
                                clause['addressed_article'] = article_id
                                all_clauses.append(clause)
                    
                    print(f"  -> Found {len(all_clauses)} clauses addressing any of the articles")
                    for clause in all_clauses:
                        results.append({
                            'id': clause['clause_id'],
                            'text': clause['clause_text'],
                            'document_name': clause['document_name'],
                            'article_id': clause.get('addressed_article'),
                            'article_ids': article_ids,
                            'type': 'clause',
                            'source': 'graph_traversal',
                            'relationship': 'ADDRESSES',
                            'note': f"Addresses {clause.get('addressed_article')} (requested: {', '.join(article_ids)})"
                        })
            elif asks_for_clauses:
                # Single article: find clauses addressing this article
                article_id = article_ids[0]
                # Recommendation 4: Validate relationship exists before querying
                if not self._validate_relationship_exists('ADDRESSES', article_id):
                    print(f"  -> Warning: Article {article_id} has no clauses linked (ADDRESSES relationship missing)")
                    print(f"  -> This may indicate missing relationships in the graph database.")
                    clauses = []  # Initialize empty list to avoid undefined variable error
                else:
                    print(f"  -> Searching for clauses addressing {article_id}...")
                    clauses = self.graph_queries.find_clauses_by_article(article_id)
                    print(f"  -> Found {len(clauses)} clauses")
                
                # Only process clauses if we have any
                if clauses:
                    for clause in clauses:
                        results.append({
                            'id': clause['clause_id'],
                            'text': clause['clause_text'],
                            'document_name': clause['document_name'],
                            'article_id': article_id,
                            'type': 'clause',
                            'source': 'graph_traversal',
                            'relationship': 'ADDRESSES'
                        })
                else:
                    print(f"  -> No clauses found for {article_id}")
            else:
                print(f"  -> Skipping clauses (query asks for incidents)")
            
            # Find incidents violating articles (handle multiple articles)
            if asks_for_incidents:
                if len(article_ids) > 1:
                    # For multiple articles, find incidents that violate ANY of them
                    print(f"  -> Searching for incidents violating any of: {', '.join(article_ids)}...")
                    all_incidents = []
                    for article_id in article_ids:
                        incidents = self.graph_queries.find_incidents_by_article(article_id)
                        all_incidents.extend(incidents)
                    # Remove duplicates based on incident_id
                    seen_ids = set()
                    unique_incidents = []
                    for incident in all_incidents:
                        inc_id = incident.get('incident_id')
                        if inc_id and inc_id not in seen_ids:
                            seen_ids.add(inc_id)
                            unique_incidents.append(incident)
                    print(f"  -> Found {len(unique_incidents)} unique incidents")
                    for incident in unique_incidents:
                        results.append({
                            'id': incident['incident_id'],
                            'text': incident['description'],
                            'title': incident['incident_title'],
                            'risk_type': incident['risk_type'],
                            'article_ids': article_ids,
                            'type': 'incident',
                            'source': 'graph_traversal',
                            'relationship': 'VIOLATES'
                        })
                else:
                    # Single article
                    article_id = article_ids[0]
                    # Recommendation 4: Validate relationship exists before querying
                    if not self._validate_relationship_exists('VIOLATES', article_id):
                        print(f"  -> Warning: Article {article_id} has no incidents linked (VIOLATES relationship missing)")
                        print(f"  -> This may indicate missing relationships in the graph database.")
                        incidents = []  # Initialize empty list to avoid undefined variable error
                    else:
                        print(f"  -> Searching for incidents violating {article_id}...")
                        incidents = self.graph_queries.find_incidents_by_article(article_id)
                        print(f"  -> Found {len(incidents)} incidents")
                    
                    # Only process incidents if we have any
                    if incidents:
                        for incident in incidents:
                            results.append({
                                'id': incident['incident_id'],
                                'text': incident['description'],
                                'title': incident['incident_title'],
                                'risk_type': incident['risk_type'],
                                'article_id': article_id,
                                'type': 'incident',
                                'source': 'graph_traversal',
                                'relationship': 'VIOLATES'
                            })
                    else:
                        print(f"  -> No incidents found for {article_id}")
            else:
                print(f"  -> Skipping incidents (query asks for clauses)")
        
        # Compliance gap and mismatch queries
        # Process gap queries FIRST (before article/topic queries) to ensure they take priority
        # Only process if we haven't already processed article-specific queries
        if is_gap_query and not article_ids:
            print(f"  -> Detected gap/mismatch query")
            # Use comprehensive mismatch analysis for better results
            if 'mismatch' in query_lower or 'compare' in query_lower or 'difference' in query_lower or 'not covered' in query_lower:
                mismatch_data = self.graph_queries.comprehensive_mismatch_analysis()
                
                # Add gaps (articles NOT covered by company documents)
                for gap in mismatch_data['gaps'][:15]:  # Top 15 gaps
                    results.append({
                        'id': gap['article_id'],
                        'text': gap.get('description', gap.get('article_title', '')),
                        'title': gap['article_title'],
                        'type': 'gap',
                        'source': 'graph_traversal',
                        'coverage_status': 'not_covered',
                        'analysis_type': 'mismatch'
                    })
                
                # Add coverage info (articles that ARE covered)
                for cov in mismatch_data['coverage'][:15]:  # Top 15 covered
                    clause_texts = [clause.get('clause_text', '')[:200] for clause in cov.get('clauses', [])[:2]]
                    results.append({
                        'id': cov['article_id'],
                        'text': cov.get('description', cov.get('article_title', '')),
                        'title': cov['article_title'],
                        'type': 'coverage',
                        'source': 'graph_traversal',
                        'coverage_status': 'covered',
                        'clause_count': cov.get('clause_count', 0),
                        'clause_examples': clause_texts,
                        'analysis_type': 'mismatch'
                    })
                
                # Add summary statistics
                results.append({
                    'id': 'summary',
                    'text': f"Coverage Analysis: {mismatch_data['covered_articles']} articles covered, {mismatch_data['uncovered_articles']} articles not covered ({mismatch_data['coverage_percentage']}% coverage)",
                    'title': 'Coverage Summary',
                    'type': 'summary',
                    'source': 'graph_traversal',
                    'coverage_percentage': mismatch_data['coverage_percentage'],
                    'covered_count': mismatch_data['covered_articles'],
                    'uncovered_count': mismatch_data['uncovered_articles'],
                    'analysis_type': 'mismatch'
                })
            else:
                # Simple gap analysis for "gap" or "missing" queries
                gaps = self.graph_queries.document_gap_analysis()
                for gap in gaps:
                    results.append({
                        'id': gap['article_id'],
                        'text': gap.get('description', ''),
                        'title': gap['article_title'],
                        'type': 'gap',
                        'source': 'graph_traversal',
                        'coverage_status': 'not_covered'
                    })
        
        # Risk mapping queries (for general risk analysis, not incident-specific)
        if 'risk' in query_lower and not asks_for_incidents and not article_ids:
            # General risk mapping (not incident-specific)
            risks = self.graph_queries.aiid_risk_mapping()
            for risk in risks[:10]:  # Limit to top 10
                results.append({
                    'id': risk['incident_id'],
                    'text': risk.get('incident_title', ''),
                    'risk_type': risk['risk_type'],
                    'violated_articles': risk['violated_articles'],
                    'type': 'risk',
                    'source': 'graph_traversal'
                })
        
        # Fallback: If no results found and query doesn't match specific patterns,
        # try to get clauses and articles using semantic similarity
        if not results:
            # Get some clauses and articles for general queries
            try:
                # Get top clauses (by coverage/popularity)
                coverage_data = self.graph_queries.gdpr_coverage()
                for cov in coverage_data[:top_k if top_k else 20]:
                    clause_texts = cov.get('clauses', [])
                    if clause_texts:
                        # Use first clause text as representative
                        results.append({
                            'id': f"{cov['article_id']}_clause",
                            'text': clause_texts[0] if isinstance(clause_texts, list) else str(clause_texts)[:500],
                            'document_name': 'Company Documents',
                            'article_id': cov['article_id'],
                            'type': 'clause',
                            'source': 'graph_traversal',
                            'relationship': 'ADDRESSES'
                        })
            except Exception as e:
                print(f"  -> Warning: Fallback graph search failed: {e}")
        
        # Score results by semantic similarity to query
        if score_results:
            results = self._score_graph_results(query, results)
        
        # Apply top_k limit if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def _score_graph_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Score graph results by semantic similarity to query.
        Uses batch embedding generation for efficiency.
        
        Args:
            query: User query
            results: List of graph results
            
        Returns:
            List of results with similarity scores, sorted by score
        """
        if not results:
            return results
        
        try:
            # Import embedding functions
            from ..utils.local_embeddings import (
                is_local_embeddings_enabled,
                generate_local_embeddings_batch
            )
            import os
            import numpy as np
            
            # Get query embedding
            query_embedding = get_query_embedding(query)
            if query_embedding is None or query_embedding.size == 0:
                # If embedding fails, return results with default score
                for result in results:
                    result['similarity'] = 0.0
                return results
            
            # Collect all texts to embed (batch processing)
            texts_to_embed = []
            result_indices = []  # Track which result each text belongs to
            
            for idx, result in enumerate(results):
                text_to_score = result.get('text', '') or result.get('title', '') or result.get('description', '')
                if text_to_score:
                    texts_to_embed.append(text_to_score[:500])  # Limit text length
                    result_indices.append(idx)
                else:
                    result['similarity'] = 0.0
            
            if not texts_to_embed:
                return results
            
            # Batch embed all texts at once (much faster!)
            use_local = is_local_embeddings_enabled()
            if use_local and os.getenv("USE_LOCAL_EMBEDDINGS", "auto").lower() in ["true", "auto"]:
                # Use batch local embeddings
                result_embeddings = generate_local_embeddings_batch(texts_to_embed)
            else:
                # Fallback: try local batch even if not primary
                result_embeddings = generate_local_embeddings_batch(texts_to_embed)
            
            if result_embeddings is not None and result_embeddings.size > 0:
                # Calculate similarities for all results at once
                query_emb = query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding
                for i, result_idx in enumerate(result_indices):
                    result_emb = result_embeddings[i]
                    similarity = float(np.dot(query_emb, result_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(result_emb)
                    ))
                    results[result_idx]['similarity'] = similarity
            else:
                # Fallback to one-by-one if batch failed
                print(f"Warning: Batch embedding failed, falling back to one-by-one (slower)")
                for i, result_idx in enumerate(result_indices):
                    try:
                        result_embedding = get_query_embedding(texts_to_embed[i])
                        if result_embedding is not None and result_embedding.size > 0:
                            query_emb = query_embedding[0] if len(query_embedding.shape) > 1 else query_embedding
                            res_emb = result_embedding[0] if len(result_embedding.shape) > 1 else result_embedding
                            similarity = float(np.dot(query_emb, res_emb) / (
                                np.linalg.norm(query_emb) * np.linalg.norm(res_emb)
                            ))
                            results[result_idx]['similarity'] = similarity
                        else:
                            results[result_idx]['similarity'] = 0.0
                    except Exception as e:
                        print(f"Warning: Failed to score result: {e}")
                        results[result_idx]['similarity'] = 0.0
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
            
        except Exception as e:
            print(f"Warning: Graph result scoring failed: {e}")
            # Return results with default scores
            for result in results:
                result['similarity'] = 0.0
        
        return results
    
    def _filter_low_quality_graph_results(
        self, 
        graph_results: List[Dict], 
        vector_similarities: List[float]
    ) -> List[Dict]:
        """
        Filter graph results based on similarity thresholds.
        
        Args:
            graph_results: Graph search results
            vector_similarities: Similarity scores from vector search (for adaptive threshold)
            
        Returns:
            Filtered graph results
        """
        if not graph_results:
            return []
        
        # Determine threshold
        import numpy as np
        if GRAPH_SIMILARITY_THRESHOLD_MODE == "adaptive":
            # Use median of vector similarities as threshold
            if vector_similarities:
                threshold = float(np.median(vector_similarities))
            else:
                threshold = 0.5  # Fallback
        elif GRAPH_SIMILARITY_THRESHOLD_MODE == "fixed":
            threshold = GRAPH_SIMILARITY_THRESHOLD_FIXED
        elif GRAPH_SIMILARITY_THRESHOLD_MODE == "percentile":
            # Filter bottom X percentile
            if graph_results:
                similarities = [r.get('similarity', 0.0) for r in graph_results]
                if similarities:
                    threshold = np.percentile(similarities, GRAPH_SIMILARITY_THRESHOLD_PERCENTILE * 100)
                else:
                    threshold = 0.0
            else:
                threshold = 0.0
        else:
            threshold = 0.5  # Default
        
        # Filter results
        filtered = [
            r for r in graph_results
            if r.get('similarity', 0.0) >= threshold
        ]
        
        return filtered
    
    def _classify_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Recommendation 3: Use LLM to classify query intent before pattern matching.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with intent classification:
            {
                'intent_type': 'article_lookup' | 'gap_analysis' | 'incident_query' | 'topic_query' | 'general',
                'confidence': float,
                'extracted_entities': {'article_ids': [], 'topics': [], ...}
            }
        """
        if not self.api_client:
            return None
        
        try:
            prompt = f"""Classify the following query about GDPR compliance and extract relevant entities.

Query: "{query}"

Respond with a JSON object containing:
- intent_type: one of "article_lookup", "gap_analysis", "incident_query", "topic_query", "clause_query", "general"
- confidence: float between 0.0 and 1.0
- extracted_entities: object with:
  - article_ids: list of article numbers mentioned (e.g., [6, 12, 13])
  - topics: list of topics mentioned (e.g., ["data subject rights", "privacy"])
  - query_asks_for: list of what the query is asking for (e.g., ["clauses", "incidents", "articles"])

Examples:
- "What clauses address Article 12?" -> {{"intent_type": "clause_query", "extracted_entities": {{"article_ids": [12], "query_asks_for": ["clauses"]}}}}
- "Which articles are not covered?" -> {{"intent_type": "gap_analysis", "extracted_entities": {{}}}}
- "What incidents violate Article 6?" -> {{"intent_type": "incident_query", "extracted_entities": {{"article_ids": [6], "query_asks_for": ["incidents"]}}}}
- "Show GDPR articles about data subject rights" -> {{"intent_type": "topic_query", "extracted_entities": {{"topics": ["data subject rights"], "query_asks_for": ["articles"]}}}}

Return only valid JSON, no other text."""

            response = self.api_client.chat.completions.create(
                model=get_llm_model(),
                messages=[
                    {"role": "system", "content": "You are a query classification expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            import json
            result_text = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()
            
            intent_data = json.loads(result_text)
            return intent_data
            
        except Exception as e:
            verbose = os.getenv("VERBOSE", "false").lower() == "true"
            if verbose:
                print(f"Warning: Query intent classification failed: {e}")
            return None
    
    def _fuzzy_match_topic(self, query_topic: str, topic_name: str, threshold: float = 0.6) -> float:
        """
        Recommendation 2: Fuzzy matching for topic names using string similarity.
        
        Args:
            query_topic: Topic from user query
            topic_name: Topic name from database
            threshold: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        query_lower = query_topic.lower()
        topic_lower = topic_name.lower()
        
        # Exact match
        if query_lower == topic_lower:
            return 1.0
        
        # Substring match
        if query_lower in topic_lower or topic_lower in query_lower:
            return 0.9
        
        # Use SequenceMatcher for fuzzy matching
        similarity = SequenceMatcher(None, query_lower, topic_lower).ratio()
        
        # Also check word-level similarity
        query_words = set(query_lower.split())
        topic_words = set(topic_lower.split())
        if query_words and topic_words:
            word_overlap = len(query_words & topic_words) / len(query_words | topic_words)
            similarity = max(similarity, word_overlap * 0.8)
        
        return similarity
    
    def _semantic_match_topic(self, query_topic: str, topic_name: str) -> float:
        """
        Recommendation 2: Semantic similarity matching for topics using embeddings.
        
        Args:
            query_topic: Topic from user query
            topic_name: Topic name from database
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            query_emb = get_query_embedding(query_topic)
            topic_emb = get_query_embedding(topic_name)
            
            if query_emb is None or topic_emb is None:
                return 0.0
            
            # Normalize embeddings
            query_emb = query_emb[0] if len(query_emb.shape) > 1 else query_emb
            topic_emb = topic_emb[0] if len(topic_emb.shape) > 1 else topic_emb
            
            # Calculate cosine similarity
            similarity = float(np.dot(query_emb, topic_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(topic_emb)
            ))
            
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            verbose = os.getenv("VERBOSE", "false").lower() == "true"
            if verbose:
                print(f"Warning: Semantic topic matching failed: {e}")
            return 0.0
    
    def _validate_relationship_exists(self, relationship_type: str, source_id: str, target_id: Optional[str] = None) -> bool:
        """
        Recommendation 4: Validate that relationships exist before querying.
        
        Args:
            relationship_type: Type of relationship ('ADDRESSES', 'VIOLATES', 'HAS_TOPIC')
            source_id: Source node ID (e.g., article_id, clause_id)
            target_id: Optional target node ID (for specific relationship validation)
            
        Returns:
            True if relationship exists, False otherwise
        """
        try:
            if relationship_type == 'ADDRESSES':
                # Check if article has clauses
                if source_id in self._relationship_cache['clauses_by_article']:
                    return self._relationship_cache['clauses_by_article'][source_id]
                
                query = """
                MATCH (c:Clause)-[:ADDRESSES]->(a:Article {id: $article_id})
                RETURN count(c) as clause_count
                LIMIT 1
                """
                result = self.graph_queries.conn.execute_query(query, {'article_id': source_id})
                has_clauses = result[0].get('clause_count', 0) > 0 if result else False
                self._relationship_cache['clauses_by_article'][source_id] = has_clauses
                return has_clauses
                
            elif relationship_type == 'VIOLATES':
                # Check if article has incidents
                if source_id in self._relationship_cache['incidents_by_article']:
                    return self._relationship_cache['incidents_by_article'][source_id]
                
                query = """
                MATCH (i:Incident)-[:VIOLATES]->(a:Article {id: $article_id})
                RETURN count(i) as incident_count
                LIMIT 1
                """
                result = self.graph_queries.conn.execute_query(query, {'article_id': source_id})
                has_incidents = result[0].get('incident_count', 0) > 0 if result else False
                self._relationship_cache['incidents_by_article'][source_id] = has_incidents
                return has_incidents
                
            elif relationship_type == 'HAS_TOPIC':
                # Check if topic exists and has articles
                if self._relationship_cache['topics'] is None:
                    # Load all topics
                    topics = self.graph_queries.topic_analysis()
                    self._relationship_cache['topics'] = [t.get('topic_name', '') for t in topics]
                
                return source_id in self._relationship_cache['topics']
            
            return True  # Default to True if relationship type not recognized
            
        except Exception as e:
            verbose = os.getenv("VERBOSE", "false").lower() == "true"
            if verbose:
                print(f"Warning: Relationship validation failed: {e}")
            return True  # Default to True on error to avoid blocking queries
    
    def close(self):
        """Close Neo4j connection."""
        if self.neo4j_conn:
            self.neo4j_conn.close()
