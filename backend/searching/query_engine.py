"""
Query Engine for Vector Database Search
Supports querying across multiple FAISS vector databases with reranking and contextualization.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configuration from environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "3000"))
RERANK_TEMPERATURE = float(os.getenv("RERANK_TEMPERATURE", "0.1"))
RERANK_MAX_TOKENS = int(os.getenv("RERANK_MAX_TOKENS", "100"))
EXPANSION_TEMPERATURE = float(os.getenv("EXPANSION_TEMPERATURE", "0.3"))
EXPANSION_MAX_TOKENS = int(os.getenv("EXPANSION_MAX_TOKENS", "300"))


def get_openai_client():
    """Get OpenAI client instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    try:
        return OpenAI(api_key=api_key)
    except TypeError:
        os.environ["OPENAI_API_KEY"] = api_key
        return OpenAI()


def load_faiss_database(db_dir: str, index_name: str) -> Tuple[faiss.Index, List[Dict], Dict]:
    """
    Load a FAISS database (index, metadata, and summary).
    
    Args:
        db_dir: Directory containing the FAISS database files
        index_name: Base name of the index files
        
    Returns:
        Tuple of (FAISS index, metadata list, summary dict)
    """
    db_path = Path(db_dir)
    
    # Load index
    index_file = db_path / f"{index_name}.index"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    index = faiss.read_index(str(index_file))
    
    # Load metadata
    metadata_file = db_path / f"{index_name}_metadata.pkl"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load summary
    summary_file = db_path / f"{index_name}_summary.json"
    summary = {}
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    
    return index, metadata, summary


def get_query_embedding(query: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    Get embedding for a query string.
    
    Args:
        query: Query text
        model: Embedding model to use
        
    Returns:
        Query embedding as numpy array
    """
    client = get_openai_client()
    
    try:
        response = client.embeddings.create(
            model=model,
            input=[query]
        )
        embedding = np.array(response.data[0].embedding, dtype='float32')
        
        # Normalize for cosine similarity (if using IndexFlatIP)
        embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)
        
        return embedding
    except Exception as e:
        raise Exception(f"Error generating query embedding: {str(e)}")


class VectorQueryEngine:
    """
    Query engine for searching across multiple FAISS vector databases.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the query engine.
        
        Args:
            base_dir: Base directory of the project
        """
        self.base_dir = Path(base_dir)
        self.faiss_dir = self.base_dir / "backend" / "building_database" / "faiss"
        
        # Available databases
        self.databases = {
            'company': {
                'dir': self.faiss_dir / "company",
                'index_name': 'company_faiss_index',
                'loaded': False,
                'index': None,
                'metadata': None,
                'summary': None
            },
            'aiid': {
                'dir': self.faiss_dir / "aiid",
                'index_name': 'aiid_faiss_index',
                'loaded': False,
                'index': None,
                'metadata': None,
                'summary': None
            },
            'standards': {
                'dir': self.faiss_dir / "standards",
                'index_name': 'standards_faiss_index',
                'loaded': False,
                'index': None,
                'metadata': None,
                'summary': None
            }
        }
        
        self.client = get_openai_client()
    
    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Expand and refine a user query to improve search results.
        
        Args:
            query: Original user query
            
        Returns:
            Dictionary with refined_query, alternative_queries, and key_terms
        """
        # Load query expansion prompt
        prompt_file = self.base_dir / "backend" / "searching" / "prompts" / "query_expansion_prompt.txt"
        expansion_prompt = ""
        if prompt_file.exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                expansion_prompt = f.read()
        else:
            expansion_prompt = """You are a query expansion expert. Given a user query, generate:
1. A refined version that clarifies the intent
2. 2-3 alternative phrasings
3. Key terms to search for

Return ONLY a JSON object:
{
  "refined_query": "improved version",
  "alternative_queries": ["alt1", "alt2"],
  "key_terms": ["term1", "term2"]
}"""
        
        full_prompt = f"""{expansion_prompt}

User Query: {query}

Generate the expanded query information:"""
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query expansion expert. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=EXPANSION_TEMPERATURE,
                max_tokens=EXPANSION_MAX_TOKENS
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
            
            expanded = json.loads(result_text)
            
            # Ensure all fields exist
            return {
                'refined_query': expanded.get('refined_query', query),
                'alternative_queries': expanded.get('alternative_queries', []),
                'key_terms': expanded.get('key_terms', [])
            }
        except Exception as e:
            print(f"Warning: Query expansion failed: {str(e)}. Using original query.")
            return {
                'refined_query': query,
                'alternative_queries': [],
                'key_terms': []
            }
    
    def search_with_expansion(
        self,
        query: str,
        db_names: Optional[List[str]] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        use_expansion: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search with query expansion for better results.
        
        Args:
            query: Original query
            db_names: Databases to search
            top_k: Number of results per query
            similarity_threshold: Minimum similarity
            use_expansion: Whether to use query expansion
            
        Returns:
            Combined search results
        """
        if not use_expansion:
            return self.search(query, db_names, top_k, similarity_threshold)
        
        # Expand query
        expanded = self.expand_query(query)
        refined_query = expanded['refined_query']
        alternative_queries = expanded['alternative_queries']
        
        # Search with refined query
        all_results = self.search(refined_query, db_names, top_k, similarity_threshold)
        
        # Also search with alternative queries (with fewer results each)
        for alt_query in alternative_queries[:2]:  # Limit to 2 alternatives
            alt_results = self.search(alt_query, db_names, top_k // 2, similarity_threshold)
            all_results.extend(alt_results)
        
        # Deduplicate by text content
        seen_texts = set()
        unique_results = []
        for result in all_results:
            text_key = result['text'][:100]  # Use first 100 chars as key
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(result)
        
        # Re-sort by similarity
        unique_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return unique_results[:top_k * 2]  # Return up to 2x top_k for better coverage
    
    def load_database(self, db_name: str):
        """
        Load a specific database into memory.
        
        Args:
            db_name: Name of database ('company', 'aiid', or 'standards')
        """
        if db_name not in self.databases:
            raise ValueError(f"Unknown database: {db_name}. Available: {list(self.databases.keys())}")
        
        if self.databases[db_name]['loaded']:
            return
        
        db_info = self.databases[db_name]
        index, metadata, summary = load_faiss_database(
            str(db_info['dir']),
            db_info['index_name']
        )
        
        self.databases[db_name]['index'] = index
        self.databases[db_name]['metadata'] = metadata
        self.databases[db_name]['summary'] = summary
        self.databases[db_name]['loaded'] = True
    
    def search(
        self,
        query: str,
        db_names: Optional[List[str]] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search across specified databases.
        
        Args:
            query: Query text
            db_names: List of database names to search (None = all)
            top_k: Number of results per database
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results with metadata
        """
        if db_names is None:
            db_names = list(self.databases.keys())
        
        # Get query embedding
        try:
            query_embedding = get_query_embedding(query)
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            raise Exception(f"Failed to generate query embedding: {str(e)}")
        
        all_results = []
        
        for db_name in db_names:
            if db_name not in self.databases:
                print(f"Warning: Database '{db_name}' not found. Available: {list(self.databases.keys())}")
                continue
            
            # Load database if not already loaded
            try:
                self.load_database(db_name)
            except Exception as e:
                print(f"Error loading database '{db_name}': {str(e)}")
                continue
            
            db_info = self.databases[db_name]
            index = db_info['index']
            metadata = db_info['metadata']
            
            if index is None:
                print(f"Warning: Index for database '{db_name}' is None")
                continue
            
            if not metadata:
                print(f"Warning: No metadata found for database '{db_name}'")
                continue
            
            # Search
            try:
                similarities, indices = index.search(query_embedding, top_k)
                
                # Process results
                for sim_score, idx in zip(similarities[0], indices[0]):
                    # For IndexFlatIP (cosine similarity), scores can be negative
                    # We'll accept all results above threshold
                    if sim_score >= similarity_threshold and idx < len(metadata) and idx >= 0:
                        result = {
                            'database': db_name,
                            'similarity': float(sim_score),
                            'index': int(idx),
                            'text': metadata[int(idx)].get('text', ''),
                            'source_name': metadata[int(idx)].get('source_name', 'Unknown'),
                            'source_file': metadata[int(idx)].get('source_file', ''),
                            'chunk_id': metadata[int(idx)].get('chunk_id', ''),
                        }
                        all_results.append(result)
            except Exception as e:
                print(f"Error searching database '{db_name}': {str(e)}")
                continue
        
        # Sort by similarity (descending)
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"Search returned {len(all_results)} results for query: '{query}'")
        if all_results:
            print(f"Similarity range: {all_results[0]['similarity']:.4f} to {all_results[-1]['similarity']:.4f}")
        
        return all_results
    
    def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: int = 5,
        rerank_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using LLM-based reranking.
        
        Args:
            query: Original query
            results: List of search results
            top_n: Number of top results to return
            rerank_prompt: Custom reranking prompt (optional)
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        # Load default prompt if not provided
        if rerank_prompt is None:
            prompt_file = self.base_dir / "backend" / "searching" / "prompts" / "rerank_prompt.txt"
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    rerank_prompt = f.read()
            else:
                # Default prompt
                rerank_prompt = """You are a search result reranking expert. 
Given a user query and a list of search results, rank them by relevance to the query.
Consider semantic similarity, context, and how well each result answers the query.
Return only the indices of the results in order of relevance (most relevant first), separated by commas.
Example: 0, 3, 1, 2"""
        
        # Format results for prompt
        results_text = []
        for i, result in enumerate(results):
            # Handle both vector search results and graph traversal results
            db_info = f"Database: {result.get('database', 'Graph')}\n" if 'database' in result else "Type: Graph Result\n"
            source_info = f"Source: {result.get('source_name', result.get('source', 'Unknown'))}\n"
            similarity = result.get('similarity', result.get('score', 0.0))
            similarity_info = f"Similarity Score: {similarity:.4f}\n"
            
            # Additional info for graph results
            extra_info = ""
            if result.get('type') == 'risk' and 'risk_type' in result:
                extra_info = f"Risk Type: {result['risk_type']}\n"
            if 'violated_articles' in result:
                articles = result['violated_articles']
                if isinstance(articles, list):
                    extra_info += f"Violated Articles: {', '.join(articles[:3])}\n"
                else:
                    extra_info += f"Violated Articles: {articles}\n"
            
            text = result.get('text', result.get('description', ''))
            text_preview = text[:500] + "..." if len(text) > 500 else text
            
            results_text.append(
                f"Result {i}:\n"
                f"{db_info}"
                f"{source_info}"
                f"{similarity_info}"
                f"{extra_info}"
                f"Text: {text_preview}\n"
            )
        
        full_prompt = f"""{rerank_prompt}

User Query: {query}

Search Results:
{chr(10).join(results_text)}

Rank the results by relevance to the query. Return only the result indices (0-{len(results)-1}) in order, separated by commas:"""
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search result reranking expert. Return only comma-separated indices."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=RERANK_TEMPERATURE,
                max_tokens=RERANK_MAX_TOKENS
            )
            
            ranked_indices_str = response.choices[0].message.content.strip()
            # Parse indices
            try:
                ranked_indices = [int(x.strip()) for x in ranked_indices_str.split(',')]
                # Filter valid indices
                ranked_indices = [idx for idx in ranked_indices if 0 <= idx < len(results)]
                
                # Create reranked results
                reranked = [results[idx] for idx in ranked_indices]
                
                # Add any remaining results that weren't ranked
                ranked_set = set(ranked_indices)
                for i, result in enumerate(results):
                    if i not in ranked_set:
                        reranked.append(result)
                
                return reranked[:top_n]
            except ValueError:
                # If parsing fails, return original results sorted by similarity
                return results[:top_n]
        
        except Exception as e:
            print(f"Warning: Reranking failed: {str(e)}. Returning original results.")
            return results[:top_n]
    
    def contextualize_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        contextualize_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a contextualized answer based on search results.
        
        Args:
            query: User query
            results: List of search results
            contextualize_prompt: Custom contextualization prompt (optional)
            
        Returns:
            Contextualized answer text
        """
        if not results:
            return "I couldn't find any relevant information to answer your query. Please try:\n- Rephrasing your question\n- Checking if the databases are loaded\n- Lowering the similarity threshold in settings"
        
        # Load default prompt if not provided
        if contextualize_prompt is None:
            prompt_file = self.base_dir / "backend" / "searching" / "prompts" / "contextualize_prompt.txt"
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    contextualize_prompt = f.read()
            else:
                # Default prompt
                contextualize_prompt = """You are a helpful assistant that answers questions based on provided search results.
Use the search results to provide a comprehensive, accurate answer to the user's query.
Cite specific sources when referencing information.
If the search results don't fully answer the query, say so and provide what information is available."""
        
        # Format results for prompt - limit text length per result to avoid token limits
        # Estimate: each result ~500 chars = ~125 tokens, 5 results = ~625 tokens
        # Plus prompt ~500 tokens, query ~50 tokens = ~1175 tokens input
        # So we can use ~3000 tokens for output safely
        max_text_length = 800  # Limit each result text to avoid exceeding context window
        context_text = []
        for i, result in enumerate(results, 1):
            # Handle both vector search results and graph traversal results
            result_text = result.get('text', result.get('description', ''))
            if len(result_text) > max_text_length:
                # Truncate at word boundary
                truncated = result_text[:max_text_length].rsplit(' ', 1)[0] + "..."
            else:
                truncated = result_text
            
            # Build source label
            source_label = result.get('source_name', result.get('source', 'Unknown'))
            if result.get('type') == 'risk':
                source_label = f"Incident: {result.get('risk_type', 'Unknown')}"
            elif result.get('type') == 'gap':
                source_label = f"Gap: {result.get('title', 'Unknown')} - Not covered by company documents"
            elif result.get('type') == 'coverage':
                clause_count = result.get('clause_count', 0)
                source_label = f"Covered: {result.get('title', 'Unknown')} - Addressed by {clause_count} clause(s)"
            elif result.get('type') == 'summary':
                source_label = f"Summary: {result.get('coverage_percentage', 0)}% coverage"
            
            similarity = result.get('similarity', result.get('score', 0.0))
            
            context_text.append(
                f"[Source {i} - {source_label}]\n"
                f"{truncated}\n"
                f"(Similarity: {similarity:.4f})\n"
            )
        
        full_prompt = f"""{contextualize_prompt}

User Query: {query}

Search Results:
{chr(10).join(context_text)}

Based on the above search results, provide a comprehensive answer to the user's query:"""
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided search results."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            
            answer = response.choices[0].message.content
            
            # Check if response was truncated
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                answer += "\n\n[Note: Response was truncated due to length limit. Consider refining your query for more specific information.]"
            
            return answer
        except Exception as e:
            raise Exception(f"Error generating contextualized answer: {str(e)}")
    
    def query(
        self,
        query: str,
        db_names: Optional[List[str]] = None,
        top_k: int = 10,
        rerank: bool = True,
        contextualize: bool = True,
        similarity_threshold: float = 0.0,
        use_expansion: bool = True
    ) -> Dict[str, Any]:
        """
        Complete query pipeline: search, rerank, and contextualize.
        
        Args:
            query: User query
            db_names: Databases to search (None = all)
            top_k: Number of initial results
            rerank: Whether to rerank results
            contextualize: Whether to generate contextualized answer
            similarity_threshold: Minimum similarity score
            use_expansion: Whether to use query expansion
            
        Returns:
            Dictionary with search results and contextualized answer
        """
        # Step 1: Search (with expansion if enabled)
        if use_expansion:
            results = self.search_with_expansion(
                query=query,
                db_names=db_names,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                use_expansion=True
            )
        else:
            results = self.search(
                query=query,
                db_names=db_names,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
        
        # Step 2: Rerank (if enabled)
        if rerank and results:
            results = self.rerank_results(query, results, top_n=min(8, len(results)))  # Increased from 5 to 8
        
        # Step 3: Contextualize (if enabled)
        answer = None
        if contextualize:
            # Use more results for better context (up to 8)
            answer = self.contextualize_results(query, results[:8])
        
        return {
            'query': query,
            'results': results,
            'answer': answer,
            'num_results': len(results)
        }

