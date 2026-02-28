"""
Interactive Chatbot Interface for Querying Processed Data
"""

import os
import sys
from pathlib import Path
from typing import Optional, List

from ..engines.query_engine import VectorQueryEngine
from ..engines.graph_query_engine import GraphQueryEngine
from ..engines.hybrid_query_engine import HybridQueryEngine


class Chatbot:
    """
    Interactive chatbot for querying vector databases and Neo4j graph.
    Supports three search modes: vector-only, graph-only, and hybrid.
    """
    
    def __init__(self, base_dir: str, search_mode: str = "vector"):
        """
        Initialize the chatbot.
        
        Args:
            base_dir: Base directory of the project
            search_mode: Search mode - 'vector', 'graph', or 'hybrid'
        """
        self.base_dir = base_dir
        self.search_mode = search_mode.lower()
        self.query_engine = None
        self.graph_engine = None
        self.hybrid_engine = None
        self.conversation_history = []
        
        # Initialize engines based on mode
        if self.search_mode == "vector":
            self.query_engine = VectorQueryEngine(base_dir)
            print("[OK] Vector search enabled (FAISS)")
        elif self.search_mode == "graph":
            try:
                self.graph_engine = GraphQueryEngine(base_dir)
                print("[OK] Graph search enabled (Neo4j)")
            except Exception as e:
                import traceback
                print(f"[ERROR] Could not initialize graph engine: {e}")
                traceback.print_exc()
                raise
        elif self.search_mode == "hybrid":
            try:
                self.query_engine = VectorQueryEngine(base_dir)
                self.hybrid_engine = HybridQueryEngine(base_dir)
                print("[OK] Hybrid search enabled (vector + graph)")
            except Exception as e:
                import traceback
                print(f"[ERROR] Could not initialize hybrid engine: {e}")
                traceback.print_exc()
                raise
        else:
            raise ValueError(f"Invalid search_mode: {search_mode}. Must be 'vector', 'graph', or 'hybrid'")
    
    def format_results(self, results: List[dict], max_display: int = 5) -> str:
        """
        Format search results for display.
        Handles both vector search results (with database, source_name, similarity)
        and graph traversal results (with type, source, risk_type, etc.)
        
        Args:
            results: List of search results
            max_display: Maximum number of results to display
            
        Returns:
            Formatted string
        """
        # For mismatch analysis, show more results and prioritize coverage
        has_mismatch_analysis = any(r.get('analysis_type') == 'mismatch' for r in results)
        if has_mismatch_analysis:
            max_display = 20  # Show more results for mismatch analysis
        if not results:
            return "No results found."
        
        formatted = []
        formatted.append(f"\n{'='*80}")
        formatted.append(f"Found {len(results)} result(s)")
        formatted.append(f"{'='*80}\n")
        
        # For mismatch analysis, prioritize: summary -> coverage -> gaps
        if has_mismatch_analysis:
            # Separate results by type
            summary_results = [r for r in results if r.get('type') == 'summary']
            coverage_results = [r for r in results if r.get('type') == 'coverage']
            gap_results = [r for r in results if r.get('type') == 'gap']
            other_results = [r for r in results if r.get('type') not in ['summary', 'coverage', 'gap']]
            
            # Display order: summary, coverage, gaps, others
            display_results = summary_results + coverage_results[:10] + gap_results[:10] + other_results
            display_results = display_results[:max_display]
        else:
            display_results = results[:max_display]
        
        for i, result in enumerate(display_results, 1):
            formatted.append(f"Result {i}:")
            
            # Handle different result types
            if 'database' in result:
                # Vector search result
                formatted.append(f"  Database: {result.get('database', 'Unknown')}")
                formatted.append(f"  Source: {result.get('source_name', result.get('source', 'Unknown'))}")
                similarity = result.get('similarity', result.get('score', 0.0))
                formatted.append(f"  Similarity: {similarity:.4f}")
            else:
                # Graph traversal result
                result_type = result.get('type', 'unknown')
                formatted.append(f"  Type: {result_type}")
                formatted.append(f"  Source: {result.get('source', 'graph_traversal')}")
                
                # Mismatch analysis results
                if result.get('analysis_type') == 'mismatch':
                    coverage_status = result.get('coverage_status', 'unknown')
                    if coverage_status == 'not_covered':
                        formatted.append(f"  Status: [GAP] Not covered by company documents")
                    elif coverage_status == 'covered':
                        clause_count = result.get('clause_count', 0)
                        formatted.append(f"  Status: [COVERED] Addressed by {clause_count} clause(s)")
                        if 'clause_examples' in result and result['clause_examples']:
                            formatted.append(f"  Example clauses from company documents:")
                            for idx, clause_example in enumerate(result['clause_examples'][:2], 1):
                                formatted.append(f"    {idx}. {clause_example[:150]}...")
                    elif result_type == 'summary':
                        formatted.append(f"  Coverage: {result.get('coverage_percentage', 0)}%")
                        formatted.append(f"  Covered: {result.get('covered_count', 0)} articles")
                        formatted.append(f"  Gaps: {result.get('uncovered_count', 0)} articles")
                
                # Risk/incident results
                if result_type == 'risk' and 'risk_type' in result:
                    formatted.append(f"  Risk Type: {result['risk_type']}")
                if 'violated_articles' in result:
                    articles = result['violated_articles']
                    if isinstance(articles, list):
                        formatted.append(f"  Violated Articles: {', '.join(articles[:5])}")
                    else:
                        formatted.append(f"  Violated Articles: {articles}")
                
                # General fields
                if 'title' in result:
                    formatted.append(f"  Title: {result['title']}")
                if 'similarity' in result or 'score' in result:
                    score = result.get('similarity', result.get('score', 0.0))
                    formatted.append(f"  Score: {score:.4f}")
            
            # Text preview
            text = result.get('text', result.get('description', ''))
            if text:
                preview = text[:300] + "..." if len(text) > 300 else text
                formatted.append(f"  Text Preview: {preview}")
            
            formatted.append("")
        
        if len(results) > max_display:
            formatted.append(f"... and {len(results) - max_display} more result(s)")
        
        return "\n".join(formatted)
    
    def query(
        self,
        query: str,
        db_names: Optional[List[str]] = None,
        top_k: int = None,
        rerank: bool = True,
        generate_answer: bool = True,
        show_results: bool = False,
        similarity_threshold: float = 0.0,
        search_mode: Optional[str] = None
    ) -> dict:
        """
        Unified query method that routes to the appropriate engine based on search mode.
        
        Args:
            query: User query
            db_names: Databases to search (None = all, only for vector search)
            top_k: Number of results to retrieve
            rerank: Whether to rerank results
            generate_answer: Whether to generate answer using LLM
            show_results: Whether to display raw results (unused here, handled in process_query)
            similarity_threshold: Minimum similarity score (vector mode only)
            search_mode: Override search mode for this query (None = use default)
            
        Returns:
            Query results dictionary
        """
        # Use override mode if provided, otherwise use instance mode
        mode = search_mode.lower() if search_mode else self.search_mode
        
        if mode == "vector":
            if not self.query_engine:
                raise ValueError("Vector engine not initialized. Cannot perform vector search.")
            return self.query_engine.query(
                query=query,
                db_names=db_names,
                top_k=top_k or 10,
                rerank=rerank,
                generate_answer=generate_answer,
                similarity_threshold=similarity_threshold
            )
        elif mode == "graph":
            if not self.graph_engine:
                raise ValueError("Graph engine not initialized. Cannot perform graph search.")
            # GraphQueryEngine uses search() method, not query()
            results = self.graph_engine.search(query=query, top_k=top_k or 10)
            
            # Generate answer if requested
            answer = None
            if generate_answer and results:
                # Try to use vector engine's answer generation (it has the LLM integration)
                # Initialize vector engine temporarily if needed
                if self.query_engine:
                    answer = self.query_engine.generate_answer(query, results)
                else:
                    # Create a temporary vector engine just for answer generation
                    try:
                        temp_vector_engine = VectorQueryEngine(self.base_dir)
                        answer = temp_vector_engine.generate_answer(query, results)
                    except Exception as e:
                        import traceback
                        verbose = os.getenv("VERBOSE", "false").lower() == "true"
                        if verbose:
                            print(f"Warning: Could not generate answer for graph results: {e}")
                            traceback.print_exc()
                        # Answer generation failed, but we still return results
            
            return {
                'query': query,
                'results': results,
                'answer': answer,
                'num_results': len(results)
            }
        elif mode == "hybrid":
            if not self.hybrid_engine:
                raise ValueError("Hybrid engine not initialized. Cannot perform hybrid search.")
            return self.hybrid_engine.hybrid_query(
                query=query,
                top_k=top_k,
                rerank=rerank,
                generate_answer=generate_answer
            )
        else:
            raise ValueError(f"Invalid search_mode: {mode}. Must be 'vector', 'graph', or 'hybrid'")
    
    def process_query(
        self,
        query: str,
        db_names: Optional[List[str]] = None,
        top_k: int = None,  # None triggers adaptive top_k
        rerank: bool = True,
        generate_answer: bool = True,
        show_results: bool = False,
        similarity_threshold: float = 0.0,
        search_mode: Optional[str] = None
    ) -> dict:
        """
        Process a user query.
        
        Args:
            query: User query
            db_names: Databases to search (None = all, only for vector search)
            top_k: Number of results to retrieve
            rerank: Whether to rerank results
            generate_answer: Whether to generate answer using LLM
            show_results: Whether to display raw results
            similarity_threshold: Minimum similarity score (vector mode only)
            search_mode: Override search mode for this query (None = use default)
            
        Returns:
            Query results dictionary
        """
        # Use the query method which handles all modes
        result = self.query(
            query=query,
            db_names=db_names,
            top_k=top_k,
            rerank=rerank,
            generate_answer=generate_answer,
            show_results=show_results,
            similarity_threshold=similarity_threshold,
            search_mode=search_mode
        )
        
        # Display generated answer (if available) - this is the main output
        if result.get('answer'):
            print(f"\n{'='*80}")
            print("ANSWER:")
            print(f"{'='*80}")
            print(result['answer'])
            print(f"{'='*80}\n")
        elif result.get('results'):
            # No answer generated - show results if enabled, otherwise show brief message
            if show_results:
                print(self.format_results(result['results']))
            else:
                # Show brief message that results were found but answer generation unavailable
                print(f"\nFound {len(result['results'])} relevant result(s). Answer generation unavailable (API limit reached). Use !settings to show results.\n")
        
        # Save to conversation history
        self.conversation_history.append({
            'query': query,
            'answer': result.get('answer'),
            'num_results': result.get('num_results', 0)
        })
        
        return result
    
    def run_interactive(self):
        """
        Run interactive chatbot session.
        """
        print("="*80)
        print("COMPLIANCE RAG CHATBOT")
        print("="*80)
        print("\nAvailable commands:")
        print("  - Type your question to search the database")
        print("  - '!help' - Show help")
        print("  - '!databases' - List available databases")
        print("  - '!history' - Show conversation history")
        print("  - '!clear' - Clear conversation history")
        print("  - '!settings' - Change search settings")
        print("  - '!mode' - Switch between vector/graph/hybrid search")
        print("  - '!quit' or '!exit' - Exit chatbot")
        mode_display = {
            'vector': 'VECTOR (FAISS)',
            'graph': 'GRAPH (Neo4j)',
            'hybrid': 'HYBRID (Vector + Graph)'
        }
        print(f"\nCurrent mode: {mode_display.get(self.search_mode, self.search_mode.upper())}")
        print("\n" + "="*80 + "\n")
        
        # Default settings
        settings = {
            'db_names': None,  # None = all databases (vector mode only)
            'top_k': 10,
            'rerank': True,
            'generate_answer': True,
            'show_results': False,  # Default: show only final answers, not raw results
            'similarity_threshold': 0.0,
            'search_mode': self.search_mode
        }
        
        # Show brief info about silent mode
        print("Note: Answer generation requires API access. If unavailable, use !settings to show raw results.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('!'):
                    command = user_input[1:].lower()
                    
                    if command in ['quit', 'exit']:
                        print("\nGoodbye! 👋\n")
                        break
                    
                    elif command == 'help':
                        print("\n" + "="*80)
                        print("HELP")
                        print("="*80)
                        print("""
Available Commands:
  !help          - Show this help message
  !databases     - List available databases
  !history       - Show conversation history
  !clear         - Clear conversation history
  !settings      - Change search settings
  !mode          - Switch between vector/hybrid search
  !quit / !exit  - Exit chatbot

Search Modes:
  - Vector Mode: FAISS vector search only (faster, semantic similarity)
  - Hybrid Mode: Combines FAISS + Neo4j graph traversal (more comprehensive)

Search Settings:
  - Databases: Search across company, aiid, and/or standards databases (vector mode only)
  - Top K: Number of results to retrieve (default: 10)
  - Rerank: Use LLM to rerank results (default: True)
  - Generate Answer: Generate answer using LLM (default: True)
  - Similarity Threshold: Minimum similarity score (default: 0.0, vector mode only)

Examples:
  "What are the privacy policies?"
  "Find clauses addressing GDPR Article 5"  (works best in hybrid mode)
  "Find incidents related to data breaches"
  "What GDPR requirements apply?"
                        """)
                        print("="*80 + "\n")
                    
                    elif command == 'mode':
                        print("\n" + "="*80)
                        print("SEARCH MODE")
                        print("="*80)
                        mode_display = {
                            'vector': 'VECTOR (FAISS)',
                            'graph': 'GRAPH (Neo4j)',
                            'hybrid': 'HYBRID (Vector + Graph)'
                        }
                        print(f"Current mode: {mode_display.get(settings['search_mode'], settings['search_mode'].upper())}")
                        print("\nAvailable modes:")
                        print("  - vector: FAISS semantic search only")
                        if self.graph_engine:
                            print("  - graph: Neo4j graph traversal only")
                        if self.hybrid_engine:
                            print("  - hybrid: Combined vector + graph search")
                        print("\nSwitch mode? (vector/graph/hybrid): ", end='')
                        mode_input = input().strip().lower()
                        if mode_input in ['vector', 'graph', 'hybrid']:
                            if mode_input == 'graph' and not self.graph_engine:
                                print("[ERROR] Graph engine not available.\n")
                            elif mode_input == 'hybrid' and not self.hybrid_engine:
                                print("[ERROR] Hybrid engine not available.\n")
                            else:
                                settings['search_mode'] = mode_input
                                print(f"[OK] Switched to {mode_display.get(mode_input, mode_input.upper())} mode\n")
                        else:
                            print("Invalid input. Mode unchanged.\n")
                    
                    elif command == 'databases':
                        print("\n" + "="*80)
                        print("AVAILABLE DATABASES")
                        print("="*80)
                        if self.query_engine:
                            for db_name in ['company', 'aiid', 'standards']:
                                db_info = self.query_engine.databases[db_name]
                                loaded = "[OK] Loaded" if db_info['loaded'] else "[ ] Not loaded"
                                print(f"  - {db_name.upper()}: {loaded}")
                        else:
                            print("  Vector databases: Not available (graph mode only)")
                        if self.graph_engine:
                            print("  - NEO4J GRAPH: [OK] Loaded")
                        print("="*80 + "\n")
                    
                    elif command == 'history':
                        if not self.conversation_history:
                            print("\nNo conversation history.\n")
                        else:
                            print("\n" + "="*80)
                            print("CONVERSATION HISTORY")
                            print("="*80)
                            for i, entry in enumerate(self.conversation_history, 1):
                                print(f"\n{i}. Query: {entry['query']}")
                                print(f"   Results: {entry['num_results']}")
                                if entry['answer']:
                                    print(f"   Answer Preview: {entry['answer'][:100]}...")
                            print("="*80 + "\n")
                    
                    elif command == 'clear':
                        self.conversation_history = []
                        print("\nConversation history cleared.\n")
                    
                    elif command == 'settings':
                        print("\n" + "="*80)
                        print("CURRENT SETTINGS")
                        print("="*80)
                        mode_display = {
                            'vector': 'VECTOR (FAISS)',
                            'graph': 'GRAPH (Neo4j)',
                            'hybrid': 'HYBRID (Vector + Graph)'
                        }
                        print(f"  Mode: {mode_display.get(settings['search_mode'], settings['search_mode'].upper())}")
                        if settings['search_mode'] == 'vector':
                            print(f"  Databases: {settings['db_names'] or 'All'}")
                            print(f"  Similarity Threshold: {settings['similarity_threshold']}")
                        print(f"  Top K: {settings['top_k']}")
                        print(f"  Rerank: {settings['rerank']}")
                        print(f"  Generate Answer: {settings['generate_answer']}")
                        print(f"  Show Results: {settings['show_results']}")
                        print("="*80)
                        
                        print("\nChange settings? (y/n): ", end='')
                        if input().strip().lower() == 'y':
                            print("Mode (vector/graph/hybrid): ", end='')
                            mode_input = input().strip().lower()
                            if mode_input in ['vector', 'graph', 'hybrid']:
                                if mode_input == 'graph' and not self.graph_engine:
                                    print("[ERROR] Graph engine not available.")
                                elif mode_input == 'hybrid' and not self.hybrid_engine:
                                    print("[ERROR] Hybrid engine not available.")
                                else:
                                    settings['search_mode'] = mode_input
                            
                            if settings['search_mode'] == 'vector':
                                print("Databases (comma-separated: company,aiid,standards, or 'all'): ", end='')
                                db_input = input().strip()
                                if db_input.lower() == 'all' or not db_input:
                                    settings['db_names'] = None
                                else:
                                    settings['db_names'] = [db.strip() for db in db_input.split(',')]
                            
                            print("Top K (default 10): ", end='')
                            top_k_input = input().strip()
                            if top_k_input:
                                try:
                                    settings['top_k'] = int(top_k_input)
                                except ValueError:
                                    print("Invalid input, keeping default.")
                            
                            print("Rerank (y/n, default y): ", end='')
                            rerank_input = input().strip().lower()
                            settings['rerank'] = rerank_input != 'n'
                            
                            print("Generate Answer (y/n, default y): ", end='')
                            ctx_input = input().strip().lower()
                            settings['generate_answer'] = ctx_input != 'n'
                            
                            print("Show Results (y/n, default n): ", end='')
                            show_input = input().strip().lower()
                            settings['show_results'] = show_input == 'y'
                            
                            if settings['search_mode'] == 'vector':
                                print("Similarity Threshold (default 0.0): ", end='')
                                thresh_input = input().strip()
                                if thresh_input:
                                    try:
                                        settings['similarity_threshold'] = float(thresh_input)
                                    except ValueError:
                                        print("Invalid input, keeping default.")
                            
                            print("\nSettings updated!\n")
                    
                    continue
                
                # Process query
                self.process_query(
                    query=user_input,
                    db_names=settings['db_names'],
                    top_k=settings['top_k'],
                    rerank=settings['rerank'],
                    generate_answer=settings['generate_answer'],
                    show_results=settings['show_results'],
                    similarity_threshold=settings['similarity_threshold'],
                    search_mode=settings['search_mode']
                )
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type '!quit' to exit.\n")
            except Exception as e:
                import traceback
                print(f"\n{'='*80}")
                print("[ERROR] Query failed")
                print(f"{'='*80}")
                print(f"Error: {str(e)}")
                
                # Show more details for common errors
                error_str = str(e).lower()
                if 'neo4j' in error_str or 'connection' in error_str:
                    print("\n💡 Tip: Make sure Neo4j is running and accessible.")
                    print("   Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env")
                elif 'api' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                    print("\n💡 Tip: Check your API key and balance.")
                    print("   Verify OPENAI_API_KEY or XAI_API_KEY in .env")
                elif 'faiss' in error_str or 'index' in error_str or 'database' in error_str:
                    print("\n💡 Tip: FAISS indexes may not be built.")
                    print("   Run: python backend/indexing/faiss/build_faiss_index.py")
                else:
                    verbose = os.getenv("VERBOSE", "false").lower() == "true"
                    if verbose:
                        print("\nFull traceback:")
                        traceback.print_exc()
                    else:
                        print("\n💡 Tip: Set VERBOSE=true to see full error details.")
                print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compliance RAG Chatbot")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid search (vector + graph)"
    )
    
    args = parser.parse_args()
    
    # Get base directory (project root)
    # This file is at backend/searching/chatbot.py
    # So we need to go up 2 levels to get to project root
    base_dir = Path(__file__).parent.parent.parent
    
    # Determine mode from args
    if hasattr(args, 'hybrid') and args.hybrid:
        mode = 'hybrid'
    elif hasattr(args, 'mode'):
        mode = args.mode
    else:
        mode = 'vector'
    
    chatbot = Chatbot(str(base_dir), search_mode=mode)
    chatbot.run_interactive()
    
    # Clean up engines
    if chatbot.graph_engine:
        chatbot.graph_engine.close()
    if chatbot.hybrid_engine:
        chatbot.hybrid_engine.graph_engine.close()

