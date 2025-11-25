"""
Interactive Chatbot Interface for Querying Processed Data
"""

import os
import sys
from pathlib import Path
from typing import Optional, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from query_engine import VectorQueryEngine


class Chatbot:
    """
    Interactive chatbot for querying vector databases and Neo4j graph.
    Supports both vector-only and hybrid search modes.
    """
    
    def __init__(self, base_dir: str, use_hybrid: bool = False):
        """
        Initialize the chatbot.
        
        Args:
            base_dir: Base directory of the project
            use_hybrid: Whether to use hybrid search (vector + graph)
        """
        self.base_dir = base_dir
        self.use_hybrid = use_hybrid
        self.query_engine = VectorQueryEngine(base_dir)
        self.hybrid_engine = None
        self.conversation_history = []
        
        # Initialize hybrid engine if requested
        if use_hybrid:
            try:
                # Import from same directory
                import sys
                from pathlib import Path
                searching_dir = Path(__file__).parent
                if str(searching_dir) not in sys.path:
                    sys.path.insert(0, str(searching_dir))
                from hybrid_query_engine import HybridQueryEngine
                self.hybrid_engine = HybridQueryEngine(base_dir)
                self.use_hybrid = True
                print("[OK] Hybrid search enabled (vector + graph)")
            except Exception as e:
                import traceback
                print(f"[WARNING] Could not initialize hybrid engine: {e}")
                print(f"  Error details: {traceback.format_exc()}")
                print("  Falling back to vector-only search")
                self.use_hybrid = False
                self.hybrid_engine = None
    
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
    
    def process_query(
        self,
        query: str,
        db_names: Optional[List[str]] = None,
        top_k: int = 10,
        rerank: bool = True,
        contextualize: bool = True,
        show_results: bool = True,
        similarity_threshold: float = 0.0,
        use_hybrid: Optional[bool] = None
    ) -> dict:
        """
        Process a user query.
        
        Args:
            query: User query
            db_names: Databases to search (None = all, only for vector search)
            top_k: Number of results to retrieve
            rerank: Whether to rerank results
            contextualize: Whether to generate contextualized answer
            show_results: Whether to display raw results
            similarity_threshold: Minimum similarity score
            use_hybrid: Override hybrid mode for this query (None = use default)
            
        Returns:
            Query results dictionary
        """
        # Determine which engine to use
        use_hybrid_mode = use_hybrid if use_hybrid is not None else self.use_hybrid
        
        if use_hybrid_mode and self.hybrid_engine:
            print(f"\n🔍 Processing query (HYBRID MODE): '{query}'...")
            result = self.hybrid_engine.hybrid_query(
                query=query,
                top_k=top_k,
                rerank=rerank,
                contextualize=contextualize
            )
            # Add query_types and sources_used to result for display
            if 'query_types' in result:
                print(f"  Query types detected: {result['query_types']}")
            if 'sources_used' in result:
                print(f"  Sources: {', '.join(result['sources_used'])}")
        else:
            print(f"\n🔍 Processing query (VECTOR MODE): '{query}'...")
            result = self.query_engine.query(
                query=query,
                db_names=db_names,
                top_k=top_k,
                rerank=rerank,
                contextualize=contextualize,
                similarity_threshold=similarity_threshold
            )
        
        # Display results
        if show_results and result['results']:
            print(self.format_results(result['results']))
        
        # Display contextualized answer
        if result['answer']:
            print(f"\n{'='*80}")
            print("ANSWER:")
            print(f"{'='*80}")
            print(result['answer'])
            print(f"{'='*80}\n")
        
        # Save to conversation history
        self.conversation_history.append({
            'query': query,
            'answer': result['answer'],
            'num_results': result['num_results']
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
        print("  - '!mode' - Switch between vector/hybrid search")
        print("  - '!quit' or '!exit' - Exit chatbot")
        print(f"\nCurrent mode: {'HYBRID' if self.use_hybrid else 'VECTOR ONLY'}")
        print("\n" + "="*80 + "\n")
        
        # Default settings
        settings = {
            'db_names': None,  # None = all databases (vector mode only)
            'top_k': 10,
            'rerank': True,
            'contextualize': True,
            'show_results': True,
            'similarity_threshold': 0.0,
            'use_hybrid': self.use_hybrid
        }
        
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
  - Contextualize: Generate contextualized answer (default: True)
  - Similarity Threshold: Minimum similarity score (default: 0.0, vector mode only)

Examples:
  "What are the privacy policies?"
  "Find clauses addressing GDPR Article 5"  (works best in hybrid mode)
  "Find incidents related to data breaches"
  "What GDPR requirements apply?"
                        """)
                        print("="*80 + "\n")
                    
                    elif command == 'mode':
                        if self.hybrid_engine:
                            print("\n" + "="*80)
                            print("SEARCH MODE")
                            print("="*80)
                            print(f"Current mode: {'HYBRID' if settings['use_hybrid'] else 'VECTOR ONLY'}")
                            print("\nSwitch mode? (hybrid/vector): ", end='')
                            mode_input = input().strip().lower()
                            if mode_input == 'hybrid':
                                settings['use_hybrid'] = True
                                print("[OK] Switched to HYBRID mode\n")
                            elif mode_input == 'vector':
                                settings['use_hybrid'] = False
                                print("[OK] Switched to VECTOR ONLY mode\n")
                            else:
                                print("Invalid input. Mode unchanged.\n")
                        else:
                            print("\n[WARNING] Hybrid engine not available. Using vector-only mode.\n")
                    
                    elif command == 'databases':
                        print("\n" + "="*80)
                        print("AVAILABLE DATABASES")
                        print("="*80)
                        for db_name in ['company', 'aiid', 'standards']:
                            db_info = self.query_engine.databases[db_name]
                            loaded = "[OK] Loaded" if db_info['loaded'] else "[ ] Not loaded"
                            print(f"  - {db_name.upper()}: {loaded}")
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
                        print(f"  Mode: {'HYBRID' if settings['use_hybrid'] else 'VECTOR ONLY'}")
                        if not settings['use_hybrid']:
                            print(f"  Databases: {settings['db_names'] or 'All'}")
                        print(f"  Top K: {settings['top_k']}")
                        print(f"  Rerank: {settings['rerank']}")
                        print(f"  Contextualize: {settings['contextualize']}")
                        print(f"  Show Results: {settings['show_results']}")
                        if not settings['use_hybrid']:
                            print(f"  Similarity Threshold: {settings['similarity_threshold']}")
                        print("="*80)
                        
                        print("\nChange settings? (y/n): ", end='')
                        if input().strip().lower() == 'y':
                            if self.hybrid_engine:
                                print("Mode (hybrid/vector, default hybrid): ", end='')
                                mode_input = input().strip().lower()
                                if mode_input == 'vector':
                                    settings['use_hybrid'] = False
                                elif mode_input == 'hybrid':
                                    settings['use_hybrid'] = True
                            
                            if not settings['use_hybrid']:
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
                            
                            print("Contextualize (y/n, default y): ", end='')
                            ctx_input = input().strip().lower()
                            settings['contextualize'] = ctx_input != 'n'
                            
                            print("Show Results (y/n, default y): ", end='')
                            show_input = input().strip().lower()
                            settings['show_results'] = show_input != 'n'
                            
                            if not settings['use_hybrid']:
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
                    contextualize=settings['contextualize'],
                    show_results=settings['show_results'],
                    similarity_threshold=settings['similarity_threshold'],
                    use_hybrid=settings['use_hybrid']
                )
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type '!quit' to exit.\n")
            except Exception as e:
                print(f"\n[ERROR] {str(e)}\n")


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
    
    chatbot = Chatbot(str(base_dir), use_hybrid=args.hybrid)
    chatbot.run_interactive()
    
    # Clean up hybrid engine if used
    if chatbot.hybrid_engine:
        chatbot.hybrid_engine.close()

