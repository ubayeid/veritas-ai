#!/usr/bin/env python3
"""
Unified CLI for Compliance RAG System
Supports interactive query/answer and evaluation.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.retrieval.interfaces.chatbot import Chatbot
from backend.evaluation.evaluate import run_retrieval_evaluation
# Lazy import for agent mode only


def run_interactive(mode: str = "vector"):
    """
    Run interactive query/answer session.
    
    Args:
        mode: Search mode - 'vector', 'graph', or 'hybrid'
    """
    print("="*80)
    print("COMPLIANCE RAG SYSTEM - Interactive Query/Answer")
    print("="*80)
    print(f"Mode: {mode.upper()}")
    print("="*80)
    
    chatbot = Chatbot(str(project_root), search_mode=mode)
    try:
        chatbot.run_interactive()
    finally:
        # Clean up engines
        if chatbot.graph_engine:
            chatbot.graph_engine.close()
        if chatbot.hybrid_engine:
            chatbot.hybrid_engine.graph_engine.close()


def run_agent_orchestration():
    """Run multi-agent orchestration system."""
    # Lazy import to avoid loading agents unless needed
    try:
        from backend.agents.orchestration.orchestration_agent import OrchestrationAgent
    except ImportError as e:
        print(f"Error: Could not import agents: {e}")
        print("Make sure langchain dependencies are installed:")
        print("  pip install langchain langchain-openai langgraph")
        sys.exit(1)
    
    print("="*80)
    print("COMPLIANCE RAG SYSTEM - Multi-Agent Orchestration")
    print("="*80)
    print("\nThis mode uses the 4-agent architecture:")
    print("  1. Monitoring Agent - Observes AI application behavior")
    print("  2. Decision Making Agent - Evaluates compliance")
    print("  3. Compliance Verification Agent - Identifies policy violations")
    print("  4. Orchestration Agent - Coordinates all agents")
    print("\n" + "="*80 + "\n")
    
    orchestrator = OrchestrationAgent(str(project_root))
    
    print("Available commands:")
    print("  - Type your question/event to evaluate compliance")
    print("  - '!audit' - Show audit log")
    print("  - '!quit' or '!exit' - Exit")
    print("\n" + "="*80 + "\n")
    
    while True:
        try:
            user_input = input("Event/Query: ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith('!'):
                command = user_input[1:].lower()
                
                if command in ['quit', 'exit']:
                    print("\nGoodbye! 👋\n")
                    break
                
                elif command == 'audit':
                    audit_log = orchestrator.get_audit_log()
                    print("\n" + "="*80)
                    print("AUDIT LOG")
                    print("="*80)
                    if not audit_log:
                        print("No audit records yet.")
                    else:
                        for i, record in enumerate(audit_log[-10:], 1):  # Show last 10
                            print(f"\n{i}. Timestamp: {record.get('timestamp', 'N/A')}")
                            print(f"   Decision: {record.get('final_decision', 'N/A')}")
                            print(f"   Compliance Score: {record.get('compliance_score', 0.0):.2f}")
                            print(f"   Actions: {', '.join(record.get('actions', []))}")
                    print("="*80 + "\n")
                    continue
            
            # Process request through orchestration
            request = {
                'type': 'user_query',
                'content': user_input,
                'source': 'audit_engineer'
            }
            
            print(f"\n{'─'*80}")
            print("PROCESSING REQUEST...")
            print(f"{'─'*80}\n")
            
            result = orchestrator.process_request(request)
            
            # Display results
            print(f"{'='*80}")
            print("ORCHESTRATION RESULTS")
            print(f"{'='*80}")
            print(f"Final Decision: {result.get('final_decision', 'N/A').upper()}")
            print(f"Compliance Score: {result.get('compliance_score', 0.0):.2f}")
            print(f"Actions Taken: {', '.join(result.get('actions', []))}")
            
            # Show agent outputs
            monitoring = result.get('monitoring', {})
            decision_making = result.get('decision_making', {})
            compliance = result.get('compliance', {})
            
            print(f"\n{'─'*80}")
            print("AGENT OUTPUTS")
            print(f"{'─'*80}")
            
            print(f"\n[Monitoring Agent]")
            print(f"  Decision: {monitoring.get('decision', 'N/A')}")
            if monitoring.get('anomalies'):
                print(f"  Anomalies: {len(monitoring.get('anomalies', []))}")
            
            print(f"\n[Decision Making Agent]")
            print(f"  Risk Level: {decision_making.get('risk_level', 'N/A')}")
            dm_decision = decision_making.get('decision', {})
            print(f"  Action: {dm_decision.get('action', 'N/A')}")
            print(f"  Reasoning: {dm_decision.get('reasoning', 'N/A')}")
            
            print(f"\n[Compliance Verification Agent]")
            cv_decision = compliance.get('compliance_decision', {})
            print(f"  Compliant: {cv_decision.get('compliant', 'N/A')}")
            violated = compliance.get('violated_articles', [])
            if violated:
                print(f"  Violated Articles: {', '.join(violated)}")
                for detail in compliance.get('violation_details', [])[:3]:
                    print(f"    - {detail.get('article', 'N/A')}: {detail.get('reasoning', 'N/A')[:60]}...")
            else:
                print(f"  No violations detected")
            
            print(f"\n{'─'*80}\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type '!quit' to exit.\n")
        except Exception as e:
            print(f"\n[ERROR] {str(e)}\n")
            import traceback
            traceback.print_exc()


def run_evaluation(
    queries_file: str = "backend/evaluation/evaluation_queries_50.json",
    output_csv: str = "backend/evaluation/retrieval_results.csv",
    top_k: int = 8,
    no_answer: bool = False,
    export_chunks: bool = False
):
    """Run evaluation on query set."""
    run_retrieval_evaluation(
        queries_file=queries_file,
        output_csv=output_csv,
        top_k=top_k,
        generate_answer=not no_answer,
        export_chunks=export_chunks
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compliance RAG System - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive query/answer (vector mode - default)
  python query.py interactive
  
  # Interactive query/answer (graph mode)
  python query.py interactive --mode graph
  
  # Interactive query/answer (hybrid mode)
  python query.py interactive --mode hybrid
  
  # Multi-agent orchestration (4-agent architecture)
  python query.py agent
  
  # Run evaluation
  python query.py evaluate
  
  # Run evaluation with custom queries
  python query.py evaluate --queries my_queries.json --output results.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Interactive command
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Run interactive query/answer session'
    )
    interactive_parser.add_argument(
        '--mode',
        type=str,
        choices=['vector', 'graph', 'hybrid'],
        default='vector',
        help='Search mode: vector (FAISS only), graph (Neo4j only), or hybrid (both)'
    )
    
    # Agent orchestration command
    agent_parser = subparsers.add_parser(
        'agent',
        help='Run multi-agent orchestration system (4-agent architecture)'
    )
    
    # Evaluation command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Run evaluation on query set'
    )
    eval_parser.add_argument(
        '--queries',
        type=str,
        default='backend/evaluation/evaluation_queries_50.json',
        help='Queries JSON file (default: backend/evaluation/evaluation_queries_50.json)'
    )
    eval_parser.add_argument(
        '--output',
        type=str,
        default='backend/evaluation/retrieval_results.csv',
        help='Output CSV file (default: backend/evaluation/retrieval_results.csv)'
    )
    eval_parser.add_argument(
        '--top-k',
        type=int,
        default=8,
        help='Top-K retrievals per method (default: 8)'
    )
    eval_parser.add_argument(
        '--no-answer',
        action='store_true',
        help='Skip answer generation (faster, no answers in CSV)'
    )
    eval_parser.add_argument(
        '--export-chunks',
        action='store_true',
        help='Export retrieval chunks for manual review (instead of answers)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'interactive':
        run_interactive(mode=args.mode)
    elif args.command == 'evaluate':
        run_evaluation(
            queries_file=args.queries,
            output_csv=args.output,
            top_k=args.top_k,
            no_answer=args.no_answer,
            export_chunks=getattr(args, 'export_chunks', False)
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
