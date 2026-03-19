"""
Run the Compliance Agent interactively
"""

import sys
import os
from pathlib import Path
import argparse
import json

# Get repository root (4 levels up from this file)
# backend/agents/scripts/run_agent.py -> repo_root
repo_root = Path(__file__).resolve().parent.parent.parent.parent
os.chdir(repo_root)

# Add paths for imports
sys.path.insert(0, str(repo_root))

# Check if virtual environment is activated
# Try to detect if we're in a venv
in_venv = (
    hasattr(sys, 'real_prefix') or 
    (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
    'venv' in sys.executable or
    'VIRTUAL_ENV' in os.environ
)

if not in_venv:
    # Try to use venv python if available
    venv_python = repo_root / "venv" / "Scripts" / "python.exe"
    if venv_python.exists() and venv_python.is_file():
        print("Warning: Virtual environment not activated!")
        print(f"   Please activate your venv (Windows): .\\venv\\Scripts\\Activate.ps1")
        print(f"   Or use: {venv_python} {__file__}")
        print(f"   Or use: make run-agent")
        print()
        # Try to use venv python anyway
        if os.access(venv_python, os.X_OK):
            print(f"   Attempting to use venv Python: {venv_python}")
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)

from backend.agents.core.langgraph_agent import ComplianceLangGraphAgent
from backend.agents.utils.agent_registry import get_registry, register_agent


def _init_agent() -> ComplianceLangGraphAgent:
    base_dir = repo_root
    print("=" * 80)
    print("COMPLIANCE AGENT")
    print("=" * 80)
    print("\nInitializing LangGraph-based agent...")
    agent = ComplianceLangGraphAgent(str(base_dir))
    register_agent("compliance", agent, ComplianceLangGraphAgent)
    print("Agent ready!")
    print("\nAvailable capabilities:")
    for tool in agent.get_tools():
        print(f"  - {tool.name}: {tool.description}")
    return agent


def main():
    """Run the agent (interactive if TTY; otherwise requires --goal)."""
    parser = argparse.ArgumentParser(description="Run the Compliance Agent")
    parser.add_argument(
        "--goal",
        type=str,
        help="Run a single goal non-interactively (useful for CI / redirected stdin).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="When used with --goal, print the raw result as JSON.",
    )
    args = parser.parse_args()

    # Non-interactive environments can't use input() reliably on Windows.
    if not sys.stdin.isatty() and not args.goal:
        print("Error: Non-interactive session detected. Provide --goal to run once.")
        print(r'Example: python backend\agents\scripts\run_agent.py --goal "Explain GDPR Article 5"')
        raise SystemExit(2)

    # Use project root as base directory
    agent = _init_agent()

    if args.goal:
        result = agent.execute(args.goal)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("\n" + "=" * 80)
            print("EXECUTION RESULTS")
            print("=" * 80)
            print(f"Goal: {result.get('goal')}")
            print(f"Success: {result.get('success')}")
            print(f"Steps executed: {result.get('steps_executed')}")
        return

    print("\n" + "=" * 80)
    print("Enter your goal (or 'quit' to exit):")
    print("=" * 80 + "\n")
    
    while True:
        try:
            goal = input("Goal: ").strip()
            
            if not goal:
                continue
            
            if goal.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!\n")
                break
            
            # Execute goal
            result = agent.execute(goal)
            
            # Display results
            print("\n" + "="*80)
            print("EXECUTION RESULTS")
            print("="*80)
            print(f"Goal: {result['goal']}")
            print(f"Success: {result['success']}")
            print(f"Steps executed: {result['steps_executed']}")
            
            if result.get('results'):
                print("\nStep Results:")
                for step_result in result['results']:
                    # LangGraph format
                    step_id = step_result.get('step_id', '?')
                    tool = step_result.get('tool', 'unknown')
                    success = step_result.get('success', False)
                    exec_result = step_result.get('result', {})
                    
                    status = "✓" if success else "✗"
                    print(f"\n{status} Step {step_id}: {tool}")
                    
                    if success:
                        tool_result = exec_result.get('result', {}) if isinstance(exec_result, dict) else exec_result
                        if isinstance(tool_result, dict):
                            # Show key results
                            for key, value in list(tool_result.items())[:3]:
                                if isinstance(value, (str, int, float, bool)):
                                    print(f"    {key}: {value}")
                        else:
                            print(f"    Result: {str(tool_result)[:200]}")
                    else:
                        print(f"    Error: {step_result.get('error', 'Unknown')}")
            
            print("\n" + "="*80 + "\n")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.\n")
        except Exception as e:
            print(f"\n[ERROR] {str(e)}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

