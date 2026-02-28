"""
Run the Compliance Agent interactively
"""

import sys
import os
from pathlib import Path

# Get project root (3 levels up from this file)
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)

# Add paths for imports
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

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
    venv_python = project_root / "venv" / "bin" / "python3"
    if venv_python.exists() and venv_python.is_file():
        print("⚠️  Warning: Virtual environment not activated!")
        print(f"   Please run: source venv/bin/activate")
        print(f"   Or use: {venv_python} {__file__}")
        print(f"   Or use: make run-agent")
        print()
        # Try to use venv python anyway
        if os.access(venv_python, os.X_OK):
            print(f"   Attempting to use venv Python: {venv_python}")
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)

from backend.agents.core.langgraph_agent import ComplianceLangGraphAgent
from backend.agents.utils.agent_registry import get_registry, register_agent


def main():
    """Run agent interactively."""
    # Use project root as base directory
    base_dir = project_root
    
    # Initialize agent
    print("="*80)
    print("COMPLIANCE AGENT")
    print("="*80)
    print("\nInitializing LangGraph-based agent...")
    
    agent = ComplianceLangGraphAgent(str(base_dir))
    # Register in global registry
    register_agent("compliance", agent, ComplianceLangGraphAgent)
    
    print("Agent ready!")
    print("\nAvailable capabilities:")
    
    tools = agent.get_tools()
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    
    print("\n" + "="*80)
    print("Enter your goal (or 'quit' to exit):")
    print("="*80 + "\n")
    
    while True:
        try:
            goal = input("Goal: ").strip()
            
            if not goal:
                continue
            
            if goal.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋\n")
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

