"""
Main Compliance Agent: Orchestrates planning, execution, and learning
"""

from typing import Dict, Any, List
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from .tools import ToolRegistry
from .planner import TaskPlanner
from .executor import TaskExecutor


class ComplianceAgent:
    """
    Main agent that autonomously plans and executes compliance analysis tasks.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize compliance agent.
        
        Args:
            base_dir: Base directory of the project
        """
        self.base_dir = Path(base_dir)
        
        # Initialize components
        self.tools = ToolRegistry(str(base_dir))
        self.planner = TaskPlanner()
        self.executor = TaskExecutor(self.tools)
        
        # Execution state
        self.current_goal = None
        self.execution_history = []
    
    def execute_goal(self, goal: str, max_steps: int = 20) -> Dict[str, Any]:
        """
        Execute a goal autonomously.
        
        Args:
            goal: The goal to achieve
            max_steps: Maximum number of steps to execute
            
        Returns:
            Execution results
        """
        self.current_goal = goal
        
        # 1. Get available tools
        available_tools = self.tools.list_tools()
        
        # 2. Create plan
        print(f"\n[AGENT] Planning for goal: {goal}")
        plan = self.planner.create_plan(goal, available_tools)
        
        print(f"[AGENT] Created plan with {len(plan)} steps:")
        for step in plan:
            print(f"  Step {step['step_id']}: {step['description']}")
        
        # 3. Execute plan
        print(f"\n[AGENT] Executing plan...")
        results = []
        
        for i, step in enumerate(plan[:max_steps]):
            print(f"\n[AGENT] Step {step['step_id']}/{len(plan)}: {step['description']}")
            
            # Execute step
            result = self.executor.execute_step(step)
            results.append({
                'step_id': step['step_id'],
                'step': step,
                'result': result
            })
            
            # Check if we can stop early
            if result.get('success') and self._is_goal_achieved(results, goal):
                print(f"[AGENT] Goal achieved early at step {step['step_id']}")
                break
            
            # Check for critical failures
            if not result.get('success'):
                print(f"[AGENT] Step {step['step_id']} failed: {result.get('error', 'Unknown error')}")
                # Try to adjust plan
                remaining_plan = self.planner.adjust_plan(plan, results, goal)
                if not remaining_plan:
                    print("[AGENT] Cannot recover from failure. Stopping.")
                    break
                plan = remaining_plan
        
        # 4. Compile final result
        final_result = {
            'goal': goal,
            'plan': plan,
            'results': results,
            'success': self._is_goal_achieved(results, goal),
            'steps_executed': len(results)
        }
        
        # Store in history
        self.execution_history.append(final_result)
        
        return final_result
    
    def _is_goal_achieved(self, results: List[Dict], goal: str) -> bool:
        """Check if goal has been achieved."""
        if not results:
            return False
        
        # Simple heuristic: check if we have successful results
        successful_results = [r for r in results if r.get('result', {}).get('success', False)]
        
        if not successful_results:
            return False
        
        # Check goal-specific indicators
        goal_lower = goal.lower()
        last_result = successful_results[-1]
        result_data = last_result.get('result', {}).get('result', {})
        
        if 'gap' in goal_lower:
            return 'gaps' in str(result_data).lower() or 'total_gaps' in str(result_data)
        elif 'report' in goal_lower:
            return 'report' in str(result_data).lower()
        elif 'compare' in goal_lower:
            return 'comparison' in str(result_data).lower() or 'results' in str(result_data)
        
        # Default: if we have results, consider it achieved
        return len(successful_results) > 0
    
    def get_execution_history(self) -> List[Dict]:
        """Get execution history."""
        return self.execution_history
    
    def clear_history(self):
        """Clear execution history."""
        self.execution_history = []

