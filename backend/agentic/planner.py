"""
Task Planner: Uses LLM to break down complex goals into executable steps
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Import unified API client
sys.path.insert(0, str(Path(__file__).parent.parent / "searching"))
from api_client import get_api_client, get_llm_model


class TaskPlanner:
    """
    Plans complex tasks by breaking them down into executable steps.
    Uses LLM to understand goals and create structured plans.
    """
    
    def __init__(self, llm_model: str = None):
        """
        Initialize task planner.
        
        Args:
            llm_model: Model to use for planning (if None, uses configured model from .env)
        """
        self.client = get_api_client()
        self.model = llm_model or get_llm_model()
    
    def create_plan(self, goal: str, available_tools: List[Dict], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Create a step-by-step plan to achieve a goal.
        
        Args:
            goal: The goal to achieve
            available_tools: List of available tools with descriptions
            context: Additional context (previous results, user preferences, etc.)
            
        Returns:
            List of planned steps
        """
        # Format tools for prompt
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in available_tools
        ])
        
        context_str = ""
        if context:
            context_str = f"\n\nContext:\n{json.dumps(context, indent=2)}"
        
        prompt = f"""You are a task planning assistant for a compliance analysis system.

Goal: {goal}

Available Tools:
{tools_description}{context_str}

Create a step-by-step plan to achieve this goal. Each step should:
1. Use a specific tool from the available tools
2. Have clear inputs/parameters
3. Define what output is expected
4. Note any dependencies on previous steps

Return your plan as a JSON object with this structure:
{{
    "steps": [
        {{
            "step_id": 1,
            "tool": "tool_name",
            "description": "what this step does",
            "parameters": {{"param_name": "value"}},
            "depends_on": [],
            "expected_output": "what we expect to get"
        }}
    ],
    "reasoning": "brief explanation of the plan"
}}

Be specific and actionable. Use only the tools listed above."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert task planner. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            plan_json = json.loads(response.choices[0].message.content)
            return plan_json.get('steps', [])
        
        except Exception as e:
            # Fallback: Create a simple plan
            print(f"Error creating plan: {e}")
            return self._create_simple_plan(goal, available_tools)
    
    def _create_simple_plan(self, goal: str, available_tools: List[Dict]) -> List[Dict[str, Any]]:
        """Fallback simple planner."""
        goal_lower = goal.lower()
        
        # Simple pattern matching
        if 'gap' in goal_lower or 'missing' in goal_lower:
            return [
                {
                    'step_id': 1,
                    'tool': 'search_compliance_gaps',
                    'description': 'Find GDPR articles not covered by company documents',
                    'parameters': {},
                    'depends_on': [],
                    'expected_output': 'List of compliance gaps'
                }
            ]
        elif 'report' in goal_lower:
            return [
                {
                    'step_id': 1,
                    'tool': 'generate_compliance_report',
                    'description': 'Generate comprehensive compliance report',
                    'parameters': {'report_type': 'full'},
                    'depends_on': [],
                    'expected_output': 'Compliance report'
                }
            ]
        else:
            return [
                {
                    'step_id': 1,
                    'tool': 'search_vector',
                    'description': f'Search for information about: {goal}',
                    'parameters': {'query': goal},
                    'depends_on': [],
                    'expected_output': 'Search results'
                }
            ]
    
    def adjust_plan(self, plan: List[Dict], intermediate_results: List[Dict], goal: str) -> List[Dict]:
        """
        Adjust plan based on intermediate results.
        
        Args:
            plan: Original plan
            intermediate_results: Results from executed steps
            goal: Original goal
            
        Returns:
            Adjusted plan
        """
        # Check if goal is achieved
        if self._is_goal_achieved(intermediate_results, goal):
            return []  # No more steps needed
        
        # Check for failures
        failed_steps = [r for r in intermediate_results if not r.get('success', False)]
        if failed_steps:
            # Try to replan with different approach
            # Note: This requires available_tools, which should be passed in
            # For now, return remaining steps
            pass
        
        # Continue with remaining steps
        executed_step_ids = {r.get('step_id') for r in intermediate_results}
        remaining_steps = [s for s in plan if s['step_id'] not in executed_step_ids]
        
        return remaining_steps
    
    def _is_goal_achieved(self, results: List[Dict], goal: str) -> bool:
        """Check if goal has been achieved based on results."""
        # Simple heuristic: if we have results and no errors
        if not results:
            return False
        
        # Check if last result indicates completion
        last_result = results[-1]
        if last_result.get('success') and last_result.get('result'):
            # Check for goal-specific indicators
            goal_lower = goal.lower()
            result_str = str(last_result.get('result', {})).lower()
            if 'gap' in goal_lower:
                return 'gaps' in result_str or 'total_gaps' in result_str
            elif 'report' in goal_lower:
                return 'report' in result_str
        
        return False

