"""
Task Executor: Executes planned steps using tools
"""

from typing import Dict, Any, List


class TaskExecutor:
    """
    Executes individual steps from a plan using the tool registry.
    """
    
    def __init__(self, tools):
        """
        Initialize executor.
        
        Args:
            tools: ToolRegistry instance
        """
        self.tools = tools
    
    def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step from a plan.
        
        Args:
            step: Step definition with tool name and parameters
            
        Returns:
            Execution result
        """
        tool_name = step.get('tool')
        parameters = step.get('parameters', {})
        
        if not tool_name:
            return {
                'success': False,
                'error': 'No tool specified in step'
            }
        
        # Execute tool
        try:
            result = self.tools.call_tool(tool_name, **parameters)
            
            return {
                'success': result.get('success', False),
                'result': result,
                'step_id': step.get('step_id'),
                'tool': tool_name
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'step_id': step.get('step_id'),
                'tool': tool_name
            }

