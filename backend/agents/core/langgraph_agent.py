"""
LangGraph-based Compliance Agent.
Implements agentic workflow using LangGraph state machine.
"""

from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import BaseTool, StructuredTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import json
import sys
from pathlib import Path

# Import base agent
from .base_agent import BaseAgent, AgentState

# Import existing components
from backend.retrieval.utils.api_client import get_api_client
from backend.retrieval.utils.model_config import get_llm_model

# Import tools
from ..utils.tools import ToolRegistry


class ComplianceAgentState(AgentState):
    """Extended state for compliance agent."""
    compliance_gaps: List[Dict[str, Any]]
    coverage_analysis: Dict[str, Any]
    report: Optional[str]


class ComplianceLangGraphAgent(BaseAgent):
    """
    Compliance agent using LangGraph for orchestration.
    Extends BaseAgent with compliance-specific functionality.
    """
    
    def __init__(
        self,
        base_dir: str,
        llm=None,
        memory=None,
        max_iterations: int = 20
    ):
        """
        Initialize compliance agent.
        
        Args:
            base_dir: Base directory of the project
            llm: LLM instance (optional)
            memory: Memory instance (optional)
            max_iterations: Maximum iterations
        """
        self.base_dir = Path(base_dir)
        
        # Initialize LLM if not provided
        if llm is None:
            from langchain_openai import ChatOpenAI
            from backend.retrieval.utils.model_config import get_agent_model, AGENT_TEMPERATURE
            llm_model = get_agent_model()
            llm = ChatOpenAI(model=llm_model, temperature=AGENT_TEMPERATURE)
        
        # Initialize tools
        tool_registry = ToolRegistry(str(base_dir))
        tools = self._create_langchain_tools(tool_registry)
        
        # Initialize base agent
        super().__init__(
            name="compliance_agent",
            description="Agent for compliance monitoring and gap analysis",
            tools=tools,
            llm=llm,
            memory=memory,
            max_iterations=max_iterations
        )
        
        self.tool_registry = tool_registry
    
    def _create_langchain_tools(self, tool_registry: ToolRegistry) -> List[BaseTool]:
        """Convert ToolRegistry tools to LangChain tools."""
        langchain_tools = []
        
        # Get all tools from registry
        available_tools = tool_registry.list_tools()
        
        for tool_info in available_tools:
            tool_name = tool_info['name']
            tool_description = tool_info['description']
            tool_params = tool_info.get('parameters', {})
            
            # Create LangChain tool
            def make_tool_func(name):
                def tool_func(**kwargs):
                    return tool_registry.call_tool(name, **kwargs)
                return tool_func
            
            tool_func = make_tool_func(tool_name)
            
            # Create structured tool
            tool = StructuredTool.from_function(
                func=tool_func,
                name=tool_name,
                description=tool_description
            )
            
            langchain_tools.append(tool)
        
        return langchain_tools
    
    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph state machine for compliance agent.
        
        Returns:
            StateGraph instance
        """
        # Create graph
        workflow = StateGraph(ComplianceAgentState)
        
        # Add nodes
        workflow.add_node("plan", self._plan)
        workflow.add_node("execute", self._execute)
        workflow.add_node("evaluate", self._evaluate)
        
        # Set entry point
        workflow.set_entry_point("plan")
        
        # Add edges
        workflow.add_edge("plan", "execute")
        workflow.add_conditional_edges(
            "execute",
            self._should_continue,
            {
                "continue": "evaluate",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {
                "continue": "execute",
                "end": END
            }
        )
        
        return workflow
    
    def _plan(self, state: ComplianceAgentState) -> ComplianceAgentState:
        """
        Planning node - creates a plan using LLM.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with plan
        """
        goal = state["goal"]
        tools_info = state["tools"]
        
        # Format tools for prompt
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in tools_info
        ])
        
        # Create planning prompt
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a task planning assistant for a compliance analysis system.
Your job is to break down complex goals into executable steps using the available tools.

Available Tools:
{tools}

Create a step-by-step plan to achieve the goal. Each step should:
1. Use a specific tool from the available tools
2. Have clear inputs/parameters
3. Define what output is expected
4. Note any dependencies on previous steps

Return your plan as a JSON array of steps, where each step has:
- step_id: sequential number
- tool: tool name to use
- description: what this step does
- parameters: dictionary of parameters
- depends_on: list of step_ids this depends on
- expected_output: what we expect to get

Be specific and actionable. Use only the tools listed above."""),
            ("user", "Goal: {goal}\n\nCreate a plan:")
        ])
        
        # Generate plan
        messages = planning_prompt.format_messages(
            tools=tools_description,
            goal=goal
        )
        
        response = self.llm.invoke(messages)
        
        # Parse plan from response
        try:
            # Try to extract JSON from response
            content = response.content
            # Find JSON array in response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                plan_json = json.loads(json_match.group())
            else:
                # Try parsing entire content
                plan_json = json.loads(content)
            
            state["plan"] = plan_json if isinstance(plan_json, list) else plan_json.get("steps", [])
        except Exception as e:
            # Fallback: create simple plan
            print(f"Warning: Failed to parse plan: {e}")
            state["plan"] = self._create_simple_plan(goal, tools_info)
        
        return state
    
    def _create_simple_plan(self, goal: str, tools_info: List[Dict]) -> List[Dict]:
        """Create a simple plan based on goal patterns."""
        goal_lower = goal.lower()
        
        if 'gap' in goal_lower or 'missing' in goal_lower:
            return [{
                'step_id': 1,
                'tool': 'search_compliance_gaps',
                'description': 'Find GDPR articles not covered by company documents',
                'parameters': {},
                'depends_on': [],
                'expected_output': 'List of compliance gaps'
            }]
        elif 'report' in goal_lower:
            return [{
                'step_id': 1,
                'tool': 'generate_compliance_report',
                'description': 'Generate comprehensive compliance report',
                'parameters': {'report_type': 'full'},
                'depends_on': [],
                'expected_output': 'Compliance report'
            }]
        else:
            return [{
                'step_id': 1,
                'tool': 'search_vector',
                'description': f'Search for information about: {goal}',
                'parameters': {'query': goal},
                'depends_on': [],
                'expected_output': 'Search results'
            }]
    
    def _execute(self, state: ComplianceAgentState) -> ComplianceAgentState:
        """
        Execution node - executes a step from the plan.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with execution results
        """
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)
        
        if current_step >= len(plan):
            state["finished"] = True
            return state
        
        step = plan[current_step]
        tool_name = step.get("tool")
        parameters = step.get("parameters", {})
        
        # Execute tool
        try:
            result = self.tool_registry.call_tool(tool_name, **parameters)
            
            # Store result
            execution_result = {
                "step_id": step.get("step_id"),
                "tool": tool_name,
                "success": result.get("success", False),
                "result": result,
                "error": result.get("error")
            }
            
            state["results"].append(execution_result)
            state["current_step"] = current_step + 1
            
            # Update state with specific results
            if tool_name == "search_compliance_gaps":
                state["compliance_gaps"] = result.get("result", {}).get("gaps", [])
            elif tool_name == "generate_compliance_report":
                state["report"] = result.get("result", {}).get("report", "")
            
        except Exception as e:
            state["error"] = str(e)
            state["results"].append({
                "step_id": step.get("step_id"),
                "tool": tool_name,
                "success": False,
                "error": str(e)
            })
            state["current_step"] = current_step + 1
        
        return state
    
    def _evaluate(self, state: ComplianceAgentState) -> ComplianceAgentState:
        """
        Evaluation node - evaluates if goal is achieved.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with evaluation
        """
        # Check if goal is achieved
        goal_lower = state["goal"].lower()
        results = state.get("results", [])
        
        if not results:
            return state
        
        # Check for goal-specific indicators
        if 'gap' in goal_lower:
            # Check if we have gaps
            gaps = state.get("compliance_gaps", [])
            if gaps:
                state["finished"] = True
        elif 'report' in goal_lower:
            # Check if we have a report
            report = state.get("report")
            if report:
                state["finished"] = True
        else:
            # Check if we have successful results
            successful_results = [r for r in results if r.get("success", False)]
            if successful_results:
                state["finished"] = True
        
        return state
    
    def _is_goal_achieved(self, state: ComplianceAgentState) -> bool:
        """Check if goal has been achieved."""
        return state.get("finished", False)
