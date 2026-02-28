"""
Base Agent class for extensible agentic system.
All agents inherit from this base class and implement their specific workflows.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypedDict
import json

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None
    END = None

# LangChain imports with fallbacks
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
except ImportError:
    HumanMessage = AIMessage = SystemMessage = None

try:
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain.memory import ConversationBufferMemory
except ImportError:
    try:
        from langchain.memory import ConversationBufferMemory
    except ImportError:
        # Fallback: create simple memory class
        class ConversationBufferMemory:
            def __init__(self, **kwargs):
                self.chat_memory = type('obj', (object,), {'messages': []})()
            def save_context(self, *args, **kwargs):
                pass
            def load_memory_variables(self, *args, **kwargs):
                return {}

try:
    from langchain.tools import BaseTool
except ImportError:
    BaseTool = ABC  # Fallback

try:
    from langchain.agents import AgentExecutor
except ImportError:
    try:
        from langchain.agents.agent import AgentExecutor
    except ImportError:
        AgentExecutor = None  # Not critical for base agent


class AgentState(TypedDict):
    """
    Base state for all agents.
    Extend this for agent-specific state.
    """
    goal: str
    messages: List[Dict[str, Any]]
    plan: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    current_step: int
    tools: List[Dict[str, Any]]
    error: Optional[str]
    finished: bool


class BaseAgent(ABC):
    """
    Base class for all agents in the system.
    Provides common functionality and extensible interface.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        tools: List[BaseTool],
        llm,
        memory: Optional[ConversationBufferMemory] = None,
        max_iterations: int = 20
    ):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            description: Agent description
            tools: List of tools available to the agent
            llm: LLM instance
            memory: Memory instance (optional)
            max_iterations: Maximum iterations for agent loop
        """
        self.name = name
        self.description = description
        self.tools = tools
        self.llm = llm
        if memory is None:
            if ConversationBufferMemory is not None:
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
            else:
                self.memory = None  # No memory if langchain not available
        else:
            self.memory = memory
        self.max_iterations = max_iterations
        
        # Build graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine for this agent.
        Must be implemented by subclasses.
        
        Returns:
            StateGraph instance
        """
        pass
    
    @abstractmethod
    def _plan(self, state: AgentState) -> AgentState:
        """
        Planning node - creates a plan to achieve the goal.
        Must be implemented by subclasses.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with plan
        """
        pass
    
    @abstractmethod
    def _execute(self, state: AgentState) -> AgentState:
        """
        Execution node - executes a step from the plan.
        Must be implemented by subclasses.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with execution results
        """
        pass
    
    def _evaluate(self, state: AgentState) -> AgentState:
        """
        Evaluation node - evaluates if goal is achieved.
        Can be overridden by subclasses.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with evaluation
        """
        # Check if goal is achieved
        if self._is_goal_achieved(state):
            state["finished"] = True
        return state
    
    def _is_goal_achieved(self, state: AgentState) -> bool:
        """
        Check if goal has been achieved.
        Can be overridden by subclasses.
        
        Args:
            state: Current agent state
            
        Returns:
            True if goal achieved, False otherwise
        """
        if not state.get("results"):
            return False
        
        # Check if we have successful results
        successful_results = [
            r for r in state["results"]
            if r.get("success", False)
        ]
        
        return len(successful_results) > 0
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Determine if agent should continue or finish.
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" or "end"
        """
        if state.get("finished", False):
            return "end"
        
        if state.get("current_step", 0) >= len(state.get("plan", [])):
            return "end"
        
        if state.get("current_step", 0) >= self.max_iterations:
            return "end"
        
        if state.get("error"):
            # Try to recover or end
            return "end"
        
        return "continue"
    
    def execute(self, goal: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent with a goal.
        
        Args:
            goal: Goal to achieve
            **kwargs: Additional arguments
            
        Returns:
            Execution results
        """
        # Initialize state
        initial_state: AgentState = {
            "goal": goal,
            "messages": [],
            "plan": [],
            "results": [],
            "current_step": 0,
            "tools": [{"name": tool.name, "description": tool.description} for tool in self.tools],
            "error": None,
            "finished": False
        }
        
        # Add initial message
        initial_state["messages"].append({
            "role": "user",
            "content": goal
        })
        
        # Run graph
        try:
            final_state = self.app.invoke(initial_state)
            return {
                "goal": goal,
                "success": final_state.get("finished", False),
                "results": final_state.get("results", []),
                "plan": final_state.get("plan", []),
                "steps_executed": final_state.get("current_step", 0),
                "error": final_state.get("error")
            }
        except Exception as e:
            return {
                "goal": goal,
                "success": False,
                "error": str(e),
                "results": [],
                "plan": [],
                "steps_executed": 0
            }
    
    def get_tools(self) -> List[BaseTool]:
        """Get list of available tools."""
        return self.tools
    
    def get_description(self) -> str:
        """Get agent description."""
        return self.description
