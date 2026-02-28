"""
Decision Making Agent
Evaluates events and determines whether system behavior complies with AI policy.
Classifies behavior as high, medium, or low risk.
"""

from typing import Dict, Any, List, Optional, TypedDict
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.core.base_agent import BaseAgent, AgentState


class DecisionState(AgentState):
    """Extended state for decision making agent."""
    percept: Optional[Dict[str, Any]]
    context: Dict[str, Any]
    options: List[Dict[str, Any]]
    decision: Optional[Dict[str, Any]]
    risk_level: Optional[str]  # 'high', 'medium', 'low'


class DecisionMakingAgent(BaseAgent):
    """
    Decision Making Agent - Evaluates compliance with AI policy.
    
    As described in the paper:
    - Reviews AI application output against established rules
    - Classifies behavior as high, medium, or low risk
    - Sends structured decision back to orchestration agent
    """
    
    def __init__(self, base_dir: str, llm=None, memory=None):
        """Initialize decision making agent."""
        self.base_dir = Path(base_dir)
        
        # Initialize LLM if not provided
        if llm is None:
            try:
                from langchain_openai import ChatOpenAI
                from backend.retrieval.utils.model_config import get_agent_model, AGENT_TEMPERATURE
                llm_model = get_agent_model()
                llm = ChatOpenAI(model=llm_model, temperature=AGENT_TEMPERATURE)
            except ImportError:
                from backend.retrieval.utils.api_client import get_api_client
                llm = get_api_client()
        
        # Decision making agent uses RAG tools for policy evaluation
        from backend.agents.utils.tools import ToolRegistry
        tool_registry = ToolRegistry(str(base_dir))
        tools = self._create_tools_from_registry(tool_registry)
        
        super().__init__(
            name="decision_making_agent",
            description="Evaluates AI application behavior for compliance with AI policy",
            tools=tools,
            llm=llm,
            memory=memory,
            max_iterations=5
        )
        
        self.tool_registry = tool_registry
        self.kb = []
        self.decision_model = {
            'rules': self._load_decision_rules(),
            'risk_classification': self._load_risk_classification()
        }
        self.time = 0
    
    def _create_tools_from_registry(self, tool_registry):
        """Create LangChain tools from tool registry."""
        from langchain.tools import StructuredTool
        
        tools = []
        available_tools = tool_registry.list_tools()
        
        for tool_info in available_tools:
            tool_name = tool_info['name']
            tool_description = tool_info['description']
            
            def make_tool_func(name):
                def tool_func(**kwargs):
                    return tool_registry.call_tool(name, **kwargs)
                return tool_func
            
            tool_func = make_tool_func(tool_name)
            tool = StructuredTool.from_function(
                func=tool_func,
                name=tool_name,
                description=tool_description
            )
            tools.append(tool)
        
        return tools
    
    def _load_decision_rules(self) -> List[Dict[str, Any]]:
        """Load decision rules for compliance evaluation."""
        return [
            {'rule': 'prohibited_ai_practice', 'risk': 'high', 'action': 'deny'},
            {'rule': 'high_risk_system', 'risk': 'high', 'action': 'require_assessment'},
            {'rule': 'transparency_required', 'risk': 'medium', 'action': 'require_disclosure'},
            {'rule': 'minimal_risk', 'risk': 'low', 'action': 'permit'},
        ]
    
    def _load_risk_classification(self) -> Dict[str, Dict[str, Any]]:
        """Load risk classification criteria."""
        return {
            'high': {
                'criteria': ['prohibited', 'high_risk', 'safety_threat', 'rights_violation'],
                'threshold': 0.8
            },
            'medium': {
                'criteria': ['transparency', 'limited_risk', 'monitoring_required'],
                'threshold': 0.5
            },
            'low': {
                'criteria': ['minimal_risk', 'permitted'],
                'threshold': 0.0
            }
        }
    
    def _build_graph(self):
        """Build decision making agent graph."""
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(DecisionState)
        
        # DM-AGENT algorithm: Percept -> Context -> Options -> Decision
        workflow.add_node("process", self._process_percept)
        workflow.add_node("decide", self._make_decision)
        
        workflow.set_entry_point("process")
        workflow.add_edge("process", "decide")
        workflow.add_edge("decide", END)
        
        return workflow
    
    def _plan(self, state: AgentState) -> AgentState:
        """Decision making doesn't need separate planning."""
        return state
    
    def _execute(self, state: AgentState) -> AgentState:
        """Decision making executes in process node."""
        return state
    
    def _process_percept(self, state: DecisionState) -> DecisionState:
        """
        Process percept and build context.
        Implements DM-AGENT algorithm from paper.
        """
        percept = state.get('percept', {})
        self.time = state.get('timestamp', self.time)
        
        # Make percept sentence
        percept_sentence = self._make_percept_sentence(percept, self.time)
        self.kb.append(percept_sentence)
        
        # Query context
        context = self._make_context_query(self.time)
        state['context'] = context
        
        # Query options
        options = self._make_option_query(context, self.time)
        state['options'] = options
        
        return state
    
    def _make_decision(self, state: DecisionState) -> DecisionState:
        """Apply decision model to make final decision."""
        context = state.get('context', {})
        options = state.get('options', [])
        
        # Apply decision model
        decision = self._apply_decision_model(options, context)
        state['decision'] = decision
        state['risk_level'] = decision.get('risk_level', 'low')
        
        # Make action sentence
        action_sentence = self._make_action_sentence(decision, self.time)
        self.kb.append(action_sentence)
        
        self.time += 1
        
        return state
    
    def _make_percept_sentence(self, percept: Dict[str, Any], t: int) -> str:
        """Convert percept to sentence for knowledge base."""
        event_type = percept.get('type', 'unknown')
        content = percept.get('content', '')
        return f"Percept[{t}]: {event_type} - {content[:200]}"
    
    def _make_context_query(self, t: int) -> Dict[str, Any]:
        """Query context from knowledge base."""
        recent_percepts = [p for p in self.kb if f"[{t}]" in p or f"[{t-1}]" in p]
        return {
            'recent_percepts': recent_percepts,
            'time': t,
            'policy_rules': self.decision_model['rules']
        }
    
    def _make_option_query(self, context: Dict[str, Any], t: int) -> List[Dict[str, Any]]:
        """Generate decision options."""
        return [
            {'action': 'permit', 'risk': 'low', 'description': 'Allow the action'},
            {'action': 'permit_with_conditions', 'risk': 'medium', 'description': 'Allow with monitoring'},
            {'action': 'deny', 'risk': 'high', 'description': 'Block the action'},
            {'action': 'require_assessment', 'risk': 'high', 'description': 'Require compliance assessment'}
        ]
    
    def _apply_decision_model(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply decision model to select best option.
        Implements DM-AGENT decision logic.
        """
        # Analyze context for risk indicators
        recent_percepts = context.get('recent_percepts', [])
        risk_score = self._calculate_risk_score(recent_percepts)
        
        # Classify risk level
        risk_level = self._classify_risk(risk_score)
        
        # Select appropriate action based on risk
        if risk_level == 'high':
            decision = {'action': 'deny', 'risk_level': 'high', 'reasoning': 'High risk detected'}
        elif risk_level == 'medium':
            decision = {'action': 'permit_with_conditions', 'risk_level': 'medium', 'reasoning': 'Medium risk - requires monitoring'}
        else:
            decision = {'action': 'permit', 'risk_level': 'low', 'reasoning': 'Low risk - permitted'}
        
        return decision
    
    def _calculate_risk_score(self, percepts: List[str]) -> float:
        """Calculate risk score from percepts."""
        if not percepts:
            return 0.0
        
        high_risk_keywords = ['prohibited', 'violation', 'harmful', 'bias', 'privacy']
        risk_count = sum(1 for p in percepts if any(kw in p.lower() for kw in high_risk_keywords))
        
        return min(1.0, risk_count / max(1, len(percepts)))
    
    def _classify_risk(self, risk_score: float) -> str:
        """Classify risk level based on score."""
        if risk_score >= 0.8:
            return 'high'
        elif risk_score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _make_action_sentence(self, decision: Dict[str, Any], t: int) -> str:
        """Convert decision to sentence for knowledge base."""
        action = decision.get('action', 'unknown')
        risk = decision.get('risk_level', 'unknown')
        return f"Decision[{t}]: {action} (risk: {risk})"
    
    def evaluate_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate event and return decision.
        
        Args:
            event: Event dictionary with 'type' and 'content'
            
        Returns:
            Decision dictionary with 'action', 'risk_level', 'reasoning'
        """
        initial_state: DecisionState = {
            "goal": "Evaluate event for compliance",
            "messages": [],
            "plan": [],
            "results": [],
            "current_step": 0,
            "tools": [{"name": tool.name, "description": tool.description} for tool in self.tools],
            "error": None,
            "finished": False,
            "percept": event,
            "context": {},
            "options": [],
            "decision": None,
            "risk_level": None,
            "timestamp": self.time
        }
        
        try:
            final_state = self.app.invoke(initial_state)
            return {
                "decision": final_state.get("decision", {}),
                "risk_level": final_state.get("risk_level", "low"),
                "success": True
            }
        except Exception as e:
            return {
                "decision": {"action": "error", "reasoning": str(e)},
                "risk_level": "unknown",
                "success": False,
                "error": str(e)
            }
