"""
Orchestration Agent
Central coordinator that manages communication and workflow between all agents.
Aggregates decisions, computes final compliance score, and initiates actions.
"""

from typing import Dict, Any, List, Optional, TypedDict
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.core.base_agent import BaseAgent, AgentState
from backend.agents.monitoring.monitoring_agent import MonitoringAgent
from backend.agents.decision_making.decision_agent import DecisionMakingAgent
from backend.agents.compliance.compliance_agent import ComplianceVerificationAgent


class OrchestrationState(AgentState):
    """Extended state for orchestration agent."""
    request: Optional[Dict[str, Any]]
    monitor_signal: Optional[Dict[str, Any]]
    dm_decision: Optional[Dict[str, Any]]
    cv_decision: Optional[Dict[str, Any]]
    context: Dict[str, Any]
    final_decision: Optional[str]  # 'deny', 'permit', 'permit_with_conditions'
    actions: List[str]
    compliance_score: Optional[float]
    audit_record: Optional[Dict[str, Any]]


class OrchestrationAgent(BaseAgent):
    """
    Orchestration Agent - Central coordinator for all agents.
    
    As described in the paper:
    - Receives requests from monitoring agent or audit engineer
    - Distributes requests to decision-making and compliance verification agents
    - Collects analysis reports from each agent
    - Computes final compliance score
    - Takes appropriate system-level actions
    """
    
    def __init__(self, base_dir: str, llm=None, memory=None):
        """Initialize orchestration agent."""
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
        
        # Orchestration agent coordinates other agents (no external tools needed)
        super().__init__(
            name="orchestration_agent",
            description="Orchestrates all agents and makes final compliance decisions",
            tools=[],  # Orchestration coordinates agents, doesn't use external tools
            llm=llm,
            memory=memory,
            max_iterations=1
        )
        
        # Initialize sub-agents
        self.monitoring_agent = MonitoringAgent(str(base_dir), llm=llm, memory=memory)
        self.decision_agent = DecisionMakingAgent(str(base_dir), llm=llm, memory=memory)
        self.compliance_agent = ComplianceVerificationAgent(str(base_dir), llm=llm, memory=memory)
        
        # Knowledge base and audit engine
        self.kb = []
        self.audit_engine = []
        self.application_interface = {
            'stop': lambda: print("[ACTION] Stopping application"),
            'halt': lambda: print("[ACTION] Halting application"),
            'continue': lambda: print("[ACTION] Continuing application")
        }
        self.time = 0
    
    def _build_graph(self):
        """Build orchestration agent graph."""
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(OrchestrationState)
        
        # ORCH-AGENT algorithm: Request -> Monitor -> DM -> CV -> Merge -> Resolve -> Execute
        workflow.add_node("orchestrate", self._orchestrate)
        
        workflow.set_entry_point("orchestrate")
        workflow.add_edge("orchestrate", END)
        
        return workflow
    
    def _plan(self, state: AgentState) -> AgentState:
        """Orchestration doesn't need separate planning."""
        return state
    
    def _execute(self, state: AgentState) -> AgentState:
        """Orchestration executes in orchestrate node."""
        return state
    
    def _orchestrate(self, state: OrchestrationState) -> OrchestrationState:
        """
        Orchestrate all agents and make final decision.
        Implements ORCH-AGENT algorithm from paper.
        """
        request = state.get('request', {})
        self.time = state.get('timestamp', self.time)
        
        # Make request sentence
        request_sentence = self._make_request_sentence(request, self.time)
        self.kb.append(request_sentence)
        
        # Step 1: Call Monitoring Agent
        monitor_signal = self.monitoring_agent.monitor_event(request)
        state['monitor_signal'] = monitor_signal
        
        # Step 2: Call Decision Making Agent
        dm_decision = self.decision_agent.evaluate_event(request)
        state['dm_decision'] = dm_decision
        
        # Step 3: Call Compliance Verification Agent
        cv_decision = self.compliance_agent.verify_compliance(request)
        state['cv_decision'] = cv_decision
        
        # Step 4: Merge context
        context = self._merge_context(monitor_signal, dm_decision, cv_decision)
        state['context'] = context
        context_sentence = self._make_context_sentence(context, self.time)
        self.kb.append(context_sentence)
        
        # Step 5: Resolve decisions
        final_decision = self._resolve_decisions(monitor_signal, dm_decision, cv_decision)
        state['final_decision'] = final_decision
        
        # Step 6: Compute compliance score
        compliance_score = self._compute_compliance_score(monitor_signal, dm_decision, cv_decision)
        state['compliance_score'] = compliance_score
        
        # Step 7: Execute actions
        actions = []
        if final_decision == 'deny':
            action = self._select_action(['stop', 'halt'], context)
            result = self._execute_action(action)
            actions.append(action)
        elif final_decision == 'permit_with_conditions':
            actions_list = self._apply_conditions(dm_decision, context)
            result = self._execute_actions(actions_list)
            actions.extend(actions_list)
        else:  # permit
            actions_list = self._materialize(dm_decision)
            result = self._execute_actions(actions_list)
            actions.extend(actions_list)
        
        state['actions'] = actions
        
        # Step 8: Create audit record
        audit_record = self._make_audit_record(self.time, request, context, final_decision, actions, result)
        state['audit_record'] = audit_record
        self.audit_engine.append(audit_record)
        
        self.time += 1
        
        return state
    
    def _make_request_sentence(self, request: Dict[str, Any], t: int) -> str:
        """Convert request to sentence for knowledge base."""
        request_type = request.get('type', 'unknown')
        content = request.get('content', '')
        return f"Request[{t}]: {request_type} - {content[:200]}"
    
    def _merge_context(self, monitor_signal: Dict[str, Any], dm_decision: Dict[str, Any], cv_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Merge context from all agents."""
        return {
            'monitoring': {
                'decision': monitor_signal.get('decision', ''),
                'anomalies': monitor_signal.get('anomalies', []),
                'requires_action': monitor_signal.get('requires_action', False)
            },
            'decision_making': {
                'decision': dm_decision.get('decision', {}),
                'risk_level': dm_decision.get('risk_level', 'low')
            },
            'compliance': {
                'compliant': cv_decision.get('compliance_decision', {}).get('compliant', True),
                'violated_articles': cv_decision.get('violated_articles', []),
                'violation_details': cv_decision.get('violation_details', [])
            }
        }
    
    def _make_context_sentence(self, context: Dict[str, Any], t: int) -> str:
        """Convert context to sentence for knowledge base."""
        risk = context.get('decision_making', {}).get('risk_level', 'unknown')
        compliant = context.get('compliance', {}).get('compliant', True)
        return f"Context[{t}]: risk={risk}, compliant={compliant}"
    
    def _resolve_decisions(self, monitor_signal: Dict[str, Any], dm_decision: Dict[str, Any], cv_decision: Dict[str, Any]) -> str:
        """
        Resolve decisions from all agents.
        Compliance and safety override all other decisions.
        """
        # Check compliance first (highest priority)
        compliance_info = cv_decision.get('compliance_decision', {})
        if not compliance_info.get('compliant', True):
            violated_articles = cv_decision.get('violated_articles', [])
            if violated_articles:
                return 'deny'  # Non-compliance -> deny
        
        # Check monitoring alerts
        if monitor_signal.get('requires_action', False):
            return 'deny'  # Anomaly detected -> deny
        
        # Check decision making risk level
        risk_level = dm_decision.get('risk_level', 'low')
        if risk_level == 'high':
            return 'deny'
        elif risk_level == 'medium':
            return 'permit_with_conditions'
        else:
            return 'permit'
    
    def _compute_compliance_score(self, monitor_signal: Dict[str, Any], dm_decision: Dict[str, Any], cv_decision: Dict[str, Any]) -> float:
        """Compute final compliance score (0.0 to 1.0)."""
        # Start with perfect score
        score = 1.0
        
        # Deduct for monitoring anomalies
        if monitor_signal.get('requires_action', False):
            score -= 0.3
        
        # Deduct for decision making risk
        risk_level = dm_decision.get('risk_level', 'low')
        if risk_level == 'high':
            score -= 0.5
        elif risk_level == 'medium':
            score -= 0.2
        
        # Deduct for compliance violations
        compliance_info = cv_decision.get('compliance_decision', {})
        if not compliance_info.get('compliant', True):
            violations = len(cv_decision.get('violated_articles', []))
            score -= min(0.5, violations * 0.1)
        
        return max(0.0, score)
    
    def _select_action(self, options: List[str], context: Dict[str, Any]) -> str:
        """Select appropriate action from options."""
        # Prefer 'halt' over 'stop' for less severe cases
        if 'halt' in options:
            return 'halt'
        return options[0] if options else 'continue'
    
    def _apply_conditions(self, dm_decision: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Apply conditions from decision making agent."""
        decision = dm_decision.get('decision', {})
        action = decision.get('action', 'permit')
        
        if action == 'permit_with_conditions':
            return ['monitor', 'log']
        return []
    
    def _materialize(self, dm_decision: Dict[str, Any]) -> List[str]:
        """Materialize decision into actions."""
        decision = dm_decision.get('decision', {})
        action = decision.get('action', 'permit')
        
        if action == 'permit':
            return ['continue']
        return []
    
    def _execute_action(self, action: str) -> Dict[str, Any]:
        """Execute a single action."""
        if action in self.application_interface:
            self.application_interface[action]()
            return {'action': action, 'status': 'executed'}
        return {'action': action, 'status': 'unknown'}
    
    def _execute_actions(self, actions: List[str]) -> Dict[str, Any]:
        """Execute multiple actions."""
        results = []
        for action in actions:
            result = self._execute_action(action)
            results.append(result)
        return {'actions': results, 'status': 'completed'}
    
    def _make_audit_record(self, t: int, request: Dict[str, Any], context: Dict[str, Any], 
                          final_decision: str, actions: List[str], result: Dict[str, Any]) -> Dict[str, Any]:
        """Create audit record."""
        return {
            'timestamp': t,
            'request': request,
            'context': context,
            'final_decision': final_decision,
            'actions': actions,
            'result': result,
            'compliance_score': self._compute_compliance_score(
                context.get('monitoring', {}),
                context.get('decision_making', {}),
                context.get('compliance', {})
            )
        }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request through the orchestrated agent system.
        
        Args:
            request: Request dictionary with 'type' and 'content'
            
        Returns:
            Orchestration result with final decision, compliance score, and actions
        """
        initial_state: OrchestrationState = {
            "goal": "Orchestrate compliance evaluation",
            "messages": [],
            "plan": [],
            "results": [],
            "current_step": 0,
            "tools": [],
            "error": None,
            "finished": False,
            "request": request,
            "monitor_signal": None,
            "dm_decision": None,
            "cv_decision": None,
            "context": {},
            "final_decision": None,
            "actions": [],
            "compliance_score": None,
            "audit_record": None,
            "timestamp": self.time
        }
        
        try:
            final_state = self.app.invoke(initial_state)
            return {
                "final_decision": final_state.get("final_decision", "permit"),
                "compliance_score": final_state.get("compliance_score", 1.0),
                "actions": final_state.get("actions", []),
                "monitoring": final_state.get("monitor_signal", {}),
                "decision_making": final_state.get("dm_decision", {}),
                "compliance": final_state.get("cv_decision", {}),
                "audit_record": final_state.get("audit_record", {}),
                "success": True
            }
        except Exception as e:
            return {
                "final_decision": "error",
                "compliance_score": 0.0,
                "actions": [],
                "success": False,
                "error": str(e)
            }
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log."""
        return self.audit_engine
