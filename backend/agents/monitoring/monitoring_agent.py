"""
Monitoring Agent
Continuously observes AI-based applications to detect actions, user inputs, 
or system behavior that require compliance evaluation.
"""

from typing import Dict, Any, List, Optional, TypedDict
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.core.base_agent import BaseAgent, AgentState


class MonitoringState(AgentState):
    """Extended state for monitoring agent."""
    event: Optional[Dict[str, Any]]
    detected_anomalies: List[Dict[str, Any]]
    monitoring_decision: Optional[str]
    metrics: Dict[str, Any]
    timestamp: int


class MonitoringAgent(BaseAgent):
    """
    Monitoring Agent - Continuously observes AI application behavior.
    
    As described in the paper:
    - Tracks real-time system events
    - Identifies potential risks
    - Forwards events to orchestration agent when unusual/high-risk behavior appears
    """
    
    def __init__(self, base_dir: str, llm=None, memory=None):
        """Initialize monitoring agent."""
        self.base_dir = Path(base_dir)
        
        # Initialize LLM if not provided
        if llm is None:
            try:
                from langchain_openai import ChatOpenAI
                from backend.retrieval.utils.model_config import get_agent_model, AGENT_TEMPERATURE
                llm_model = get_agent_model()
                llm = ChatOpenAI(model=llm_model, temperature=AGENT_TEMPERATURE)
            except ImportError:
                # Fallback: use OpenAI client directly
                from backend.retrieval.utils.api_client import get_api_client
                from backend.retrieval.utils.model_config import get_agent_model
                llm = get_api_client()  # Use as LLM proxy
        
        # Monitoring agent doesn't need external tools - it observes and detects
        super().__init__(
            name="monitoring_agent",
            description="Monitors AI application behavior and detects compliance-related anomalies",
            tools=[],  # Monitoring uses internal logic, not external tools
            llm=llm,
            memory=memory,
            max_iterations=1  # Single-step: observe and decide
        )
        
        # Knowledge base of monitored data
        self.kb = []
        self.monitoring_model = {
            'rules': self._load_monitoring_rules(),
            'thresholds': self._load_thresholds(),
            'anomaly_detection': True
        }
        self.time = 0
    
    def _load_monitoring_rules(self) -> List[Dict[str, Any]]:
        """Load monitoring rules for detecting anomalies."""
        return [
            {'pattern': 'harmful_output', 'risk_level': 'high'},
            {'pattern': 'policy_sensitive_topic', 'risk_level': 'medium'},
            {'pattern': 'safety_violation', 'risk_level': 'high'},
            {'pattern': 'unexpected_behavior', 'risk_level': 'medium'},
            {'pattern': 'bias_indicator', 'risk_level': 'high'},
            {'pattern': 'privacy_violation', 'risk_level': 'high'},
        ]
    
    def _load_thresholds(self) -> Dict[str, float]:
        """Load thresholds for anomaly detection."""
        return {
            'similarity_threshold': 0.5,
            'risk_score_threshold': 0.7,
            'anomaly_confidence': 0.6
        }
    
    def _build_graph(self):
        """Build monitoring agent graph."""
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(MonitoringState)
        
        # Single node: monitor and decide
        workflow.add_node("monitor", self._monitor)
        workflow.set_entry_point("monitor")
        workflow.add_edge("monitor", END)
        
        return workflow
    
    def _plan(self, state: AgentState) -> AgentState:
        """Monitoring doesn't need planning - it observes."""
        return state
    
    def _execute(self, state: AgentState) -> AgentState:
        """Monitoring doesn't execute - it observes."""
        return state
    
    def _monitor(self, state: MonitoringState) -> MonitoringState:
        """
        Monitor event and make monitoring decision.
        Implements MONITORING-AGENT algorithm from paper.
        """
        event = state.get('event', {})
        self.time = state.get('timestamp', self.time)
        
        # Make event sentence
        event_sentence = self._make_event_sentence(event, self.time)
        self.kb.append(event_sentence)
        
        # Query state and metrics
        current_state = self._make_state_query(self.time)
        metrics = self._make_metric_query(current_state, self.time)
        
        # Apply monitoring model
        assessment = self._apply_monitoring_model(metrics, current_state)
        
        if assessment == 'anomaly':
            # Generate monitoring decision (alert)
            monitoring_decision = self._generate_event(assessment, self.time)
            state['monitoring_decision'] = monitoring_decision
            state['detected_anomalies'] = state.get('detected_anomalies', []) + [{
                'type': assessment,
                'timestamp': self.time,
                'event': event,
                'metrics': metrics
            }]
            # Tell KB about alert
            self.kb.append(f"ALERT: {monitoring_decision} at time {self.time}")
        else:
            # Generate status update
            update = self._generate_status_update(assessment, self.time)
            state['monitoring_decision'] = update
            state['metrics'] = metrics
        
        self.time += 1
        state['timestamp'] = self.time
        
        return state
    
    def _make_event_sentence(self, event: Dict[str, Any], t: int) -> str:
        """Convert event to sentence for knowledge base."""
        event_type = event.get('type', 'unknown')
        content = event.get('content', '')
        return f"Event[{t}]: {event_type} - {content[:100]}"
    
    def _make_state_query(self, t: int) -> Dict[str, Any]:
        """Query current state from knowledge base."""
        recent_events = [e for e in self.kb if f"[{t}]" in e or f"[{t-1}]" in e]
        return {
            'recent_events': recent_events,
            'time': t,
            'total_events': len(self.kb)
        }
    
    def _make_metric_query(self, state: Dict[str, Any], t: int) -> Dict[str, Any]:
        """Calculate metrics from state."""
        return {
            'event_count': len(state.get('recent_events', [])),
            'anomaly_count': len([e for e in state.get('recent_events', []) if 'ALERT' in e]),
            'risk_score': self._calculate_risk_score(state)
        }
    
    def _calculate_risk_score(self, state: Dict[str, Any]) -> float:
        """Calculate risk score from state."""
        recent_events = state.get('recent_events', [])
        if not recent_events:
            return 0.0
        
        # Check for high-risk patterns
        high_risk_patterns = ['harmful', 'violation', 'bias', 'privacy']
        risk_count = sum(1 for event in recent_events if any(pattern in event.lower() for pattern in high_risk_patterns))
        
        return min(1.0, risk_count / max(1, len(recent_events)))
    
    def _apply_monitoring_model(self, metrics: Dict[str, Any], state: Dict[str, Any]) -> str:
        """
        Apply monitoring model to detect anomalies.
        Returns: 'anomaly' or 'normal'
        """
        risk_score = metrics.get('risk_score', 0.0)
        threshold = self.monitoring_model['thresholds']['risk_score_threshold']
        
        if risk_score >= threshold:
            return 'anomaly'
        
        # Check for specific anomaly patterns
        recent_events = state.get('recent_events', [])
        for rule in self.monitoring_model['rules']:
            pattern = rule['pattern']
            if any(pattern in event.lower() for event in recent_events):
                return 'anomaly'
        
        return 'normal'
    
    def _generate_event(self, assessment: str, t: int) -> str:
        """Generate monitoring decision/alert."""
        return f"MONITORING_ALERT: {assessment} detected at time {t}"
    
    def _generate_status_update(self, assessment: str, t: int) -> str:
        """Generate status update."""
        return f"MONITORING_STATUS: {assessment} at time {t}"
    
    def monitor_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor a single event and return monitoring decision.
        
        Args:
            event: Event dictionary with 'type' and 'content'
            
        Returns:
            Monitoring decision dictionary
        """
        initial_state: MonitoringState = {
            "goal": "Monitor event for compliance",
            "messages": [],
            "plan": [],
            "results": [],
            "current_step": 0,
            "tools": [],
            "error": None,
            "finished": False,
            "event": event,
            "detected_anomalies": [],
            "monitoring_decision": None,
            "metrics": {},
            "timestamp": self.time
        }
        
        try:
            final_state = self.app.invoke(initial_state)
            return {
                "decision": final_state.get("monitoring_decision", "normal"),
                "anomalies": final_state.get("detected_anomalies", []),
                "metrics": final_state.get("metrics", {}),
                "requires_action": final_state.get("monitoring_decision", "").startswith("MONITORING_ALERT")
            }
        except Exception as e:
            return {
                "decision": "error",
                "anomalies": [],
                "metrics": {},
                "requires_action": False,
                "error": str(e)
            }
