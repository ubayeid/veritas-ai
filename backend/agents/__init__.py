"""
Agentic System Module
Multi-agent architecture for compliance monitoring.
"""

# Lazy imports to avoid breaking if langchain not available
# Only import when actually needed (agent mode)

def _lazy_import_agents():
    """Lazy import agents - only when needed."""
    try:
        from .core.langgraph_agent import ComplianceLangGraphAgent
        from .monitoring.monitoring_agent import MonitoringAgent
        from .decision_making.decision_agent import DecisionMakingAgent
        from .compliance.compliance_agent import ComplianceVerificationAgent
        from .orchestration.orchestration_agent import OrchestrationAgent
        from .utils.agent_registry import get_agent, register_agent, get_registry
        
        return {
            'ComplianceLangGraphAgent': ComplianceLangGraphAgent,
            'MonitoringAgent': MonitoringAgent,
            'DecisionMakingAgent': DecisionMakingAgent,
            'ComplianceVerificationAgent': ComplianceVerificationAgent,
            'OrchestrationAgent': OrchestrationAgent,
            'get_agent': get_agent,
            'register_agent': register_agent,
            'get_registry': get_registry
        }
    except ImportError as e:
        raise ImportError(
            f"Agent modules require langchain dependencies. "
            f"Install with: pip install langchain langchain-openai langgraph langchain-community\n"
            f"Original error: {e}"
        )

# Export lazy loader - actual imports happen on access
__all__ = [
    'ComplianceLangGraphAgent',
    'MonitoringAgent',
    'DecisionMakingAgent',
    'ComplianceVerificationAgent',
    'OrchestrationAgent',
    'get_agent',
    'register_agent',
    'get_registry'
]

# Create lazy accessors
def __getattr__(name):
    if name in __all__:
        agents = _lazy_import_agents()
        return agents.get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
