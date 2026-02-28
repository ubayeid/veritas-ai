"""
Compliance Verification Agent
Provides rule-based diagnostic that identifies which specific AI Act policy 
article or requirement the AI application is violating.
"""

from typing import Dict, Any, List, Optional, TypedDict
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.core.base_agent import BaseAgent, AgentState


class ComplianceState(AgentState):
    """Extended state for compliance verification agent."""
    percept: Optional[Dict[str, Any]]
    facts: Dict[str, Any]
    obligations: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]
    compliance_decision: Optional[Dict[str, Any]]
    violated_articles: List[str]
    violation_details: List[Dict[str, Any]]


class ComplianceVerificationAgent(BaseAgent):
    """
    Compliance Verification Agent - Identifies specific policy violations.
    
    As described in the paper:
    - Maps requests to structured AI policy database
    - Identifies exact type and category of policy violation
    - Explains detailed reasoning behind the violation
    """
    
    def __init__(self, base_dir: str, llm=None, memory=None):
        """Initialize compliance verification agent."""
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
        
        # Compliance agent uses RAG tools to query AI Act policies
        from backend.agents.utils.tools import ToolRegistry
        tool_registry = ToolRegistry(str(base_dir))
        tools = self._create_tools_from_registry(tool_registry)
        
        super().__init__(
            name="compliance_verification_agent",
            description="Identifies specific AI Act policy violations and provides detailed diagnostics",
            tools=tools,
            llm=llm,
            memory=memory,
            max_iterations=5
        )
        
        self.tool_registry = tool_registry
        self.kb = []
        self.ruleset = self._load_ai_act_rules()
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
    
    def _load_ai_act_rules(self) -> Dict[str, Any]:
        """Load AI Act ruleset for compliance checking."""
        return {
            'prohibited_practices': {
                'article': 'Article 5',
                'practices': [
                    'cognitive_behavior_manipulation',
                    'exploiting_vulnerabilities',
                    'social_scoring',
                    'real_time_biometric_identification'
                ]
            },
            'high_risk_requirements': {
                'article': 'Article 6',
                'requirements': [
                    'risk_management',
                    'data_governance',
                    'technical_documentation',
                    'record_keeping',
                    'transparency',
                    'human_oversight',
                    'accuracy_robustness_security'
                ]
            },
            'transparency_obligations': {
                'article': 'Article 50',
                'obligations': [
                    'disclose_ai_generated_content',
                    'prevent_illegal_content',
                    'publish_copyright_summaries'
                ]
            }
        }
    
    def _build_graph(self):
        """Build compliance verification agent graph."""
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(ComplianceState)
        
        # CV-AGENT algorithm: Percept -> Facts -> Obligations -> Risks -> Decision
        workflow.add_node("analyze", self._analyze_percept)
        workflow.add_node("verify", self._verify_compliance)
        
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "verify")
        workflow.add_edge("verify", END)
        
        return workflow
    
    def _plan(self, state: AgentState) -> AgentState:
        """Compliance verification doesn't need separate planning."""
        return state
    
    def _execute(self, state: AgentState) -> AgentState:
        """Compliance verification executes in analyze node."""
        return state
    
    def _analyze_percept(self, state: ComplianceState) -> ComplianceState:
        """
        Analyze percept and extract facts.
        Implements CV-AGENT algorithm from paper.
        """
        percept = state.get('percept', {})
        self.time = state.get('timestamp', self.time)
        
        # Make percept sentence
        percept_sentence = self._make_percept_sentence(percept, self.time)
        self.kb.append(percept_sentence)
        
        # Query facts
        facts = self._make_fact_query(self.time)
        state['facts'] = facts
        
        # Query obligations
        obligations = self._make_obligation_query(facts, self.time)
        state['obligations'] = obligations
        
        # Query risks
        risks = self._make_risk_query(facts, self.time)
        state['risks'] = risks
        
        return state
    
    def _verify_compliance(self, state: ComplianceState) -> ComplianceState:
        """Apply compliance rules to make verification decision."""
        facts = state.get('facts', {})
        obligations = state.get('obligations', [])
        risks = state.get('risks', [])
        
        # Apply compliance rules
        decision = self._apply_compliance_rules(facts, obligations, risks)
        state['compliance_decision'] = decision
        state['violated_articles'] = decision.get('violated_articles', [])
        state['violation_details'] = decision.get('violation_details', [])
        
        # Make compliance sentence
        compliance_sentence = self._make_compliance_sentence(decision, self.time)
        self.kb.append(compliance_sentence)
        
        self.time += 1
        
        return state
    
    def _make_percept_sentence(self, percept: Dict[str, Any], t: int) -> str:
        """Convert percept to sentence for knowledge base."""
        event_type = percept.get('type', 'unknown')
        content = percept.get('content', '')
        return f"Percept[{t}]: {event_type} - {content[:200]}"
    
    def _make_fact_query(self, t: int) -> Dict[str, Any]:
        """Query facts from knowledge base and percept."""
        recent_percepts = [p for p in self.kb if f"[{t}]" in p or f"[{t-1}]" in p]
        return {
            'recent_percepts': recent_percepts,
            'time': t,
            'ai_system_type': self._extract_system_type(recent_percepts),
            'behavior': self._extract_behavior(recent_percepts)
        }
    
    def _make_obligation_query(self, facts: Dict[str, Any], t: int) -> List[Dict[str, Any]]:
        """Query obligations based on facts."""
        system_type = facts.get('ai_system_type', 'unknown')
        behavior = facts.get('behavior', {})
        
        obligations = []
        
        # Check against AI Act ruleset
        for category, rules in self.ruleset.items():
            if self._matches_category(system_type, behavior, category):
                obligations.append({
                    'category': category,
                    'article': rules.get('article', ''),
                    'requirements': rules.get('requirements', rules.get('practices', rules.get('obligations', []))),
                    'applies': True
                })
        
        return obligations
    
    def _make_risk_query(self, facts: Dict[str, Any], t: int) -> List[Dict[str, Any]]:
        """Query risks based on facts."""
        behavior = facts.get('behavior', {})
        risks = []
        
        # Check for prohibited practices
        if self._has_prohibited_practice(behavior):
            risks.append({
                'type': 'prohibited_practice',
                'severity': 'high',
                'article': 'Article 5'
            })
        
        # Check for high-risk indicators
        if self._has_high_risk_indicator(behavior):
            risks.append({
                'type': 'high_risk',
                'severity': 'high',
                'article': 'Article 6'
            })
        
        return risks
    
    def _apply_compliance_rules(self, facts: Dict[str, Any], obligations: List[Dict[str, Any]], risks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply compliance rules to determine violations.
        Implements CV-AGENT compliance logic.
        """
        violated_articles = []
        violation_details = []
        
        # Check for violations based on risks
        for risk in risks:
            article = risk.get('article', '')
            if article and article not in violated_articles:
                violated_articles.append(article)
                violation_details.append({
                    'article': article,
                    'violation_type': risk.get('type', ''),
                    'severity': risk.get('severity', 'medium'),
                    'reasoning': f"Detected {risk.get('type', 'violation')} requiring {article}"
                })
        
        # Check obligations for unmet requirements
        for obligation in obligations:
            if obligation.get('applies', False):
                article = obligation.get('article', '')
                requirements = obligation.get('requirements', [])
                
                # Check if requirements are met (simplified - would need actual verification)
                unmet = self._check_unmet_requirements(facts, requirements)
                if unmet:
                    if article not in violated_articles:
                        violated_articles.append(article)
                    violation_details.append({
                        'article': article,
                        'violation_type': 'unmet_requirement',
                        'severity': 'high',
                        'reasoning': f"Unmet requirements: {', '.join(unmet)}"
                    })
        
        return {
            'compliant': len(violated_articles) == 0,
            'violated_articles': violated_articles,
            'violation_details': violation_details,
            'obligations_checked': len(obligations),
            'risks_identified': len(risks)
        }
    
    def _extract_system_type(self, percepts: List[str]) -> str:
        """Extract AI system type from percepts."""
        # Simplified extraction
        if any('chatbot' in p.lower() or 'llm' in p.lower() for p in percepts):
            return 'generative_ai'
        elif any('biometric' in p.lower() for p in percepts):
            return 'biometric_system'
        elif any('scoring' in p.lower() for p in percepts):
            return 'scoring_system'
        return 'general_ai'
    
    def _extract_behavior(self, percepts: List[str]) -> Dict[str, Any]:
        """Extract behavior indicators from percepts."""
        behavior = {
            'has_manipulation': any('manipulation' in p.lower() for p in percepts),
            'has_biometric': any('biometric' in p.lower() for p in percepts),
            'has_scoring': any('scoring' in p.lower() for p in percepts),
            'has_harmful_content': any('harmful' in p.lower() or 'violation' in p.lower() for p in percepts)
        }
        return behavior
    
    def _matches_category(self, system_type: str, behavior: Dict[str, Any], category: str) -> bool:
        """Check if system/behavior matches AI Act category."""
        if category == 'prohibited_practices':
            return behavior.get('has_manipulation') or behavior.get('has_biometric') or behavior.get('has_scoring')
        elif category == 'high_risk_requirements':
            return system_type in ['biometric_system', 'scoring_system']
        elif category == 'transparency_obligations':
            return system_type == 'generative_ai'
        return False
    
    def _has_prohibited_practice(self, behavior: Dict[str, Any]) -> bool:
        """Check if behavior contains prohibited practices."""
        return behavior.get('has_manipulation') or behavior.get('has_scoring')
    
    def _has_high_risk_indicator(self, behavior: Dict[str, Any]) -> bool:
        """Check if behavior has high-risk indicators."""
        return behavior.get('has_biometric') or behavior.get('has_harmful_content')
    
    def _check_unmet_requirements(self, facts: Dict[str, Any], requirements: List[str]) -> List[str]:
        """Check which requirements are unmet (simplified)."""
        # In real implementation, would verify each requirement
        # For now, return empty list (all met) or sample unmet requirements
        return []
    
    def _make_compliance_sentence(self, decision: Dict[str, Any], t: int) -> str:
        """Convert compliance decision to sentence for knowledge base."""
        compliant = decision.get('compliant', False)
        articles = decision.get('violated_articles', [])
        if compliant:
            return f"Compliance[{t}]: COMPLIANT - No violations detected"
        else:
            return f"Compliance[{t}]: NON-COMPLIANT - Violated articles: {', '.join(articles)}"
    
    def verify_compliance(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify compliance and return violation details.
        
        Args:
            event: Event dictionary with 'type' and 'content'
            
        Returns:
            Compliance verification dictionary with violations and details
        """
        initial_state: ComplianceState = {
            "goal": "Verify compliance with AI Act",
            "messages": [],
            "plan": [],
            "results": [],
            "current_step": 0,
            "tools": [{"name": tool.name, "description": tool.description} for tool in self.tools],
            "error": None,
            "finished": False,
            "percept": event,
            "facts": {},
            "obligations": [],
            "risks": [],
            "compliance_decision": None,
            "violated_articles": [],
            "violation_details": [],
            "timestamp": self.time
        }
        
        try:
            final_state = self.app.invoke(initial_state)
            return {
                "compliance_decision": final_state.get("compliance_decision", {}),
                "violated_articles": final_state.get("violated_articles", []),
                "violation_details": final_state.get("violation_details", []),
                "success": True
            }
        except Exception as e:
            return {
                "compliance_decision": {"compliant": False, "error": str(e)},
                "violated_articles": [],
                "violation_details": [],
                "success": False,
                "error": str(e)
            }
