# Multi-Agent Architecture

This module implements the 4-agent architecture described in the research paper:
"Securing AI Systems with AI: An Agentic Approach for European AI Act"

## Architecture Overview

The system consists of four specialized agents working together:

1. **Monitoring Agent** - Continuously observes AI application behavior
2. **Decision Making Agent** - Evaluates compliance with AI policy
3. **Compliance Verification Agent** - Identifies specific policy violations
4. **Orchestration Agent** - Coordinates all agents and makes final decisions

## Directory Structure

```
backend/agents/
├── monitoring/              # Monitoring Agent
│   ├── __init__.py
│   └── monitoring_agent.py
├── decision_making/         # Decision Making Agent
│   ├── __init__.py
│   └── decision_agent.py
├── compliance/             # Compliance Verification Agent
│   ├── __init__.py
│   └── compliance_agent.py
├── orchestration/          # Orchestration Agent
│   ├── __init__.py
│   └── orchestration_agent.py
├── core/                   # Base classes
│   ├── base_agent.py
│   └── langgraph_agent.py  # Legacy single agent
└── utils/                  # Utilities
    ├── agent_registry.py
    ├── tools.py
    └── prompts.py
```

## Usage

### Via CLI

```bash
# Run multi-agent orchestration system
python query.py agent
```

### Programmatically

```python
from backend.agents import OrchestrationAgent

# Initialize orchestrator
orchestrator = OrchestrationAgent(base_dir=".")

# Process a request
request = {
    'type': 'user_query',
    'content': 'Does this AI system comply with AI Act?',
    'source': 'audit_engineer'
}

result = orchestrator.process_request(request)
print(f"Decision: {result['final_decision']}")
print(f"Compliance Score: {result['compliance_score']}")
```

## Agent Workflow

```
Request → Orchestration Agent
    ↓
    ├─→ Monitoring Agent (detects anomalies)
    ├─→ Decision Making Agent (evaluates risk)
    └─→ Compliance Verification Agent (identifies violations)
    ↓
Orchestration Agent merges results
    ↓
Final Decision + Compliance Score + Actions
```

## Individual Agents

### Monitoring Agent

- **Purpose**: Continuously observes AI application behavior
- **Output**: Monitoring decision (alert or status update)
- **Key Features**:
  - Real-time event monitoring
  - Anomaly detection
  - Risk score calculation

### Decision Making Agent

- **Purpose**: Evaluates compliance with AI policy
- **Output**: Decision with risk level (high/medium/low)
- **Key Features**:
  - Policy rule evaluation
  - Risk classification
  - Action recommendation

### Compliance Verification Agent

- **Purpose**: Identifies specific AI Act policy violations
- **Output**: Violation details with article references
- **Key Features**:
  - Article-level violation detection
  - Detailed reasoning
  - Policy mapping

### Orchestration Agent

- **Purpose**: Coordinates all agents and makes final decisions
- **Output**: Final decision, compliance score, actions
- **Key Features**:
  - Agent coordination
  - Decision resolution
  - Action execution
  - Audit logging

## Integration with RAG

All agents leverage the RAG system:
- **Vector Search** (FAISS) for semantic similarity
- **Graph Traversal** (Neo4j) for relationship queries
- **Hybrid Search** (RRF) for comprehensive retrieval

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture documentation
- `query.py` - Main CLI entry point
- `backend/retrieval/` - RAG system used by agents
