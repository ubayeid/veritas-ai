# Agentic System Module

This module implements an agentic system for autonomous compliance analysis.

**📚 For complete documentation, see [docs/AGENTIC_SYSTEM.md](../../docs/AGENTIC_SYSTEM.md)**

## Quick Overview

The agentic system provides:
- **Planning**: Breaks down complex goals into executable steps
- **Tool Execution**: Structured way to call capabilities
- **Autonomous Operation**: Works without constant user input
- **Learning**: Remembers past executions and improves

## Components

### 1. `ComplianceAgent` (`agent.py`)
Main orchestrator that coordinates planning and execution.

### 2. `ToolRegistry` (`tools.py`)
Registry of available tools:
- `search_compliance_gaps` - Find GDPR articles not covered
- `map_clauses_to_articles` - Map clauses to articles
- `generate_compliance_report` - Generate compliance reports
- `compare_documents` - Compare company docs vs GDPR
- `find_related_incidents` - Find AIID incidents
- `search_vector` - Semantic vector search

### 3. `TaskPlanner` (`planner.py`)
Uses LLM to break down goals into steps.

### 4. `TaskExecutor` (`executor.py`)
Executes planned steps using tools.

### 5. `AgentMemory` (`memory.py`)
Manages conversation context and task history.

## Usage

### Basic Usage

```python
from backend.agentic import ComplianceAgent

# Initialize agent
agent = ComplianceAgent(base_dir="/path/to/project")

# Execute a goal
result = agent.execute_goal("Find compliance gaps and generate a report")

print(f"Success: {result['success']}")
print(f"Steps executed: {result['steps_executed']}")
```

### Interactive Mode

```bash
python backend/agentic/run_agent.py
```

Example session:
```
Goal: Find compliance gaps
[AGENT] Planning for goal: Find compliance gaps
[AGENT] Created plan with 2 steps:
  Step 1: Find GDPR articles not covered by company documents
  Step 2: Generate comprehensive compliance report

[AGENT] Executing plan...
[AGENT] Step 1/2: Find GDPR articles not covered...
✓ Step 1: Found 94 compliance gaps
[AGENT] Step 2/2: Generate comprehensive compliance report...
✓ Step 2: Report generated
```

## Example Goals

- "Find compliance gaps"
- "Compare company privacy policy with GDPR requirements"
- "Find incidents related to data breaches"
- "Generate a full compliance assessment report"
- "Map all clauses to GDPR articles"

## Architecture

```
User Goal
    ↓
ComplianceAgent
    ↓
TaskPlanner (LLM) → Creates Plan
    ↓
TaskExecutor → Executes Steps
    ↓
ToolRegistry → Calls Tools
    ↓
Results → AgentMemory → Learning
```

## Next Steps

1. **Add More Tools**: Wrap more existing functions
2. **Improve Planning**: Better LLM prompts for complex tasks
3. **Add Reflection**: Analyze results and improve plans
4. **Persistent Memory**: Store learnings across sessions
5. **Integration**: Add agent mode to existing chatbot

## Testing

```python
# Test individual components
from backend.agentic.tools import ToolRegistry
from backend.agentic.planner import TaskPlanner

# Test tools
tools = ToolRegistry("/path/to/project")
result = tools.call_tool('search_compliance_gaps')
print(result)

# Test planner
planner = TaskPlanner()
plan = planner.create_plan(
    "Find compliance gaps",
    tools.list_tools()
)
print(plan)
```

