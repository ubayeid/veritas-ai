# Agentic System Architecture

## Overview

The agentic system provides an extensible framework for building autonomous agents using LangGraph. The architecture supports multiple agents beyond compliance monitoring.

## Architecture Components

### 1. Base Agent (`base_agent.py`)

The `BaseAgent` class provides the foundation for all agents:

- **State Management**: TypedDict-based state management
- **Graph Building**: Abstract method for building LangGraph workflows
- **Execution Loop**: Standardized execution with planning, execution, and evaluation
- **Tool Integration**: LangChain tool integration
- **Memory**: Conversation memory support

### 2. LangGraph Agent (`langgraph_agent.py`)

The `ComplianceLangGraphAgent` extends `BaseAgent` with compliance-specific functionality:

- **Planning Node**: LLM-based planning with tool selection
- **Execution Node**: Tool execution with error handling
- **Evaluation Node**: Goal achievement checking
- **State Machine**: LangGraph workflow with conditional edges

### 3. Agent Registry (`agent_registry.py`)

Centralized registry for managing multiple agents:

- **Registration**: Register agent instances and classes
- **Retrieval**: Get agents by name
- **Discovery**: List available agents
- **Factory**: Create new agent instances

### 4. Prompt Management (`prompts.py`)

Centralized prompt management using LangChain:

- **Template Loading**: Load prompts from files
- **Programmatic Creation**: Create prompts programmatically
- **Caching**: Cache loaded prompts
- **Standardization**: Consistent prompt format

## State Machine Flow

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌─────────┐
│  PLAN   │  ← Creates plan using LLM
└────┬────┘
     │
     ▼
┌──────────┐
│ EXECUTE  │  ← Executes tool from plan
└────┬─────┘
     │
     ├───→ Continue? ──┐
     │                 │
     ▼                 │
┌──────────┐           │
│ EVALUATE │  ← Checks if goal achieved
└────┬─────┘           │
     │                 │
     └───→ Continue? ──┘
           │
           ▼
        ┌─────┐
        │ END │
        └─────┘
```

## Extending the System

### Adding a New Agent

1. **Create Agent Class**:

```python
from backend.agents.core.base_agent import BaseAgent, AgentState
from langgraph.graph import StateGraph

class MyAgent(BaseAgent):
    def __init__(self, ...):
        super().__init__(
            name="my_agent",
            description="My custom agent",
            tools=[...],
            llm=llm
        )
    
    def _build_graph(self):
        # Build your LangGraph workflow
        workflow = StateGraph(AgentState)
        # Add nodes and edges
        return workflow
    
    def _plan(self, state: AgentState):
        # Implement planning logic
        return state
    
    def _execute(self, state: AgentState):
        # Implement execution logic
        return state
```

2. **Register Agent**:

```python
from backend.agents.utils.agent_registry import register_agent

agent = MyAgent(...)
register_agent("my_agent", agent, MyAgent)
```

3. **Use Agent**:

```python
from backend.agents.utils.agent_registry import get_agent

agent = get_agent("my_agent")
result = agent.execute("My goal")
```

## State Structure

### Base State (`AgentState`)

```python
{
    "goal": str,
    "messages": List[Dict],
    "plan": List[Dict],
    "results": List[Dict],
    "current_step": int,
    "tools": List[Dict],
    "error": Optional[str],
    "finished": bool
}
```

### Extended State (e.g., `ComplianceAgentState`)

```python
{
    ...AgentState fields...,
    "compliance_gaps": List[Dict],
    "coverage_analysis": Dict,
    "report": Optional[str]
}
```

## Tool Integration

Tools are integrated using LangChain's `BaseTool` interface:

```python
from langchain.tools import StructuredTool

tool = StructuredTool.from_function(
    func=my_function,
    name="my_tool",
    description="Tool description"
)
```

## Memory Management

Memory is managed using LangChain's memory classes:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

## Benefits

1. **Extensibility**: Easy to add new agents
2. **Standardization**: Consistent interface across agents
3. **State Management**: Typed state with validation
4. **Error Handling**: Built-in error handling and recovery
5. **Observability**: LangGraph provides visualization
6. **Testing**: Easier to test individual nodes

## Future Agents

The architecture supports adding:

- **Risk Assessment Agent**: Analyze risks from incidents
- **Policy Generation Agent**: Generate compliance policies
- **Audit Agent**: Perform compliance audits
- **Reporting Agent**: Generate various reports

Each agent extends `BaseAgent` and implements its specific workflow.
