# Agentic System Guide

Complete guide to the agentic compliance analysis system.

## Table of Contents
- [What Makes It Agentic](#what-makes-it-agentic)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Tools](#tools)
- [Examples](#examples)

---

## What Makes It Agentic

An **agentic system** can:
1. **Plan** - Break down complex goals into steps
2. **Decide** - Choose which tools/actions to use
3. **Execute** - Run multiple steps autonomously
4. **Observe** - Check if goals are achieved
5. **Adapt** - Adjust plans based on results
6. **Learn** - Remember patterns from past executions

### Reactive vs Agentic

**❌ Reactive System (Original Chatbot):**
```
User: "Find compliance gaps"
System: [Searches once, returns results]
User: "Now generate a report"
System: [Searches again, generates report]
```

**✅ Agentic System:**
```
User: "Find compliance gaps and generate a report"
Agent: 
  [PLAN] Creates 2-step plan
  [EXECUTE] Runs both steps autonomously
  [RETURN] Complete result
```

---

## Quick Start

### Run the Agent

```bash
# Option 1: Using Makefile (Recommended)
make run-agent

# Option 2: Activate venv first
source venv/bin/activate
python3 backend/agentic/run_agent.py

# Option 3: Direct venv Python
venv/bin/python3 backend/agentic/run_agent.py
```

### Use in Code

```python
from backend.agentic import ComplianceAgent

agent = ComplianceAgent("/path/to/comp_rag")
result = agent.execute_goal("Find compliance gaps and generate report")
```

---

## Usage

### Basic Usage

```python
from backend.agentic import ComplianceAgent

# Initialize
agent = ComplianceAgent(base_dir="/path/to/project")

# Execute a goal
result = agent.execute_goal("Find compliance gaps")

# Check results
print(f"Success: {result['success']}")
print(f"Steps executed: {result['steps_executed']}")
for step_result in result['results']:
    print(f"Step {step_result['step_id']}: {step_result['result']}")
```

### Interactive Mode

Run `make run-agent` and enter goals like:
- "Find compliance gaps"
- "Compare privacy policy with GDPR"
- "Generate full compliance report"
- "Find incidents related to data breaches"

---

## Architecture

### Components

1. **ComplianceAgent** (`agent.py`) - Main orchestrator
2. **ToolRegistry** (`tools.py`) - Available tools
3. **TaskPlanner** (`planner.py`) - LLM-based planning
4. **TaskExecutor** (`executor.py`) - Step execution
5. **AgentMemory** (`memory.py`) - Context and history

### Execution Flow

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

---

## Tools

Available tools the agent can use:

### 1. `search_compliance_gaps`
Find GDPR articles not covered by company documents.

**Parameters:**
- `similarity_threshold` (float, default: 0.45) - Minimum similarity for matching
- `article_ids` (list, optional) - Specific articles to check

**Example:**
```python
result = tools.call_tool('search_compliance_gaps', similarity_threshold=0.45)
# Returns: {total_gaps: 94, gaps: [...], coverage: [...]}
```

### 2. `map_clauses_to_articles`
Find which company clauses address which GDPR articles.

**Parameters:**
- `article_id` (str, optional) - Specific article to find clauses for
- `top_k` (int, default: 10) - Number of matches to return

### 3. `generate_compliance_report`
Generate comprehensive compliance assessment report.

**Parameters:**
- `report_type` (str, default: 'full') - Type: "gaps", "coverage", "full"
- `include_recommendations` (bool, default: True)

### 4. `compare_documents`
Compare company documents against GDPR standards.

**Parameters:**
- `document_name` (str, optional) - Specific document to compare
- `top_k` (int, default: 5) - Number of matches per query

### 5. `find_related_incidents`
Find AIID incidents related to GDPR articles or topics.

**Parameters:**
- `article_id` (str, optional) - GDPR article ID
- `topic` (str, optional) - Topic/keyword to search
- `top_k` (int, default: 10) - Number of incidents

### 6. `search_vector`
Perform semantic vector search across all databases.

**Parameters:**
- `query` (str, required) - Search query
- `db_names` (list, optional) - Databases: ["company", "aiid", "standards"]
- `top_k` (int, default: 10) - Number of results

---

## Examples

### Example 1: Compliance Gap Analysis

```python
agent = ComplianceAgent(base_dir)

result = agent.execute_goal("Find compliance gaps")

# Agent automatically:
# 1. Plans: [search_compliance_gaps()]
# 2. Executes: Searches for gaps
# 3. Returns: {total_gaps: 94, gaps: [...], coverage: [...]}
```

### Example 2: Multi-Step Analysis

```python
result = agent.execute_goal(
    "Compare privacy policy with GDPR and find related incidents"
)

# Agent automatically:
# 1. Plans: [compare_documents, search_gaps, find_incidents]
# 2. Executes all steps
# 3. Returns comprehensive analysis
```

### Example 3: Report Generation

```python
result = agent.execute_goal("Generate full compliance report")

# Agent automatically:
# 1. Plans: [gather_data, analyze_gaps, generate_report]
# 2. Executes autonomously
# 3. Returns complete compliance assessment
```

---

## Key Features

### ✅ Autonomous Planning
- Agent decides what steps are needed
- Uses LLM to understand goals
- Creates structured execution plans

### ✅ Multi-Step Execution
- Executes multiple tools in sequence
- No user intervention between steps
- Handles dependencies automatically

### ✅ Decision Making
- Chooses which tools to use
- Decides when to stop
- Adapts plans based on results

### ✅ State Management
- Tracks execution progress
- Maintains context across steps
- Remembers past executions

---

## Troubleshooting

### Issue: "No module named 'numpy'"
**Solution**: Activate virtual environment or use `make run-agent`

### Issue: "Failed to import query engines"
**Solution**: Make sure you're running from project root directory

### Issue: "Neo4j connection failed"
**Solution**: OK for basic testing - agent falls back to vector-only mode

---

## Next Steps

1. **Try It**: Run `make run-agent` and test with simple goals
2. **Add Tools**: Extend `ToolRegistry` with more capabilities
3. **Improve Planning**: Enhance LLM prompts in `planner.py`
4. **Integrate**: Add agent mode to existing chatbot
5. **Add Learning**: Implement reflection and pattern learning

For more details, see the code in `backend/agentic/`.

