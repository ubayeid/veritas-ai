"""
Centralized prompt management for the compliance RAG system.
All prompts are stored here for easy access and versioning.
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

# Prompt file paths
COMPLIANCE_MONITORING_PROMPT = PROMPTS_DIR / "compliance_monitoring.txt"
AIID_INCIDENT_ANALYSIS_PROMPT = PROMPTS_DIR / "aiid_incident_analysis.txt"
STANDARDS_COMPLIANCE_PROMPT = PROMPTS_DIR / "standards_compliance.txt"
ANSWER_GENERATION_PROMPT = PROMPTS_DIR / "answer_generation.txt"
RERANK_PROMPT = PROMPTS_DIR / "rerank.txt"


def load_prompt(prompt_path: Path) -> str:
    """
    Load a prompt from file.
    
    Args:
        prompt_path: Path to prompt file
        
    Returns:
        Prompt text as string
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


__all__ = [
    "PROMPTS_DIR",
    "COMPLIANCE_MONITORING_PROMPT",
    "AIID_INCIDENT_ANALYSIS_PROMPT",
    "STANDARDS_COMPLIANCE_PROMPT",
    "ANSWER_GENERATION_PROMPT",
    "RERANK_PROMPT",
    "load_prompt",
]
