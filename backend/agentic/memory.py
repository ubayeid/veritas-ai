"""
Agent Memory: Short-term and long-term memory for the agent
"""

from typing import Dict, Any, List
from datetime import datetime
import json
from pathlib import Path


class AgentMemory:
    """
    Manages agent memory: conversation context, task history, and learnings.
    """
    
    def __init__(self, memory_dir: str = None):
        """
        Initialize memory system.
        
        Args:
            memory_dir: Directory to store persistent memory (optional)
        """
        self.memory_dir = Path(memory_dir) if memory_dir else None
        if self.memory_dir:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Short-term memory (current session)
        self.conversation_context = []
        self.current_task_state = {}
        
        # Long-term memory (persistent)
        self.task_history = []
        self.learnings = []
    
    def add_conversation(self, role: str, content: str):
        """Add to conversation context."""
        self.conversation_context.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_conversation_context(self, last_n: int = 10) -> List[Dict]:
        """Get recent conversation context."""
        return self.conversation_context[-last_n:]
    
    def store_task(self, goal: str, plan: List[Dict], results: List[Dict]):
        """Store completed task."""
        task_record = {
            'goal': goal,
            'plan': plan,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'success': any(r.get('result', {}).get('success', False) for r in results)
        }
        
        self.task_history.append(task_record)
        
        # Persist if memory_dir is set
        if self.memory_dir:
            self._save_task_history()
    
    def get_task_history(self, limit: int = 10) -> List[Dict]:
        """Get recent task history."""
        return self.task_history[-limit:]
    
    def store_learning(self, learning: Dict[str, Any]):
        """Store a learning/pattern."""
        learning_record = {
            'learning': learning,
            'timestamp': datetime.now().isoformat()
        }
        self.learnings.append(learning_record)
    
    def _save_task_history(self):
        """Save task history to disk."""
        if not self.memory_dir:
            return
        
        history_file = self.memory_dir / 'task_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.task_history, f, indent=2)
    
    def load_task_history(self):
        """Load task history from disk."""
        if not self.memory_dir:
            return
        
        history_file = self.memory_dir / 'task_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.task_history = json.load(f)

