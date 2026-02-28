"""
Agent Registry for managing multiple agents.
Enables extensibility for adding new agents beyond compliance monitoring.
"""

from typing import Dict, List, Optional, Type
from ..core.base_agent import BaseAgent
import importlib
from pathlib import Path


class AgentRegistry:
    """
    Registry for managing multiple agents.
    Allows dynamic registration and retrieval of agents.
    """
    
    def __init__(self):
        """Initialize agent registry."""
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}
    
    def register_agent(
        self,
        name: str,
        agent: BaseAgent,
        agent_class: Optional[Type[BaseAgent]] = None
    ):
        """
        Register an agent instance.
        
        Args:
            name: Agent name
            agent: Agent instance
            agent_class: Agent class (for creating new instances)
        """
        self._agents[name] = agent
        if agent_class:
            self._agent_classes[name] = agent_class
    
    def register_agent_class(
        self,
        name: str,
        agent_class: Type[BaseAgent]
    ):
        """
        Register an agent class.
        
        Args:
            name: Agent name
            agent_class: Agent class
        """
        self._agent_classes[name] = agent_class
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        Get an agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance or None
        """
        return self._agents.get(name)
    
    def create_agent(
        self,
        name: str,
        **kwargs
    ) -> Optional[BaseAgent]:
        """
        Create a new agent instance from registered class.
        
        Args:
            name: Agent name
            **kwargs: Arguments to pass to agent constructor
            
        Returns:
            New agent instance or None
        """
        agent_class = self._agent_classes.get(name)
        if agent_class:
            return agent_class(**kwargs)
        return None
    
    def list_agents(self) -> List[str]:
        """
        List all registered agent names.
        
        Returns:
            List of agent names
        """
        return list(self._agents.keys())
    
    def list_agent_classes(self) -> List[str]:
        """
        List all registered agent class names.
        
        Returns:
            List of agent class names
        """
        return list(self._agent_classes.keys())
    
    def get_agent_info(self, name: str) -> Optional[Dict]:
        """
        Get information about an agent.
        
        Args:
            name: Agent name
            
        Returns:
            Dictionary with agent information or None
        """
        agent = self._agents.get(name)
        if agent:
            return {
                "name": agent.name,
                "description": agent.description,
                "tools": [tool.name for tool in agent.get_tools()],
                "type": type(agent).__name__
            }
        return None


# Global registry instance
_global_registry = AgentRegistry()


def get_registry() -> AgentRegistry:
    """Get the global agent registry."""
    return _global_registry


def register_agent(name: str, agent: BaseAgent, agent_class: Optional[Type[BaseAgent]] = None):
    """Register an agent in the global registry."""
    _global_registry.register_agent(name, agent, agent_class)


def get_agent(name: str) -> Optional[BaseAgent]:
    """Get an agent from the global registry."""
    return _global_registry.get_agent(name)
