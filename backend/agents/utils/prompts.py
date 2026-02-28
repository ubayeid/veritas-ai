"""
Prompt templates using LangChain PromptTemplate.
Centralized prompt management for all agents.
"""

from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from pathlib import Path
import os


class PromptManager:
    """
    Manages prompts for agents using LangChain templates.
    Supports loading from files and programmatic creation.
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt files (optional)
        """
        if prompts_dir is None:
            # Default to consolidated prompts directory
            self.prompts_dir = Path(__file__).parent.parent / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
        
        self._prompts: Dict[str, PromptTemplate] = {}
        self._chat_prompts: Dict[str, ChatPromptTemplate] = {}
    
    def load_prompt_from_file(self, name: str, file_path: Optional[Path] = None) -> PromptTemplate:
        """
        Load a prompt template from a file.
        
        Args:
            name: Prompt name
            file_path: Path to prompt file (if None, uses prompts_dir)
            
        Returns:
            PromptTemplate instance
        """
        if file_path is None:
            file_path = self.prompts_dir / f"{name}.txt"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        prompt = PromptTemplate.from_template(template)
        self._prompts[name] = prompt
        return prompt
    
    def get_planning_prompt(self) -> ChatPromptTemplate:
        """
        Get planning prompt for agents.
        
        Returns:
            ChatPromptTemplate for planning
        """
        if "planning" not in self._chat_prompts:
            self._chat_prompts["planning"] = ChatPromptTemplate.from_messages([
                ("system", """You are a task planning assistant for a compliance analysis system.
Your job is to break down complex goals into executable steps using the available tools.

Available Tools:
{tools}

Create a step-by-step plan to achieve the goal. Each step should:
1. Use a specific tool from the available tools
2. Have clear inputs/parameters
3. Define what output is expected
4. Note any dependencies on previous steps

Return your plan as a JSON array of steps, where each step has:
- step_id: sequential number
- tool: tool name to use
- description: what this step does
- parameters: dictionary of parameters
- depends_on: list of step_ids this depends on
- expected_output: what we expect to get

Be specific and actionable. Use only the tools listed above."""),
                ("user", "Goal: {goal}\n\nCreate a plan:")
            ])
        
        return self._chat_prompts["planning"]
    
    def get_answer_generation_prompt(self) -> ChatPromptTemplate:
        """
        Get answer generation prompt.
        
        Returns:
            ChatPromptTemplate for answer generation
        """
        if "answer_generation" not in self._chat_prompts:
            # Try to load from file (check both new and old locations)
            prompt_file = self.prompts_dir / "answer_generation.txt"
            if not prompt_file.exists():
                prompt_file = self.prompts_dir / "answer_generation_prompt.txt"
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    template = f.read()
            else:
                template = """You are a helpful assistant that answers questions based on provided search results.
Use the search results to provide a comprehensive, accurate answer to the user's query.
Cite specific sources when referencing information.
If the search results don't fully answer the query, say so and provide what information is available.

User Query: {query}

Search Results:
{context}

Based on the above search results, provide a comprehensive answer to the user's query:"""
            
            self._chat_prompts["answer_generation"] = ChatPromptTemplate.from_messages([
                ("system", template.split("\n\n")[0]),
                ("user", "\n\n".join(template.split("\n\n")[1:]))
            ])
        
        return self._chat_prompts["answer_generation"]
    
    def get_rerank_prompt(self) -> ChatPromptTemplate:
        """
        Get reranking prompt.
        
        Returns:
            ChatPromptTemplate for reranking
        """
        if "rerank" not in self._chat_prompts:
            # Try to load from file (check both new and old locations)
            prompt_file = self.prompts_dir / "rerank.txt"
            if not prompt_file.exists():
                prompt_file = self.prompts_dir / "rerank_prompt.txt"
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    template = f.read()
            else:
                template = """You are a search result reranking expert.
Given a user query and a list of search results, rank them by relevance to the query.
Consider semantic similarity, context, and how well each result answers the query.
Return only the indices of the results in order of relevance (most relevant first), separated by commas.

User Query: {query}

Search Results:
{results}

Rank the results by relevance to the query. Return only the result indices (0-{num_results}) in order, separated by commas:"""
            
            self._chat_prompts["rerank"] = ChatPromptTemplate.from_messages([
                ("system", "You are a search result reranking expert. Return only comma-separated indices."),
                ("user", template)
            ])
        
        return self._chat_prompts["rerank"]
    
    def get_compliance_analysis_prompt(self) -> ChatPromptTemplate:
        """
        Get compliance analysis prompt.
        
        Returns:
            ChatPromptTemplate for compliance analysis
        """
        if "compliance_analysis" not in self._chat_prompts:
            # Try consolidated prompts directory first
            prompt_file = self.prompts_dir / "compliance_monitoring.txt"
            if not prompt_file.exists():
                # Fallback to old location
                prompt_file = Path(__file__).parent.parent / "generation" / "prompts" / "compliance_monitoring_prompt.txt"
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    template = f.read()
            else:
                template = """Analyze the compliance gaps and coverage between company documents and GDPR requirements.
Provide a comprehensive analysis including:
1. Identified gaps (articles not covered)
2. Coverage analysis (articles that are covered)
3. Recommendations for improvement

Company Documents: {company_docs}
GDPR Requirements: {gdpr_requirements}
Gap Analysis: {gap_analysis}

Provide your compliance analysis:"""
            
            self._chat_prompts["compliance_analysis"] = ChatPromptTemplate.from_messages([
                ("system", "You are a compliance analysis expert."),
                ("user", template)
            ])
        
        return self._chat_prompts["compliance_analysis"]


# Global prompt manager instance
_prompt_manager = PromptManager()


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager."""
    return _prompt_manager
