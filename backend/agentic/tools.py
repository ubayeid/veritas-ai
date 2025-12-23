"""
Tool Registry: Structured way to expose capabilities as callable tools
"""

from typing import Dict, Any, List, Callable
from pathlib import Path
import sys

# Add paths for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

try:
    from searching.query_engine import VectorQueryEngine
    from searching.hybrid_query_engine import HybridQueryEngine
except ImportError as e:
    # If imports fail, provide helpful error
    raise ImportError(
        f"Failed to import query engines. Make sure you're running from the project root "
        f"and have activated the virtual environment. Error: {e}"
    )


class ToolRegistry:
    """
    Registry of available tools that the agent can call.
    Each tool has a name, description, parameters, and function.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize tool registry with query engines.
        
        Args:
            base_dir: Base directory of the project
        """
        self.base_dir = Path(base_dir)
        self.vector_engine = VectorQueryEngine(str(base_dir))
        self.hybrid_engine = None
        
        # Try to initialize hybrid engine
        try:
            self.hybrid_engine = HybridQueryEngine(str(base_dir))
        except Exception as e:
            print(f"Warning: Hybrid engine not available: {e}")
        
        # Register all tools
        self.tools = {}
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools."""
        
        # Tool 1: Search compliance gaps
        self.tools['search_compliance_gaps'] = {
            'function': self._search_compliance_gaps,
            'description': 'Find GDPR articles that are not covered by company documents',
            'parameters': {
                'similarity_threshold': {
                    'type': 'float',
                    'description': 'Minimum similarity threshold for matching (0.0-1.0)',
                    'default': 0.45,
                    'required': False
                },
                'article_ids': {
                    'type': 'list',
                    'description': 'Specific article IDs to check (optional, checks all if not provided)',
                    'default': None,
                    'required': False
                }
            }
        }
        
        # Tool 2: Map clauses to articles
        self.tools['map_clauses_to_articles'] = {
            'function': self._map_clauses_to_articles,
            'description': 'Find which company clauses address which GDPR articles',
            'parameters': {
                'article_id': {
                    'type': 'str',
                    'description': 'Specific GDPR article ID to find clauses for',
                    'default': None,
                    'required': False
                },
                'top_k': {
                    'type': 'int',
                    'description': 'Number of top matches to return',
                    'default': 10,
                    'required': False
                }
            }
        }
        
        # Tool 3: Generate compliance report
        self.tools['generate_compliance_report'] = {
            'function': self._generate_compliance_report,
            'description': 'Generate a comprehensive compliance assessment report',
            'parameters': {
                'report_type': {
                    'type': 'str',
                    'description': 'Type of report: "gaps", "coverage", "full"',
                    'default': 'full',
                    'required': False
                },
                'include_recommendations': {
                    'type': 'bool',
                    'description': 'Include actionable recommendations',
                    'default': True,
                    'required': False
                }
            }
        }
        
        # Tool 4: Compare documents
        self.tools['compare_documents'] = {
            'function': self._compare_documents,
            'description': 'Compare company documents against GDPR standards',
            'parameters': {
                'document_name': {
                    'type': 'str',
                    'description': 'Specific company document to compare (optional)',
                    'default': None,
                    'required': False
                },
                'top_k': {
                    'type': 'int',
                    'description': 'Number of top matches per query',
                    'default': 5,
                    'required': False
                }
            }
        }
        
        # Tool 5: Find related incidents
        self.tools['find_related_incidents'] = {
            'function': self._find_related_incidents,
            'description': 'Find AIID incidents related to specific GDPR articles or topics',
            'parameters': {
                'article_id': {
                    'type': 'str',
                    'description': 'GDPR article ID to find incidents for',
                    'default': None,
                    'required': False
                },
                'topic': {
                    'type': 'str',
                    'description': 'Topic/keyword to search for',
                    'default': None,
                    'required': False
                },
                'top_k': {
                    'type': 'int',
                    'description': 'Number of incidents to return',
                    'default': 10,
                    'required': False
                }
            }
        }
        
        # Tool 6: Search vector database
        self.tools['search_vector'] = {
            'function': self._search_vector,
            'description': 'Perform semantic vector search across all databases',
            'parameters': {
                'query': {
                    'type': 'str',
                    'description': 'Search query',
                    'required': True
                },
                'db_names': {
                    'type': 'list',
                    'description': 'Databases to search: ["company", "aiid", "standards"]',
                    'default': None,
                    'required': False
                },
                'top_k': {
                    'type': 'int',
                    'description': 'Number of results',
                    'default': 10,
                    'required': False
                }
            }
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools with their descriptions."""
        return [
            {
                'name': name,
                'description': info['description'],
                'parameters': info['parameters']
            }
            for name, info in self.tools.items()
        ]
    
    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool by name with parameters.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")
        
        tool = self.tools[tool_name]
        
        # Validate and set default parameters
        params = {}
        for param_name, param_info in tool['parameters'].items():
            if param_name in kwargs:
                params[param_name] = kwargs[param_name]
            elif param_info.get('required', False):
                raise ValueError(f"Required parameter '{param_name}' missing for tool '{tool_name}'")
            elif 'default' in param_info:
                params[param_name] = param_info['default']
        
        # Execute tool
        try:
            result = tool['function'](**params)
            return {
                'success': True,
                'tool': tool_name,
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'tool': tool_name,
                'error': str(e)
            }
    
    # Tool implementations
    
    def _search_compliance_gaps(self, similarity_threshold: float = 0.45, article_ids: List[str] = None) -> Dict[str, Any]:
        """Find GDPR articles not covered by company documents."""
        if not self.hybrid_engine:
            raise ValueError("Hybrid engine required for gap analysis")
        
        # Use hybrid engine to find gaps
        query = "What GDPR articles are not covered by company documents?"
        result = self.hybrid_engine.hybrid_query(
            query=query,
            top_k=100,
            rerank=True,
            generate_answer=False
        )
        
        # Extract gap information
        gaps = []
        coverage = []
        
        for res in result['results']:
            if res.get('coverage_status') == 'not_covered':
                gaps.append({
                    'article_id': res.get('article_id', 'Unknown'),
                    'title': res.get('title', ''),
                    'description': res.get('text', res.get('description', ''))
                })
            elif res.get('coverage_status') == 'covered':
                coverage.append({
                    'article_id': res.get('article_id', 'Unknown'),
                    'clause_count': res.get('clause_count', 0)
                })
        
        return {
            'total_gaps': len(gaps),
            'total_covered': len(coverage),
            'gaps': gaps[:20],  # Limit results
            'coverage': coverage[:20]
        }
    
    def _map_clauses_to_articles(self, article_id: str = None, top_k: int = 10) -> Dict[str, Any]:
        """Find clauses addressing specific articles."""
        if not self.hybrid_engine:
            raise ValueError("Hybrid engine required for clause mapping")
        
        if article_id:
            query = f"Find clauses addressing GDPR Article {article_id}"
        else:
            query = "Find all clauses that address GDPR articles"
        
        result = self.hybrid_engine.hybrid_query(
            query=query,
            top_k=top_k,
            rerank=True,
            generate_answer=False
        )
        
        mappings = []
        for res in result['results']:
            if res.get('type') == 'clause' or 'clause' in res.get('source', '').lower():
                mappings.append({
                    'clause_text': res.get('text', res.get('description', ''))[:200],
                    'article_id': res.get('article_id', 'Unknown'),
                    'similarity': res.get('similarity', res.get('score', 0.0)),
                    'source': res.get('source', 'Unknown')
                })
        
        return {
            'mappings': mappings,
            'count': len(mappings)
        }
    
    def _generate_compliance_report(self, report_type: str = 'full', include_recommendations: bool = True) -> Dict[str, Any]:
        """Generate compliance assessment report."""
        # This would integrate with your compliance_monitoring_system.py
        # For now, return a placeholder structure
        return {
            'report_type': report_type,
            'status': 'not_implemented',
            'message': 'This tool needs integration with compliance_monitoring_system.py'
        }
    
    def _compare_documents(self, document_name: str = None, top_k: int = 5) -> Dict[str, Any]:
        """Compare company documents against GDPR."""
        query = "Compare company documents with GDPR requirements"
        if document_name:
            query += f" focusing on {document_name}"
        
        result = self.vector_engine.query(
            query=query,
            db_names=['company', 'standards'],
            top_k=top_k,
            rerank=True,
            generate_answer=True
        )
        
        return {
            'comparison_results': result['results'],
            'answer': result['answer'],
            'num_results': result['num_results']
        }
    
    def _find_related_incidents(self, article_id: str = None, topic: str = None, top_k: int = 10) -> Dict[str, Any]:
        """Find incidents related to articles or topics."""
        if article_id:
            query = f"Find AIID incidents that violate GDPR Article {article_id}"
        elif topic:
            query = f"Find AIID incidents related to {topic}"
        else:
            query = "Find AIID incidents related to GDPR violations"
        
        result = self.vector_engine.query(
            query=query,
            db_names=['aiid'],
            top_k=top_k,
            rerank=True,
            generate_answer=False
        )
        
        incidents = []
        for res in result['results']:
            incidents.append({
                'text': res.get('text', '')[:300],
                'source': res.get('source_name', 'Unknown'),
                'similarity': res.get('similarity', 0.0)
            })
        
        return {
            'incidents': incidents,
            'count': len(incidents)
        }
    
    def _search_vector(self, query: str, db_names: List[str] = None, top_k: int = 10) -> Dict[str, Any]:
        """Perform vector search."""
        result = self.vector_engine.query(
            query=query,
            db_names=db_names,
            top_k=top_k,
            rerank=True,
            generate_answer=True
        )
        
        return {
            'results': result['results'],
            'answer': result['answer'],
            'num_results': result['num_results']
        }

