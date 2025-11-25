"""
Neo4j Database Connection Module
Handles connection to Neo4j database and provides utility functions.
"""

import os
from typing import Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Neo4jConnection:
    """Manages Neo4j database connection."""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j database URI (defaults to NEO4J_URI env var or bolt://localhost:7687)
            user: Neo4j username (defaults to NEO4J_USER env var or 'neo4j')
            password: Neo4j password (defaults to NEO4J_PASSWORD env var or 'password')
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
    
    def verify_connectivity(self) -> bool:
        """Verify connection to Neo4j database."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            return False
    
    def execute_query(self, query: str, parameters: Optional[dict] = None) -> list:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters dictionary
            
        Returns:
            List of result records
        """
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record for record in result]
    
    def execute_write(self, query: str, parameters: Optional[dict] = None) -> list:
        """
        Execute a write transaction.
        
        Args:
            query: Cypher query string
            parameters: Query parameters dictionary
            
        Returns:
            List of result records
        """
        with self.driver.session() as session:
            # Neo4j 5.x uses execute_write instead of write_transaction
            try:
                # Try new API (Neo4j 5.x)
                result = session.execute_write(lambda tx: list(tx.run(query, parameters or {})))
                return result
            except AttributeError:
                # Fallback to old API (Neo4j 4.x)
                result = session.write_transaction(lambda tx: list(tx.run(query, parameters or {})))
                return result
    
    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_write(query)
        print("Database cleared successfully.")
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        node_count_query = """
        MATCH (n)
        RETURN labels(n)[0] as label, count(n) as count
        ORDER BY label
        """
        
        rel_count_query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        ORDER BY type
        """
        
        nodes = self.execute_query(node_count_query)
        relationships = self.execute_query(rel_count_query)
        
        return {
            "nodes": {record["label"]: record["count"] for record in nodes},
            "relationships": {record["type"]: record["count"] for record in relationships}
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

