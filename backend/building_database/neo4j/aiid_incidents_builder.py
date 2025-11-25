"""
AIID Incidents Builder for Neo4j
Processes AIID database incidents and creates Incident nodes with relationships to GDPR Articles.
"""

import csv
import sys
from typing import List, Dict, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from neo4j_connection import Neo4jConnection


class AIIDIncidentsBuilder:
    """Builds AIID incidents structure in Neo4j."""
    
    def __init__(self, neo4j_conn: Neo4jConnection):
        """
        Initialize AIID incidents builder.
        
        Args:
            neo4j_conn: Neo4jConnection instance
        """
        self.conn = neo4j_conn
    
    def read_incidents_csv(self, csv_path: str) -> List[Dict]:
        """
        Read incidents from CSV file.
        
        Args:
            csv_path: Path to incidents CSV file
            
        Returns:
            List of incident dictionaries
        """
        incidents = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                incidents.append(row)
        
        return incidents
    
    def determine_risk_type(self, description: str, title: str) -> str:
        """
        Determine risk type from incident description and title.
        
        Args:
            description: Incident description
            title: Incident title
            
        Returns:
            Risk type ('Privacy', 'Security', 'Bias', 'Safety', 'Other')
        """
        text = (description + " " + title).lower()
        
        privacy_keywords = ['privacy', 'data leak', 'data breach', 'unauthorized access', 
                           'personal information', 'data exposure', 'data misuse']
        security_keywords = ['security', 'hack', 'vulnerability', 'attack', 'breach', 
                           'unauthorized', 'malware', 'exploit']
        bias_keywords = ['bias', 'discrimination', 'unfair', 'discriminatory', 'racial', 
                        'gender', 'prejudice']
        safety_keywords = ['accident', 'crash', 'injury', 'death', 'harm', 'malfunction', 
                          'failure', 'safety']
        
        if any(kw in text for kw in privacy_keywords):
            return 'Privacy'
        elif any(kw in text for kw in security_keywords):
            return 'Security'
        elif any(kw in text for kw in bias_keywords):
            return 'Bias'
        elif any(kw in text for kw in safety_keywords):
            return 'Safety'
        else:
            return 'Other'
    
    def determine_system_type(self, description: str, title: str) -> str:
        """
        Determine system type from incident description.
        
        Args:
            description: Incident description
            title: Incident title
            
        Returns:
            System type
        """
        text = (description + " " + title).lower()
        
        if 'autonomous' in text or 'self-driving' in text or 'autopilot' in text:
            return 'Autonomous Vehicle'
        elif 'facial recognition' in text or 'biometric' in text:
            return 'Biometric AI'
        elif 'chatbot' in text or 'conversational' in text:
            return 'Conversational AI'
        elif 'recommendation' in text or 'algorithm' in text:
            return 'Recommendation System'
        elif 'scheduling' in text:
            return 'Scheduling System'
        elif 'risk assessment' in text or 'risk model' in text:
            return 'Risk Assessment System'
        elif 'surgery' in text or 'medical' in text:
            return 'Medical AI'
        else:
            return 'AI System'
    
    def create_incident_node(self, incident: Dict) -> Optional[str]:
        """
        Create an Incident node.
        
        Args:
            incident: Incident dictionary from CSV
            
        Returns:
            Incident ID if successful, None otherwise
        """
        incident_id = incident.get('incident_id', incident.get('_id', ''))
        if not incident_id:
            # Generate ID from title
            incident_id = f"AIID_{abs(hash(incident.get('title', '')))}"
        
        description = incident.get('description', '') or ''
        title = incident.get('title', '') or ''
        
        risk_type = self.determine_risk_type(description, title)
        system_type = self.determine_system_type(description, title)
        
        query = """
        MERGE (i:Incident {id: $id})
        SET i.description = $description,
            i.title = $title,
            i.system_type = $system_type,
            i.risk_type = $risk_type,
            i.source = 'AIID Database',
            i.date = $date
        RETURN i
        """
        
        try:
            self.conn.execute_write(query, {
                'id': incident_id,
                'description': description[:1000] if len(description) > 1000 else description,  # Limit length
                'title': title[:500] if len(title) > 500 else title,
                'system_type': system_type,
                'risk_type': risk_type,
                'date': incident.get('date', '')
            })
            return incident_id
        except Exception as e:
            print(f"Error creating incident {incident_id}: {e}")
            return None
    
    def link_incident_to_article(self, incident_id: str, article_id: str, 
                                 violation_type: Optional[str] = None) -> bool:
        """
        Link an Incident to a GDPR Article via VIOLATES relationship.
        
        Args:
            incident_id: ID of the incident
            article_id: ID of the GDPR article
            violation_type: Optional type of violation
        """
        query = """
        MATCH (i:Incident {id: $incident_id})
        MATCH (a:Article {id: $article_id})
        MERGE (i)-[r:VIOLATES]->(a)
        SET r.violation_type = $violation_type
        RETURN i, a, r
        """
        
        try:
            self.conn.execute_write(query, {
                'incident_id': incident_id,
                'article_id': article_id,
                'violation_type': violation_type
            })
            return True
        except Exception as e:
            print(f"Error linking incident {incident_id} to article {article_id}: {e}")
            return False
    
    def auto_link_incidents_to_articles(self, similarity_threshold: float = 0.7):
        """
        Automatically link incidents to articles based on keywords and risk types.
        This is a simplified approach - in production, you'd use semantic similarity.
        
        Args:
            similarity_threshold: Threshold for linking (not used in simple keyword matching)
        """
        # Mapping of risk types and keywords to GDPR articles
        risk_to_article_mapping = {
            'Privacy': ['Art5', 'Art32', 'Art33', 'Art34'],  # Data protection, security, breach notification
            'Security': ['Art32', 'Art33', 'Art34'],  # Security, breach notification
            'Bias': ['Art5', 'Art22'],  # Fairness, automated decision-making
            'Safety': ['Art32', 'Art35'],  # Security, impact assessment
            'Other': ['Art5', 'Art32']  # General data protection
        }
        
        # Get all incidents
        query = """
        MATCH (i:Incident)
        RETURN i.id as id, i.risk_type as risk_type, i.description as description
        """
        
        incidents = self.conn.execute_query(query)
        
        for incident in incidents:
            incident_id = incident['id']
            risk_type = incident['risk_type'] or 'Other'
            description = incident['description'] or ''
            
            # Get relevant articles based on risk type
            article_ids = risk_to_article_mapping.get(risk_type, ['Art5'])
            
            # Also check for specific keywords in description
            description_lower = description.lower()
            if 'breach' in description_lower or 'leak' in description_lower:
                article_ids.extend(['Art33', 'Art34'])
            if 'consent' in description_lower:
                article_ids.extend(['Art6', 'Art7'])
            if 'access' in description_lower or 'right' in description_lower:
                article_ids.extend(['Art15', 'Art16', 'Art17'])
            
            # Link to articles
            for article_id in set(article_ids):  # Remove duplicates
                self.link_incident_to_article(incident_id, article_id)
    
    def build_from_json(self, json_path: str):
        """
        Build incidents from processed graph JSON file.
        
        Args:
            json_path: Path to processed graph JSON file
        """
        import json
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        incidents = data.get('incidents', [])
        print(f"Loading {len(incidents)} incidents from {json_path}")
        
        processed = 0
        for incident_data in incidents:
            incident_id = self.create_incident_node(incident_data)
            if incident_id:
                processed += 1
            
            if processed % 100 == 0:
                print(f"Processed {processed} incidents...")
        
        print(f"Created {processed} incident nodes")
        
        # Auto-link to articles
        print("Linking incidents to GDPR articles...")
        self.auto_link_incidents_to_articles()
        print("Completed linking incidents to articles")
    
    def process_incidents_csv(self, csv_path: str, limit: Optional[int] = None):
        """
        Process incidents from CSV file.
        
        Args:
            csv_path: Path to incidents CSV file
            limit: Optional limit on number of incidents to process
        """
        print(f"Reading incidents from {csv_path}...")
        incidents = self.read_incidents_csv(csv_path)
        
        if limit:
            incidents = incidents[:limit]
        
        print(f"Processing {len(incidents)} incidents...")
        
        processed = 0
        for incident in incidents:
            incident_id = self.create_incident_node(incident)
            if incident_id:
                processed += 1
            
            if processed % 100 == 0:
                print(f"Processed {processed} incidents...")
        
        print(f"Created {processed} incident nodes")
        
        # Auto-link to articles
        print("Linking incidents to GDPR articles...")
        self.auto_link_incidents_to_articles()
        print("Completed linking incidents to articles")


if __name__ == "__main__":
    # Example usage
    with Neo4jConnection() as conn:
        if not conn.verify_connectivity():
            print("Failed to connect to Neo4j. Please check your connection settings.")
            exit(1)
        
        builder = AIIDIncidentsBuilder(conn)
        
        # Process AIID incidents
        incidents_csv = Path(__file__).parent.parent.parent.parent / "data" / "aiid" / "incidents.csv"
        if incidents_csv.exists():
            # Process first 100 incidents as example (remove limit to process all)
            builder.process_incidents_csv(str(incidents_csv), limit=100)
        else:
            print(f"File not found: {incidents_csv}")
        
        # Print statistics
        stats = conn.get_stats()
        print("\nDatabase Statistics:")
        print(f"Nodes: {stats['nodes']}")
        print(f"Relationships: {stats['relationships']}")

