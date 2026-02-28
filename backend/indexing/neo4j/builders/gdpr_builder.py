"""
GDPR Structure Builder for Neo4j
Creates Article, SubObligation, and Topic nodes with relationships.
"""

import re
import sys
from typing import List, Dict, Optional
from pathlib import Path
import PyPDF2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from backend.indexing.neo4j.utils.neo4j_connection import Neo4jConnection


class GDPRBuilder:
    """Builds GDPR structure in Neo4j."""
    
    def __init__(self, neo4j_conn: Neo4jConnection):
        """
        Initialize GDPR builder.
        
        Args:
            neo4j_conn: Neo4jConnection instance
        """
        self.conn = neo4j_conn
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from GDPR PDF."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
        return text
    
    def parse_gdpr_articles(self, text: str) -> List[Dict]:
        """
        Parse GDPR text to extract articles.
        This is a simplified parser - you may need to enhance it based on your PDF structure.
        
        Args:
            text: Extracted text from GDPR PDF
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        # Pattern to match article numbers (e.g., "Article 5", "Art. 5", etc.)
        article_pattern = r'(?:Article|Art\.?)\s*(\d+)[\s\n]+(.+?)(?=(?:Article|Art\.?)\s*\d+|$)'
        
        matches = re.finditer(article_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            article_num = match.group(1)
            article_text = match.group(2).strip()
            
            # Extract title (first sentence or line)
            lines = article_text.split('\n')
            title = lines[0].strip() if lines else f"Article {article_num}"
            
            # Extract description (first paragraph or first few sentences)
            description = article_text[:500] if len(article_text) > 500 else article_text
            
            # Extract keywords (simple keyword extraction)
            keywords = self._extract_keywords(article_text)
            
            articles.append({
                'id': f'Art{article_num}',
                'number': article_num,
                'title': title,
                'description': description,
                'keywords': keywords
            })
        
        return articles
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Common GDPR-related keywords
        gdpr_keywords = [
            'data protection', 'personal data', 'processing', 'consent', 'right to access',
            'right to erasure', 'data minimization', 'purpose limitation', 'storage limitation',
            'accuracy', 'integrity', 'confidentiality', 'accountability', 'transparency',
            'data subject', 'controller', 'processor', 'breach', 'notification', 'security',
            'privacy by design', 'privacy by default', 'impact assessment', 'supervisory authority'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in gdpr_keywords if kw in text_lower]
        
        # Also extract some common words (simplified)
        words = re.findall(r'\b[a-z]{4,}\b', text_lower)
        common_words = [w for w in set(words) if len(w) > 4][:5]
        
        return list(set(found_keywords + common_words))
    
    def create_article_node(self, article: Dict) -> bool:
        """Create an Article node in Neo4j."""
        query = """
        MERGE (a:Article {id: $id})
        SET a.title = $title,
            a.description = $description,
            a.keywords = $keywords,
            a.number = $number
        RETURN a
        """
        
        try:
            self.conn.execute_write(query, {
                'id': article['id'],
                'title': article['title'],
                'description': article['description'],
                'keywords': article['keywords'],
                'number': article['number']
            })
            return True
        except Exception as e:
            print(f"Error creating article {article['id']}: {e}")
            return False
    
    def create_sub_obligation(self, sub_obligation_id: str, description: str, 
                              keywords: List[str], article_id: str) -> bool:
        """
        Create a SubObligation node and connect it to an Article.
        
        Args:
            sub_obligation_id: Unique ID for the sub-obligation
            description: Description of the sub-obligation
            keywords: List of keywords
            article_id: ID of the parent Article
        """
        # Create sub-obligation node
        create_query = """
        MERGE (so:SubObligation {id: $id})
        SET so.description = $description,
            so.keywords = $keywords
        RETURN so
        """
        
        # Connect to article
        connect_query = """
        MATCH (a:Article {id: $article_id})
        MATCH (so:SubObligation {id: $so_id})
        MERGE (a)-[:HAS_SUB_OBLIGATION]->(so)
        RETURN a, so
        """
        
        try:
            self.conn.execute_write(create_query, {
                'id': sub_obligation_id,
                'description': description,
                'keywords': keywords
            })
            
            self.conn.execute_write(connect_query, {
                'article_id': article_id,
                'so_id': sub_obligation_id
            })
            return True
        except Exception as e:
            print(f"Error creating sub-obligation {sub_obligation_id}: {e}")
            return False
    
    def create_topic(self, topic_name: str, article_id: str) -> bool:
        """
        Create a Topic node and connect it to an Article.
        
        Args:
            topic_name: Name of the topic
            article_id: ID of the Article
        """
        # Create topic node (merge to avoid duplicates)
        create_query = """
        MERGE (t:Topic {name: $name})
        RETURN t
        """
        
        # Connect article to topic
        connect_query = """
        MATCH (a:Article {id: $article_id})
        MATCH (t:Topic {name: $topic_name})
        MERGE (a)-[:HAS_TOPIC]->(t)
        RETURN a, t
        """
        
        try:
            self.conn.execute_write(create_query, {'name': topic_name})
            self.conn.execute_write(connect_query, {
                'article_id': article_id,
                'topic_name': topic_name
            })
            return True
        except Exception as e:
            print(f"Error creating topic {topic_name}: {e}")
            return False
    
    def build_from_pdf(self, pdf_path: str):
        """
        Build GDPR structure from PDF file.
        
        Args:
            pdf_path: Path to GDPR PDF file
        """
        print(f"Extracting text from {pdf_path}...")
        text = self.extract_text_from_pdf(pdf_path)
        
        print("Parsing articles...")
        articles = self.parse_gdpr_articles(text)
        print(f"Found {len(articles)} articles")
        
        # Create article nodes
        for article in articles:
            print(f"Creating article {article['id']}...")
            self.create_article_node(article)
            
            # Extract topics from keywords and create topic relationships
            for keyword in article['keywords'][:3]:  # Top 3 keywords as topics
                if len(keyword) > 5:  # Only meaningful keywords
                    self.create_topic(keyword.title(), article['id'])
        
        print(f"Created {len(articles)} articles")
    
    def build_from_json(self, json_path: str):
        """
        Build GDPR structure from processed graph JSON file.
        
        Args:
            json_path: Path to processed graph JSON file
        """
        import json
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = data.get('articles', [])
        print(f"Loading {len(articles)} articles from {json_path}")
        
        for article_data in articles:
            # Create article
            self.create_article_node(article_data)
            
            # Create sub-obligations
            for sub_obligation in article_data.get('sub_obligations', []):
                self.create_sub_obligation(
                    sub_obligation['id'],
                    sub_obligation['description'],
                    sub_obligation.get('keywords', []),
                    article_data['id']
                )
            
            # Create topics
            for topic in article_data.get('topics', []):
                self.create_topic(topic, article_data['id'])
    
    def build_from_manual_data(self, articles_data: List[Dict]):
        """
        Build GDPR structure from manual data structure.
        
        Args:
            articles_data: List of article dictionaries with structure:
                {
                    'id': 'Art5',
                    'title': 'Data Minimization',
                    'description': '...',
                    'keywords': ['data minimization', 'limited', 'necessary'],
                    'sub_obligations': [
                        {'id': 'Art5.1', 'description': '...', 'keywords': [...]}
                    ],
                    'topics': ['Data Minimization', 'Purpose Limitation']
                }
        """
        for article_data in articles_data:
            # Create article
            self.create_article_node(article_data)
            
            # Create sub-obligations
            for sub_obligation in article_data.get('sub_obligations', []):
                self.create_sub_obligation(
                    sub_obligation['id'],
                    sub_obligation['description'],
                    sub_obligation.get('keywords', []),
                    article_data['id']
                )
            
            # Create topics
            for topic in article_data.get('topics', []):
                self.create_topic(topic, article_data['id'])


def create_sample_gdpr_data() -> List[Dict]:
    """Create sample GDPR data for testing."""
    return [
        {
            'id': 'Art5',
            'title': 'Principles relating to processing of personal data',
            'description': 'Personal data shall be: (a) processed lawfully, fairly and in a transparent manner; (b) collected for specified, explicit and legitimate purposes; (c) adequate, relevant and limited to what is necessary; (d) accurate and kept up to date; (e) kept in a form which permits identification for no longer than necessary; (f) processed in a manner that ensures appropriate security.',
            'keywords': ['data minimization', 'purpose limitation', 'storage limitation', 'accuracy', 'lawfulness', 'fairness', 'transparency'],
            'sub_obligations': [
                {
                    'id': 'Art5.1.c',
                    'description': 'Personal data must be adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed (data minimisation)',
                    'keywords': ['data minimization', 'limited', 'necessary', 'relevant']
                },
                {
                    'id': 'Art5.1.f',
                    'description': 'Personal data must be processed in a manner that ensures appropriate security, including protection against unauthorised or unlawful processing and against accidental loss, destruction or damage',
                    'keywords': ['security', 'protection', 'unauthorized', 'loss', 'destruction']
                }
            ],
            'topics': ['Data Minimization', 'Purpose Limitation', 'Storage Limitation', 'Security']
        },
        {
            'id': 'Art32',
            'title': 'Security of processing',
            'description': 'Taking into account the state of the art, the costs of implementation and the nature, scope, context and purposes of processing as well as the risk of varying likelihood and severity for the rights and freedoms of natural persons, the controller and the processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk.',
            'keywords': ['security', 'technical measures', 'organizational measures', 'risk assessment', 'data protection'],
            'sub_obligations': [
                {
                    'id': 'Art32.1',
                    'description': 'Implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk',
                    'keywords': ['technical measures', 'organizational measures', 'security', 'risk']
                }
            ],
            'topics': ['Security', 'Risk Assessment', 'Technical Measures']
        },
        {
            'id': 'Art15',
            'title': 'Right of access by the data subject',
            'description': 'The data subject shall have the right to obtain from the controller confirmation as to whether or not personal data concerning him or her are being processed, and, where that is the case, access to the personal data.',
            'keywords': ['right to access', 'data subject rights', 'transparency', 'access'],
            'sub_obligations': [
                {
                    'id': 'Art15.1',
                    'description': 'Data subject has the right to obtain confirmation of processing and access to personal data',
                    'keywords': ['right to access', 'confirmation', 'data subject']
                }
            ],
            'topics': ['Data Subject Rights', 'Transparency', 'Access']
        }
    ]


if __name__ == "__main__":
    # Example usage
    with Neo4jConnection() as conn:
        if not conn.verify_connectivity():
            print("Failed to connect to Neo4j. Please check your connection settings.")
            exit(1)
        
        builder = GDPRBuilder(conn)
        
        # Option 1: Build from manual data
        print("Building GDPR structure from sample data...")
        sample_data = create_sample_gdpr_data()
        builder.build_from_manual_data(sample_data)
        
        # Option 2: Build from PDF (uncomment if you have GDPR PDF)
        # pdf_path = Path(__file__).parent.parent.parent.parent / "data" / "standards" / "gdpr.pdf"
        # if pdf_path.exists():
        #     builder.build_from_pdf(str(pdf_path))
        
        # Print statistics
        stats = conn.get_stats()
        print("\nDatabase Statistics:")
        print(f"Nodes: {stats['nodes']}")
        print(f"Relationships: {stats['relationships']}")

