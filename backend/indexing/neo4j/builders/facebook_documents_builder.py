"""
Facebook Documents Builder for Neo4j
Processes Facebook documents (Privacy Policy, Terms of Service, Cookie Policy)
and creates Document and Clause nodes with relationships to GDPR Articles.
"""

import re
import sys
from typing import List, Dict, Optional
from pathlib import Path
import PyPDF2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from backend.indexing.neo4j.utils.neo4j_connection import Neo4jConnection


class FacebookDocumentsBuilder:
    """Builds Facebook documents structure in Neo4j."""
    
    def __init__(self, neo4j_conn: Neo4jConnection):
        """
        Initialize Facebook documents builder.
        
        Args:
            neo4j_conn: Neo4jConnection instance
        """
        self.conn = neo4j_conn
        
        # Document URLs mapping
        self.document_urls = {
            'Privacy Policy': 'https://www.facebook.com/privacy/policy',
            'Terms of Service': 'https://www.facebook.com/legal/terms',
            'Cookie Policy': 'https://www.facebook.com/policies/cookies'
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
        return text
    
    def split_into_clauses(self, text: str, min_length: int = 50) -> List[str]:
        """
        Split text into clauses (sentences or logical units).
        
        Args:
            text: Text to split
            min_length: Minimum length for a clause
            
        Returns:
            List of clause texts
        """
        clauses = []
        
        # Split by sentences (period, exclamation, question mark followed by space)
        sentences = re.split(r'[.!?]+\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= min_length:
                clauses.append(sentence)
        
        # Also try splitting by newlines (for structured documents)
        if len(clauses) < 5:  # If we didn't get many clauses from sentences
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) >= min_length and line:
                    clauses.append(line)
        
        return clauses
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from clause text."""
        # Common privacy/data protection keywords
        keywords = [
            'data', 'personal', 'information', 'privacy', 'collect', 'use', 'share',
            'consent', 'right', 'access', 'delete', 'security', 'protection', 'policy',
            'cookies', 'tracking', 'advertising', 'third party', 'user', 'account'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in keywords if kw in text_lower]
        
        # Extract some meaningful words
        words = re.findall(r'\b[a-z]{4,}\b', text_lower)
        meaningful_words = [w for w in set(words) if len(w) > 4][:5]
        
        return list(set(found_keywords + meaningful_words))
    
    def create_document_node(self, document_name: str, source_url: Optional[str] = None) -> bool:
        """
        Create a Document node.
        
        Args:
            document_name: Name of the document
            source_url: URL of the document
        """
        url = source_url or self.document_urls.get(document_name, '')
        
        query = """
        MERGE (d:Document {name: $name})
        SET d.source_url = $url
        RETURN d
        """
        
        try:
            self.conn.execute_write(query, {
                'name': document_name,
                'url': url
            })
            return True
        except Exception as e:
            print(f"Error creating document {document_name}: {e}")
            return False
    
    def create_clause_node(self, clause_text: str, document_name: str, 
                          section: Optional[str] = None, clause_id: Optional[str] = None) -> str:
        """
        Create a Clause node and connect it to a Document.
        
        Args:
            clause_text: Text of the clause
            document_name: Name of the parent document
            section: Optional section name
            clause_id: Optional unique ID for the clause
            
        Returns:
            The clause ID (generated or provided)
        """
        if not clause_id:
            # Generate a unique ID based on document name and text hash
            clause_id = f"{document_name}_{abs(hash(clause_text[:50]))}"
        
        keywords = self.extract_keywords(clause_text)
        
        # Create clause node
        create_query = """
        MERGE (c:Clause {id: $id})
        SET c.text = $text,
            c.document_name = $document_name,
            c.keywords = $keywords,
            c.section = $section
        RETURN c
        """
        
        # Connect to document
        connect_query = """
        MATCH (d:Document {name: $document_name})
        MATCH (c:Clause {id: $clause_id})
        MERGE (d)-[:COVERS]->(c)
        RETURN d, c
        """
        
        try:
            self.conn.execute_write(create_query, {
                'id': clause_id,
                'text': clause_text,
                'document_name': document_name,
                'keywords': keywords,
                'section': section or ''
            })
            
            self.conn.execute_write(connect_query, {
                'document_name': document_name,
                'clause_id': clause_id
            })
            
            return clause_id
        except Exception as e:
            print(f"Error creating clause: {e}")
            return clause_id
    
    def link_clause_to_article(self, clause_id: str, article_id: str, 
                               similarity_score: Optional[float] = None) -> bool:
        """
        Link a Clause to a GDPR Article via ADDRESSES relationship.
        
        Args:
            clause_id: ID of the clause
            article_id: ID of the GDPR article
            similarity_score: Optional similarity score (for future use)
        """
        query = """
        MATCH (c:Clause {id: $clause_id})
        MATCH (a:Article {id: $article_id})
        MERGE (c)-[r:ADDRESSES]->(a)
        SET r.similarity_score = $similarity_score
        RETURN c, a, r
        """
        
        try:
            self.conn.execute_write(query, {
                'clause_id': clause_id,
                'article_id': article_id,
                'similarity_score': similarity_score
            })
            return True
        except Exception as e:
            print(f"Error linking clause {clause_id} to article {article_id}: {e}")
            return False
    
    def set_clause_compliance_status(self, clause_id: str, status: str) -> bool:
        """
        Set compliance status for a clause.
        
        Args:
            clause_id: ID of the clause
            status: Compliance status ('compliant', 'partial', 'non-compliant')
        """
        if status not in ['compliant', 'partial', 'non-compliant']:
            raise ValueError(f"Invalid status: {status}. Must be 'compliant', 'partial', or 'non-compliant'")
        
        query = """
        MATCH (c:Clause {id: $clause_id})
        SET c.compliance_status = $status
        RETURN c
        """
        
        try:
            self.conn.execute_write(query, {
                'clause_id': clause_id,
                'status': status
            })
            return True
        except Exception as e:
            print(f"Error setting compliance status for clause {clause_id}: {e}")
            return False
    
    def process_pdf_document(self, pdf_path: str, document_name: str):
        """
        Process a PDF document and create nodes.
        
        Args:
            pdf_path: Path to the PDF file
            document_name: Name of the document
        """
        print(f"Processing document: {document_name}")
        
        # Create document node
        self.create_document_node(document_name)
        
        # Extract text
        print(f"Extracting text from {pdf_path}...")
        text = self.extract_text_from_pdf(pdf_path)
        
        # Split into clauses
        print("Splitting into clauses...")
        clauses = self.split_into_clauses(text)
        print(f"Created {len(clauses)} clauses")
        
        # Create clause nodes
        for i, clause_text in enumerate(clauses):
            clause_id = f"{document_name}_clause_{i+1}"
            self.create_clause_node(clause_text, document_name, clause_id=clause_id)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} clauses...")
        
        print(f"Completed processing {document_name}")
    
    def build_from_json(self, json_path: str):
        """
        Build documents and clauses from processed graph JSON file.
        
        Args:
            json_path: Path to processed graph JSON file
        """
        import json
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data.get('documents', [])
        clauses = data.get('clauses', [])
        
        print(f"Loading {len(documents)} documents and {len(clauses)} clauses from {json_path}")
        
        # Create document nodes
        for doc_data in documents:
            self.create_document_node(
                doc_data['name'],
                doc_data.get('source_url', '')
            )
        
        # Create clause nodes
        for clause_data in clauses:
            self.create_clause_node(
                clause_data['text'],
                clause_data['document_name'],
                section=clause_data.get('section'),
                clause_id=clause_data['id']
            )
    
    def process_directory(self, directory_path: str):
        """
        Process all PDF documents in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
        """
        dir_path = Path(directory_path)
        
        # Map file names to document names
        file_mapping = {
            'Meta Privacy Policy': 'Privacy Policy',
            'Meta Terms of Service': 'Terms of Service',
            'Meta Cookies Policy': 'Cookie Policy'
        }
        
        pdf_files = list(dir_path.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            # Try to match document name
            document_name = None
            for file_key, doc_name in file_mapping.items():
                if file_key.lower() in pdf_file.name.lower():
                    document_name = doc_name
                    break
            
            if not document_name:
                # Use file name as document name
                document_name = pdf_file.stem
            
            self.process_pdf_document(str(pdf_file), document_name)


if __name__ == "__main__":
    # Example usage
    with Neo4jConnection() as conn:
        if not conn.verify_connectivity():
            print("Failed to connect to Neo4j. Please check your connection settings.")
            exit(1)
        
        builder = FacebookDocumentsBuilder(conn)
        
        # Process Facebook documents
        company_dir = Path(__file__).parent.parent.parent.parent / "data" / "company"
        if company_dir.exists():
            builder.process_directory(str(company_dir))
        else:
            print(f"Directory not found: {company_dir}")
        
        # Print statistics
        stats = conn.get_stats()
        print("\nDatabase Statistics:")
        print(f"Nodes: {stats['nodes']}")
        print(f"Relationships: {stats['relationships']}")

