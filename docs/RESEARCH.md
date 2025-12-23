# Research Documentation

Complete research documentation for the Compliance RAG System, including dataset information, experimental setup, and ethical considerations.

## Table of Contents

1. [Dataset Documentation](#dataset-documentation)
2. [Experimental Setup](#experimental-setup)
3. [Legal, Ethical, and Professional Considerations](#legal-ethical-and-professional-considerations)

---

## Dataset Documentation

This section describes the datasets used in the Compliance RAG System for publication purposes.

### Data Sources

#### 1. GDPR Regulation (Standards)

- **Source**: [Specify source, e.g., "Official Journal of the European Union"]
- **Version**: [Specify version/date, e.g., "Regulation (EU) 2016/679, as of 2018-05-25"]
- **Format**: PDF
- **Location**: `data/standards/gdpr.pdf`
- **License**: Public domain / EU legislation
- **Preprocessing**:
  - Text extraction using PyPDF2
  - Structured parsing into Articles, Topics, and SubObligations
  - Total Articles: 102
  - Total Topics: [Specify]
  - Total SubObligations: [Specify]

#### 2. Company Documents

- **Source**: [Specify company name or "Anonymized Company"]
- **Documents**:
  1. Privacy Policy (`data/company/Meta Privacy Policy.pdf`)
  2. Terms of Service (`data/company/Meta Terms of Service.pdf`)
  3. Cookie Policy (`data/company/Meta Cookies Policy.pdf`)
- **Anonymization**: [Specify if documents were anonymized]
- **Format**: PDF
- **Preprocessing**:
  - Text extraction using PyPDF2
  - Clause segmentation
  - Total Clauses: ~292
  - Total Documents: 3

#### 3. AIID (AI Incident Database)

- **Source**: [Specify source, e.g., "AIID - AI Incident Database"]
- **Version**: [Specify version]
- **License**: [Specify license]
- **Format**: CSV
- **Location**: `data/aiid/incidents.csv`
- **Preprocessing**:
  - CSV parsing
  - Incident extraction
  - Total Incidents: [Specify count]
- **Additional Files**:
  - Classification CSVs (used for embeddings only, not graph structure)

### Dataset Statistics

#### Overall Statistics

| Dataset | Documents | Chunks/Clauses | Tokens (approx) | Embeddings |
|---------|-----------|----------------|-----------------|------------|
| GDPR    | 1         | 102 Articles   | [Specify]       | 102        |
| Company | 3         | ~292 Clauses   | [Specify]       | ~292       |
| AIID    | 1 CSV     | [Specify]       | [Specify]       | [Specify]   |

#### Graph Structure Statistics

- **Total Nodes**: [Specify]
  - Article nodes: 102
  - Clause nodes: ~292
  - Document nodes: 3
  - Incident nodes: [Specify]
  - Topic nodes: [Specify]
  - SubObligation nodes: [Specify]

- **Total Relationships**: [Specify]
  - ADDRESSES (Clause → Article): 21
  - VIOLATES (Incident → Article): [Specify]
  - COVERS (Document → Clause): [Specify]
  - HAS_TOPIC (Article → Topic): [Specify]
  - HAS_SUB_OBLIGATION (Article → SubObligation): [Specify]

#### Coverage Statistics

- **Articles Covered**: 8 (7.84%)
- **Articles Not Covered**: 94 (92.16%)
- **Average Clauses per Covered Article**: [Specify]
- **Most Covered Article**: Article 12 (10 clauses)

### Preprocessing Details

#### Text Extraction

- **Method**: PyPDF2 PDF text extraction
- **Quality Control**: Manual verification of extraction accuracy
- **Issues Encountered**: [Document any issues, e.g., formatting problems, OCR errors]

#### Chunking Strategy

- **Chunk Size**: 1000 tokens (default)
- **Chunk Overlap**: 200 tokens (default)
- **Chunking Method**: Sliding window
- **Rationale**: Balance between context preservation and granularity

#### Embedding Generation

- **Model**: `text-embedding-3-small` (OpenAI)
- **Dimensions**: 1536
- **Normalization**: L2 normalization for cosine similarity
- **Total Embeddings Generated**: [Specify]

#### Graph Construction

- **Similarity Threshold**: 0.45 (for clause-article linking)
- **Linking Method**: Embedding similarity comparison
- **Validation**: [Specify validation process]

### Data Quality

#### Quality Metrics

- **Extraction Accuracy**: [Specify if measured]
- **Chunking Quality**: [Specify assessment]
- **Embedding Quality**: [Specify if validated]
- **Graph Accuracy**: [Specify validation method]

#### Known Issues

- [List any known data quality issues]
- [Document any limitations]

### Ground Truth Dataset

#### Test Queries

- **Total Queries**: [Specify]
- **Query Types**:
  - Semantic queries: [Count]
  - Graph queries: [Count]
  - Hybrid queries: [Count]

#### Annotation Process

- **Annotators**: [Specify number and expertise]
- **Annotation Guidelines**: [Reference guidelines document]
- **Inter-annotator Agreement**: [Specify if measured]
- **Annotation Format**: JSON (see `backend/evaluation/ground_truth_template.json`)

#### Ground Truth Structure

```json
{
  "query": "Example query",
  "relevant_ids": ["id1", "id2", ...],
  "relevance_scores": {
    "id1": 1.0,
    "id2": 0.8,
    ...
  }
}
```

### Data Privacy and Ethics

#### Privacy Considerations

- **Company Documents**: [Specify anonymization process]
- **Personal Data**: [Specify if any personal data is present]
- **Data Handling**: [Specify data handling procedures]

#### Ethical Considerations

- **Data Usage Rights**: [Specify]
- **Consent**: [Specify if consent was obtained]
- **Bias**: [Document any potential biases]

### Reproducibility

#### Data Availability

- **Public Data**: GDPR regulation (public domain)
- **Proprietary Data**: Company documents (may require permission)
- **Third-Party Data**: AIID dataset (check license)

#### Data Access

- **Public Repositories**: [Specify if data is available]
- **Contact**: [Specify contact for data access requests]

---

## Experimental Setup

This section describes the experimental setup used for reproducibility in the publication.

### Hardware Configuration

#### Test Environment 1 (Primary)
- **CPU**: [Specify CPU model, e.g., Intel Core i7-12700K or AMD Ryzen 9 5900X]
- **RAM**: [Specify amount, e.g., 32 GB DDR4]
- **GPU**: [If used, specify model, e.g., NVIDIA RTX 3090 or None]
- **Storage**: [Specify type, e.g., SSD NVMe]
- **OS**: [Specify OS and version, e.g., Windows 11 Pro 22H2 or Ubuntu 22.04 LTS]

#### Test Environment 2 (Validation) [Optional]
- **CPU**: [Specify]
- **RAM**: [Specify]
- **GPU**: [Specify]
- **Storage**: [Specify]
- **OS**: [Specify]

### Software Configuration

#### Python Environment
- **Python Version**: 3.8.0 or higher (tested with Python 3.10.12)
- **Virtual Environment**: venv (included in repository)
- **Package Manager**: pip

#### Key Software Versions
- **Neo4j**: [Specify version, e.g., 5.15.0]
- **OpenAI API**: [Specify API version used, e.g., v1]
- **FAISS**: [Specify version from requirements.txt]

#### Dependencies
All dependencies are specified in `requirements.txt` with version constraints.
Install with:
```bash
pip install -r requirements.txt
```

### Random Seed Configuration

For reproducibility, random seeds are set in the following components:

#### Python Random Seed
```python
import random
random.seed(42)
```

#### NumPy Random Seed
```python
import numpy as np
np.random.seed(42)
```

#### FAISS Random Seed
FAISS uses deterministic algorithms by default, but ensure consistent initialization.

### API Configuration

#### OpenAI API
- **Model Snapshots**: [Specify exact model versions used]
  - Embedding Model: `text-embedding-3-small` (snapshot: [date if available])
  - LLM Model: `gpt-4` (snapshot: [date if available])
  - Judge LLM Model: `gpt-4o` (snapshot: [date if available])
- **API Version**: [Specify, e.g., v1]
- **Rate Limits**: [Document any rate limits encountered]

#### Neo4j Configuration
- **Version**: [Specify Neo4j version]
- **Connection**: bolt://localhost:7687
- **Memory Settings**: [Specify heap size, page cache if relevant]
- **Database**: [Specify database name, e.g., neo4j]

### Dataset Configuration

#### Data Sources
- **GDPR PDF**: Version [specify], Source [specify], Date [specify]
- **Company Documents**: [Specify which documents, anonymization process]
- **AIID Dataset**: Version [specify], License [specify]

#### Preprocessing Parameters
- **Chunk Size**: 1000 tokens (default)
- **Chunk Overlap**: 200 tokens (default)
- **Embedding Model**: text-embedding-3-small
- **Similarity Threshold**: 0.45 (for clause-article linking)

### Experimental Parameters

#### Search Parameters
- **top_k**: 10 (default, varied in ablation studies)
- **similarity_threshold**: 0.0 (default)
- **RRF_k**: 60 (default, varied in ablation studies)

#### Evaluation Parameters
- **Number of Runs**: 5 (for statistical significance)
- **Random Seeds**: [42, 123, 456, 789, 101112] (for multiple runs)
- **Judge LLM Temperature**: 0.1 (for consistency)
- **Confidence Level**: 95% (for confidence intervals)

### Reproducibility Checklist

- [ ] All dependencies installed from `requirements.txt`
- [ ] Random seeds set as specified
- [ ] Neo4j running with correct version
- [ ] OpenAI API keys configured in `.env`
- [ ] Data files in correct locations (`data/` directory)
- [ ] FAISS indexes built (`backend/building_database/faiss/`)
- [ ] Neo4j graph built (`backend/building_database/neo4j/`)
- [ ] Environment variables set (`.env` file)

### Running Experiments

#### Single Run
```bash
python backend/evaluation/evaluate_search.py \
    --queries-file backend/evaluation/test_queries.json \
    --top-k 10 \
    --output evaluation_results.json
```

#### Multiple Runs (for statistical analysis)
```bash
# Run evaluation 5 times with different seeds
for seed in 42 123 456 789 101112; do
    python backend/evaluation/evaluate_search.py \
        --queries-file backend/evaluation/test_queries.json \
        --top-k 10 \
        --output evaluation_results_seed_${seed}.json \
        --seed ${seed}
done
```

#### Ablation Studies
```bash
# Test different RRF_k values
for rrf_k in 30 60 90 120; do
    python backend/evaluation/evaluate_search.py \
        --queries-file backend/evaluation/test_queries.json \
        --rrf-k ${rrf_k} \
        --output ablation_rrf_${rrf_k}.json
done
```

---

## Legal, Ethical, and Professional Considerations

This section outlines the legal, ethical, and professional considerations implemented in the Compliance RAG System.

### Ethical AI Practices

#### Transparency and Explainability

**1. Transparent AI Decision-Making**
- **Source Citations**: All AI-generated answers include citations to source documents
- **Similarity Scores**: Search results display similarity scores
- **Query Expansion Disclosure**: Users are informed when query expansion is used
- **Reranking Transparency**: Users can see original and reranked results

**2. Explainable Search Results**
- Clear explanations of which databases were searched
- Information about result ranking
- Relationship discovery in knowledge graph

**3. User Control**
- Choose which databases to search
- Adjust similarity thresholds
- Enable/disable AI features (reranking, contextualization)
- Switch between search modes (vector-only vs. hybrid)
- View raw search results without AI processing

#### Fairness and Bias Mitigation

**1. Data Representation**
- Multiple data sources provide balanced perspectives
- No single data source dominates results
- Users can select specific databases to avoid bias

**2. Algorithmic Fairness**
- Equal treatment of all documents
- Configurable similarity thresholds
- No demographic bias

**3. Continuous Monitoring**
- Search queries and results logged for analysis
- Regular evaluation for potential biases
- User feedback mechanisms

#### Accountability and Responsibility

**1. Human-in-the-Loop**
- AI-generated answers presented as suggestions
- Users responsible for verifying compliance information
- System explicitly states it is a tool, not legal advice

**2. Error Handling**
- Clear error messages when AI operations fail
- Fallback mechanisms when API calls fail
- Users notified when confidence is low

**3. Audit Trail**
- Conversation history maintained (when enabled)
- Search queries and results logged for review
- Users can export results for documentation

### Privacy & Security Considerations

#### Data Privacy

**1. Data Handling**
- Local processing on user's machine or controlled servers
- Minimal data exposure (only embeddings and queries sent to API)
- User control over which documents are processed

**2. API Key Management**
- Secure storage in `.env` files, excluded from version control
- Environment variables for sensitive credentials
- Clear instructions on securing API keys

**3. Data Retention**
- No persistent storage of user data
- Optional logging can be disabled
- Users control database content

#### Security Measures

**1. Input Validation**
- All user inputs validated before processing
- Parameterized queries for Neo4j
- Path traversal prevention
- API input sanitization

**2. Authentication & Authorization**
- Secure Neo4j connection with username/password
- API keys never exposed in client-side code
- HTTPS support for production
- Users control access to local databases

**3. Secure Communication**
- HTTPS support for frontend
- Proper CORS configuration
- Security headers in backend API
- Environment isolation via virtual environments

### Accessibility

#### Web Interface Accessibility

**1. Keyboard Navigation**
- All interactive elements keyboard accessible
- Logical tab order
- Keyboard shortcuts for common actions
- Focus indicators for screen readers

**2. Screen Reader Support**
- Semantic HTML structure
- ARIA labels for interactive elements
- Alt text for icons and images
- Proper heading hierarchy

**3. Visual Accessibility**
- Sufficient color contrast ratios
- Responsive text sizing
- Flexible layout for different screen sizes
- High contrast mode compatibility

### Professional Coding Standards

#### Code Quality

**1. Code Organization**
- Modular structure with logical modules
- Clear separation of concerns
- DRY principle (no code duplication)
- Single responsibility per module

**2. Documentation**
- Comprehensive docstrings for functions and classes
- Comments explaining complex logic
- README files for major components
- Python type hints where applicable

**3. Error Handling**
- Proper try-except blocks throughout
- Clear and actionable error messages
- Graceful degradation when possible
- Appropriate logging for debugging

#### Version Control

**1. Git Best Practices**
- Meaningful commit messages
- Organized branch structure
- Proper `.gitignore` exclusions
- No secrets in version control

**2. Dependencies Management**
- Pinned versions in `requirements.txt`
- Minimal dependencies
- Regular updates for security patches
- License compliance

### Compliance & Legal Considerations

#### Intellectual Property

**1. Open Source Compliance**
- Open-source compatible licenses
- Properly licensed dependencies
- Proper attribution of third-party code
- LICENSE file included

**2. Data Usage Rights**
- Users responsible for ensuring document processing rights
- No redistribution of source documents
- Fair use for compliance analysis

#### Regulatory Compliance

**1. GDPR Compliance**
- System designed to help organizations comply with GDPR
- Follows GDPR principles in its own operations
- Respects user rights to access, correct, and delete data

**2. Industry Standards**
- Follows industry best practices for AI systems
- Adheres to ethical AI guidelines
- Meets professional software development standards

#### Liability and Disclaimers

**1. Use Disclaimer**
- System provides information, not legal advice
- Users responsible for verifying compliance
- System provided "as-is" without warranties
- Results should be reviewed by legal professionals

**2. Accuracy Limitations**
- AI-generated content may contain errors
- Results depend on quality of source documents
- Human review recommended for critical decisions

### Ethical AI Framework Alignment

This system aligns with established ethical AI frameworks:

#### ACM Code of Ethics
- Contribute to Society: Helps organizations improve compliance
- Avoid Harm: Includes safeguards and disclaimers
- Be Honest: Transparent about capabilities and limitations
- Strive for Quality: High-quality code and documentation
- Give Credit: Proper attribution of sources

#### IEEE Ethically Aligned Design
- Human Rights: Respects user autonomy and privacy
- Well-being: Designed to benefit organizations and society
- Accountability: Clear responsibility and audit trails
- Transparency: Explainable and transparent operations

#### EU AI Act Principles
- Human Agency: Users maintain control
- Technical Robustness: Reliable and secure system
- Privacy: Privacy-by-design approach
- Transparency: Clear information about AI use
- Diversity: Accessible to diverse users
- Accountability: Mechanisms for accountability

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0     | 2024-12-XX | Initial research documentation |

## Contact

For questions about research documentation, contact: [Your contact information]

