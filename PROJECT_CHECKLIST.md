# Project Checklist for Paper Submission

## ✅ Completed

1. **Database Setup**
   - ✅ FAISS indices built (company, aiid, standards)
   - ✅ Neo4j knowledge graph populated (1,648 nodes, 9,516 relationships)
   - ✅ Embeddings generated and linked to nodes
   - ✅ Data quality verification script created

2. **Evaluation Framework**
   - ✅ 50 evaluation queries (30 standard, 20 long-tail)
   - ✅ Chunk pooling and deduplication implemented
   - ✅ Retrieval chunks exported (1,164 chunks) for manual labeling
   - ✅ IR metrics calculation code ready (Precision@K, Recall@K, F1, MRR, MAP, NDCG)
   - ✅ Evaluation script supports answer generation

3. **Code Improvements**
   - ✅ Sentence-aware chunking implemented
   - ✅ Text cleaning pipeline added
   - ✅ Vector search returns top-8 (not 24) after combining databases
   - ✅ Unicode encoding issues fixed
   - ✅ Path resolution issues fixed

4. **Documentation**
   - ✅ Paper additions document created (PAPER_ADDITIONS.md)
   - ✅ README files updated
   - ✅ API documentation included

## 🔄 In Progress / To Do

### 1. Answer Generation via API ⚠️ **CRITICAL**

**Status**: API server exists, but needs to be used for evaluation

**Action Required**:
```bash
# Terminal 1: Start API server
python backend/retrieval/interfaces/api_server.py

# Terminal 2: Generate answers via API
python backend/evaluation/generate_answers_via_api.py
```

**What it does**:
- Calls API server for each query and method (vector, graph, hybrid)
- Generates answers using the API endpoint
- Saves answers to JSON file for IR metrics calculation

**Output**: `backend/evaluation/answers_via_api.json`

### 2. Manual Relevance Labeling ⚠️ **CRITICAL**

**Status**: CSV file ready, needs manual labeling

**File**: `backend/evaluation/retrieval_results.csv` (1,164 rows)

**Action Required**:
1. Open `retrieval_results.csv` in Excel/Google Sheets
2. For each chunk, fill in "Relevance" column:
   - **2** = Highly relevant
   - **1** = Partially relevant  
   - **0** = Not relevant
3. Save the labeled CSV

**Time Estimate**: 2-4 hours depending on thoroughness

### 3. IR Metrics Calculation ⚠️ **CRITICAL**

**Status**: Code ready, needs labeled data

**Action Required**:
```bash
# After labeling retrieval_results.csv:
python backend/evaluation/evaluate.py metrics \
    --labels backend/evaluation/retrieval_results.csv \
    --output backend/evaluation/metrics_results.json \
    --k 8 \
    --per-method
```

**Output**: JSON file with Precision@8, Recall@8, F1@8, MRR, MAP, NDCG@8 for each method

### 4. Results Integration

**Action Required**:
- Fill in results tables in paper using metrics from step 3
- Add statistical analysis (if applicable)
- Create visualizations (bar charts comparing methods)

### 5. Reproducibility Documentation

**Files to Create/Update**:

**a) `.env.example`** (if not exists):
```env
# API Configuration
OPENAI_API_KEY=your_key_here
API_PROVIDER=openai
LLM_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small

# Local Embeddings (fallback)
USE_LOCAL_EMBEDDINGS=false
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Server Configuration
PORT=5000
FLASK_DEBUG=False
```

**b) `SETUP.md`** (detailed setup instructions):
- Step-by-step database setup
- How to run rebuild_database.py
- How to start API server
- How to run evaluation

**c) `REPRODUCIBILITY.md`**:
- Exact versions of dependencies
- System requirements
- Step-by-step reproduction guide
- Expected outputs and file sizes

### 6. Paper-Specific Additions

**Already Created**: `PAPER_ADDITIONS.md` contains:
- ✅ Evaluation methodology section
- ✅ IR metrics explanations
- ✅ Data quality verification
- ✅ Performance results framework
- ✅ API documentation
- ✅ Limitations and future work

**Still Needed**:
- Fill in actual results numbers (after metrics calculation)
- Add figures/tables with results
- Statistical significance tests (if applicable)

### 7. Code Cleanup (Optional but Recommended)

**Files to Review**:
- Remove any temporary test files
- Ensure all scripts have proper error handling
- Add docstrings where missing
- Verify all imports work correctly

### 8. Testing (Optional but Recommended)

**Quick Verification**:
```bash
# Test API server
python backend/retrieval/interfaces/api_server.py
# In another terminal:
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the privacy policies?", "mode": "hybrid", "generate_answer": true}'

# Test evaluation
python backend/evaluation/evaluate.py run --export-chunks --no-answer

# Test metrics calculation (with sample labels)
# (Create a small test CSV with labels first)
```

## 📋 Summary: What You Need to Do Next

### Immediate (Before IR Metrics):

1. **Start API Server**:
   ```bash
   python backend/retrieval/interfaces/api_server.py
   ```

2. **Generate Answers**:
   ```bash
   python backend/evaluation/generate_answers_via_api.py
   ```
   (Script created: `backend/evaluation/generate_answers_via_api.py`)

3. **Manual Labeling**:
   - Open `backend/evaluation/retrieval_results.csv`
   - Label all 1,164 chunks (0, 1, or 2)
   - Save the file

4. **Calculate Metrics**:
   ```bash
   python backend/evaluation/evaluate.py metrics \
       --labels backend/evaluation/retrieval_results.csv \
       --output backend/evaluation/metrics_results.json \
       --k 8
   ```

### For Paper Submission:

5. **Fill in Results**:
   - Copy metrics from `metrics_results.json` into paper tables
   - Add to `PAPER_ADDITIONS.md` Performance Results section

6. **Create Reproducibility Docs**:
   - `.env.example` file
   - `SETUP.md` with detailed instructions
   - `REPRODUCIBILITY.md` with exact versions

7. **Final Paper Review**:
   - Ensure all sections are complete
   - Add actual numbers to tables
   - Review limitations section
   - Check all citations

## 🎯 Critical Path to Completion

```
1. Start API Server
   ↓
2. Generate Answers via API
   ↓
3. Manual Labeling (2-4 hours)
   ↓
4. Calculate IR Metrics
   ↓
5. Fill Results in Paper
   ↓
6. Create Reproducibility Docs
   ↓
7. Final Review & Submission
```

## 📝 Notes

- **API Server**: Must be running for answer generation. Uses your API key from `.env`
- **Manual Labeling**: This is the most time-consuming step but critical for evaluation
- **Metrics**: Code is ready, just needs labeled data
- **Paper**: Framework is complete, needs actual results filled in

## ✅ You're Almost There!

The hard technical work is done. What remains:
1. API-based answer generation (script created)
2. Manual labeling (your task)
3. Metrics calculation (automated)
4. Results integration (copy-paste numbers)
5. Documentation (templates provided)

Good luck with the paper submission! 🎉
