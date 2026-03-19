# Compliance RAG System - Makefile

.PHONY: help install clean test process-graph process-vector build-faiss build-neo4j complete start-web-app run-agent run-evaluation

.DEFAULT_GOAL := help

VENV_DIR := venv
PYTHON := $(shell [ -f "$(VENV_DIR)/bin/python" ] && echo "$(VENV_DIR)/bin/python" || [ -f "$(VENV_DIR)/Scripts/python.exe" ] && echo "$(VENV_DIR)/Scripts/python.exe" || command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)
PIP := $(shell [ -f "$(VENV_DIR)/bin/pip" ] && echo "$(VENV_DIR)/bin/pip" || [ -f "$(VENV_DIR)/Scripts/pip.exe" ] && echo "$(VENV_DIR)/Scripts/pip.exe" || echo "$(PYTHON) -m pip")

help:
	@echo "Compliance RAG System"
	@echo "  make install          - Setup environment"
	@echo "  make process-graph    - Process data to graphs"
	@echo "  make process-vector    - Generate embeddings"
	@echo "  make build-faiss      - Build FAISS indexes"
	@echo "  make build-neo4j      - Build Neo4j graph"
	@echo "  make complete         - Full pipeline"
	@echo "  make start-web-app    - Start web app"
	@echo "  make run-agent        - Run agent"
	@echo "  make clean            - Clean outputs"

install:
	@[ -d "$(VENV_DIR)" ] || $(PYTHON) -m venv $(VENV_DIR)
	@$(PIP) install -r requirements.txt
	@echo "✓ Setup complete"

process-graph:
	@$(PYTHON) backend/processing/graph/gdpr_to_graph.py
	@$(PYTHON) backend/processing/graph/company_to_graph.py
	@$(PYTHON) backend/processing/graph/aiid_to_graph.py
	@echo "✓ Graph processing complete"

process-vector:
	@$(PYTHON) backend/processing/vector/standards_to_embeddings.py
	@$(PYTHON) backend/processing/vector/company_to_embeddings.py
	@$(PYTHON) backend/processing/vector/aiid_to_embeddings.py
	@echo "✓ Vector processing complete"

build-faiss:
	@$(PYTHON) backend/indexing/faiss/build_faiss_index.py --source company
	@$(PYTHON) backend/indexing/faiss/build_faiss_index.py --source aiid
	@$(PYTHON) backend/indexing/faiss/build_faiss_index.py --source standards
	@echo "✓ FAISS indexes built"

build-neo4j:
	@$(PYTHON) backend/indexing/neo4j/build_knowledge_graph.py
	@$(PYTHON) backend/indexing/neo4j/scripts/add_embeddings.py --json-dir backend/processed/vector/company --node-type Clause
	@$(PYTHON) backend/indexing/neo4j/scripts/add_embeddings.py --json-dir backend/processed/vector/standards --node-type Article
	@$(PYTHON) backend/indexing/neo4j/scripts/link_clauses_to_articles.py
	@echo "✓ Neo4j graph built"

complete: process-graph process-vector build-faiss build-neo4j
	@echo "✓ Complete pipeline finished"

test:
	@test -f backend/processed/graph/gdpr_graph.json && echo "✓ GDPR" || echo "✗ GDPR"
	@test -f backend/processed/graph/company_graph.json && echo "✓ Company" || echo "✗ Company"
	@test -f backend/processed/graph/aiid_graph.json && echo "✓ AIID" || echo "✗ AIID"
	@test -f backend/indexing/faiss/output/company_faiss_index.index && echo "✓ Company FAISS" || echo "✗ Company FAISS"
	@test -f backend/indexing/faiss/output/standards_faiss_index.index && echo "✓ Standards FAISS" || echo "✗ Standards FAISS"
	@test -f backend/indexing/faiss/output/aiid_faiss_index.index && echo "✓ AIID FAISS" || echo "✗ AIID FAISS"

clean:
	@rm -f backend/processed/graph/*.json
	@rm -rf backend/processed/vector/*/*.json
	@rm -rf backend/indexing/faiss/output/*.{index,pkl,json}
	@echo "✓ Cleanup complete"

start-web-app:
	@VENV_PYTHON="$$([ -x "$(VENV_DIR)/bin/python3" ] && echo "$(VENV_DIR)/bin/python3" || echo "$(VENV_DIR)/bin/python")"; \
	$$VENV_PYTHON query.py api & BACKEND_PID=$$!; sleep 2; \
	$$VENV_PYTHON frontend/scripts/start_server.py & FRONTEND_PID=$$!; \
	trap "kill $$BACKEND_PID $$FRONTEND_PID 2>/dev/null; exit" INT TERM; wait $$BACKEND_PID $$FRONTEND_PID

run-agent:
	@VENV_PYTHON="$$([ -x "$(VENV_DIR)/bin/python3" ] && echo "$(VENV_DIR)/bin/python3" || echo "$(VENV_DIR)/bin/python")"; \
	$$VENV_PYTHON query.py agent

run-evaluation:
	@VENV_PYTHON="$$([ -x "$(VENV_DIR)/bin/python3" ] && echo "$(VENV_DIR)/bin/python3" || echo "$(VENV_DIR)/bin/python")"; \
	$$VENV_PYTHON backend/evaluation/evaluate.py run $(ARGS)
