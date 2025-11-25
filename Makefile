# Compliance RAG System - Makefile
# Automates the complete data processing and database building pipeline
# 
# NOTE: This Makefile is for Linux/Mac/WSL systems
# For Windows PowerShell, use build.ps1 instead

.PHONY: help install venv clean clean-venv all process-graph process-vector build-faiss build-faiss-only build-neo4j setup-neo4j-docker check-neo4j get-windows-ip add-embeddings link-relationships setup complete start-web-app stop-web-app run-agent

# Default target
.DEFAULT_GOAL := help

# Python executable - try python3 first (common in WSL/Linux), fallback to python
# This detects which Python executable is available
PYTHON_CMD := $(shell sh -c 'if command -v python3 >/dev/null 2>&1; then echo python3; elif command -v python >/dev/null 2>&1; then echo python; else echo python3; fi')

# Virtual environment directory
VENV_DIR := venv

# Python executable - use venv if it exists, otherwise use system Python
PYTHON := $(shell if [ -f "$(VENV_DIR)/bin/python" ]; then echo "$(VENV_DIR)/bin/python"; elif [ -f "$(VENV_DIR)/bin/python3" ]; then echo "$(VENV_DIR)/bin/python3"; elif [ -f "$(VENV_DIR)/bin/python3.12" ]; then echo "$(VENV_DIR)/bin/python3.12"; elif [ -f "$(VENV_DIR)/Scripts/python.exe" ]; then echo "$(VENV_DIR)/Scripts/python.exe"; else echo "$(PYTHON_CMD)"; fi)
PIP := $(shell if [ -f "$(VENV_DIR)/bin/pip" ]; then echo "$(VENV_DIR)/bin/pip"; elif [ -f "$(VENV_DIR)/bin/pip3" ]; then echo "$(VENV_DIR)/bin/pip3"; elif [ -f "$(VENV_DIR)/Scripts/pip.exe" ]; then echo "$(VENV_DIR)/Scripts/pip.exe"; else echo "$(PYTHON_CMD) -m pip"; fi)

# Project root directory
ROOT_DIR := $(shell pwd)

# Directories
BACKEND_DIR := backend
DATA_PROCESSING_DIR := $(BACKEND_DIR)/data_processing
BUILDING_DB_DIR := $(BACKEND_DIR)/building_database
SEARCHING_DIR := $(BACKEND_DIR)/searching
FRONTEND_DIR := frontend

# Data paths
DATA_DIR := data
PROCESSED_DIR := $(DATA_PROCESSING_DIR)/processed
GRAPH_DIR := $(PROCESSED_DIR)/graph
VECTOR_DIR := $(PROCESSED_DIR)/vector

# Scripts
GDPR_TO_GRAPH := $(DATA_PROCESSING_DIR)/graph/gdpr_to_graph.py
COMPANY_TO_GRAPH := $(DATA_PROCESSING_DIR)/graph/company_to_graph.py
AIID_TO_GRAPH := $(DATA_PROCESSING_DIR)/graph/aiid_to_graph.py

GDPR_TO_EMBEDDINGS := $(DATA_PROCESSING_DIR)/vector/standards_to_embeddings.py
COMPANY_TO_EMBEDDINGS := $(DATA_PROCESSING_DIR)/vector/company_to_embeddings.py
AIID_TO_EMBEDDINGS := $(DATA_PROCESSING_DIR)/vector/aiid_to_embeddings.py

BUILD_FAISS_COMPANY := $(BUILDING_DB_DIR)/faiss/company_to_faiss_database.py
BUILD_FAISS_STANDARDS := $(BUILDING_DB_DIR)/faiss/standards_to_faiss_database.py
BUILD_FAISS_AIID := $(BUILDING_DB_DIR)/faiss/aiid_to_faiss_database.py

BUILD_NEO4J := $(BUILDING_DB_DIR)/neo4j/build_knowledge_graph.py
ADD_EMBEDDINGS := $(BUILDING_DB_DIR)/neo4j/add_embeddings.py
LINK_RELATIONSHIPS := $(BUILDING_DB_DIR)/neo4j/link_clauses_to_articles.py

# Help target
help:
	@echo "Compliance RAG System - Makefile"
	@echo "================================="
	@echo ""
	@echo "Python executable: $(PYTHON)"
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment: ✓ Active ($(VENV_DIR))"; \
	else \
		echo "Virtual environment: ✗ Not created (run 'make install' to create)"; \
	fi
	@echo ""
	@echo "Available targets:"
	@echo "  make venv             - Create virtual environment"
	@echo "  make install          - Create venv and install dependencies"
	@echo "  make process-graph    - Process raw data to graph JSON files"
	@echo "  make process-vector - Process raw data to embeddings JSON files"
	@echo "  make process-vector   - Generate embeddings from raw data"
	@echo "  make build-faiss      - Build FAISS vector databases"
	@echo "  make build-faiss-only - Build FAISS only (skip Neo4j)"
	@echo "  make setup-neo4j-docker - Setup Neo4j in Docker (WSL users)"
	@echo "  make check-neo4j      - Check Neo4j connection"
	@echo "  make get-windows-ip  - Get Windows host IP (for WSL users)"
	@echo "  make build-neo4j      - Build Neo4j knowledge graph"
	@echo "  make add-embeddings   - Add embeddings to Neo4j nodes"
	@echo "  make link-relationships - Link clauses to articles"
	@echo "  make complete         - Full pipeline (all steps)"
	@echo "  make start-web-app    - Start web application (API + frontend)"
	@echo "  make stop-web-app     - Stop web application servers"
	@echo "  make run-agent        - Run agentic system interactively"
	@echo "  make clean            - Clean processed data and databases"
	@echo "  make test             - Test the system"
	@echo ""
	@echo "Individual steps:"
	@echo "  make gdpr-graph       - Process GDPR PDF to graph JSON"
	@echo "  make company-graph    - Process company PDFs to graph JSON"
	@echo "  make aiid-graph       - Process AIID CSV to graph JSON"
	@echo "  make gdpr-embeddings  - Generate GDPR embeddings"
	@echo "  make company-embeddings - Generate company embeddings"
	@echo "  make aiid-embeddings  - Generate AIID embeddings"
	@echo ""

# Create virtual environment
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON_CMD) -m venv $(VENV_DIR); \
		echo "✓ Virtual environment created"; \
	else \
		echo "Virtual environment already exists"; \
	fi

# Install dependencies
install: venv
	@echo "Installing Python dependencies..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Error: Virtual environment not created!"; \
		exit 1; \
	fi
	@PIP_EXE=$$(if [ -f "$(VENV_DIR)/bin/pip" ]; then echo "$(VENV_DIR)/bin/pip"; elif [ -f "$(VENV_DIR)/bin/pip3" ]; then echo "$(VENV_DIR)/bin/pip3"; elif [ -f "$(VENV_DIR)/Scripts/pip.exe" ]; then echo "$(VENV_DIR)/Scripts/pip.exe"; else echo "$(PYTHON_CMD) -m pip"; fi); \
	$$PIP_EXE install -r requirements.txt
	@echo "✓ Dependencies installed"

# Check if .env exists
check-env:
	@if [ ! -f .env ]; then \
		echo "⚠ Warning: .env file not found!"; \
		echo "  Create .env file with OPENAI_API_KEY and other settings"; \
		echo "  See .env.example for template"; \
	fi
	@if ! $(PYTHON) --version >/dev/null 2>&1; then \
		echo "⚠ Error: $(PYTHON) not found!"; \
		echo "  Install Python 3: sudo apt-get install python3"; \
		exit 1; \
	fi

# ========== Graph Processing ==========

gdpr-graph: check-env
	@if [ -f "$(GRAPH_DIR)/gdpr_graph.json" ]; then \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "⚠ GDPR graph JSON already exists: $(GRAPH_DIR)/gdpr_graph.json"; \
		echo "  Skipping processing. Run 'make clean-graph' to reprocess."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	else \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "Processing GDPR PDF to graph JSON..."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		$(PYTHON) $(GDPR_TO_GRAPH); \
		echo "✓ GDPR graph processing complete"; \
	fi

company-graph: check-env
	@if [ -f "$(GRAPH_DIR)/company_graph.json" ]; then \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "⚠ Company graph JSON already exists: $(GRAPH_DIR)/company_graph.json"; \
		echo "  Skipping processing. Run 'make clean-graph' to reprocess."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	else \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "Processing company PDFs to graph JSON..."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		$(PYTHON) $(COMPANY_TO_GRAPH); \
		echo "✓ Company graph processing complete"; \
	fi

aiid-graph: check-env
	@if [ -f "$(GRAPH_DIR)/aiid_graph.json" ]; then \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "⚠ AIID graph JSON already exists: $(GRAPH_DIR)/aiid_graph.json"; \
		echo "  Skipping processing. Run 'make clean-graph' to reprocess."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	else \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "Processing AIID CSV to graph JSON..."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		$(PYTHON) $(AIID_TO_GRAPH); \
		echo "✓ AIID graph processing complete"; \
	fi

process-graph: gdpr-graph company-graph aiid-graph
	@echo ""
	@echo "✓ All graph processing complete!"

# ========== Vector Processing ==========

gdpr-embeddings: check-env
	@if [ -d "$(VECTOR_DIR)/standards" ] && [ -n "$$(ls -A $(VECTOR_DIR)/standards/*.json 2>/dev/null)" ]; then \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "⚠ GDPR embeddings already exist in $(VECTOR_DIR)/standards/"; \
		echo "  Skipping processing. Run 'make clean-vector' to reprocess."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	else \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "Generating GDPR embeddings..."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		$(PYTHON) $(GDPR_TO_EMBEDDINGS); \
		echo "✓ GDPR embeddings complete"; \
	fi

company-embeddings: check-env
	@if [ -d "$(VECTOR_DIR)/company" ] && [ -n "$$(ls -A $(VECTOR_DIR)/company/*.json 2>/dev/null)" ]; then \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "⚠ Company embeddings already exist in $(VECTOR_DIR)/company/"; \
		echo "  Skipping processing. Run 'make clean-vector' to reprocess."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	else \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "Generating company embeddings..."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		$(PYTHON) $(COMPANY_TO_EMBEDDINGS); \
		echo "✓ Company embeddings complete"; \
	fi

aiid-embeddings: check-env
	@if [ -d "$(VECTOR_DIR)/aiid" ] && [ -n "$$(ls -A $(VECTOR_DIR)/aiid/*.json 2>/dev/null)" ]; then \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "⚠ AIID embeddings already exist in $(VECTOR_DIR)/aiid/"; \
		echo "  Skipping processing. Run 'make clean-vector' to reprocess."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	else \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "Generating AIID embeddings..."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		$(PYTHON) $(AIID_TO_EMBEDDINGS); \
		echo "✓ AIID embeddings complete"; \
	fi

process-vector: gdpr-embeddings company-embeddings aiid-embeddings
	@echo ""
	@echo "✓ All vector processing complete!"

# ========== FAISS Database Building ==========

faiss-company: check-env
	@if [ -f "$(BUILDING_DB_DIR)/faiss/company/company_faiss_index.index" ]; then \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "⚠ Company FAISS index already exists"; \
		echo "  Skipping build. Run 'make clean-faiss' to rebuild."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	else \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "Building FAISS index for company documents..."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		$(PYTHON) $(BUILD_FAISS_COMPANY); \
		echo "✓ Company FAISS index complete"; \
	fi

faiss-standards: check-env
	@if [ -f "$(BUILDING_DB_DIR)/faiss/standards/standards_faiss_index.index" ]; then \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "⚠ Standards FAISS index already exists"; \
		echo "  Skipping build. Run 'make clean-faiss' to rebuild."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	else \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "Building FAISS index for standards (GDPR)..."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		$(PYTHON) $(BUILD_FAISS_STANDARDS); \
		echo "✓ Standards FAISS index complete"; \
	fi

faiss-aiid: check-env
	@if [ -f "$(BUILDING_DB_DIR)/faiss/aiid/aiid_faiss_index.index" ]; then \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "⚠ AIID FAISS index already exists"; \
		echo "  Skipping build. Run 'make clean-faiss' to rebuild."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	else \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "Building FAISS index for AIID..."; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		$(PYTHON) $(BUILD_FAISS_AIID); \
		echo "✓ AIID FAISS index complete"; \
	fi

build-faiss: faiss-company faiss-standards faiss-aiid
	@echo ""
	@echo "✓ All FAISS indexes built!"

# ========== Neo4j Graph Building ==========

# Get Windows host IP (for WSL users)
get-windows-ip:
	@echo "Finding Windows host IP address..."
	@echo ""
	@echo "Method 1: From /etc/resolv.conf (WSL default gateway):"
	@if [ -f /etc/resolv.conf ]; then \
		windows_ip=$$(cat /etc/resolv.conf | grep nameserver | awk '{print $$2}' | head -1); \
		if [ -n "$$windows_ip" ]; then \
			echo "  Windows host IP: $$windows_ip"; \
		else \
			echo "  Could not find IP"; \
		fi; \
	else \
		echo "  /etc/resolv.conf not found"; \
	fi
	@echo ""
	@echo "Method 2: From default route:"
	@default_ip=$$(ip route show | grep -i default | awk '{print $$3}' | head -1); \
	if [ -n "$$default_ip" ]; then \
		echo "  Default gateway IP: $$default_ip"; \
	else \
		echo "  Could not find default gateway"; \
	fi
	@echo ""
	@echo "Method 3: Test connectivity:"
	@echo "  Try: nc -zv 10.255.255.254 7687"
	@echo ""
	@echo "Update your .env file with one of these:"
	@if [ -f /etc/resolv.conf ]; then \
		windows_ip=$$(cat /etc/resolv.conf | grep nameserver | awk '{print $$2}' | head -1); \
		if [ -n "$$windows_ip" ]; then \
			echo "  NEO4J_URI=bolt://$$windows_ip:7687"; \
		fi; \
	fi
	@default_ip=$$(ip route show | grep -i default | awk '{print $$3}' | head -1); \
	if [ -n "$$default_ip" ]; then \
		echo "  NEO4J_URI=bolt://$$default_ip:7687"; \
	fi
	@echo ""
	@echo "Note: Windows Firewall may block port 7687. Check firewall settings."

# Check Neo4j connection
check-neo4j:
	@echo "Checking Neo4j connection..."
	@echo "Testing connection with current settings..."
	@$(PYTHON) -c "from neo4j import GraphDatabase; import os; from dotenv import load_dotenv; load_dotenv(); uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687'); user = os.getenv('NEO4J_USER', 'neo4j'); password = os.getenv('NEO4J_PASSWORD', 'password'); print(f'Attempting connection to: {uri}'); print(f'User: {user}'); driver = GraphDatabase.driver(uri, auth=(user, password)); driver.verify_connectivity(); driver.close(); print('✓ Neo4j connection successful')" 2>&1 || ( \
		echo ""; \
		echo "✗ Error: Cannot connect to Neo4j!"; \
		echo ""; \
		echo "Current settings from .env:"; \
		$(PYTHON) -c "import os; from dotenv import load_dotenv; load_dotenv(); print(f\"  NEO4J_URI={os.getenv('NEO4J_URI', 'not set')}\"); print(f\"  NEO4J_USER={os.getenv('NEO4J_USER', 'not set')}\"); print(f\"  NEO4J_PASSWORD={'*' * len(os.getenv('NEO4J_PASSWORD', '')) if os.getenv('NEO4J_PASSWORD') else 'not set'}\")" 2>/dev/null || echo "  (Could not read .env file)"; \
		echo ""; \
		echo "Troubleshooting steps:"; \
		echo "  1. Verify Neo4j Desktop is running and database is started"; \
		echo "  2. Check Windows Firewall - port 7687 may be blocked"; \
		echo "  3. Try testing port connectivity:"; \
		echo "     nc -zv 10.255.255.254 7687"; \
		echo "  4. Neo4j Desktop may need to allow external connections:"; \
		echo "     - Check Neo4j Desktop settings"; \
		echo "     - May need to bind to 0.0.0.0 instead of 127.0.0.1"; \
		echo "  5. Alternative: Try using Windows hostname:"; \
		echo "     NEO4J_URI=bolt://\$$(hostname).local:7687"; \
		echo "  6. Or try connecting via Windows localhost from WSL:"; \
		echo "     Get actual Windows IP: ip route show | grep -i default | awk '{ print \$$3}'"; \
		exit 1 \
	)

# Set similarity threshold for linking relationships
set-similarity-threshold:
	@read -p "Enter similarity threshold (0.0-1.0, suggested: 0.53): " threshold; \
	if [ -f .env ]; then \
		if grep -q "^ADDRESSES_SIMILARITY_THRESHOLD=" .env; then \
			sed -i "s|^ADDRESSES_SIMILARITY_THRESHOLD=.*|ADDRESSES_SIMILARITY_THRESHOLD=$$threshold|" .env; \
		else \
			echo "ADDRESSES_SIMILARITY_THRESHOLD=$$threshold" >> .env; \
		fi; \
		echo "✓ Updated ADDRESSES_SIMILARITY_THRESHOLD=$$threshold in .env"; \
	else \
		echo "ADDRESSES_SIMILARITY_THRESHOLD=$$threshold" > .env; \
		echo "✓ Created .env with ADDRESSES_SIMILARITY_THRESHOLD=$$threshold"; \
	fi

# Update .env file with Docker Neo4j settings
update-neo4j-env:
	@echo "Updating .env file with Docker Neo4j settings..."
	@BOLT_PORT=$$(docker port neo4j 2>/dev/null | grep 7687 | cut -d: -f2 || echo "7689"); \
	if [ -f .env ]; then \
		if grep -q "^NEO4J_URI=" .env; then \
			sed -i "s|^NEO4J_URI=.*|NEO4J_URI=bolt://localhost:$$BOLT_PORT|" .env; \
		else \
			echo "NEO4J_URI=bolt://localhost:$$BOLT_PORT" >> .env; \
		fi; \
		if grep -q "^NEO4J_USER=" .env; then \
			sed -i "s|^NEO4J_USER=.*|NEO4J_USER=neo4j|" .env; \
		else \
			echo "NEO4J_USER=neo4j" >> .env; \
		fi; \
		if grep -q "^NEO4J_PASSWORD=" .env; then \
			sed -i "s|^NEO4J_PASSWORD=.*|NEO4J_PASSWORD=password|" .env; \
		else \
			echo "NEO4J_PASSWORD=password" >> .env; \
		fi; \
		echo "✓ .env file updated!"; \
	else \
		echo "Creating .env file..."; \
		echo "NEO4J_URI=bolt://localhost:$$BOLT_PORT" > .env; \
		echo "NEO4J_USER=neo4j" >> .env; \
		echo "NEO4J_PASSWORD=password" >> .env; \
		echo "✓ .env file created!"; \
	fi; \
	echo ""; \
	echo "Current Neo4j Docker settings:"; \
	grep "^NEO4J_" .env || echo "  (No NEO4J_ settings found)"

build-neo4j: check-env
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Building Neo4j knowledge graph..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "⚠ Note: This will rebuild the entire graph. Existing data will be cleared."
	@echo "  Checking if graph JSON files exist..."
	@if [ ! -f "$(GRAPH_DIR)/gdpr_graph.json" ] || [ ! -f "$(GRAPH_DIR)/company_graph.json" ] || [ ! -f "$(GRAPH_DIR)/aiid_graph.json" ]; then \
		echo "✗ Error: Graph JSON files missing. Run 'make process-graph' first."; \
		exit 1; \
	fi
	@echo "  Checking Neo4j connection..."
	@if ! $(PYTHON) -c "from neo4j import GraphDatabase; import os; from dotenv import load_dotenv; load_dotenv(); uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687'); user = os.getenv('NEO4J_USER', 'neo4j'); password = os.getenv('NEO4J_PASSWORD', 'password'); driver = GraphDatabase.driver(uri, auth=(user, password)); driver.verify_connectivity(); driver.close()" 2>/dev/null; then \
		echo ""; \
		echo "⚠ Warning: Cannot connect to Neo4j!"; \
		echo "  Options:"; \
		echo "  1. Run Neo4j in WSL/Docker: make setup-neo4j-docker"; \
		echo "  2. Skip Neo4j and continue with FAISS only: make build-faiss"; \
		echo "  3. Fix Windows Neo4j connection (see troubleshooting above)"; \
		echo ""; \
		read -p "Continue anyway? (y/N): " confirm; \
		if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
			echo "Aborted."; \
			exit 1; \
		fi; \
	fi
	@echo "✓ All graph JSON files found. Building Neo4j graph..."
	$(PYTHON) $(BUILD_NEO4J)
	@echo "✓ Neo4j graph build complete"

# ========== Embeddings to Neo4j ==========

add-embeddings-company: check-env
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Adding company embeddings to Neo4j nodes..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	$(PYTHON) $(ADD_EMBEDDINGS) --json-dir $(VECTOR_DIR)/company --node-type Clause
	@echo "✓ Company embeddings added to Neo4j"

add-embeddings-standards: check-env
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Adding standards embeddings to Neo4j nodes..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	$(PYTHON) $(ADD_EMBEDDINGS) --json-dir $(VECTOR_DIR)/standards --node-type Article
	@echo "✓ Standards embeddings added to Neo4j"

add-embeddings: add-embeddings-company add-embeddings-standards
	@echo ""
	@echo "✓ All embeddings added to Neo4j!"

# ========== Link Relationships ==========

link-relationships: check-env
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Linking clauses to articles based on embedding similarity..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@if [ -f .env ] && grep -q "^ADDRESSES_SIMILARITY_THRESHOLD=" .env; then \
		THRESHOLD=$$(grep "^ADDRESSES_SIMILARITY_THRESHOLD=" .env | cut -d= -f2); \
		echo "Using similarity threshold from .env: $$THRESHOLD"; \
	else \
		echo "Using default similarity threshold: 0.45"; \
		echo "  (Set ADDRESSES_SIMILARITY_THRESHOLD in .env to customize)"; \
	fi
	$(PYTHON) $(LINK_RELATIONSHIPS)
	@echo "✓ Clause-to-article relationships created"

# ========== Complete Setup ==========

# Setup Neo4j in Docker (for WSL users)
setup-neo4j-docker:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Setting up Neo4j in Docker (WSL)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "Installing Docker..."; \
		sudo apt update && sudo apt install -y docker.io; \
		sudo service docker start; \
		sleep 3; \
	fi
	@echo "Cleaning up any existing containers..."
	@docker stop neo4j 2>/dev/null || true; \
	docker rm neo4j 2>/dev/null || true; \
	echo "Finding available ports to avoid conflicts..."
	@BOLT_PORT=7688; \
	HTTP_PORT=7475; \
	CONTAINER_CREATED=0; \
	for bolt in 7688 7689 7690 7691; do \
		for http in 7475 7476 7477 7478; do \
			if [ "$$CONTAINER_CREATED" = "0" ]; then \
				echo "Trying ports $$http (HTTP) and $$bolt (Bolt)..."; \
				if docker run -d \
					--name neo4j \
					-p $$http:7474 -p $$bolt:7687 \
					-e NEO4J_AUTH=neo4j/password \
					-e NEO4J_PLUGINS='["apoc"]' \
					neo4j:latest >/dev/null 2>&1; then \
					BOLT_PORT=$$bolt; \
					HTTP_PORT=$$http; \
					CONTAINER_CREATED=1; \
					echo "✓ Successfully created container on ports $$http (HTTP) and $$bolt (Bolt)"; \
					break; \
				else \
					docker rm neo4j 2>/dev/null || true; \
				fi \
			fi \
		done; \
		if [ "$$CONTAINER_CREATED" = "1" ]; then break; fi \
	done; \
	if [ "$$CONTAINER_CREATED" = "0" ]; then \
		echo "✗ Error: Could not find available ports. Please check what's using ports 7474-7478 and 7687-7691"; \
		echo "  You can check with: ss -tuln | grep -E ':(747[4-8]|768[7-9]|769[0-1])'"; \
		exit 1; \
	fi
	@echo "Waiting for Neo4j to start (this may take 30-60 seconds)..."
	@sleep 20
	@echo "Checking if Neo4j is ready..."
	@for i in 1 2 3 4 5; do \
		if docker exec neo4j cypher-shell -u neo4j -p password "RETURN 1" >/dev/null 2>&1; then \
			echo "✓ Neo4j is ready!"; \
			break; \
		fi; \
		echo "  Waiting... (attempt $$i/5)"; \
		sleep 10; \
	done
	@echo ""
	@echo "✓ Neo4j Docker container is running!"
	@echo ""
	@BOLT_PORT=$$(docker port neo4j 2>/dev/null | grep 7687 | cut -d: -f2 || echo "7689"); \
	echo "Updating .env file with Docker Neo4j settings..."; \
	if [ -f .env ]; then \
		if grep -q "^NEO4J_URI=" .env; then \
			sed -i "s|^NEO4J_URI=.*|NEO4J_URI=bolt://localhost:$$BOLT_PORT|" .env; \
		else \
			echo "NEO4J_URI=bolt://localhost:$$BOLT_PORT" >> .env; \
		fi; \
		if grep -q "^NEO4J_USER=" .env; then \
			sed -i "s|^NEO4J_USER=.*|NEO4J_USER=neo4j|" .env; \
		else \
			echo "NEO4J_USER=neo4j" >> .env; \
		fi; \
		if grep -q "^NEO4J_PASSWORD=" .env; then \
			sed -i "s|^NEO4J_PASSWORD=.*|NEO4J_PASSWORD=password|" .env; \
		else \
			echo "NEO4J_PASSWORD=password" >> .env; \
		fi; \
		echo "✓ .env file updated!"; \
	else \
		echo "Creating .env file..."; \
		echo "NEO4J_URI=bolt://localhost:$$BOLT_PORT" > .env; \
		echo "NEO4J_USER=neo4j" >> .env; \
		echo "NEO4J_PASSWORD=password" >> .env; \
		echo "✓ .env file created!"; \
	fi; \
	echo ""; \
	echo "Current Neo4j Docker settings:"; \
	echo "  NEO4J_URI=bolt://localhost:$$BOLT_PORT"; \
	echo "  NEO4J_USER=neo4j"; \
	echo "  NEO4J_PASSWORD=password"; \
	echo ""; \
	echo "Test connection: make check-neo4j"

# Build without Neo4j (FAISS only)
build-faiss-only: process-graph process-vector build-faiss
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✓ FAISS-only build complete!"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "System ready for vector search (FAISS only)"
	@echo "To add Neo4j later: make setup-neo4j-docker && make build-neo4j"

# Complete: Full pipeline including Neo4j
complete: process-graph process-vector build-faiss build-neo4j add-embeddings link-relationships
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🎉 COMPLETE PIPELINE FINISHED!"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Verifying all outputs..."
	@missing=0; \
	if [ ! -f "$(GRAPH_DIR)/gdpr_graph.json" ]; then echo "✗ Missing: GDPR graph JSON"; missing=1; fi; \
	if [ ! -f "$(GRAPH_DIR)/company_graph.json" ]; then echo "✗ Missing: Company graph JSON"; missing=1; fi; \
	if [ ! -f "$(GRAPH_DIR)/aiid_graph.json" ]; then echo "✗ Missing: AIID graph JSON"; missing=1; fi; \
	if [ ! -f "$(BUILDING_DB_DIR)/faiss/company/company_faiss_index.index" ]; then echo "✗ Missing: Company FAISS index"; missing=1; fi; \
	if [ ! -f "$(BUILDING_DB_DIR)/faiss/standards/standards_faiss_index.index" ]; then echo "✗ Missing: Standards FAISS index"; missing=1; fi; \
	if [ ! -f "$(BUILDING_DB_DIR)/faiss/aiid/aiid_faiss_index.index" ]; then echo "✗ Missing: AIID FAISS index"; missing=1; fi; \
	if [ $$missing -eq 0 ]; then \
		echo "✓ All outputs verified!"; \
		echo ""; \
		echo "System is ready to use:"; \
		echo "  • FAISS vector databases built"; \
		echo "  • Neo4j knowledge graph built"; \
		echo "  • Embeddings connected to nodes"; \
		echo "  • Clauses linked to articles"; \
		echo ""; \
		echo "Start the chatbot:"; \
		echo "  python $(SEARCHING_DIR)/run_chatbot.py --hybrid"; \
	else \
		echo ""; \
		echo "⚠ Some outputs are missing. Review errors above."; \
	fi
	@echo ""

# ========== Testing ==========

test:
	@echo "Testing system..."
	@echo "Checking if required files exist..."
	@test -f $(GRAPH_DIR)/gdpr_graph.json && echo "✓ GDPR graph JSON exists" || echo "✗ GDPR graph JSON missing"
	@test -f $(GRAPH_DIR)/company_graph.json && echo "✓ Company graph JSON exists" || echo "✗ Company graph JSON missing"
	@test -f $(GRAPH_DIR)/aiid_graph.json && echo "✓ AIID graph JSON exists" || echo "✗ AIID graph JSON missing"
	@test -d $(BUILDING_DB_DIR)/faiss/company && echo "✓ FAISS company index exists" || echo "✗ FAISS company index missing"
	@test -d $(BUILDING_DB_DIR)/faiss/standards && echo "✓ FAISS standards index exists" || echo "✗ FAISS standards index missing"
	@test -d $(BUILDING_DB_DIR)/faiss/aiid && echo "✓ FAISS AIID index exists" || echo "✗ FAISS AIID index missing"
	@echo ""
	@echo "Test complete. Check Neo4j separately."

# ========== Cleanup ==========

clean-graph:
	@echo "Cleaning graph JSON files..."
	@rm -f $(GRAPH_DIR)/*.json
	@echo "✓ Graph JSON files cleaned"

clean-vector:
	@echo "Cleaning embedding JSON files..."
	@rm -rf $(VECTOR_DIR)/company/*.json
	@rm -rf $(VECTOR_DIR)/standards/*.json
	@rm -rf $(VECTOR_DIR)/aiid/*.json
	@echo "✓ Embedding JSON files cleaned"

clean-faiss:
	@echo "Cleaning FAISS indexes..."
	@rm -rf $(BUILDING_DB_DIR)/faiss/company/*.index
	@rm -rf $(BUILDING_DB_DIR)/faiss/company/*.pkl
	@rm -rf $(BUILDING_DB_DIR)/faiss/company/*.json
	@rm -rf $(BUILDING_DB_DIR)/faiss/standards/*.index
	@rm -rf $(BUILDING_DB_DIR)/faiss/standards/*.pkl
	@rm -rf $(BUILDING_DB_DIR)/faiss/standards/*.json
	@rm -rf $(BUILDING_DB_DIR)/faiss/aiid/*.index
	@rm -rf $(BUILDING_DB_DIR)/faiss/aiid/*.pkl
	@rm -rf $(BUILDING_DB_DIR)/faiss/aiid/*.json
	@echo "✓ FAISS indexes cleaned"

clean: clean-graph clean-vector clean-faiss
	@echo ""
	@echo "✓ Cleanup complete!"
	@echo "Note: Neo4j database must be cleared manually if needed"

# Clean virtual environment
clean-venv:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "✓ Virtual environment removed"

# ========== Quick Commands ==========

# Quick rebuild: Clean and rebuild everything
rebuild: clean complete

# Quick rebuild without Neo4j
rebuild-faiss: clean-graph clean-vector clean-faiss process-graph process-vector build-faiss

# Rebuild Neo4j only (assumes graph JSONs exist)
rebuild-neo4j: build-neo4j add-embeddings link-relationships

# ========== Web Application ==========

# Start web application (API server + Frontend)
start-web-app: check-env
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Starting Compliance RAG Web Application..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@if [ -d "$(VENV_DIR)" ]; then \
		if [ -x "$(ROOT_DIR)/$(VENV_DIR)/bin/python3" ]; then \
			VENV_PYTHON="$(ROOT_DIR)/$(VENV_DIR)/bin/python3"; \
		elif [ -x "$(ROOT_DIR)/$(VENV_DIR)/bin/python" ]; then \
			VENV_PYTHON="$(ROOT_DIR)/$(VENV_DIR)/bin/python"; \
		else \
			VENV_PYTHON="$(PYTHON_CMD)"; \
		fi; \
	else \
		echo "⚠ Warning: Virtual environment not found. Creating one..."; \
		$(MAKE) venv; \
		VENV_PYTHON="$(ROOT_DIR)/$(VENV_DIR)/bin/python3"; \
	fi; \
	echo "Checking dependencies..."; \
	if ! "$$VENV_PYTHON" -c "import flask" >/dev/null 2>&1; then \
		echo "⚠ Warning: Dependencies not installed. Installing..."; \
		cd "$(ROOT_DIR)" && "$$VENV_PYTHON" -m pip install -r requirements.txt || exit 1; \
		echo "✓ Dependencies installed"; \
	fi; \
	echo "Using Python: $$VENV_PYTHON"; \
	echo ""; \
	echo "Starting servers:"; \
	echo "  • Backend API: http://localhost:5000"; \
	echo "  • Frontend UI: http://localhost:8000"; \
	echo ""; \
	echo "Press Ctrl+C to stop both servers"; \
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	echo ""; \
	cd "$(SEARCHING_DIR)" && PATH="$(ROOT_DIR)/$(VENV_DIR)/bin:$$PATH" "$$VENV_PYTHON" start_server.py & \
	BACKEND_PID=$$!; \
	sleep 2; \
	cd "$(ROOT_DIR)/$(FRONTEND_DIR)" && "$$VENV_PYTHON" start_server.py & \
	FRONTEND_PID=$$!; \
	trap "kill $$BACKEND_PID $$FRONTEND_PID 2>/dev/null; exit" INT TERM; \
	wait $$BACKEND_PID $$FRONTEND_PID

# Stop web application (if running in background)
stop-web-app:
	@echo "Stopping web application..."
	@pkill -f "backend/searching/start_server.py" || pkill -f "frontend/start_server.py" || pkill -f "api_server.py" || echo "No web app process found"
	@echo "✓ Web application stopped"

# ========== Agentic System ==========

# Run agentic system interactively
run-agent: check-env
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Starting Compliance Agent..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@if [ -d "$(VENV_DIR)" ]; then \
		if [ -x "$(ROOT_DIR)/$(VENV_DIR)/bin/python3" ]; then \
			VENV_PYTHON="$(ROOT_DIR)/$(VENV_DIR)/bin/python3"; \
		elif [ -x "$(ROOT_DIR)/$(VENV_DIR)/bin/python" ]; then \
			VENV_PYTHON="$(ROOT_DIR)/$(VENV_DIR)/bin/python"; \
		else \
			VENV_PYTHON="$(PYTHON_CMD)"; \
		fi; \
	else \
		echo "⚠ Warning: Virtual environment not found. Creating one..."; \
		$(MAKE) venv; \
		VENV_PYTHON="$(ROOT_DIR)/$(VENV_DIR)/bin/python3"; \
	fi; \
	echo "Using Python: $$VENV_PYTHON"; \
	echo ""; \
	cd "$(ROOT_DIR)" && "$$VENV_PYTHON" backend/agentic/run_agent.py

