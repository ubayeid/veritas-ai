"""
Startup script for the Flask API server.
Can be used as an alternative entry point.
"""

import sys
from pathlib import Path

# Add backend/searching to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from api_server import app, init_query_engine

if __name__ == '__main__':
    # Initialize query engine
    if not init_query_engine():
        print("Failed to initialize query engine. Exiting.")
        sys.exit(1)
    
    # Import here to avoid circular imports
    import os
    
    # Run Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\n{'='*80}")
    print("COMPLIANCE RAG CHATBOT API SERVER")
    print(f"{'='*80}")
    print(f"Server starting on http://localhost:{port}")
    print(f"Debug mode: {debug}")
    print(f"{'='*80}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

