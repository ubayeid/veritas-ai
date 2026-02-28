"""
Flask API Server for Compliance RAG Chatbot
Provides REST API endpoints for the chatbot interface.
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Optional, List
import traceback

# Add current directory to path for imports
from ..engines.query_engine import VectorQueryEngine

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize query engines
BASE_DIR = Path(__file__).parent.parent.parent
query_engine = None
graph_engine = None
hybrid_engine = None

def init_query_engine():
    """Initialize the vector query engine."""
    global query_engine
    if query_engine is not None:
        return True
    try:
        query_engine = VectorQueryEngine(str(BASE_DIR))
        print(f"Vector query engine initialized with base directory: {BASE_DIR}")
        return True
    except Exception as e:
        print(f"Error initializing vector query engine: {str(e)}")
        traceback.print_exc()
        return False

def init_graph_engine():
    """Initialize the graph query engine."""
    global graph_engine
    if graph_engine is not None:
        return True
    try:
        from ..engines.graph_query_engine import GraphQueryEngine
        graph_engine = GraphQueryEngine(str(BASE_DIR))
        print(f"Graph query engine initialized")
        return True
    except Exception as e:
        print(f"Error initializing graph engine: {str(e)}")
        traceback.print_exc()
        return False

def init_hybrid_engine():
    """Initialize the hybrid query engine."""
    global hybrid_engine
    if hybrid_engine is not None:
        return True
    try:
        from ..engines.hybrid_query_engine import HybridQueryEngine
        hybrid_engine = HybridQueryEngine(str(BASE_DIR))
        print(f"Hybrid query engine initialized")
        return True
    except Exception as e:
        print(f"Error initializing hybrid engine: {str(e)}")
        traceback.print_exc()
        return False

@app.before_request
def ensure_query_engine():
    """Ensure query engines are initialized before handling requests."""
    if query_engine is None:
        init_query_engine()


@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API information."""
    return jsonify({
        'name': 'Compliance RAG Chatbot API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'databases': '/api/databases',
            'query': '/api/query (POST) - Unified endpoint: vector/graph/hybrid (specify mode)',
            'vector_query': '/api/vector_query (POST) - Vector search only (legacy)',
            'graph_query': '/api/graph_query (POST) - Graph search only (legacy)',
            'hybrid_query': '/api/hybrid_query (POST) - Hybrid search (legacy)',
            'search': '/api/search (POST) - Vector search without answer'
        },
        'status': 'running'
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'query_engine_loaded': query_engine is not None
    })


@app.route('/api/databases', methods=['GET'])
def get_databases():
    """Get list of available databases."""
    if query_engine is None:
        return jsonify({'error': 'Query engine not initialized'}), 500
    
    databases = []
    for db_name, db_info in query_engine.databases.items():
        databases.append({
            'name': db_name,
            'loaded': db_info['loaded'],
            'index_name': db_info['index_name']
        })
    
    return jsonify({'databases': databases})


@app.route('/api/query', methods=['POST'])
def query():
    """
    Unified query endpoint supporting vector, graph, and hybrid modes.
    
    Expected JSON body:
    {
        "query": "user query string",
        "mode": "vector" | "graph" | "hybrid" (default: "vector"),
        "db_names": ["company", "aiid", "standards"] or null (vector mode only),
        "top_k": 10,
        "rerank": true,
        "generate_answer": true,
        "similarity_threshold": 0.0 (vector mode only)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing required field: query'}), 400
        
        query_text = data['query']
        mode = data.get('mode', 'vector').lower()
        top_k = data.get('top_k', None)
        rerank = data.get('rerank', True)
        generate_answer = data.get('generate_answer', True)
        
        # Route to appropriate engine based on mode
        if mode == 'vector':
            if query_engine is None:
                init_query_engine()
            if query_engine is None:
                return jsonify({'error': 'Vector query engine not initialized'}), 500
            
            db_names = data.get('db_names', None)
            similarity_threshold = data.get('similarity_threshold', 0.0)
            
            result = query_engine.query(
                query=query_text,
                db_names=db_names,
                top_k=top_k,
                rerank=rerank,
                generate_answer=generate_answer,
                similarity_threshold=similarity_threshold
            )
            
            # Format vector results
            formatted_results = []
            for res in result['results']:
                formatted_results.append({
                    'database': res.get('database', ''),
                    'similarity': res.get('similarity', 0.0),
                    'text': res.get('text', ''),
                    'source_name': res.get('source_name', ''),
                    'source_file': res.get('source_file', ''),
                    'chunk_id': res.get('chunk_id', '')
                })
            
            return jsonify({
                'success': True,
                'mode': 'vector',
                'query': query_text,
                'answer': result.get('answer'),
                'results': formatted_results,
                'num_results': result.get('num_results', 0)
            })
            
        elif mode == 'graph':
            if graph_engine is None:
                init_graph_engine()
            if graph_engine is None:
                return jsonify({'error': 'Graph query engine not initialized'}), 500
            
            # Graph search
            graph_results = graph_engine.search(query_text, top_k=top_k, score_results=True)
            
            # Generate answer if requested
            answer = None
            if generate_answer and graph_results:
                try:
                    from ..engines.query_engine import generate_answer_with_attribution
                    answer = generate_answer_with_attribution(
                        query=query_text,
                        results=graph_results[:top_k or 8],
                        max_tokens=3000
                    )
                except Exception as e:
                    print(f"Answer generation failed: {e}")
            
            # Format graph results
            formatted_results = []
            for res in graph_results:
                formatted_results.append({
                    'id': res.get('id', ''),
                    'text': res.get('text', res.get('description', '')),
                    'type': res.get('type', ''),
                    'source': res.get('source', 'graph_traversal'),
                    'similarity': res.get('similarity', 0.0),
                    'article_id': res.get('article_id', ''),
                    'document_name': res.get('document_name', '')
                })
            
            return jsonify({
                'success': True,
                'mode': 'graph',
                'query': query_text,
                'answer': answer,
                'results': formatted_results,
                'num_results': len(formatted_results)
            })
            
        elif mode == 'hybrid':
            if hybrid_engine is None:
                init_hybrid_engine()
            if hybrid_engine is None:
                return jsonify({'error': 'Hybrid query engine not initialized'}), 500
            
            result = hybrid_engine.hybrid_query(
                query=query_text,
                top_k=top_k,
                rerank=rerank,
                generate_answer=generate_answer
            )
            
            # Format hybrid results
            formatted_results = []
            for res in result.get('results', []):
                formatted_results.append({
                    'text': res.get('text', res.get('description', '')),
                    'source': res.get('source', 'hybrid'),
                    'similarity': res.get('similarity', res.get('rrf_score', 0.0)),
                    'database': res.get('database', ''),
                    'type': res.get('type', '')
                })
            
            return jsonify({
                'success': True,
                'mode': 'hybrid',
                'query': query_text,
                'answer': result.get('answer'),
                'results': formatted_results,
                'num_results': result.get('num_results', 0),
                'sources_used': result.get('sources_used', {})
            })
        else:
            return jsonify({'error': f'Invalid mode: {mode}. Must be "vector", "graph", or "hybrid"'}), 400
    
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/search', methods=['POST'])
def search():
    """
    Search endpoint (without answer generation).
    
    Expected JSON body:
    {
        "query": "user query string",
        "db_names": ["company", "aiid", "standards"] or null,
        "top_k": 10,
        "similarity_threshold": 0.0
    }
    """
    if query_engine is None:
        return jsonify({'error': 'Query engine not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing required field: query'}), 400
        
        query_text = data['query']
        db_names = data.get('db_names', None)
        top_k = data.get('top_k', None)  # None triggers adaptive top_k
        similarity_threshold = data.get('similarity_threshold', 0.0)
        
        # Execute search
        results = query_engine.search(
            query=query_text,
            db_names=db_names,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Format results for frontend
        formatted_results = []
        for res in results:
            formatted_results.append({
                'database': res['database'],
                'similarity': res['similarity'],
                'text': res['text'],
                'source_name': res['source_name'],
                'source_file': res.get('source_file', ''),
                'chunk_id': res.get('chunk_id', '')
            })
        
        return jsonify({
            'success': True,
            'query': query_text,
            'results': formatted_results,
            'num_results': len(formatted_results)
        })
    
    except Exception as e:
        print(f"Error processing search: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/graph_query', methods=['POST'])
def graph_query():
    """
    Graph query endpoint using Neo4j graph traversal (standalone).
    Use /api/query with mode='graph' for unified interface.
    
    Expected JSON body:
    {
        "query": "user query string",
        "top_k": 10,
        "rerank": true,
        "generate_answer": true
    }
    """
    if graph_engine is None:
        init_graph_engine()
    
    if graph_engine is None:
        return jsonify({'error': 'Graph query engine not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing required field: query'}), 400
        
        query_text = data['query']
        top_k = data.get('top_k', None)
        rerank = data.get('rerank', True)
        generate_answer = data.get('generate_answer', True)
        
        # Execute graph search
        graph_results = graph_engine.search(query_text, top_k=top_k, score_results=True)
        
        # Generate answer if requested
        answer = None
        if generate_answer and graph_results:
            try:
                from ..engines.query_engine import generate_answer_with_attribution
                answer = generate_answer_with_attribution(
                    query=query_text,
                    results=graph_results[:top_k or 8],
                    max_tokens=3000
                )
            except Exception as e:
                print(f"Answer generation failed: {e}")
        
        # Format results for frontend
        formatted_results = []
        for res in graph_results:
            formatted_results.append({
                'id': res.get('id', ''),
                'text': res.get('text', res.get('description', '')),
                'title': res.get('title', ''),
                'type': res.get('type', 'unknown'),
                'source': res.get('source', 'graph_traversal'),
                'similarity': res.get('similarity', 0.0),
                'database': res.get('document_name', 'Graph'),
                'article_id': res.get('article_id', ''),
                'risk_type': res.get('risk_type', ''),
                'violated_articles': res.get('violated_articles', [])
            })
        
        return jsonify({
            'success': True,
            'query': query_text,
            'results': formatted_results,
            'answer': answer,
            'num_results': len(formatted_results),
            'sources_used': {'graph': True}
        })
    
    except Exception as e:
        print(f"Error processing graph query: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/hybrid_query', methods=['POST'])
def hybrid_query():
    """
    Hybrid query endpoint combining FAISS vector search + Neo4j graph traversal.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    
    Expected JSON body:
    {
        "query": "user query string",
        "top_k": 10,
        "rerank": true,
        "generate_answer": true,
        "use_faiss": true,
        "use_graph_traversal": true,
        "rrf_k": 60
    }
    """
    if hybrid_engine is None:
        init_hybrid_engine()
    
    if hybrid_engine is None:
        return jsonify({'error': 'Hybrid query engine not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing required field: query'}), 400
        
        query_text = data['query']
        top_k = data.get('top_k', None)  # None triggers adaptive top_k
        rerank = data.get('rerank', True)
        generate_answer = data.get('generate_answer', True)
        use_faiss = data.get('use_faiss', True)
        use_graph_traversal = data.get('use_graph_traversal', True)
        rrf_k = data.get('rrf_k', None)  # Optional, uses default from .env if not provided
        
        # Execute hybrid query
        result = hybrid_engine.hybrid_query(
            query=query_text,
            top_k=top_k,
            rerank=rerank,
            generate_answer=generate_answer
        )
        
        # Format results for frontend
        formatted_results = []
        for res in result['results']:
            formatted_results.append({
                'id': res.get('id', ''),
                'text': res.get('text', res.get('description', '')),
                'title': res.get('title', ''),
                'similarity': res.get('similarity', res.get('score', 0.0)),
                'source': res.get('source', 'hybrid'),
                'sources': res.get('sources', []),
                'type': res.get('type', 'unknown'),
                'document_name': res.get('document_name', ''),
                'article_id': res.get('article_id', ''),
                'risk_type': res.get('risk_type', '')
            })
        
        return jsonify({
            'success': True,
            'query': query_text,
            'answer': result['answer'],
            'results': formatted_results,
            'num_results': result['num_results'],
            'query_types': result['query_types'],
            'sources_used': result['sources_used']
        })
    
    except Exception as e:
        print(f"Error processing hybrid query: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Initialize query engines (lazy loading - will initialize on first use)
    print("Initializing query engines...")
    init_query_engine()  # Vector engine
    init_graph_engine()  # Graph engine (may fail if Neo4j not available)
    init_hybrid_engine()  # Hybrid engine (may fail if Neo4j not available)
    
    # Run Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\n{'='*80}")
    print("COMPLIANCE RAG CHATBOT API SERVER")
    print(f"{'='*80}")
    print(f"Server starting on http://localhost:{port}")
    print(f"Debug mode: {debug}")
    print(f"\nAvailable modes:")
    print(f"  - Vector: {'✓' if query_engine else '✗'}")
    print(f"  - Graph: {'✓' if graph_engine else '✗'}")
    print(f"  - Hybrid: {'✓' if hybrid_engine else '✗'}")
    print(f"{'='*80}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

