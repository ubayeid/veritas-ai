"""
Simple HTTP server for serving the frontend static files.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent.parent / 'static'), **kwargs)

if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent / 'static')
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"\n{'='*80}")
        print("FRONTEND SERVER")
        print(f"{'='*80}")
        print(f"Frontend available at: http://localhost:{PORT}")
        print(f"Make sure the API server is running on http://localhost:5000")
        print(f"{'='*80}\n")
        httpd.serve_forever()
