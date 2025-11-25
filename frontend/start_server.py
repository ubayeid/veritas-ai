"""
Simple HTTP server to serve the frontend files.
Run this script from the frontend directory.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def main():
    # Change to the directory where this script is located
    os.chdir(Path(__file__).parent)
    
    Handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\n{'='*80}")
        print("FRONTEND SERVER")
        print(f"{'='*80}")
        print(f"Server running on http://localhost:{PORT}")
        print(f"Serving files from: {os.getcwd()}")
        print(f"\nOpen your browser and navigate to: http://localhost:{PORT}")
        print(f"Press Ctrl+C to stop the server")
        print(f"{'='*80}\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")

if __name__ == "__main__":
    main()

